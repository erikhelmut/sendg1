#!/usr/bin/env python
"""Find the largest num_envs that fits on this GPU for each task.

    python scripts/tune_num_envs.py --task Sendg1-Velocity-Rough-G1-Blind

mjlab's command curriculum advances on common_step_counter, which increments
once per env.step() regardless of batch size, so its stages land at fixed
iteration counts and do not shift with num_envs. Varying num_envs per task is
therefore safe; it changes how much data the policy has seen by each stage, not
when the stages fire.
"""

from __future__ import annotations

import argparse
import time


def main() -> None:
  import torch
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.registry import load_env_cfg

  import sendg1.tasks  # noqa: F401

  ap = argparse.ArgumentParser()
  ap.add_argument("--task", required=True)
  ap.add_argument("--candidates", nargs="+", type=int,
                  default=[1024, 2048, 4096, 8192, 16384])
  ap.add_argument("--steps", type=int, default=20)
  args = ap.parse_args()

  best = None
  results: list[tuple[int, float, float]] = []
  for n in sorted(args.candidates):
    try:
      cfg = load_env_cfg(args.task)
      cfg.scene.num_envs = n
      env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
      env.reset()
      act = torch.zeros(n, env.action_manager.total_action_dim, device="cuda:0")
      for _ in range(5):  # warm up kernels before timing
        env.step(act)
      torch.cuda.synchronize()
      t0 = time.perf_counter()
      for _ in range(args.steps):
        env.step(act)
      torch.cuda.synchronize()
      dt = time.perf_counter() - t0

      # Whole-device usage. torch.cuda.max_memory_allocated only sees torch's
      # caching allocator, and MuJoCo-Warp allocates outside it -- it reports
      # ~0.1 GiB for a scene actually using several, which is useless here.
      free, total = torch.cuda.mem_get_info()
      used_gib = (total - free) / 2**30
      sps = n * args.steps / dt

      print(f"  num_envs={n:6d}  OK   device_mem={used_gib:5.2f} GiB  "
            f"{sps:9,.0f} steps/s")
      results.append((n, used_gib, sps))
      best = n
      env.close()
      del env, act
      torch.cuda.empty_cache()
    except Exception as exc:  # noqa: BLE001
      print(f"  num_envs={n:6d}  FAILED  {type(exc).__name__}: {str(exc)[:120]}")
      break

  if results:
    fastest = max(results, key=lambda r: r[2])
    print(f"\nlargest that fits: {best}")
    print(f"fastest:           {fastest[0]} at {fastest[2]:,.0f} steps/s "
          f"({fastest[1]:.2f} GiB)")
    print("More envs is not automatically better: throughput saturates once the "
          "GPU is busy, while larger batches change PPO's effective batch size.")
    print("Set DEFAULT_NUM_ENVS in src/sendg1/tasks/velocity/env_cfg.py, or pass "
          "--env.scene.num-envs at train time.")


if __name__ == "__main__":
  main()
