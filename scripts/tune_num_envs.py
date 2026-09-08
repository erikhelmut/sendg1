#!/usr/bin/env python
"""Find the largest num_envs that fits on this GPU for each task.

    python scripts/tune_num_envs.py --task Sendg1-Velocity-Rough-G1-Blind

Comparisons are baseline-vs-tactile *within* a task, and num_envs is fixed per
task, so both conditions see the same command curriculum -- which is why a
per-task value is safe even though mjlab's command curriculum keys off a global
step counter rather than steps-per-env.
"""

from __future__ import annotations

import argparse


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
  for n in sorted(args.candidates):
    torch.cuda.empty_cache()
    try:
      cfg = load_env_cfg(args.task)
      cfg.scene.num_envs = n
      env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
      env.reset()
      for _ in range(args.steps):
        env.step(torch.zeros(n, env.action_manager.total_action_dim, device="cuda:0"))
      peak = torch.cuda.max_memory_allocated() / 2**30
      print(f"  num_envs={n:6d}  OK   peak={peak:.2f} GiB")
      best = n
      env.close()
      del env
      torch.cuda.reset_peak_memory_stats()
    except Exception as exc:  # noqa: BLE001
      print(f"  num_envs={n:6d}  FAILED  {type(exc).__name__}: {str(exc)[:120]}")
      break

  print(f"\nlargest working num_envs for {args.task}: {best}")
  print("Set it in DEFAULT_NUM_ENVS (src/sendg1/tasks/velocity/env_cfg.py) or pass "
        "--env.scene.num-envs at train time.")


if __name__ == "__main__":
  main()
