#!/usr/bin/env python
"""Initialise a policy from another task's actor, then train.

    python scripts/warm_start.py \
      --task Sendg1-Velocity-Rough-G1-Blind \
      --from-checkpoint runs/velocity_flat__blind/<run>/model_29999.pt

Why this exists
---------------
Training the rough task from scratch converged to a local optimum where the
robot stands still instead of walking: measured lin_vel_error 0.557 against
0.515 for a robot that never moves, and strayed_frac 0.000 across 2,071
episodes. Walking on rough terrain risks a fall that ends the episode and
forfeits all future return, while `upright` + `pose` pay out safely for
standing, so standing wins.

The flat-trained policy does not have that problem: dropped on the same terrain
it walks (lin_vel_error 0.232) but falls 68% of the time. "Walks but falls" is a
much better basin to start from than "stands still" -- learning robustness is
easier than escaping a standing attractor.

mjlab's own `train --agent.resume` cannot do this: it calls `runner.load(path)`
with no `load_cfg`, which also loads the critic, and the critic dimensions
differ between tasks (140 on flat vs 327 on rough, because the rough critic
takes the height scan). Loading the ACTOR only is what makes the transfer legal
-- and it is legal precisely because the actor observation space was kept
identical at 96 dims on both terrains.

For the ablation
----------------
Initialisation is part of the experimental condition. If the blind rough policy
is warm-started, the tactile rough policy must be warm-started the same way --
from its own flat counterpart -- or the comparison is confounded by the starting
point rather than the observations. Record the source checkpoint (this script
writes it into params/warm_start.yaml) for every run.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--task", required=True)
  ap.add_argument("--from-checkpoint", required=True)
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--max-iterations", type=int, default=30_000)
  ap.add_argument("--run-name", default=None)
  ap.add_argument("--log-root", default=str(REPO / "runs"))
  ap.add_argument("--num-envs", type=int, default=None)
  ap.add_argument(
    "--action-std",
    type=float,
    default=None,
    help="Override the loaded policy's action std. The source policy arrives "
    "confident (flat ended at ~0.42); raising it buys exploration on the new "
    "terrain. Leave unset to keep whatever the checkpoint had.",
  )
  args = ap.parse_args()

  import torch
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
  from mjlab.utils.os import dump_yaml
  from mjlab.utils.torch import configure_torch_backends

  import sendg1.tasks  # noqa: F401
  from sendg1 import MJLAB_COMMIT, MJLAB_VERSION

  configure_torch_backends()
  device = "cuda:0" if torch.cuda.is_available() else "cpu"

  ckpt = Path(args.from_checkpoint).resolve()
  if not ckpt.is_file():
    raise SystemExit(f"checkpoint not found: {ckpt}")

  env_cfg = load_env_cfg(args.task)
  rl_cfg = load_rl_cfg(args.task)
  rl_cfg.seed = args.seed
  rl_cfg.max_iterations = args.max_iterations
  rl_cfg.run_name = args.run_name or f"seed{args.seed}_warmstart"
  if args.num_envs is not None:
    env_cfg.scene.num_envs = args.num_envs
  env_cfg.seed = args.seed

  log_dir = (
    Path(args.log_root)
    / rl_cfg.experiment_name
    / f"{datetime.now():%Y-%m-%d_%H-%M-%S}_{rl_cfg.run_name}"
  )
  print(f"[warm-start] source actor : {ckpt}")
  print(f"[warm-start] target task  : {args.task}")
  print(f"[warm-start] log dir      : {log_dir}")

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=rl_cfg.clip_actions)

  runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
  runner = runner_cls(wrapped, asdict(rl_cfg), str(log_dir), device)

  # Actor only. The critic is task-specific (different observation groups) and
  # is trained from scratch; strict=True so a genuine shape mismatch in the
  # actor is an error rather than a silently skipped layer.
  runner.load(str(ckpt), load_cfg={"actor": True}, strict=True, map_location=device)
  # Count iterations from zero: this is a new run, not a continuation.
  runner.current_learning_iteration = 0

  if args.action_std is not None:
    with torch.no_grad():
      dist = runner.alg.actor.distribution
      for name in ("std", "log_std", "_std", "_log_std"):
        p = getattr(dist, name, None)
        if isinstance(p, torch.nn.Parameter):
          p.fill_(
            args.action_std if "log" not in name else float(torch.tensor(args.action_std).log())
          )
          print(f"[warm-start] action std -> {args.action_std} (param '{name}')")
          break
      else:
        print("[warm-start] WARNING: could not locate an action-std parameter")

  dump_yaml(log_dir / "params" / "env.yaml", asdict(env_cfg))
  dump_yaml(log_dir / "params" / "agent.yaml", asdict(rl_cfg))
  dump_yaml(
    log_dir / "params" / "warm_start.yaml",
    {
      "source_checkpoint": str(ckpt),
      "target_task": args.task,
      "loaded": "actor_only",
      "action_std_override": args.action_std,
      "mjlab_version": MJLAB_VERSION,
      "mjlab_commit": MJLAB_COMMIT,
    },
  )

  runner.add_git_repo_to_log(__file__)
  runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)
  env.close()


if __name__ == "__main__":
  main()
