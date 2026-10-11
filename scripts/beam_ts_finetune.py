#!/usr/bin/env python
"""Continue PPO training of the beam teacher from a checkpoint, with envs
spread over all width levels.

    python scripts/beam_ts_finetune.py --checkpoint runs/beam_ts_teacher/<run>/model_11999.pt \
        --max-iterations 2000

Used after the curriculum gained level replay (``mdp.REPLAY_SHARE``): the first
run had drained the wide levels and forgotten them. Loads actor, critic and
optimizer, keeps the iteration count, and writes to a new run directory.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--checkpoint", required=True)
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--num-envs", type=int, default=4096)
  ap.add_argument("--max-iterations", type=int, default=2000)
  ap.add_argument("--logger", default="wandb", choices=["wandb", "tensorboard"])
  ap.add_argument("--run-name", default=None)
  ap.add_argument("--log-root", default=str(REPO / "runs"))
  args = ap.parse_args()

  import torch
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
  from mjlab.utils.torch import configure_torch_backends

  import sendg1.tasks  # noqa: F401
  from sendg1.common.beam_ts.mdp import set_levels
  from sendg1.tasks.beam_ts import TEACHER_TASK

  configure_torch_backends()
  device = "cuda:0"
  env_cfg = load_env_cfg(TEACHER_TASK)
  env_cfg.scene.num_envs = args.num_envs
  env_cfg.seed = args.seed
  rl_cfg = load_rl_cfg(TEACHER_TASK)
  rl_cfg.seed = args.seed
  rl_cfg.logger = args.logger
  rl_cfg.run_name = args.run_name or f"seed{args.seed}_finetune"

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  geo_levels = env.scene.terrain.terrain_origins.shape[0]
  set_levels(env, torch.randint(0, geo_levels, (env.num_envs,), device=env.device))
  wrapped = RslRlVecEnvWrapper(env, clip_actions=rl_cfg.clip_actions)

  # Fresh episodes on the assigned levels, before a checkpoint restores the
  # step counter (a reset with counter > 0 would score the empty episodes).
  wrapped.reset()

  stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  log_dir = Path(args.log_root) / rl_cfg.experiment_name / f"{stamp}_{rl_cfg.run_name}"
  os.makedirs(log_dir, exist_ok=True)
  (log_dir / "parent.txt").write_text(str(Path(args.checkpoint).resolve()) + "\n")

  runner = MjlabOnPolicyRunner(wrapped, asdict(rl_cfg), str(log_dir), device)
  runner.load(args.checkpoint, map_location=device)
  runner.learn(args.max_iterations)
  env.close()


if __name__ == "__main__":
  main()
