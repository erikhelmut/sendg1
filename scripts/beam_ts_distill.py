#!/usr/bin/env python
"""Distil the beam teacher into the deployable student (DAgger).

    python scripts/beam_ts_distill.py --teacher runs/beam_ts_teacher/<run>/model_14999.pt \
        --init-max-level 6 --max-iterations 3000

The student drives (after the teacher-control fade), the frozen teacher labels
every visited state, and the student regresses onto the labels. Envs start
spread uniformly over levels 0..``--init-max-level`` -- the levels the teacher
masters -- and the per-env curriculum then moves them on the student's own
performance.
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
  ap.add_argument("--teacher", required=True, help="teacher PPO checkpoint (.pt)")
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--num-envs", type=int, default=4096)
  ap.add_argument("--max-iterations", type=int, default=3000)
  ap.add_argument("--init-max-level", type=int, default=0)
  ap.add_argument("--fade-iterations", type=int, default=None)
  ap.add_argument("--logger", default="wandb", choices=["wandb", "tensorboard"])
  ap.add_argument("--run-name", default=None)
  ap.add_argument("--log-root", default=str(REPO / "runs"))
  args = ap.parse_args()

  import torch
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
  from mjlab.utils.torch import configure_torch_backends

  import sendg1.tasks  # noqa: F401
  from sendg1.common.beam_ts.mdp import set_levels
  from sendg1.rl.dagger import DistillationRunner
  from sendg1.tasks.beam_ts import STUDENT_TASK

  configure_torch_backends()
  device = "cuda:0"
  env_cfg = load_env_cfg(STUDENT_TASK)
  env_cfg.scene.num_envs = args.num_envs
  env_cfg.seed = args.seed
  rl_cfg = load_rl_cfg(STUDENT_TASK)
  rl_cfg.seed = args.seed
  rl_cfg.max_iterations = args.max_iterations
  rl_cfg.logger = args.logger
  rl_cfg.run_name = args.run_name or f"seed{args.seed}"
  if args.fade_iterations is not None:
    rl_cfg.algorithm.fade_iterations = args.fade_iterations

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  if args.init_max_level > 0:
    lv = torch.randint(0, args.init_max_level + 1, (env.num_envs,), device=env.device)
    set_levels(env, lv)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=rl_cfg.clip_actions)

  # Fresh episodes on the assigned levels, before a checkpoint restores the
  # step counter (a reset with counter > 0 would score the empty episodes).
  wrapped.reset()

  stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  log_dir = Path(args.log_root) / rl_cfg.experiment_name / f"{stamp}_{rl_cfg.run_name}"
  os.makedirs(log_dir, exist_ok=True)
  (log_dir / "teacher.txt").write_text(str(Path(args.teacher).resolve()) + "\n")

  runner = DistillationRunner(wrapped, asdict(rl_cfg), str(log_dir), device)
  runner.load_teacher(args.teacher, map_location=device)
  runner.learn(rl_cfg.max_iterations)
  env.close()


if __name__ == "__main__":
  main()
