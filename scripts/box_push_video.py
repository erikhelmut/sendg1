#!/usr/bin/env python
"""Headless video of box-pushing episodes: random actions or a trained policy.

    # scene check: random actions, a few episodes
    python scripts/box_push_video.py --random --episodes 4

    # a trained policy, on a pinned floor mu and box mass
    python scripts/box_push_video.py --task Sendg1-BoxPush-G1-Blind --seed 0 \
        --floor-mu 0.3 --box-mass 30

Random actions are N(0, 1) per action dimension -- what PPO's freshly
initialised policy (init_std 1.0) emits around a zero mean.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
os.environ.setdefault("MUJOCO_GL", "egl")


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--task", default="Sendg1-BoxPush-G1-Blind")
  ap.add_argument("--random", action="store_true", help="random actions, no checkpoint")
  ap.add_argument("--seed", type=int, default=0, help="training seed of the checkpoint")
  ap.add_argument("--checkpoint", default=None)
  ap.add_argument("--log-root", default=str(REPO / "runs"))
  ap.add_argument("--episodes", type=int, default=3)
  ap.add_argument("--env-seed", type=int, default=1)
  ap.add_argument("--floor-mu", type=float, default=None)
  ap.add_argument("--box-mass", type=float, default=None)
  ap.add_argument("--box-depth", type=float, default=None, help="m, along the push")
  ap.add_argument("--box-width", type=float, default=None)
  ap.add_argument("--box-height", type=float, default=None)
  ap.add_argument("--patches", action="store_true", help="enable low-friction patches")
  ap.add_argument("--out", default=None)
  ap.add_argument("--width", type=int, default=1280)
  ap.add_argument("--height", type=int, default=720)
  ap.add_argument("--theme", default="studio", help="studio | gray | mujoco")
  args = ap.parse_args()

  import torch
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_rl_cfg, load_runner_cls
  from mjlab.utils.os import get_checkpoint_path
  from mjlab.utils.torch import configure_torch_backends

  import sendg1.tasks  # noqa: F401
  from sendg1.common.box_push.mdp import BoxPushState
  from sendg1.eval.box_push import configure_video_view, roll_out_video
  from sendg1.tasks._register import variants
  from sendg1.tasks.box_push.env_cfg import make_g1_box_push_env_cfg
  from sendg1.video import EvalVideo

  configure_torch_backends()
  device = "cuda:0" if torch.cuda.is_available() else "cpu"

  _, obs_set = variants()[args.task]
  env_cfg = make_g1_box_push_env_cfg(
    obs_set, num_envs=1, low_friction_patches=args.patches
  )
  env_cfg.seed = args.env_seed
  env_cfg.observations["proprio"].enable_corruption = False
  configure_video_view(env_cfg, args.width, args.height, args.theme)
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode="rgb_array")
  rl_cfg = load_rl_cfg(args.task)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=rl_cfg.clip_actions)
  video = EvalVideo(env, rl_cfg, title=args.task, theme=args.theme)

  st = BoxPushState.get(env)
  if args.floor_mu is not None:
    st.pins["floor_mu"][:] = args.floor_mu
  for name in ("box_mass", "box_depth", "box_width", "box_height"):
    value = getattr(args, name)
    if value is not None:
      st.pins[name][:] = value

  reset_policy = None
  if args.random:
    dim = env.action_manager.total_action_dim
    act = lambda obs: torch.randn(1, dim, device=device)  # noqa: E731
    label = "random actions"
  else:
    ckpt = (
      Path(args.checkpoint)
      if args.checkpoint
      else get_checkpoint_path(
        Path(args.log_root) / rl_cfg.experiment_name, f".*seed{args.seed}$", "model_.*.pt"
      )
    )
    print(f"[video] checkpoint: {ckpt}")
    runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
    runner = runner_cls(wrapped, asdict(rl_cfg), device=device)
    runner.load(str(ckpt), load_cfg={"actor": True}, strict=True, map_location=device)
    policy = runner.get_inference_policy(device=device)
    act = policy
    reset_policy = getattr(policy, "reset", None)
    label = f"{args.task} {ckpt.parent.name}/{ckpt.name}"

  obs, _ = wrapped.reset()
  frames = []
  for ep in range(args.episodes):
    f, rec, obs = roll_out_video(
      wrapped, act, obs, max_steps=env.max_episode_length + 1,
      label=f"{label}  ep {ep}", video=video, reset_policy=reset_policy,
    )
    frames += f
    print(f"[video] episode {ep}: " + ", ".join(f"{k}={v:.3g}" for k, v in rec.items()))

  out = Path(args.out) if args.out else (
    REPO / "videos" / f"box_push_{'random' if args.random else obs_set}.mp4"
  )
  out.parent.mkdir(parents=True, exist_ok=True)
  video.save(out, frames)
  print(f"[video] wrote {out} ({len(frames)} frames)")
  env.close()


if __name__ == "__main__":
  main()
