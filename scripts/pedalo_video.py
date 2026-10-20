#!/usr/bin/env python
"""Render one pedalo episode of a policy, headless.

    python scripts/pedalo_video.py --checkpoint <pt> --speed 0.45
    python scripts/pedalo_video.py --task G1-Pedalo-Sport --checkpoint <pt>

One env, a constant forward command, one full episode (or ``--seconds``),
written as an MP4 through ``sendg1.video.EvalVideo`` (studio theme, caption,
foot-tactile card). If the episode ends in a fall the clip ends on the
terminating step and its caption names the termination.
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
  ap.add_argument("--checkpoint", required=True)
  ap.add_argument("--task", default="G1-Pedalo")
  ap.add_argument("--label", default="", help="caption title, e.g. the run name (default: the task)")
  ap.add_argument("--speed", type=float, default=0.45)
  ap.add_argument("--seconds", type=float, default=None, help="clip length (default: the full episode)")
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--out", default=None)
  ap.add_argument("--width-px", type=int, default=1280)
  ap.add_argument("--height-px", type=int, default=720)
  ap.add_argument("--theme", default="studio", help="studio | gray | mujoco")
  ap.add_argument("--body", default="pelvis", help="robot body the camera tracks (frame centre)")
  ap.add_argument("--distance", type=float, default=3.0)
  ap.add_argument("--elevation", type=float, default=-15.0)
  ap.add_argument("--azimuth", type=float, default=90.0)
  args = ap.parse_args()

  import torch
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

  import sendg1.tasks  # noqa: F401
  from sendg1.common.pedalo.mdp import PedaloState
  from sendg1.video import EvalVideo, prepare_video_cfg

  task = args.task
  cfg = load_env_cfg(task)
  cfg.scene.num_envs = 1
  cfg.seed = args.seed
  twist = cfg.commands["twist"]
  twist.rel_standing_envs = 0.0
  twist.resampling_time_range = (1e6, 1e6)
  prepare_video_cfg(cfg, args.width_px, args.height_px, args.theme)
  cfg.viewer.body_name = args.body
  cfg.viewer.distance = args.distance
  cfg.viewer.elevation = args.elevation
  cfg.viewer.azimuth = args.azimuth
  env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0", render_mode="rgb_array")

  rl_cfg = load_rl_cfg(task)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=rl_cfg.clip_actions)
  runner = MjlabOnPolicyRunner(wrapped, asdict(rl_cfg), device="cuda:0")
  runner.load(args.checkpoint, load_cfg={"actor": True}, strict=True, map_location="cuda:0")
  policy = runner.get_inference_policy(device="cuda:0")
  video = EvalVideo(env, rl_cfg, title=args.label or task, theme=args.theme)
  tm = env.termination_manager
  pedalo = "Pedalo Sport" if "Sport" in task else "Pedalo Classic"

  seconds = args.seconds or cfg.episode_length_s
  reason, dist = "", 0.0
  with torch.inference_mode():
    obs, _ = wrapped.reset()
    policy.reset(torch.ones(1, dtype=torch.bool, device=env.device))
    term = env.command_manager.get_term("twist")
    st = PedaloState.get(env)
    term.vel_command_b[:, 0] = args.speed
    term.vel_command_b[:, 1:] = 0.0
    obs = wrapped.get_observations()  # with the fixed command
    video.new_episode()
    for _ in range(int(seconds / env.step_dt)):
      term.vel_command_b[:, 0] = args.speed
      term.vel_command_b[:, 1:] = 0.0
      obs, _, dones, _ = wrapped.step(policy(obs))
      if bool(dones[0]):
        fired = [n for n in tm.active_terms if n != "time_out" and bool(tm.get_term(n)[0])]
        reason = fired[0] if fired else ""
        break
      v, _ = st.velocity()
      dist += float(v[0]) * env.step_dt
      video.capture([
        f"{pedalo}   command {args.speed:.2f} m/s",
        f"pedalo {float(v[0]):+.2f} m/s   {dist:.2f} m",
      ])
  if reason:
    video.hold(0.8, [f"{pedalo}   command {args.speed:.2f} m/s", f"FELL: {reason}"])
  env.close()

  name = f"{'sport' if 'Sport' in task else 'classic'}_v{args.speed:.2f}"
  out = Path(args.out or REPO / "media" / "pedalo" / f"{name}.mp4")
  video.save(out, qp=28)
  print(f"[video] wrote {out} ({len(video.frames)} frames, {dist:.2f} m, {'fell: ' + reason if reason else 'no fall'})")


if __name__ == "__main__":
  main()
