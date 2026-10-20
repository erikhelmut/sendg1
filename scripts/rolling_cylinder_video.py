#!/usr/bin/env python
"""Render one rolling-cylinder episode of a policy, headless.

    python scripts/rolling_cylinder_video.py --checkpoint <pt>
    python scripts/rolling_cylinder_video.py --task G1-RollingCylinder-Blind --checkpoint <pt>

One env, one full episode (or ``--seconds``), written as an MP4 through
``sendg1.video.EvalVideo`` (studio theme, caption, foot-tactile card). The task
has no command, so the caption shows the cylinder's rolling speed and drift
from where it started. If the episode ends in a fall the clip ends on the
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
  ap.add_argument("--task", default="G1-RollingCylinder-Touch")
  ap.add_argument("--label", default="", help="caption title, e.g. the run name (default: the task)")
  ap.add_argument("--seconds", type=float, default=None)
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--out", default=None)
  ap.add_argument("--theme", default="studio")
  args = ap.parse_args()

  import torch
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

  import sendg1.tasks  # noqa: F401
  from sendg1.common.rolling_cylinder.mdp import CylinderState
  from sendg1.video import EvalVideo, prepare_video_cfg

  cfg = load_env_cfg(args.task)
  cfg.scene.num_envs = 1
  cfg.seed = args.seed
  prepare_video_cfg(cfg, 1280, 720, args.theme)
  env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0", render_mode="rgb_array")

  rl_cfg = load_rl_cfg(args.task)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=rl_cfg.clip_actions)
  runner = MjlabOnPolicyRunner(wrapped, asdict(rl_cfg), device="cuda:0")
  runner.load(args.checkpoint, load_cfg={"actor": True}, strict=True, map_location="cuda:0")
  policy = runner.get_inference_policy(device="cuda:0")
  video = EvalVideo(env, rl_cfg, title=args.label or args.task, theme=args.theme)
  tm = env.termination_manager

  seconds = args.seconds or cfg.episode_length_s
  reason, t = "", 0.0
  with torch.inference_mode():
    obs, _ = wrapped.reset()
    policy.reset(torch.ones(1, dtype=torch.bool, device=env.device))
    st = CylinderState.get(env)
    video.new_episode()
    for _ in range(int(seconds / env.step_dt)):
      obs, _, dones, _ = wrapped.step(policy(obs))
      if bool(dones[0]):
        fired = [n for n in tm.active_terms if n != "time_out" and bool(tm.get_term(n)[0])]
        reason = fired[0] if fired else ""
        break
      t += env.step_dt
      v, _ = st.velocity()
      drift = st.drift()[0]
      video.capture([
        f"balance on a rolling cylinder   t {t:4.1f} s",
        f"cylinder {float(v[0]):+.2f} m/s   drift {float(drift[0]):+.2f} m",
      ])
  if reason:
    video.hold(0.8, ["balance on a rolling cylinder", f"FELL at {t:.1f} s: {reason}"])
  env.close()
  name = "touch" if "Touch" in args.task else "blind"
  out = Path(args.out or REPO / "media" / "rolling_cylinder" / f"{name}.mp4")
  video.save(out, qp=28)
  print(f"[video] wrote {out} ({len(video.frames)} frames, {t:.1f} s, "
        f"{'fell: ' + reason if reason else 'no fall'})")


if __name__ == "__main__":
  main()
