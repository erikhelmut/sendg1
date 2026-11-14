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
  ap.add_argument("--width-px", type=int, default=1280)
  ap.add_argument("--height-px", type=int, default=720)
  ap.add_argument("--qp", type=int, default=28, help="H.264 quantiser (lower = better, ~18 is near lossless)")
  ap.add_argument("--no-caption", action="store_true", help="drop the top-left caption card")
  ap.add_argument("--logo", action="append", default=[],
                  help="top-left logo, repeatable (left to right): an image path, 'ias' or 'flexoo'; default ias, 'none' for no logo")
  ap.add_argument("--sensor-brand", default=None,
                  help="logo for the tactile card's 'sensor by' row: a path or 'flexoo' (default); 'none' drops the row")
  ap.add_argument("--azimuth", type=float, default=None,
                  help="camera azimuth [deg], fixed in the world (task default 90 = from behind; "
                       "0 or 180 = side on, cylinder end-on)")
  ap.add_argument("--cylinder-yaw", type=float, default=None,
                  help="pin the cylinder's spawn yaw [deg] (0 = axis along the floor grid's y, "
                       "so --azimuth 90 looks along it); default: random")
  ap.add_argument("--view-angle", type=float, default=None,
                  help="camera azimuth relative to the robot's starting heading [deg] (180 = "
                       "facing it, 145 / 215 = front three-quarter); then fixed in the world. "
                       "Overrides --azimuth")
  ap.add_argument("--distance", type=float, default=None)
  ap.add_argument("--elevation", type=float, default=None)
  args = ap.parse_args()

  import torch
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

  import sendg1.tasks  # noqa: F401
  from sendg1.common.rolling_cylinder.mdp import CylinderState
  from sendg1.video import LOGOS, EvalVideo, prepare_video_cfg

  cfg = load_env_cfg(args.task)
  cfg.scene.num_envs = 1
  cfg.seed = args.seed
  if args.cylinder_yaw is not None:
    import math

    yaw = math.radians(args.cylinder_yaw)
    cfg.events["reset_on_cylinder"].params["yaw_range"] = (yaw, yaw)
  prepare_video_cfg(cfg, args.width_px, args.height_px, args.theme)
  for key in ("azimuth", "distance", "elevation"):
    if getattr(args, key) is not None:
      setattr(cfg.viewer, key, getattr(args, key))
  env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0", render_mode="rgb_array")

  rl_cfg = load_rl_cfg(args.task)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=rl_cfg.clip_actions)
  runner = MjlabOnPolicyRunner(wrapped, asdict(rl_cfg), device="cuda:0")
  runner.load(args.checkpoint, load_cfg={"actor": True}, strict=True, map_location="cuda:0")
  policy = runner.get_inference_policy(device="cuda:0")
  video = EvalVideo(env, rl_cfg, title=args.label or args.task, theme=args.theme,
                    caption=not args.no_caption,
                    logo=[LOGOS.get(name, name) for name in args.logo] or None,
                    sensor_brand=LOGOS.get(args.sensor_brand, args.sensor_brand))
  tm = env.termination_manager

  seconds = args.seconds or cfg.episode_length_s
  reason, t = "", 0.0
  with torch.inference_mode():
    obs, _ = wrapped.reset()
    policy.reset(torch.ones(1, dtype=torch.bool, device=env.device))
    st = CylinderState.get(env)
    if args.view_angle is not None:
      # The robot faces either way along the cylinder (random): aim from its
      # actual start heading, then leave the camera fixed in the world.
      import math

      heading = float(env.scene["robot"].data.heading_w[0])
      env._offline_renderer._cam.azimuth = math.degrees(heading) + args.view_angle
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
  video.save(out, qp=args.qp)
  print(f"[video] wrote {out} ({len(video.frames)} frames, {t:.1f} s, "
        f"{'fell: ' + reason if reason else 'no fall'})")


if __name__ == "__main__":
  main()
