#!/usr/bin/env python
"""Render the G1 with the tactile slipper and the live tactile HUD.

    MUJOCO_GL=egl python scripts/demo_video.py            # -> media/tactile_demo.mp4

No policy needed: mjlab's stock G1 flat env wearing the slipper, holding its
default pose. Without a controller the G1 cannot balance, so a virtual harness
(a spring-damper wrench on the pelvis) holds it upright and pulls it slowly
from side to side, and the joints are stiffer than mjlab's (whose gains are
tuned for a learned policy and let the knees fold without one). Height stays
free, so the feet carry the full body weight, and the taxels, loads and force
trace move from foot to foot. ``--no-harness``
drops it (the robot then falls). To film a policy instead, build your env the
same way and call ``video.capture()`` after each step.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import replace

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg
from mjlab.utils.lab_api.math import quat_box_minus

from g1_tactile.env_cfg import wear_tactile_slipper
from g1_tactile.video import TactileVideo, prepare_video_cfg


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--out", default="media/tactile_demo.mp4")
  ap.add_argument("--seconds", type=float, default=8.0)
  ap.add_argument("--width", type=int, default=1280)
  ap.add_argument("--height", type=int, default=720)
  ap.add_argument("--sway", type=float, default=0.04, help="side-to-side sway, m")
  ap.add_argument("--stiffen", type=float, default=6.0,
                  help="joint stiffness factor over mjlab's G1 gains")
  ap.add_argument("--no-harness", action="store_true", help="no virtual pelvis harness")
  ap.add_argument("--period", type=float, default=2.5, help="sway period, s")
  ap.add_argument("--logo", nargs="*", default=None,
                  help="top-left logos: ias, flexoo, paths, or none (default: ias)")
  ap.add_argument("--sensor-brand", default=None,
                  help="logo in the tactile card's 'sensor by' row (default: flexoo; none)")
  ap.add_argument("--theme", default="studio", choices=("studio", "gray", "mujoco"))
  ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
  args = ap.parse_args()

  cfg = wear_tactile_slipper(unitree_g1_flat_env_cfg(play=True), noisy=False)
  cfg.scene.num_envs = 1
  cfg.commands["twist"].ranges.lin_vel_x = (0.0, 0.0)
  cfg.commands["twist"].ranges.lin_vel_y = (0.0, 0.0)
  cfg.commands["twist"].ranges.ang_vel_z = (0.0, 0.0)
  cfg.commands["twist"].debug_vis = False
  # A repeatable clip: no pushes, no random heading, COM offset or encoder bias.
  cfg.events = {k: v for k, v in cfg.events.items()
                if "push" not in k and k not in ("base_com", "encoder_bias")}
  if "reset_base" in cfg.events:
    cfg.events["reset_base"].params["pose_range"] = {"z": (0.01, 0.01)}
  if not args.no_harness:
    cfg.terminations = {k: v for k, v in cfg.terminations.items() if k == "time_out"}
  if args.stiffen != 1.0:
    robot_cfg = cfg.scene.entities["robot"]
    k = args.stiffen
    robot_cfg.articulation = replace(robot_cfg.articulation, actuators=tuple(
      replace(a, stiffness=a.stiffness * k, damping=a.damping * math.sqrt(k))
      for a in robot_cfg.articulation.actuators
    ))
  cfg.episode_length_s = args.seconds + 5.0
  # Three-quarter front view, close enough for the sway to read.
  cfg.viewer.azimuth, cfg.viewer.elevation, cfg.viewer.distance = 150.0, -15.0, 3.2
  prepare_video_cfg(cfg, args.width, args.height, theme=args.theme)
  env = ManagerBasedRlEnv(cfg=cfg, device=args.device, render_mode="rgb_array")

  robot = env.scene["robot"]
  pelvis, _ = robot.find_bodies("pelvis")
  upright = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=env.device)
  zero_action = torch.zeros(1, env.action_manager.total_action_dim, device=env.device)

  def harness(home: torch.Tensor, t: float) -> None:
    """Spring-damper on the pelvis toward ``home`` swayed sideways, and upright.
    Applied once per control step; stiffer gains go unstable at that rate."""
    d = robot.data
    target = home.clone()
    target[:, 1] += args.sway * math.sin(2 * math.pi * t / args.period)
    force = torch.zeros(1, 1, 3, device=env.device)
    force[:, 0, :2] = 1500.0 * (target - d.root_link_pos_w[:, :2])
    force[:, 0, :2] -= 150.0 * d.root_link_lin_vel_w[:, :2]
    torque = -150.0 * quat_box_minus(d.root_link_quat_w, upright)
    torque -= 10.0 * d.root_link_ang_vel_w
    robot.write_external_wrench_to_sim(force, torque[:, None], body_ids=pelvis)

  video = TactileVideo(env, title="G1 tactile slipper", theme=args.theme,
                       logo=args.logo or None, sensor_brand=args.sensor_brand)
  env.reset()
  home = robot.data.root_link_pos_w[:, :2].clone()
  steps = int(args.seconds / env.step_dt)
  settle = int(1.0 / env.step_dt)
  with torch.inference_mode():
    for k in range(steps + settle):
      t = max(0, k - settle) * env.step_dt
      if not args.no_harness:
        harness(home, t)
      env.step(zero_action)
      if k >= settle:
        video.capture([f"t = {t:4.1f} s", "default pose, swaying side to side"])
  path = video.save(args.out)
  env.close()
  print(f"[demo] wrote {path}")


if __name__ == "__main__":
  main()
