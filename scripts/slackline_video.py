"""How the slackline behaves, before any policy exists. Two parts, one clip:

1. The line alone (CPU MuJoCo, the task's line model and 5 ms step): a
   35 kg block, the G1's weight, dropped on at mid-span, shoved sideways,
   then pushed down. It sags, swings and rebounds.
2. The G1 on it, through ``EvalVideo`` (the real env, ``G1-Slackline-Touch``):
   a few spawns holding the solved tandem stance as fixed joint targets, no
   policy, in slow motion. The line sags and rolls under it and it falls:
   what a policy has to learn to prevent.

  uv run python scripts/slackline_video.py --out results/slackline/line_demo.mp4
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
import torch

BLOCK_KG = 35.0
LINE_SCRIPT: tuple[tuple[float, float, str, tuple[float, float, float]], ...] = (
  # start (s), end (s), caption, force on the block (N, world = line frame)
  (0.0, 2.5, "35 kg dropped on at mid-span: the line sags and settles", (0.0, 0.0, 0.0)),
  (2.5, 2.7, "sideways shove: 100 N for 0.2 s", (0.0, 100.0, 0.0)),
  (2.7, 5.5, "the line swings back and forth", (0.0, 0.0, 0.0)),
  (5.5, 5.7, "downward push: 400 N for 0.2 s", (0.0, 0.0, -400.0)),
  (5.7, 8.5, "it rebounds", (0.0, 0.0, 0.0)),
)
SLOW = 4
"""Slow-motion factor of the robot part."""


def line_alone(width: int, height: int) -> list[np.ndarray]:
  """Part 1: frames at 50 fps, real time."""
  from sendg1.common.slackline import model as sm
  from sendg1.video import hud

  spec = sm.slackline_spec()
  spec.option.timestep = 0.005
  spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
  spec.visual.global_.offwidth, spec.visual.global_.offheight = width, height
  spec.visual.quality.shadowsize = 4096
  spec.visual.headlight.ambient = (0.35, 0.35, 0.38)
  spec.add_texture(
    name="grid",
    type=mujoco.mjtTexture.mjTEXTURE_2D,
    builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
    rgb1=(0.11, 0.13, 0.16),
    rgb2=(0.08, 0.10, 0.12),
    width=512,
    height=512,
  )
  mat = spec.add_material(name="floor", texrepeat=(10, 10), reflectance=0.1)
  mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "grid"
  spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=(8, 8, 0.1), material="floor")
  spec.worldbody.add_light(pos=(1.0, -1.5, 3.5), dir=(-0.2, 0.4, -1.0), castshadow=True)
  block = spec.worldbody.add_body(name="block", pos=(0.0, 0.0, sm.HEIGHT + 0.12))
  # Sliding only: a block balanced on 5 cm webbing would tip off at once.
  for axis in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
    block.add_joint(type=mujoco.mjtJoint.mjJNT_SLIDE, axis=axis)
  block.add_geom(
    type=mujoco.mjtGeom.mjGEOM_BOX, size=(0.12, 0.06, 0.06), mass=BLOCK_KG, rgba=(0.98, 0.75, 0.33, 1)
  )
  m = spec.compile()
  d = mujoco.MjData(m)
  bid = m.body("block").id
  z_adr = [m.jnt_qposadr[m.joint(sm.SLIDE_Z.format(i)).id] for i in range(sm.NUM_NODES)]
  y_adr = [m.jnt_qposadr[m.joint(sm.SLIDE_Y.format(i)).id] for i in range(sm.NUM_NODES)]
  mid = sm.NUM_NODES // 2
  cam = mujoco.MjvCamera()
  cam.lookat[:] = (0.0, 0.0, 0.8)
  cam.distance, cam.azimuth, cam.elevation = 3.6, 150.0, -10.0
  vignette = hud.vignette(height, width)
  frames = []
  with mujoco.Renderer(m, height, width) as r:
    for k in range(round(LINE_SCRIPT[-1][1] * 50)):
      t = k / 50
      caption, force = next((c, f) for s, e, c, f in LINE_SCRIPT if s <= t < e)
      d.xfrc_applied[bid, :3] = force
      for _ in range(4):  # 5 ms physics, 50 fps
        mujoco.mj_step(m, d)
      r.update_scene(d, cam)
      frame = (r.render() * vignette).astype(np.uint8)
      lines = [
        f"t = {t:4.1f} s   {caption}",
        f"mid-span: sag {-100 * d.qpos[z_adr[mid]]:4.1f} cm, sideways {100 * d.qpos[y_adr[mid]]:+5.1f} cm",
        f"4 m line at {sm.TENSION:.0f} N (randomised 1000-2000 N in training)",
      ]
      frames.append(hud.caption(frame, "The slackline alone", "LINE", False, lines))
  return frames


def robot_on_line(width: int, height: int, episodes: int, seconds: float) -> list[np.ndarray]:
  """Part 2: each control step repeated ``SLOW`` times, for 50 fps playback."""
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.termination_manager import TerminationTermCfg
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
  from mjlab.tasks.velocity import mdp as vel_mdp

  import sendg1.tasks  # noqa: F401
  from sendg1.common.slackline import mdp
  from sendg1.tasks.slackline import TOUCH_TASK
  from sendg1.video import EvalVideo, prepare_video_cfg

  cfg = load_env_cfg(TOUCH_TASK, play=True)
  cfg.scene.num_envs = 1
  # Let each fall play out: only the clip's own clock ends it.
  cfg.terminations = {"time_out": TerminationTermCfg(func=vel_mdp.time_out, time_out=True)}
  cfg.events["reset_on_line"].params = {
    "joint_noise": 0.0,
    "heading_noise": 0.0,
    "along_range": (-0.3, -0.3),
  }
  prepare_video_cfg(cfg, width, height)
  cfg.viewer.distance, cfg.viewer.azimuth, cfg.viewer.elevation = 3.2, 150.0, -8.0
  env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0", render_mode="rgb_array")
  frames: list[np.ndarray] = []
  try:
    st = mdp.LineState.get(env)
    robot = env.scene["robot"]
    scale = env.action_manager.get_term("joint_pos")._scale
    stances = mdp._stance_tensors(env)
    video = EvalVideo(env, load_rl_cfg(TOUCH_TASK), title="G1 on the slackline, no policy yet")
    for ep in range(episodes):
      env.reset()
      video.new_episode()
      q = torch.where(st.front_left[:, None], stances["left"], stances["right"])
      action = (q - robot.data.default_joint_pos[0]) / scale
      front = "left" if bool(st.front_left[0]) else "right"
      for k in range(round(seconds / env.step_dt)):
        env.step(action)
        line = st.line_under_base()
        sag = float(line.z[0, 0]) - float(st.origin()[0, 2]) - 1.0
        frame = video.capture(
          [
            (f"spawn {ep + 1}/{episodes}, t = {k * env.step_dt:4.2f} s, {SLOW}x slow motion; "
            f"{front} foot in front, fixed joint targets"),
            (f"line under the robot: sag {-100 * sag:4.1f} cm, sideways {100 * float(line.y[0, 0]):+5.1f} cm, "
            f"roll {math.degrees(float(line.roll[0, 0])):+5.1f} deg"),
            f"line tension {float(mdp.line_tension(env)[0]):.0f} N",
          ]
        )
        assert frame is not None
        frames += [frame] * SLOW
  finally:
    env.close()
  return frames


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--out", type=Path, default=Path("results/slackline/line_demo.mp4"))
  ap.add_argument("--episodes", type=int, default=3)
  ap.add_argument("--seconds", type=float, default=1.2, help="sim time per robot spawn")
  ap.add_argument("--width", type=int, default=1280)
  ap.add_argument("--height", type=int, default=720)
  args = ap.parse_args()

  import mediapy

  frames = line_alone(args.width, args.height)
  frames += robot_on_line(args.width, args.height, args.episodes, args.seconds)
  args.out.parent.mkdir(parents=True, exist_ok=True)
  mediapy.write_video(str(args.out), frames, fps=50, qp=24)
  print(f"wrote {args.out} ({len(frames) / 50:.1f} s)")


if __name__ == "__main__":
  main()
