#!/usr/bin/env python
"""Pictures and a driving clip of a pedalo model, Classic or Sport (CPU
MuJoCo, no env, no policy). Sport files carry a ``_sport`` suffix.

- ``pedalo.png``       the pedalo alone, boards half a stroke apart.
- ``g1_on_pedalo.png`` the G1 in the task's spawn pose on it.
- ``pedalo_drive.mp4`` the pedalo driven by a scripted "rider": the board
                       ahead of the wheel centre is pressed with 240 N, the
                       other with 100 N (a 35 kg rider shifting weight), until
                       0.6 m/s. Red arrows are the pushes. On the Sport a
                       stiff torque also holds each step level, standing in
                       for the foot on it.

  python scripts/pedalo_render.py --out results/pedalo [--variant sport]
"""

from __future__ import annotations

import argparse
import math
import os
import re
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from PIL import Image

from sendg1.common.pedalo import mdp
from sendg1.common.pedalo import model as pm
from sendg1.common.pedalo.variants import CLASSIC, VARIANTS, PedaloVariant
from sendg1.common.robots.g1 import get_g1_cfg

W, H = 1280, 800
FPS = 50
PUSH_HIGH_N = 240.0
PUSH_LOW_N = 100.0
TOP_SPEED = 0.6
STEP_HOLD = (0.5, 0.005)
"""Sport only: N m / rad and N m s / rad holding each step level. Applied
explicitly, so bounded by the step's tiny inertia (~1.4e-4 kg m^2 with
armature) at the 5 ms step."""


def scene(with_robot: bool, variant: PedaloVariant = CLASSIC) -> mujoco.MjSpec:
  spec = mujoco.MjSpec()
  spec.visual.global_.offwidth = W
  spec.visual.global_.offheight = H
  spec.visual.quality.shadowsize = 4096
  spec.visual.headlight.ambient = (0.35, 0.35, 0.35)
  spec.visual.headlight.diffuse = (0.45, 0.45, 0.45)
  spec.add_texture(
    name="sky",
    type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
    builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
    rgb1=(0.96, 0.97, 0.99),
    rgb2=(0.80, 0.84, 0.90),
    width=512,
    height=512,
  )
  spec.add_texture(
    name="grid",
    type=mujoco.mjtTexture.mjTEXTURE_2D,
    builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
    rgb1=(0.82, 0.82, 0.80),
    rgb2=(0.74, 0.74, 0.72),
    width=512,
    height=512,
  )
  mat = spec.add_material(name="floor", texrepeat=(8, 8), reflectance=0.05)
  mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "grid"
  spec.worldbody.add_geom(
    name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=(30, 30, 0.1), material="floor"
  )
  spec.worldbody.add_light(pos=(1.5, -1.0, 3.0), dir=(-0.4, 0.3, -1.0), castshadow=True)
  spec.worldbody.add_light(pos=(-2.0, 2.0, 2.5), dir=(0.5, -0.5, -1.0), castshadow=False, diffuse=(0.4, 0.4, 0.4))
  frame = spec.worldbody.add_frame(pos=(variant.root_ahead, 0, pm.WHEEL_RADIUS))
  spec.attach(variant.spec_fn(), prefix="pedalo/", frame=frame)
  if with_robot:
    spec.attach(get_g1_cfg().spec_fn(), prefix="robot/", frame=spec.worldbody.add_frame())
  return spec


def set_crank(m: mujoco.MjModel, d: mujoco.MjData, phi: float, variant: PedaloVariant) -> None:
  adr = m.joint("pedalo/pedalo_free").qposadr[0]
  d.qpos[adr : adr + 3] = (variant.root_ahead, 0, pm.WHEEL_RADIUS)
  d.qpos[adr + 3 : adr + 7] = (math.cos(phi / 2), 0, math.sin(phi / 2), 0)
  for name, per_phi in variant.hinge_phi:
    d.qpos[m.joint(f"pedalo/{name}").qposadr[0]] = per_phi * phi


def set_robot_on_pedalo(m: mujoco.MjModel, d: mujoco.MjData, phi: float, variant: PedaloVariant) -> None:
  """The task's spawn pose (``mdp.reset_on_pedalo``), soles on the boards."""
  cfg = get_g1_cfg()
  adr = m.joint("robot/floating_base_joint").qposadr[0] if _has(m, "robot/floating_base_joint") else _free_adr(m, "robot/")
  q = {}
  for i in range(m.njnt):
    name = m.joint(i).name
    if not name.startswith("robot/") or m.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE:
      continue
    short = name[len("robot/") :]
    for pat, v in cfg.init_state.joint_pos.items():
      if re.fullmatch(pat, short):
        q[short] = v
  ahead = pm.ECCENTRIC * math.cos(phi)
  for side, sign, dx in (("left", 1.0, ahead), ("right", -1.0, -ahead)):
    dpitch = -dx / mdp.LEG_LENGTH
    q[f"{side}_hip_pitch_joint"] = q.get(f"{side}_hip_pitch_joint", 0.0) + dpitch
    q[f"{side}_ankle_pitch_joint"] = q.get(f"{side}_ankle_pitch_joint", 0.0) - dpitch
    q[f"{side}_hip_roll_joint"] = q.get(f"{side}_hip_roll_joint", 0.0) - sign * mdp.SPAWN_STANCE_Y
    q[f"{side}_ankle_roll_joint"] = q.get(f"{side}_ankle_roll_joint", 0.0) + sign * mdp.SPAWN_STANCE_Y
  for short, v in q.items():
    d.qpos[m.joint(f"robot/{short}").qposadr[0]] = v
  top = pm.WHEEL_RADIUS + variant.step_top_above_axle + pm.ECCENTRIC * abs(math.sin(phi))
  d.qpos[adr : adr + 3] = (-mdp.FOOT_SITE_X, 0, top + mdp.STANDING_BASE_Z)
  d.qpos[adr + 3 : adr + 7] = (1, 0, 0, 0)
  mujoco.mj_forward(m, d)
  # Settle the soles exactly onto the boards.
  sole = min(d.site("robot/left_foot").xpos[2], d.site("robot/right_foot").xpos[2])
  d.qpos[adr + 2] -= sole - top
  mujoco.mj_forward(m, d)


def _has(m: mujoco.MjModel, name: str) -> bool:
  try:
    m.joint(name)
    return True
  except KeyError:
    return False


def _free_adr(m: mujoco.MjModel, prefix: str) -> int:
  for i in range(m.njnt):
    if m.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE and m.joint(i).name.startswith(prefix):
      return m.jnt_qposadr[i]
  raise KeyError(prefix)


def camera(lookat, distance: float, azimuth: float, elevation: float) -> mujoco.MjvCamera:
  cam = mujoco.MjvCamera()
  cam.type = mujoco.mjtCamera.mjCAMERA_FREE
  cam.lookat[:] = lookat
  cam.distance = distance
  cam.azimuth = azimuth
  cam.elevation = elevation
  return cam


def add_arrow(scn: mujoco.MjvScene, start, end, rgba=(0.9, 0.1, 0.1, 1.0)) -> None:
  if scn.ngeom >= scn.maxgeom:
    return
  g = scn.geoms[scn.ngeom]
  mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3), np.zeros(9), np.array(rgba, dtype=np.float32))
  mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, 0.012, np.asarray(start, float), np.asarray(end, float))
  scn.ngeom += 1


def picture_pedalo(out: Path, variant: PedaloVariant) -> None:
  m = scene(with_robot=False, variant=variant).compile()
  d = mujoco.MjData(m)
  set_crank(m, d, 0.9, variant)
  mujoco.mj_forward(m, d)
  with mujoco.Renderer(m, H, W) as r:
    r.update_scene(d, camera((0.0, 0.0, 0.11), 0.95, 135.0, -22.0))
    Image.fromarray(r.render()).save(out)


def picture_g1(out: Path, variant: PedaloVariant) -> None:
  m = scene(with_robot=True, variant=variant).compile()
  d = mujoco.MjData(m)
  set_crank(m, d, 0.0, variant)
  set_robot_on_pedalo(m, d, 0.0, variant)
  with mujoco.Renderer(m, H, W) as r:
    r.update_scene(d, camera((0.0, 0.0, 0.6), 2.6, 140.0, -12.0))
    Image.fromarray(r.render()).save(out)


def drive_video(out: Path, seconds: float, variant: PedaloVariant) -> dict[str, float]:
  import mediapy

  m = scene(with_robot=False, variant=variant).compile()
  m.opt.timestep = 0.005  # the task's physics step (mjlab velocity cfg)
  m.opt.iterations = 10
  m.opt.ls_iterations = 20
  d = mujoco.MjData(m)
  set_crank(m, d, 0.3, variant)
  mujoco.mj_forward(m, d)
  boards = [m.body(f"pedalo/{b}").id for b in variant.step_bodies]
  hinges = [m.joint(f"pedalo/{j}").dofadr[0] for j, _ in variant.hinge_phi]
  front = m.body(f"pedalo/{variant.root_body}").id
  top_local = np.array(variant.step_top_local)
  substeps = round(1.0 / (FPS * m.opt.timestep))
  frames, speeds = [], []
  with mujoco.Renderer(m, H, W) as r:
    for _ in range(int(seconds * FPS)):
      for _ in range(substeps):
        d.xfrc_applied[:] = 0
        ahead = d.xpos[boards[0], 0] > d.xpos[front, 0]  # left axle ahead of the wheel centre
        push = PUSH_HIGH_N if d.qvel[0] < TOP_SPEED else PUSH_LOW_N
        d.xfrc_applied[boards[0], 2] = -(push if ahead else PUSH_LOW_N)
        d.xfrc_applied[boards[1], 2] = -(PUSH_LOW_N if ahead else push)
        if variant.steps_turn:
          for b, dof in zip(boards, hinges):
            x = d.xmat[b].reshape(3, 3)[:, 0]
            pitch = math.atan2(-x[2], x[0])
            rate = d.qvel[4] + d.qvel[dof]  # wheelset pitch rate (body y) + hinge
            d.xfrc_applied[b, 4] = -STEP_HOLD[0] * pitch - STEP_HOLD[1] * rate
        mujoco.mj_step(m, d)
      speeds.append(float(d.qvel[0]))
      wheels = [m.body(f"pedalo/{w}").id for w in variant.wheel_bodies]
      centre = d.xpos[wheels].mean(axis=0)
      r.update_scene(d, camera((centre[0], 0.0, 0.15), 1.3, 60.0, -14.0))
      for b in boards:
        top = d.xpos[b] + d.xmat[b].reshape(3, 3) @ top_local
        f = -d.xfrc_applied[b, 2]
        add_arrow(r.scene, top + (0, 0, 0.05 + f / 1000.0), top + (0, 0, 0.01))
      frames.append(r.render())
  mediapy.write_video(out, frames, fps=FPS)
  dist = float(d.xpos[front, 0] - variant.root_ahead)
  return {"distance_m": dist, "mean_speed": float(np.mean(speeds[FPS:])), "seconds": seconds}


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--out", type=Path, default=Path("results/pedalo"))
  ap.add_argument("--seconds", type=float, default=8.0)
  ap.add_argument("--variant", choices=[v.name for v in VARIANTS], default=CLASSIC.name)
  args = ap.parse_args()
  variant = next(v for v in VARIANTS if v.name == args.variant)
  sfx = "" if variant is CLASSIC else f"_{variant.name}"
  args.out.mkdir(parents=True, exist_ok=True)
  picture_pedalo(args.out / f"pedalo{sfx}.png", variant)
  picture_g1(args.out / f"g1_on_pedalo{sfx}.png", variant)
  print(drive_video(args.out / f"pedalo{sfx}_drive.mp4", args.seconds, variant))


if __name__ == "__main__":
  main()
