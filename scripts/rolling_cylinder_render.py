"""A still of the G1 in the rolling-cylinder task's spawn pose (CPU MuJoCo,
no env, no policy): standing across the cylinder, soles centred on its top
line, side and front views.

  python scripts/rolling_cylinder_render.py --out results/rolling_cylinder
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from PIL import Image

from sendg1.common.robots.g1 import get_g1_cfg
from sendg1.common.rolling_cylinder import mdp
from sendg1.common.rolling_cylinder import model as cm

W, H = 960, 720


def scene() -> mujoco.MjSpec:
  spec = mujoco.MjSpec()
  spec.visual.global_.offwidth = 2 * W
  spec.visual.global_.offheight = H
  spec.visual.quality.shadowsize = 4096
  spec.visual.headlight.ambient = (0.3, 0.3, 0.3)
  spec.visual.headlight.diffuse = (0.4, 0.4, 0.4)
  spec.add_texture(
    name="sky",
    type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
    builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
    rgb1=(0.07, 0.09, 0.12),
    rgb2=(0.02, 0.03, 0.05),
    width=512,
    height=512,
  )
  spec.add_texture(
    name="grid",
    type=mujoco.mjtTexture.mjTEXTURE_2D,
    builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
    rgb1=(0.11, 0.13, 0.16),
    rgb2=(0.08, 0.10, 0.12),
    width=512,
    height=512,
  )
  mat = spec.add_material(name="floor", texrepeat=(8, 8), reflectance=0.1)
  mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "grid"
  spec.worldbody.add_geom(
    name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=(30, 30, 0.1), material="floor"
  )
  spec.worldbody.add_light(pos=(1.5, -1.0, 3.0), dir=(-0.4, 0.3, -1.0), castshadow=True)
  spec.attach(cm.cylinder_spec(), prefix="cylinder/", frame=spec.worldbody.add_frame(pos=(0, 0, cm.RADIUS)))
  spec.attach(get_g1_cfg().spec_fn(), prefix="robot/", frame=spec.worldbody.add_frame())
  return spec


def spawn_pose(m: mujoco.MjModel, d: mujoco.MjData) -> None:
  """``mdp.reset_on_cylinder`` without noise: default joints, sole centres on
  the top line, soles just touching it."""
  cfg = get_g1_cfg()
  for i in range(m.njnt):
    name = m.joint(i).name
    if not name.startswith("robot/") or m.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE:
      continue
    for pat, v in cfg.init_state.joint_pos.items():
      if re.fullmatch(pat, name[len("robot/") :]):
        d.qpos[m.jnt_qposadr[i]] = v
  free = next(i for i in range(m.njnt) if m.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE and m.joint(i).name.startswith("robot/"))
  adr = m.jnt_qposadr[free]
  top = 2 * cm.RADIUS
  d.qpos[adr : adr + 7] = (-mdp.FOOT_SITE_X, 0, top + mdp.STANDING_BASE_Z, 1, 0, 0, 0)
  mujoco.mj_forward(m, d)
  sole = min(d.site("robot/left_foot").xpos[2], d.site("robot/right_foot").xpos[2])
  d.qpos[adr + 2] -= sole - top
  mujoco.mj_forward(m, d)


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--out", type=Path, default=Path("results/rolling_cylinder"))
  args = ap.parse_args()
  args.out.mkdir(parents=True, exist_ok=True)

  m = scene().compile()
  d = mujoco.MjData(m)
  spawn_pose(m, d)
  views = []
  with mujoco.Renderer(m, H, W) as r:
    for azimuth in (90.0, 180.0):
      cam = mujoco.MjvCamera()
      cam.lookat[:] = (0.0, 0.0, 0.7)
      cam.distance, cam.azimuth, cam.elevation = 2.8, azimuth, -12.0
      r.update_scene(d, cam)
      views.append(r.render())
  path = args.out / "g1_on_cylinder.png"
  Image.fromarray(np.concatenate(views, axis=1)).save(path)
  print(f"wrote {path}")


if __name__ == "__main__":
  main()
