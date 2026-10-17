"""Render the G1 in a posed, low-angle "sole to the camera" shot, with the
slipper_v1 taxel array drawn on both soles.

The pose is hand-fit to a reference photo (no pose estimator involved): joint
angles live in POSE below, the camera in CAMERA.

  python scripts/render_pose_picture.py --out results/picture/g1_kick.png
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from PIL import Image

from sendg1.common.tactile.layout import SOLE_Z, csv_layout
from sendg1.common.tactile.viz import _g1_xml_dir, foot_mesh

# Radians, G1 conventions: hip pitch < 0 swings the thigh forward, knee > 0
# flexes, ankle pitch < 0 lifts the toes.
POSE = {
  # Kicking leg: thigh forward and up, knee almost straight, sole to camera.
  "right_hip_pitch_joint": -1.0,
  "right_hip_roll_joint": -0.12,
  "right_knee_joint": 0.2,
  "right_ankle_pitch_joint": -0.35,
  # Stance leg, nearly straight.
  "left_hip_pitch_joint": 0.1,
  "left_hip_roll_joint": 0.08,
  "left_knee_joint": 0.05,
  "left_ankle_pitch_joint": -0.2,
  "waist_pitch_joint": -0.1,
  # Arms in a walking swing, opposite the legs: left a little forward, right a
  # little back, both nearly straight (G1 elbow: ~1.3 is straight, 0 is 90 deg).
  # Pitches are relative to a torso that already leans back ~0.25 rad.
  "left_shoulder_pitch_joint": -0.05,
  "left_shoulder_roll_joint": 0.12,
  "left_elbow_joint": 1.2,
  "right_shoulder_pitch_joint": 0.7,
  "right_shoulder_roll_joint": -0.12,
  "right_elbow_joint": 1.2,
}
BASE_POS = (0.0, 0.0, 0.80)
BASE_PITCH = -0.15  # lean back (about +y; < 0 tips the torso backwards)

# World frame: robot faces +x, left is +y. The camera looks along `look`
# (from low in front, up at the kicking sole) and is pulled back along it until
# the whole robot fits in frame with `margin` to spare.
CAMERA = dict(look=(-0.48, 0.08, 0.20), roll_deg=-38.0, fovy=74.0, margin=0.06)

BACKGROUND = (1.0, 1.0, 1.0)
SOLE_RGBA = (0.10, 0.10, 0.11, 1.0)
TAXEL_RGBA = (0.43, 0.65, 0.93, 1.0)  # #6da7ec, light blue from viz.FORCE_RAMP


def sole_plate(side: str, thickness: float = 0.003) -> np.ndarray:
  """Vertices of the foot mesh's plan-view convex hull, as a prism under the
  sole; MuJoCo takes the convex hull of them."""
  from scipy.spatial import ConvexHull

  xy = foot_mesh(side).vertices[:, :2]
  hull = xy[ConvexHull(xy).vertices]
  top, bottom = SOLE_Z + 0.001, SOLE_Z - thickness
  return np.concatenate([
    np.column_stack([hull, np.full(len(hull), top)]),
    np.column_stack([hull, np.full(len(hull), bottom)]),
  ])


def build_model() -> mujoco.MjModel:
  spec = mujoco.MjSpec.from_file(str(_g1_xml_dir() / "g1.xml"))
  spec.visual.global_.offwidth = 2400
  spec.visual.global_.offheight = 3200
  spec.visual.quality.shadowsize = 8192
  spec.visual.headlight.ambient = (0.35, 0.35, 0.38)
  spec.visual.headlight.diffuse = (0.45, 0.45, 0.45)

  spec.add_texture(
    name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
    builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
    rgb1=BACKGROUND, rgb2=BACKGROUND, width=64, height=384,
  )
  sun = spec.worldbody.add_light(pos=(1.5, -1.0, 4.0), dir=(-0.3, 0.2, -1.0))
  sun.diffuse = (0.75, 0.73, 0.68)
  sun.castshadow = True
  fill = spec.worldbody.add_light(pos=(2.0, 0.5, -1.0), dir=(-1.0, -0.2, 0.6))
  fill.diffuse = (0.35, 0.38, 0.45)
  fill.castshadow = False

  layout = csv_layout("slipper_v1")
  for side in ("left", "right"):
    body = spec.body(f"{side}_ankle_roll_link")
    # Slipper sole: the foot's own outline, extruded into a thin dark plate.
    spec.add_mesh(name=f"{side}_slipper", uservert=sole_plate(side).ravel())
    body.add_geom(
      type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"{side}_slipper",
      rgba=SOLE_RGBA, contype=0, conaffinity=0, group=2,
    )
    for (cx, cy), (hx, hy) in zip(layout.for_side(side), layout.half_sizes):
      body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(hx, hy, 0.0012),
        pos=(cx, cy, SOLE_Z - 0.0035),
        rgba=TAXEL_RGBA, contype=0, conaffinity=0, group=2,
      )

  right, up, _ = camera_axes()
  cam = spec.worldbody.add_camera(name="shot", fovy=CAMERA["fovy"])
  cam.alt.type = mujoco.mjtOrientation.mjORIENTATION_XYAXES
  cam.alt.xyaxes = np.concatenate([right, up])
  return spec.compile()


def camera_axes() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  fwd = np.array(CAMERA["look"], dtype=float)
  fwd /= np.linalg.norm(fwd)
  right = np.cross(fwd, (0, 0, 1))
  right /= np.linalg.norm(right)
  up = np.cross(right, fwd)
  r = np.deg2rad(CAMERA["roll_deg"])
  right, up = np.cos(r) * right + np.sin(r) * up, -np.sin(r) * right + np.cos(r) * up
  return right, up, fwd


def visible_points(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
  """World-frame corners of every visual geom's bounding box."""
  signs = np.array(np.meshgrid([-1, 1], [-1, 1], [-1, 1])).reshape(3, -1).T
  pts = []
  for g in np.flatnonzero(model.geom_group == 2):
    c, h = model.geom_aabb[g, :3], model.geom_aabb[g, 3:]
    local = c + signs * h
    pts.append(data.geom_xpos[g] + local @ data.geom_xmat[g].reshape(3, 3).T)
  return np.concatenate(pts)


def frame_camera(model: mujoco.MjModel, data: mujoco.MjData, aspect: float) -> None:
  """Move the fixed-orientation camera as close as it can get with every
  visual geom in frame, centred on the binding image axis.

  In camera axes a point at (a, b, d) is visible iff |a - sx| <= t (d - cf)
  per image axis, so sx - t cf >= max(a - t d) =: U and sx + t cf <= min(a +
  t d) =: W. The closest camera has cf = (W - U) / 2t, sx = (U + W) / 2.
  """
  right, up, fwd = camera_axes()
  pts = visible_points(model, data)
  ty = np.tan(np.deg2rad(CAMERA["fovy"]) / 2) * (1 - CAMERA["margin"])
  tx = ty * aspect
  d = pts @ fwd
  shift, depth = [], []
  for coord, t in ((pts @ right, tx), (pts @ up, ty)):
    U, W = np.max(coord - t * d), np.min(coord + t * d)
    shift.append(0.5 * (U + W))
    depth.append((W - U) / (2 * t))
  model.cam_pos[model.camera("shot").id] = (
    shift[0] * right + shift[1] * up + min(depth) * fwd
  )
  mujoco.mj_forward(model, data)


def crop_to_content(img: np.ndarray, pad: int) -> np.ndarray:
  """Trim the white border down to ``pad`` pixels around the robot."""
  ink = np.any(img < 250, axis=-1)
  rows, cols = np.flatnonzero(ink.any(1)), np.flatnonzero(ink.any(0))
  r0, r1 = max(rows[0] - pad, 0), min(rows[-1] + pad + 1, img.shape[0])
  c0, c1 = max(cols[0] - pad, 0), min(cols[-1] + pad + 1, img.shape[1])
  return img[r0:r1, c0:c1]


def pose(model: mujoco.MjModel, data: mujoco.MjData) -> None:
  data.qpos[:3] = BASE_POS
  data.qpos[3:7] = (np.cos(BASE_PITCH / 2), 0, np.sin(BASE_PITCH / 2), 0)
  for name, q in POSE.items():
    j = model.joint(name)
    lo, hi = model.jnt_range[j.id]
    if not lo <= q <= hi:
      raise ValueError(f"{name}={q} outside [{lo:.3f}, {hi:.3f}]")
    data.qpos[model.jnt_qposadr[j.id]] = q
  mujoco.mj_forward(model, data)


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--out", type=Path, default=Path("results/picture/g1_kick.png"))
  ap.add_argument("--size", type=int, default=2400, help="render is size x size, then cropped")
  args = ap.parse_args()

  model = build_model()
  data = mujoco.MjData(model)
  pose(model, data)
  frame_camera(model, data, 1.0)
  opt = mujoco.MjvOption()
  opt.geomgroup[:] = 0
  opt.geomgroup[2] = 1
  with mujoco.Renderer(model, args.size, args.size) as r:
    r.update_scene(data, camera="shot", scene_option=opt)
    r.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = True
    img = r.render()
  img = crop_to_content(img, pad=int(0.05 * args.size))
  args.out.parent.mkdir(parents=True, exist_ok=True)
  Image.fromarray(img).save(args.out)
  print(args.out)


if __name__ == "__main__":
  main()
