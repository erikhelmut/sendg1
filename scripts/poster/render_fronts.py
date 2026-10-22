"""Front views of the standing human skeleton and the standing G1, same camera."""

from pathlib import Path

import math
import os

import mujoco
import numpy as np
from PIL import Image

import human_pose as hp

# Work dir for intermediate renders and outputs (media/ is gitignored).
D = os.environ.get("POSTER_DIR", str(Path(__file__).resolve().parents[2] / "media" / "poster")) + "/"
os.makedirs(D, exist_ok=True)
PX, SS = 1600, 2
DIST, ELEV, Z = 3.4, -3.0, 0.85
STAND = 60  # G1 standing still (command 0 for the first 2 s of walk.npz)


def masked(m, d, rend, cam, opt, drop):
  rend.disable_segmentation_rendering()
  rend.update_scene(d, cam, opt)
  rgb = rend.render().astype(np.float32)
  rend.enable_segmentation_rendering()
  rend.update_scene(d, cam, opt)
  seg = rend.render()
  is_geom = seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)
  ids = np.where(is_geom, seg[..., 0], 0)
  a = is_geom & ~drop[ids]
  return Image.fromarray(np.dstack([rgb, a * 255]).astype(np.uint8), "RGBA").resize(
    (PX, PX), Image.LANCZOS)


# Human, standing: neutral pose, arms relaxed.
hm, hd = hp.m, hp.d
hp.pose(dict(r=(0, 3, 0), l=(0, 3, 0), arm_r=0, arm_l=0))
pel = hd.xpos[mujoco.mj_name2id(hm, mujoco.mjtObj.mjOBJ_BODY, "pelvis")]
cam = mujoco.MjvCamera()
cam.lookat[:] = [pel[0], pel[1], Z]
cam.distance, cam.azimuth, cam.elevation = DIST, 90, ELEV
hm.vis.global_.offwidth = hm.vis.global_.offheight = PX * SS
hr = mujoco.Renderer(hm, PX * SS, PX * SS)
hdrop = np.zeros(hm.ngeom, bool)
hdrop[hp.floor] = True
masked(hm, hd, hr, cam, hp.opt, hdrop).save(D + f"front_human{os.environ.get('SUFFIX', '')}.png")

# G1, standing.
gm = mujoco.MjModel.from_binary_path(D + "walk.mjb")
gd = mujoco.MjData(gm)
foot = mujoco.mj_name2id(gm, mujoco.mjtObj.mjOBJ_BODY, "robot/right_ankle_roll_link")
for g in range(gm.ngeom):
  if not os.environ.get("NO_TINT") and gm.geom_bodyid[g] == foot and gm.geom_contype[g] == 0 and gm.geom_conaffinity[g] == 0:
    gm.geom_matid[g] = -1
    gm.geom_rgba[g] = [0.30, 0.79, 0.94, 1.0]
gm.mat_reflectance[:] = 0
gm.vis.global_.offwidth = gm.vis.global_.offheight = PX * SS
gr = mujoco.Renderer(gm, PX * SS, PX * SS)
gd.qpos[:] = np.load(D + "walk.npz")["qpos"][STAND]
mujoco.mj_forward(gm, gd)
q = gd.qpos[3:7]
yaw = math.degrees(math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2)))
cam = mujoco.MjvCamera()
cam.lookat[:] = [gd.qpos[0], gd.qpos[1], Z]
sfx = os.environ.get('SUFFIX', '')
# Straight front (split figure) and turned towards the human (face-off figure).
for off, name in ((180, f"front_g1{sfx}.png"), (210, f"front_g1_turn{sfx}.png")):
  cam.distance, cam.azimuth, cam.elevation = DIST, yaw + off, ELEV
  masked(gm, gd, gr, cam, mujoco.MjvOption(), gm.geom_bodyid < 2).save(D + name)
print("ok")
