"""Pose the MyoSuite full-body skeleton at textbook gait phases; render on white.

Angles are illustrative (typical sagittal gait values), not measured data.
"""

from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

# Work dir for intermediate renders and outputs (media/ is gitignored).
D = os.environ.get("POSTER_DIR", str(Path(__file__).resolve().parents[2] / "media" / "poster")) + "/"
os.makedirs(D, exist_ok=True)
PX, SS = 1400, 2

# degrees: (hip flexion +, knee flexion +, ankle dorsiflexion +) per leg, and
# shoulder swing (+ forward) per arm.
import os
MTP = float(os.environ.get('MTP', '-30'))  # toe extension at push-off
PHASES = {
  "heel_strike": dict(r=(25, 3, 0), l=(-10, 15, -10), arm_r=-15, arm_l=15, mtp_l=MTP),
  "mid_stance": dict(r=(0, 12, 8), l=(25, 55, 0), arm_r=0, arm_l=0),
  "push_off": dict(r=(-10, 15, -10), l=(25, 3, 0), arm_r=15, arm_l=-15, mtp_r=MTP),
}

m = mujoco.MjModel.from_binary_path(D + "human.mjb")
d = mujoco.MjData(m)
jid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)  # noqa: E731
qadr = lambda n: m.jnt_qposadr[jid(n)]  # noqa: E731


def apply_couplings():
  # Joint equalities (knee translations etc.): q1 = q1_0 + poly(q2 - q2_0).
  for _ in range(3):
    for e in range(m.neq):
      if m.eq_type[e] != mujoco.mjtEq.mjEQ_JOINT:
        continue
      j1, j2 = m.eq_obj1id[e], m.eq_obj2id[e]
      c = m.eq_data[e, :5]
      a1 = m.jnt_qposadr[j1]
      if j2 < 0:
        d.qpos[a1] = m.qpos0[a1] + c[0]
        continue
      a2 = m.jnt_qposadr[j2]
      x = d.qpos[a2] - m.qpos0[a2]
      d.qpos[a1] = m.qpos0[a1] + c[0] + c[1] * x + c[2] * x**2 + c[3] * x**3 + c[4] * x**4


foot_bodies = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
               for n in ("calcn_r", "toes_r", "calcn_l", "toes_l")]


def lowest_foot_z():
  z = np.inf
  for g in range(m.ngeom):
    if m.geom_bodyid[g] in foot_bodies and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
      mid = m.geom_dataid[g]
      v = m.mesh_vert[m.mesh_vertadr[mid]: m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
      w = d.geom_xpos[g] + v @ d.geom_xmat[g].reshape(3, 3).T
      z = min(z, w[:, 2].min())
  return z


def pose(p):
  d.qpos[:] = m.qpos0
  r = np.radians
  for side in ("r", "l"):
    hip, knee, ankle = p[side]
    d.qpos[qadr(f"hip_flexion_{side}")] = r(hip)
    d.qpos[qadr(f"knee_angle_{side}")] = r(knee)
    d.qpos[qadr(f"ankle_angle_{side}")] = r(ankle)
    d.qpos[qadr(f"elbow_flexion_{side}")] = r(20)
    d.qpos[qadr(f"mtp_angle_{side}")] = r(p.get(f"mtp_{side}", 0))
    # Arm swing: elevate slightly in the sagittal plane (plane angle ~ forward/back).
    sw = p[f"arm_{side}"]
    d.qpos[qadr(f"shoulder_elv_{side}")] = r(abs(sw) + 3)
    d.qpos[qadr(f"shoulder1_r2_{side}")] = r(90 if sw >= 0 else -90)
  apply_couplings()
  mujoco.mj_forward(m, d)
  d.qpos[2] -= lowest_foot_z()
  mujoco.mj_forward(m, d)


# Right foot bones in kyber blue, as on the G1.
for g in range(m.ngeom):
  if not os.environ.get("NO_TINT") and m.geom_bodyid[g] in foot_bodies[:2] + [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "talus_r")]:
    m.geom_matid[g] = -1
    m.geom_rgba[g] = [0.30, 0.79, 0.94, 1]
m.mat_reflectance[:] = 0
m.vis.global_.offwidth = m.vis.global_.offheight = PX * SS
rend = mujoco.Renderer(m, PX * SS, PX * SS)
opt = mujoco.MjvOption()
opt.flags[mujoco.mjtVisFlag.mjVIS_TENDON] = 0
for k in range(6):
  opt.sitegroup[k] = 0
floor = [g for g in range(m.ngeom) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE]

for name, p in PHASES.items():
  pose(p)
  cam = mujoco.MjvCamera()
  pelvis = d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")]
  cam.lookat[:] = [pelvis[0], pelvis[1], 0.9]
  cam.distance, cam.azimuth, cam.elevation = 3.2, 0, -4
  rend.disable_segmentation_rendering()
  rend.update_scene(d, cam, opt)
  rgb = rend.render().astype(np.float32)
  rend.enable_segmentation_rendering()
  rend.update_scene(d, cam, opt)
  seg = rend.render()
  is_geom = seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)
  ids = np.where(is_geom, seg[..., 0], 0)
  alpha = is_geom & ~np.isin(ids, floor)
  rgba = np.dstack([rgb, alpha * 255]).astype(np.uint8)
  img = Image.fromarray(rgba, "RGBA").resize((PX, PX), Image.LANCZOS)
  img.save(D + f"human_{name}.png")
  print(name, "ok")
