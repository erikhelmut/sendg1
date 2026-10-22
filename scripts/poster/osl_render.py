"""MyoSuite skeleton with the Open-Source Leg prosthesis, on white."""

import os
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image
sys.path.insert(0, sys.argv[1])
import myo_sim
# Work dir for intermediate renders and outputs (media/ is gitignored).
D = os.environ.get("POSTER_DIR", str(Path(__file__).resolve().parents[2] / "media" / "poster")) + "/"
os.makedirs(D, exist_ok=True)
PX, SS = 1600, 2
spec = mujoco.MjSpec.from_file(str(myo_sim.MODELS_DIR / "legacy/osl/myolegs_osl.xml"))
if os.environ.get("FULLBODY_LIGHT"):
  # Same lighting as the full-body model, so the bones match the plain skeleton.
  spec.worldbody.add_light(pos=[0, 0, 10.3], dir=[0, 0, -1], diffuse=[1.5, 1.5, 1.5])
m = spec.compile()
d = mujoco.MjData(m)
bid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
mujoco.mj_forward(m, d)
# Stand on the floor: lowest point of any geom at z = 0.
low = min((d.geom_xpos[g, 2] - m.geom_rbound[g]) for g in range(m.ngeom)
          if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_PLANE and m.geom_bodyid[g] > 0)
# The rbound bound is loose; refine using mesh vertices of the feet.
def lowest():
  z = np.inf
  for g in range(m.ngeom):
    if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and m.geom_bodyid[g] in (bid("calcn_l"), bid("toes_l"), bid("osl_foot_assembly")):
      mid = m.geom_dataid[g]
      v = m.mesh_vert[m.mesh_vertadr[mid]: m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
      z = min(z, (d.geom_xpos[g] + v @ d.geom_xmat[g].reshape(3, 3).T)[:, 2].min())
  return z
d.qpos[2] -= lowest(); mujoco.mj_forward(m, d)
osl = [bid(n) for n in ("prosthetic_socket", "osl_femoral_pylon", "osl_proximal_adapter",
                        "osl_knee_assembly", "osl_tibial_pylon", "osl_ankle_assembly")]
for g in range(m.ngeom):
  if m.geom_bodyid[g] == bid("osl_foot_assembly"):
    m.geom_matid[g] = -1; m.geom_rgba[g] = [0.17, 0.21, 0.27, 1] if os.environ.get("NO_TINT") else [0.30, 0.79, 0.94, 1]
  elif m.geom_bodyid[g] in osl:
    m.geom_matid[g] = -1; m.geom_rgba[g] = [0.17, 0.21, 0.27, 1]   # prosthesis: dark slate
m.mat_reflectance[:] = 0
m.vis.headlight.ambient[:] = float(os.environ.get("AMB", 0.35)); m.vis.headlight.diffuse[:] = float(os.environ.get("DIF", 0.65))
if os.environ.get("FULLBODY_LIGHT"):
  m.vis.headlight.specular[:] = 0
m.vis.global_.offwidth = m.vis.global_.offheight = PX * SS
r = mujoco.Renderer(m, PX * SS, PX * SS)
opt = mujoco.MjvOption(); opt.flags[mujoco.mjtVisFlag.mjVIS_TENDON] = 0
for k in range(6): opt.sitegroup[k] = 0
for az in (float(a) for a in sys.argv[2:]):
  cam = mujoco.MjvCamera(); cam.lookat[:] = [d.xpos[bid("pelvis")][0], d.xpos[bid("pelvis")][1], 0.85]
  cam.distance, cam.azimuth, cam.elevation = 3.4, az, -3
  r.disable_segmentation_rendering(); r.update_scene(d, cam, opt); rgb = r.render().astype(np.float32)
  r.enable_segmentation_rendering(); r.update_scene(d, cam, opt); seg = r.render()
  isg = seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM); ids = np.where(isg, seg[..., 0], 0)
  a = isg & (m.geom_type[ids] != mujoco.mjtGeom.mjGEOM_PLANE) & (m.geom_bodyid[ids] > 0)
  im = Image.fromarray(np.dstack([rgb, a * 255]).astype(np.uint8), "RGBA").resize((PX, PX), Image.LANCZOS)
  im = im.crop(im.getchannel("A").point(lambda v: 255 if v > 8 else 0).getbbox())
  im.save(D + f"osl34_{int(az)}{os.environ.get('SUFFIX', '')}.png"); print("az", az, im.size)
