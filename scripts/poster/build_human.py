import sys
sys.path.insert(0, sys.argv[1])
import mujoco
from myo_sim.build.compose import build_spec
m = build_spec("myofullbody").compile()
print("nq", m.nq, "njnt", m.njnt, "nbody", m.nbody, "ngeom", m.ngeom)
for j in [j for j in range(m.njnt) if any(k in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or "") for k in ("hip","knee","ankle","root","free","lumbar","shoulder","elbow","arm","subtalar","mtp","flex","bend","rot"))][:70]:
  print(j, mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j), m.jnt_type[j], m.jnt_qposadr[j], m.jnt_range[j].round(2))
mujoco.mj_saveModel(m, sys.argv[2], None)
