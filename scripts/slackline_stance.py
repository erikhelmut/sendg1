"""Solve the G1's tandem slackline stance and write it to
``src/sendg1/common/slackline/stance.py``.

The stance, as slackliners teach it: both feet on the line, the front foot
pointing straight along it, the back foot turned out ~45 deg with its arch
across the line; pelvis level and facing along the line, the whole-body CoM
over the midpoint of the feet. Solved by least squares on the leg and waist
joints (arms stay at the default pose) for both mirror images: left foot in
front and right foot in front.

  uv run python scripts/slackline_stance.py
"""

from __future__ import annotations

import math
from pathlib import Path

import mujoco
import numpy as np
from mjlab.entity import Entity
from scipy.optimize import least_squares

from sendg1.common.robots.g1 import G1_FOOT_SITES, get_g1_cfg
from sendg1.common.slackline.mdp import BACK_FOOT_YAW, STANCE_LENGTH

OUT = Path(__file__).resolve().parents[1] / "src" / "sendg1" / "common" / "slackline" / "stance.py"
FREE = (".*hip.*", ".*knee.*", ".*ankle.*", "waist.*")
REG = 0.02
"""Pull toward the default pose (per rad), so the solution stays natural."""
BRACE_CLEARANCE = 0.035
"""Lowest point of each ankle-linkage brace (a capsule on the knee link,
just behind the heel) above the soles (m). A deeply flexed back ankle puts it
at sole height, on the webbing that runs out behind the turned-out heel."""


def _model() -> tuple[mujoco.MjModel, mujoco.MjData]:
  m = Entity(get_g1_cfg()).spec.compile()
  d = mujoco.MjData(m)
  mujoco.mj_resetDataKeyframe(m, d, 0)
  return m, d


def _yaw(mat: np.ndarray) -> float:
  return math.atan2(mat[1, 0], mat[0, 0])


def solve(front: str) -> dict[str, float]:
  import re

  m, d = _model()
  q0 = d.qpos.copy()
  names = [m.joint(j).name for j in range(1, m.njnt)]
  free = [j for j, n in enumerate(names) if any(re.fullmatch(p, n) for p in FREE)]
  adr = [m.jnt_qposadr[j + 1] for j in free]
  lo = np.array([m.jnt_range[j + 1, 0] for j in free])
  hi = np.array([m.jnt_range[j + 1, 1] for j in free])
  sites = {s.split("_")[0]: m.site(s).id for s in G1_FOOT_SITES}
  back = "right" if front == "left" else "left"
  # Turned out: the left foot's toes to the left (+yaw), the right's to the right.
  back_yaw = BACK_FOOT_YAW if back == "left" else -BACK_FOOT_YAW
  targets = {front: (STANCE_LENGTH / 2, 0.0), back: (-STANCE_LENGTH / 2, back_yaw)}
  root = m.body("pelvis").id
  braces = [m.geom(f"{side}_linkage_brace_collision").id for side in ("left", "right")]

  def brace_bottom(g: int) -> float:
    axis = d.geom_xmat[g].reshape(3, 3)[:, 2]
    half, radius = m.geom_size[g, 1], m.geom_size[g, 0]
    return float(d.geom_xpos[g, 2] - abs(axis[2]) * half - radius)

  def residual(x: np.ndarray) -> np.ndarray:
    d.qpos[:] = q0
    d.qpos[0] = x[0]
    d.qpos[2] = x[1]
    d.qpos[adr] = x[2:]
    mujoco.mj_kinematics(m, d)
    mujoco.mj_comPos(m, d)
    r = []
    for side, (fx, fyaw) in targets.items():
      sid = sites[side]
      p, mat = d.site_xpos[sid], d.site_xmat[sid].reshape(3, 3)
      r += [10 * (p[0] - fx), 10 * p[1], 10 * p[2]]
      r += [3 * math.sin(_yaw(mat) - fyaw), 3 * mat[2, 0], 3 * mat[2, 1]]  # yaw, flat sole
    r += [20 * max(0.0, BRACE_CLEARANCE - brace_bottom(g)) for g in braces]
    com = d.subtree_com[root]
    r += [10 * com[0], 10 * com[1]]
    r += list(REG * (x[2:] - q0[adr]))
    return np.array(r)

  x0 = np.concatenate([[0.0, q0[2]], q0[adr]])
  sol = least_squares(
    residual,
    x0,
    bounds=(np.concatenate([[-0.3, 0.5], lo]), np.concatenate([[0.3, 1.0], hi])),
  )
  res = residual(sol.x)
  clear = [brace_bottom(g) for g in braces]
  print(
    f"{front} front: cost {sol.cost:.2e}, foot/CoM residual {np.abs(res[:16]).max():.4f}, "
    f"base z {sol.x[1]:.3f}, brace clearance {min(clear):.3f} m"
  )
  out = {names[j]: float(v) for j, v in zip(free, sol.x[2:])}
  out["__base_x__"] = float(sol.x[0])
  out["__base_z__"] = float(sol.x[1])
  return out


def main() -> None:
  stances = {front: solve(front) for front in ("left", "right")}
  lines = [
    '"""The tandem slackline stance, solved by ``scripts/slackline_stance.py``.',
    "",
    "Generated; do not edit by hand. Per front foot: leg and waist joint angles",
    "(rad; joints not listed stay at the default pose), and the base's offset",
    "along the line (``__base_x__``) and height over the soles (``__base_z__``)",
    'with the stance centred on the origin."""',
    "",
    "STANCES: dict[str, dict[str, float]] = {",
  ]
  for front, joints in stances.items():
    lines.append(f'  "{front}": {{')
    lines += [f'    "{k}": {v:.5f},' for k, v in joints.items()]
    lines.append("  },")
  lines.append("}")
  OUT.write_text("\n".join(lines) + "\n")
  print(f"wrote {OUT}")


if __name__ == "__main__":
  main()
