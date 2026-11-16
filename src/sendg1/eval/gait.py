"""Gait analysis: from recorded walking to per-stride human-gait measures.

Pure numpy, no simulator: ``scripts/gait_analysis.py`` records a policy's
walking (``Recording``), this module cuts it into strides and measures them,
and ``scripts/gait_report.py`` compares conditions. Everything here reads
simulator ground truth (true joint angles, actuator torques, the soles' true
contact forces), the same for a blind and a tactile policy.

A stride runs from one foot's touchdown to its next touchdown (0-100 % of the
gait cycle); stance ends at that foot's toe-off. Angles follow the clinical
gait conventions, so the curves read like a human gait report:

=============  ==================  =============================  ==============
plane          joint               positive                       G1 joint
=============  ==================  =============================  ==============
sagittal       hip                 flexion (thigh forward)        -hip_pitch
sagittal       knee                flexion                        knee
sagittal       ankle               dorsiflexion (toes up)         -ankle_pitch
frontal        hip                 adduction (towards midline)    -/+ hip_roll
frontal        ankle               inversion (sole inwards)       -/+ ankle_roll
transverse     hip                 internal rotation (toes in)    -/+ hip_yaw
=============  ==================  =============================  ==============

(-/+: left / right; the G1's joints share one axis direction on both legs, so
the frontal and transverse signs flip on the left leg.) The G1 knee is a
single hinge: it has no ab/adduction. Torques use the same signs (a positive
hip torque is a flexion moment), per kg of robot mass; power is torque times
joint velocity, positive when the motor does work on the leg.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

SIDES = ("left", "right")
N_POINTS = 101
"""Samples per normalised cycle (0, 1, ..., 100 %)."""

ON_N = 30.0
OFF_N = 10.0
"""Contact hysteresis on the sole's true normal force [N]: on above ON_N,
off below OFF_N (~9 % and ~3 % of body weight). One threshold would chatter
on a foot that is just brushing the ground."""
MIN_SWING_S = 0.06
"""Shorter breaks in contact are chatter (a sole bouncing or rocking on an
edge), not swings: they are filled in."""
MIN_STRIDE_S = 0.35
MAX_STRIDE_S = 2.0
SATURATION = 0.95
"""A joint counts as torque-saturated at |torque| >= 95 % of its limit."""
G = 9.81

# (name, plane, G1 joint, sign on the left leg, sign on the right leg)
ANGLES = (
  ("hip_flexion", "sagittal", "hip_pitch", -1.0, -1.0),
  ("knee_flexion", "sagittal", "knee", 1.0, 1.0),
  ("ankle_dorsiflexion", "sagittal", "ankle_pitch", -1.0, -1.0),
  ("hip_adduction", "frontal", "hip_roll", -1.0, 1.0),
  ("ankle_inversion", "frontal", "ankle_roll", -1.0, 1.0),
  ("hip_internal_rotation", "transverse", "hip_yaw", -1.0, 1.0),
)
LEG_JOINTS = ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")


@dataclass
class Recording:
  """One policy's walking at one command, ``T`` policy steps x ``N`` envs.

  Feet are ordered (left, right); leg joints as in ``LEG_JOINTS``; frames are
  the world's unless named."""

  dt: float
  mass: float
  """Robot mass [kg]."""
  leg_q: np.ndarray
  """[T, N, 2, 6] leg joint angles [rad]."""
  leg_qd: np.ndarray
  """[T, N, 2, 6] [rad/s]."""
  leg_qdd: np.ndarray
  """[T, N, 2, 6] [rad/s^2]."""
  leg_tau: np.ndarray
  """[T, N, 2, 6] actuator torques [Nm]."""
  leg_limit: np.ndarray
  """[6] actuator torque limits [Nm] (``LEG_JOINTS`` order)."""
  all_tau: np.ndarray
  """[T, N, J] every actuated joint's torque, for the energy budget."""
  all_qd: np.ndarray
  """[T, N, J]."""
  grf: np.ndarray
  """[T, N, 2, 3] each sole's true contact force in the sole frame (x forward,
  y left, z up) [N]."""
  cop: np.ndarray
  """[T, N, 2, 2] centre of pressure in the sole frame [m]; NaN in the air."""
  foot_pos: np.ndarray
  """[T, N, 2, 3] ankle-roll link position [m]."""
  foot_vel: np.ndarray
  """[T, N, 2, 3] [m/s]."""
  base_pos: np.ndarray
  """[T, N, 3] pelvis position [m]."""
  base_rpy: np.ndarray
  """[T, N, 3] pelvis roll, pitch, yaw [rad]."""
  valid: np.ndarray
  """[T, N] bool: past the warm-up and before the env's first termination."""


def contact_mask(fz: np.ndarray, dt: float) -> np.ndarray:
  """Hysteresis contact along axis 0 of ``fz`` [T, ...], with breaks shorter
  than ``MIN_SWING_S`` filled in."""
  on = np.zeros(fz.shape, dtype=bool)
  state = fz[0] > ON_N
  on[0] = state
  for t in range(1, fz.shape[0]):
    state = np.where(state, fz[t] > OFF_N, fz[t] > ON_N)
    on[t] = state
  max_gap = round(MIN_SWING_S / dt)
  flat = on.reshape(on.shape[0], -1)
  for k in range(flat.shape[1]):
    col = flat[:, k]
    tds, tos = _events(col)
    for to in tos:
      nxt = tds[tds > to]
      if nxt.size and nxt[0] - to < max_gap and col[:to].any():
        col[to : nxt[0]] = True
  return on


def _events(on: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  """Touchdown and toe-off indices of one foot's contact series [T]."""
  edge = np.diff(on.astype(np.int8))
  return np.flatnonzero(edge == 1) + 1, np.flatnonzero(edge == -1) + 1


def _resample(y: np.ndarray, start: int, stop: int) -> np.ndarray:
  """``y[start:stop + 1]`` (along axis 0) onto ``N_POINTS`` evenly spaced
  points, linear in between."""
  seg = y[start : stop + 1]
  src = np.linspace(0.0, 1.0, seg.shape[0])
  dst = np.linspace(0.0, 1.0, N_POINTS)
  flat = seg.reshape(seg.shape[0], -1)
  out = np.stack([np.interp(dst, src, flat[:, k]) for k in range(flat.shape[1])], -1)
  return out.reshape((N_POINTS, *seg.shape[1:]))


def _heading_frame(yaw: float) -> np.ndarray:
  """Rows: forward and left unit vectors of a heading."""
  c, s = np.cos(yaw), np.sin(yaw)
  return np.array([[c, s], [-s, c]])


def _wrap(a: np.ndarray) -> np.ndarray:
  return (a + np.pi) % (2 * np.pi) - np.pi


def strides(rec: Recording) -> list[dict]:
  """Cut every env's walking into strides of each foot and measure them.

  A stride is kept only if it lies wholly inside ``rec.valid``, lasts
  ``MIN_STRIDE_S``-``MAX_STRIDE_S`` and has exactly one toe-off.
  """
  n_envs = rec.valid.shape[1]
  on = contact_mask(rec.grf[..., 2], rec.dt)  # [T, N, 2]
  out: list[dict] = []
  joint_idx = {j: k for k, j in enumerate(LEG_JOINTS)}
  for n in range(n_envs):
    for f, side in enumerate(SIDES):
      other = 1 - f
      tds, tos = _events(on[:, n, f])
      other_tds, _ = _events(on[:, n, other])
      for a, b in itertools.pairwise(tds):
        if not rec.valid[a : b + 1, n].all():
          continue
        dur = (b - a) * rec.dt
        if not MIN_STRIDE_S <= dur <= MAX_STRIDE_S:
          continue
        offs = tos[(tos > a) & (tos < b)]
        if offs.size != 1:
          continue
        to = int(offs[0])
        out.append(
          _measure(rec, n, f, side, other, int(a), to, int(b), other_tds, on, joint_idx)
        )
  return out


def _measure(rec, n, f, side, other, a, to, b, other_tds, on, joint_idx) -> dict:
  dt, m = rec.dt, rec.mass
  dur = (b - a) * dt
  sign = np.array([[s_l, s_r][f] for *_, s_l, s_r in ANGLES])
  cols = [joint_idx[j] for _, _, j, *_ in ANGLES]

  # -- kinematics and kinetics over the cycle --------------------------------
  q = rec.leg_q[:, n, f]
  tau = rec.leg_tau[:, n, f]
  qd = rec.leg_qd[:, n, f]
  curves = {
    "angle": np.degrees(_resample(q[:, cols] * sign, a, b)),
    "torque": _resample(tau[:, cols] * sign, a, b) / m,
    "power": _resample(tau[:, cols] * qd[:, cols], a, b) / m,
  }
  yaw0 = float(np.mean(np.unwrap(rec.base_rpy[a : b + 1, n, 2])))
  pelvis = rec.base_rpy[:, n].copy()
  pelvis[:, 2] = _wrap(pelvis[:, 2] - yaw0)
  # Pelvis tilt (anterior positive), obliquity and rotation as seen from this
  # leg: up on its side and forward on its side positive, the clinical
  # convention, so left and right strides can be averaged.
  side_sign = 1.0 if f == 0 else -1.0
  pel = np.stack([pelvis[:, 1], -side_sign * pelvis[:, 0], side_sign * pelvis[:, 2]], -1)
  curves["pelvis"] = np.degrees(_resample(pel, a, b))

  # -- ground reaction and centre of pressure, over stance --------------------
  grf = rec.grf[:, n, f] / (m * G)
  cop = rec.cop[:, n, f].copy()
  cop[:, 1] *= side_sign  # lateral positive on both feet
  curves["grf_stance"] = _resample(grf[:, [0, 2]], a, to - 1)
  cop_st = cop[a:to]
  if np.isnan(cop_st).any():  # brief force dips: carry the last value
    for k in range(1, cop_st.shape[0]):
      bad = np.isnan(cop_st[k])
      cop_st[k, bad] = cop_st[k - 1, bad]
  curves["cop_stance"] = _resample(np.nan_to_num(cop_st), 0, cop_st.shape[0] - 1)

  # -- spatio-temporal -------------------------------------------------------
  fwd_left = _heading_frame(yaw0)
  step = rec.foot_pos[b, n, f, :2] - rec.foot_pos[a, n, f, :2]
  base_step = rec.base_pos[b, n, :2] - rec.base_pos[a, n, :2]
  opp = other_tds[(other_tds > a) & (other_tds < b)]
  width = np.nan
  if opp.size:
    rel = rec.foot_pos[a, n, f, :2] - rec.foot_pos[a, n, other, :2]
    width = abs(float(fwd_left[1] @ rel))
  both = on[a:b, n, f] & on[a:b, n, other]
  stance = on[a:b, n, f]

  # -- foot ------------------------------------------------------------------
  z = rec.foot_pos[a : b + 1, n, f, 2]
  z_stance = np.median(rec.foot_pos[a:to, n, f, 2])
  v_td = rec.foot_vel[a - 1, n, f] if a > 0 else rec.foot_vel[a, n, f]
  slide = np.linalg.norm(rec.foot_vel[a:to, n, f, :2], axis=-1).sum() * dt
  fz = rec.grf[a:to, n, f, 2]
  rise = np.diff(fz) / dt if fz.size > 1 else np.zeros(1)
  early = max(1, round(0.3 * rise.size))

  # -- energy and effort -----------------------------------------------------
  p_all = rec.all_tau[a:b, n] * rec.all_qd[a:b, n]
  work_pos_all = np.clip(p_all, 0, None).sum() * dt
  p_leg = (tau * qd)[a:b]
  dist = float(np.linalg.norm(base_step))
  frac = np.abs(tau[a:b]) / rec.leg_limit
  dtau = np.diff(tau[a : b + 1], axis=0) / dt

  scalars = {
    "stride_time": dur,
    "cadence": 120.0 / dur,
    "stride_length": float(np.linalg.norm(step)),
    "speed": dist / dur,
    "step_width": width,
    "stance_pct": 100.0 * stance.mean(),
    "double_support_pct": 100.0 * both.mean(),
    "clearance": float(z.max() - z_stance),
    "touchdown_vz": float(-v_td[2]),
    "touchdown_vxy": float(np.linalg.norm(v_td[:2])),
    "stance_slide": float(slide),
    "loading_rate": float(rise[:early].max()) / (m * G),
    "grf_peak": float(fz.max()) / (m * G),
    "cop_start": float(cop_st[0, 0]),
    "cop_end": float(cop_st[-1, 0]),
    "cop_range_x": float(np.nanmax(cop_st[:, 0]) - np.nanmin(cop_st[:, 0])),
    "cop_range_y": float(np.nanmax(cop_st[:, 1]) - np.nanmin(cop_st[:, 1])),
    "cot_mech": work_pos_all / (m * G * max(dist, 1e-3)),
    "qdd_rms": float(np.sqrt(np.mean(rec.leg_qdd[a:b, n, f] ** 2))),
  }
  for k, j in enumerate(LEG_JOINTS):
    scalars[f"{j}_peak_torque"] = float(np.abs(tau[a:b, k]).max()) / m
    scalars[f"{j}_peak_frac"] = float(frac[:, k].max())
    scalars[f"{j}_saturated_pct"] = 100.0 * float((frac[:, k] >= SATURATION).mean())
    scalars[f"{j}_work_pos"] = float(np.clip(p_leg[:, k], 0, None).sum() * dt) / m
    scalars[f"{j}_work_neg"] = float(np.clip(p_leg[:, k], None, 0).sum() * dt) / m
    scalars[f"{j}_torque_rate_rms"] = float(np.sqrt(np.mean(dtau[:, k] ** 2)))
    scalars[f"{j}_qdd_rms"] = float(np.sqrt(np.mean(rec.leg_qdd[a:b, n, f, k] ** 2)))
  for (name, *_), k in zip(ANGLES, range(len(ANGLES))):
    c = curves["angle"][:, k]
    scalars[f"{name}_rom"] = float(c.max() - c.min())
  return {"env": n, "side": side, "start": a, "scalars": scalars, "curves": curves}


def summarize(rows: list[dict]) -> dict:
  """Mean and SD of every scalar and curve over strides, plus stride-to-stride
  variability (coefficient of variation per env and foot, then averaged)."""
  if not rows:
    return {"n_strides": 0}
  keys = rows[0]["scalars"].keys()
  sc = {k: np.array([r["scalars"][k] for r in rows], dtype=float) for k in keys}
  out = {
    "n_strides": len(rows),
    "scalars": {k: [float(np.nanmean(v)), float(np.nanstd(v))] for k, v in sc.items()},
  }
  cv = {}
  for k in ("stride_time", "stride_length", "step_width", "stance_pct"):
    per = {}
    for r in rows:
      per.setdefault((r["env"], r["side"]), []).append(r["scalars"][k])
    vals = [np.nanstd(v) / abs(np.nanmean(v)) for v in per.values() if len(v) >= 3]
    cv[k] = float(np.nanmean(vals)) * 100.0 if vals else float("nan")
  out["variability_cv_pct"] = cv
  out["curves"] = {}
  for name in rows[0]["curves"]:
    stack = np.stack([r["curves"][name] for r in rows])
    out["curves"][name] = {
      "mean": stack.mean(0).tolist(),
      "sd": stack.std(0).tolist(),
    }
  return out
