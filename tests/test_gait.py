"""The gait module on synthetic walking: stride cutting, chatter, signs."""

from __future__ import annotations

import numpy as np
import pytest

from sendg1.eval import gait

DT = 0.02
CYCLE = 40  # steps: a 0.8 s stride
STANCE = 24  # 60 % stance


def _walk(steps: int = 400, chatter: bool = False, roll: float = 0.0) -> gait.Recording:
  """Two feet half a cycle apart; foot force 300 N in stance, 0 in swing."""
  t = np.arange(steps)
  fz = np.zeros((steps, 1, 2))
  for f, phase in enumerate((0, CYCLE // 2)):
    fz[:, 0, f] = np.where((t - phase) % CYCLE < STANCE, 300.0, 0.0)
  if chatter:  # one-step dips inside every left stance
    fz[(t % CYCLE) == 10, 0, 0] = 0.0
  grf = np.zeros((steps, 1, 2, 3))
  grf[..., 2] = fz
  q = np.zeros((steps, 1, 2, 6))
  q[..., 1] = roll  # hip roll, same value on both legs
  z = np.zeros((steps, 1, 2, 6))
  foot = np.zeros((steps, 1, 2, 3))
  foot[:, 0, 0, 1], foot[:, 0, 1, 1] = 0.1, -0.1
  return gait.Recording(
    dt=DT,
    mass=33.7,
    leg_q=q,
    leg_qd=z,
    leg_qdd=z,
    leg_tau=z,
    leg_limit=np.array([88.0, 139.0, 88.0, 139.0, 50.0, 50.0]),
    all_tau=np.zeros((steps, 1, 29)),
    all_qd=np.zeros((steps, 1, 29)),
    grf=grf,
    cop=np.zeros((steps, 1, 2, 2)),
    foot_pos=foot,
    foot_vel=np.zeros((steps, 1, 2, 3)),
    base_pos=np.zeros((steps, 1, 3)),
    base_rpy=np.zeros((steps, 1, 3)),
    valid=np.ones((steps, 1), dtype=bool),
  )


@pytest.mark.parametrize("chatter", [False, True])
def test_strides_are_cut_at_touchdowns_and_chatter_is_ignored(chatter: bool) -> None:
  rows = gait.strides(_walk(chatter=chatter))
  left = [r for r in rows if r["side"] == "left"]
  assert len(left) == 8  # touchdowns at 40, 80, ..., 360 (it starts in stance)
  for r in rows:
    assert r["scalars"]["stride_time"] == pytest.approx(CYCLE * DT)
    assert r["scalars"]["stance_pct"] == pytest.approx(100 * STANCE / CYCLE)
    assert r["scalars"]["double_support_pct"] == pytest.approx(20.0)


def test_frontal_angles_follow_the_clinical_sign_on_both_legs() -> None:
  # The same hip roll on both legs swings both feet to the robot's left: the
  # left leg abducts, the right adducts.
  rows = gait.strides(_walk(roll=0.1))
  k = [a[0] for a in gait.ANGLES].index("hip_adduction")
  by_side = {r["side"]: r["curves"]["angle"][0, k] for r in rows}
  assert by_side["left"] == pytest.approx(-np.degrees(0.1))
  assert by_side["right"] == pytest.approx(np.degrees(0.1))


def test_invalid_steps_drop_their_strides() -> None:
  rec = _walk()
  rec.valid[200:] = False
  assert all(r["start"] + CYCLE <= 200 for r in gait.strides(rec))


def test_summary_has_curves_on_the_normalised_cycle() -> None:
  s = gait.summarize(gait.strides(_walk()))
  assert len(s["curves"]["angle"]["mean"]) == gait.N_POINTS
  assert s["variability_cv_pct"]["stride_time"] == pytest.approx(0.0)
