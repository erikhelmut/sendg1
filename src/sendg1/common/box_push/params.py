"""Every number that defines the box-pushing task's world, in one place.

The ranges below are the task specification. Changing one changes the task:
every run made before must be treated as a different experiment (the config
fingerprints say so).

Friction model
--------------
MuJoCo does not have a per-pair friction coefficient by default: for two geoms
of equal ``priority`` it uses the element-wise MAX of their frictions, and for
unequal priorities it uses the higher-priority geom's. The G1's foot capsules
ship with priority 1 and are randomised elsewhere (0.3-1.2), so with max-mixing
a floor of mu 0.2 would silently become >= 0.3 under the feet. The priorities
below make every contact the study cares about take exactly one sampled value:

  foot  / floor   -> floor mu     (floor priority 2 > foot 1)
  box   / floor   -> box mu       (box priority 3 > floor 2)
  ball  / box     -> ball mu 0.8  (ball-hand priority 4 > box 3)

With the stock rubber hands (``ball_hands=False``) the last line becomes
hand / box -> box mu (box priority 3 > hand 0): the hands would grip with the
box-floor coefficient, so a slippery box would also be slippery to push. The
ball hands remove that coupling.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

G1_MASS_KG = 33.74
"""Total mass of mjlab's 29-DoF G1 with the slippers on (33.34 kg + 2 x 0.2 kg;
sum of body_mass; checked in
tests/test_box_push.py so it cannot drift from the model)."""


FLOOR_PRIORITY = 2
BOX_PRIORITY = 3


@dataclass(frozen=True)
class BoxPushRanges:
  """Per-episode sampling ranges. Uniform unless stated otherwise.

  Mass is NOT sampled directly. The episode's difficulty is the traction
  margin -- how much of the robot's foot traction is left over after sliding
  the box -- and the mass follows from it:

      m_box = (1 - margin) * mu_floor * m_robot / mu_box,  clipped to box_mass

  Sampling the margin instead of the mass is what makes a chosen share of
  episodes sit near the limit. With mass and floor mu drawn independently the
  near-limit band is whatever falls out (11% for the first version's ranges,
  against 23% infeasible); here it is a design parameter.
  """

  traction_margin: tuple[float, float] = (-0.05, 0.40)
  """1 - required/available push force. Negative: the feet must slip to move
  the box at all. The upper end keeps a comfortable share in the mix."""
  floor_mu: tuple[float, float] = (0.25, 0.7)
  """Floor friction, i.e. foot-floor friction (see the module docstring).

  Capped at 0.7 deliberately. Near the traction limit the required push force
  IS the available traction, mu_floor * m_robot * g, so training up to mu 1.0
  would demand pushes of a full body weight -- far past the ~50% body weight
  VOFA reports as its hardest real-world case. At 0.7 the median episode needs
  125 N (38% of the robot's weight). Evaluation still scores mu above this, as
  generalisation."""
  box_mu: tuple[float, float] = (0.35, 0.7)
  """Box-floor friction, independent of the floor's."""
  box_mass: tuple[float, float] = (8.0, 50.0)
  """kg. Clip on the mass the margin implies; ~10% of draws hit an end."""

  box_width: tuple[float, float] = (0.7, 0.9)
  """m, across the push direction."""
  box_depth: tuple[float, float] = (0.7, 0.9)
  """m, along the push direction. Wide and low on purpose: a box tips before
  it slides when the push height exceeds depth / (2 mu_box), so the first
  version's 0.4-0.6 m depth tipped 72% of boxes at a 0.7 m push. At 0.7-0.9 m
  it is ~3% at the 0.55 m push height the G1 reaches."""
  box_height: tuple[float, float] = (0.6, 0.8)
  """m. The top face is the box height: the box sits on the floor."""

  robot_gap: tuple[float, float] = (0.6, 1.0)
  """m from the robot's base to the box's NEAR FACE (not its centre)."""
  robot_yaw_noise: float = 0.1
  """rad, +/- about facing the box."""

  target_distance: tuple[float, float] = (2.0, 4.0)
  """m from the box's initial centre to the target centre."""
  target_angle: tuple[float, float] = (-np.pi / 6, np.pi / 6)
  """rad, relative to straight ahead (+x, the robot's initial facing)."""
  target_radius: float = 0.5
  """m. The box centre must be inside this disc (xy) ..."""
  success_dwell_s: float = 2.0
  """... and stay there this long. A dwell time, as in VOFA (0.3 m for 2 s):
  it does not reward a box that skids through the target."""

  push_height_ref: float = 0.55
  """m. Reference push height for the tipping report only; the G1's hands
  reach roughly this high when leaning into a box."""


def mass_from_margin(
  margin: np.ndarray, floor_mu: np.ndarray, box_mu: np.ndarray,
  mass_limits: tuple[float, float], robot_mass: float = G1_MASS_KG,
) -> np.ndarray:
  """Box mass that puts an episode at the requested traction margin."""
  m = (1.0 - margin) * floor_mu * robot_mass / box_mu
  return np.clip(m, *mass_limits)


# ---------------------------------------------------------------------------
# Feasibility (quasi-static; tests check the ranges with these)
# ---------------------------------------------------------------------------


def traction_margin(
  floor_mu: np.ndarray, box_mass: np.ndarray, box_mu: np.ndarray, robot_mass: float = G1_MASS_KG
) -> np.ndarray:
  """Fraction of the robot's traction budget left over after sliding the box.

  ``1 - (mu_box * m_box * g) / (mu_foot * m_robot * g)``. Negative: the robot
  cannot push the box at all without its feet slipping. Idealised: horizontal
  push only, no vertical hand force changing either normal load, no dynamics.
  """
  return 1.0 - (box_mu * box_mass) / (floor_mu * robot_mass)


def tips_before_sliding(
  push_height: np.ndarray, box_depth: np.ndarray, box_mu: np.ndarray
) -> np.ndarray:
  """Whether a horizontal push at ``push_height`` tips the box before it slides.

  Tipping about the leading bottom edge needs ``F * h > m g d / 2``; sliding
  needs ``F > mu m g``. So a push tips first when ``h > d / (2 mu)``, whatever
  the mass. Ignores any downward hand force, which would help.
  """
  return push_height > box_depth / (2.0 * box_mu)
