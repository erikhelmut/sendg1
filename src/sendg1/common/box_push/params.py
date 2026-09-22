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
  foot  / patch   -> patch mu     (patch priority 2 > foot 1)
  box   / floor   -> box mu       (box priority 3 > floor 2)
  box   / patch   -> box mu       (box priority 3 > patch 2)
  hand  / box     -> box mu       (box priority 3 > hand 0)   <- a coupling

The last line is an assumption worth knowing about: the hands grip the box with
the box-floor coefficient, so a slippery box is also slippery to push.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

G1_MASS_KG = 33.34
"""Total mass of mjlab's 29-DoF G1 (sum of body_mass; checked in
tests/test_box_push.py so it cannot drift from the model)."""

GRAVITY = 9.81

FLOOR_PRIORITY = 2
BOX_PRIORITY = 3


@dataclass(frozen=True)
class BoxPushRanges:
  """Per-episode sampling ranges. Uniform unless stated otherwise."""

  floor_mu: tuple[float, float] = (0.2, 1.0)
  """Floor friction, i.e. foot-floor friction (see the module docstring)."""
  box_mass: tuple[float, float] = (10.0, 40.0)
  """kg. Uniform density: inertia follows mass and size."""
  box_mu: tuple[float, float] = (0.2, 0.8)
  """Box-floor friction, independent of the floor's."""
  box_width: tuple[float, float] = (0.4, 0.6)
  """m, along world y at spawn (across the push direction)."""
  box_depth: tuple[float, float] = (0.4, 0.6)
  """m, along world x at spawn (along the push direction)."""
  box_height: tuple[float, float] = (0.6, 0.9)
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
  """m. Success when the box centre is inside this disc (xy)."""


@dataclass(frozen=True)
class PatchCfg:
  """Optional low-friction floor patches (``low_friction_patches=True``)."""

  count: tuple[int, int] = (1, 3)
  """Inclusive. Unused patch geoms are parked below the floor."""
  mu: tuple[float, float] = (0.1, 0.3)
  size: tuple[float, float] = (0.4, 1.0)
  """m, side length, sampled independently for x and y."""
  path_fraction: tuple[float, float] = (0.1, 0.9)
  """Where along the robot-start -> target segment a patch centre lands."""
  lateral: float = 0.3
  """m, +/- offset of a patch centre across that segment."""
  thickness: float = 0.01
  """m. The patch top sits PATCH_LIFT above the floor, so feet touch the patch
  rather than the plane beneath it."""


PATCH_LIFT = 0.001
MAX_PATCHES = 3


@dataclass(frozen=True)
class CurriculumCfg:
  """Global difficulty level alpha in [0, 1], moved by success rate.

  At alpha = 0: box mass 10 kg, floor mu 1.0, box mu 0.2 (the easiest corner).
  At alpha = 1: the full ranges. In between each range grows linearly from its
  easy end. Box size, target and spawn geometry are always full-range.
  """

  window: int = 512
  """Episodes pooled before alpha is reconsidered."""
  promote: float = 0.5
  """Success rate over the window at or above which alpha rises by ``step``."""
  demote: float = 0.2
  """Success rate below which alpha falls by ``step``."""
  step: float = 0.1


def curriculum_ranges(r: BoxPushRanges, alpha: float) -> dict[str, tuple[float, float]]:
  """The sampling ranges at difficulty ``alpha``: easy end first."""
  a = float(np.clip(alpha, 0.0, 1.0))
  m0, m1 = r.box_mass
  f0, f1 = r.floor_mu
  b0, b1 = r.box_mu
  return {
    "box_mass": (m0, m0 + a * (m1 - m0)),
    "floor_mu": (f0 + (1.0 - a) * (f1 - f0), f1),
    "box_mu": (b0, b0 + a * (b1 - b0)),
  }


# ---------------------------------------------------------------------------
# Feasibility (quasi-static, used by scripts/box_push_feasibility.py and tests)
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
