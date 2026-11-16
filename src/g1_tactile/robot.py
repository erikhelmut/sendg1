"""mjlab's 29-DoF Unitree G1 (``g1_29dof_rev_1_0``) wearing the tactile slipper.

Use :func:`get_g1_cfg` in place of mjlab's ``get_g1_robot_cfg``. It swaps in
the slipper (mesh, 200 g per foot, one box collision sole per foot instead of
the 7 foot capsules) and spawns the robot 8 mm higher, by the thickness of the
sole stack.
"""

from __future__ import annotations

from dataclasses import replace

import mujoco
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import (
  FULL_COLLISION,
  KNEES_BENT_KEYFRAME,
  get_g1_robot_cfg,
)
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import get_spec as _get_g1_spec
from mjlab.entity import EntityCfg

from g1_tactile.layout import G1_SOLE_Z, SOLE_Z
from g1_tactile.slipper import G1_SOLE_GEOMS, add_slipper

G1_FOOT_GEOMS: tuple[str, ...] = G1_SOLE_GEOMS
"""One collision box per foot: the slipper's rubber sole (``g1_tactile.slipper``).
It replaces mjlab's 7 capsules per foot. On flat ground MuJoCo contacts its 4
corners, which is why the tactile model (``g1_tactile.model``) takes the net
wrench from the contacts rather than binning contact points onto taxels."""

SOLE_RE = r"^(left|right)_sole_collision$"

STANDING_BASE_Z = KNEES_BENT_KEYFRAME.pos[2] + (G1_SOLE_Z - SOLE_Z)
"""Base height with the soles on the floor: mjlab's knees-bent keyframe
(0.76 m) lifted by the slipper stack (8 mm)."""

SLIPPER_COLLISION = replace(
  FULL_COLLISION,
  condim={SOLE_RE: 3, ".*_collision": 1},
  priority={SOLE_RE: 1, ".*": 0},
  friction={SOLE_RE: (0.6,)},
)
"""mjlab's FULL_COLLISION with its foot-capsule rules moved to the sole box
(friction 0.6, mjlab's stock foot value)."""


def get_g1_spec() -> mujoco.MjSpec:
  """mjlab's 29-DoF G1 wearing the tactile slipper."""
  return add_slipper(_get_g1_spec())


def get_g1_cfg() -> EntityCfg:
  """A fresh 29-DoF G1 entity config (knees-bent keyframe, full collisions),
  wearing the slipper."""
  cfg = get_g1_robot_cfg()
  cfg.spec_fn = get_g1_spec
  cfg.collisions = (SLIPPER_COLLISION,)
  cfg.init_state = replace(cfg.init_state, pos=(*cfg.init_state.pos[:2], STANDING_BASE_Z))
  return cfg
