"""Unitree G1 robot definition for this study.

We standardise on mjlab's 29-DoF MJCF (``g1_29dof_rev_1_0``), all 29 joints
actuated. mjlab ships no other G1 variant.

The actuated set is exposed as a config axis (:class:`DofSet`) rather than
hard-coded, because a legs-only ablation is a plausible robustness check if the
29-DoF results turn out noisy -- upper-body motion is a confound for a study
about the feet. It is NOT the Phase 1 default, for three reasons:

  1. Every reward weight mjlab ships was tuned against 29 actuated DoF.
  2. Un-actuating a joint does not freeze it. The position actuators are
     compliant, so arm joints excluded from the action term still drift up to
     ~0.5 rad from the default pose under gravity and gait dynamics. You get
     passive spring-loaded limbs, which match neither the real robot nor a clean
     legs-only model. Truly freezing the upper body means editing the spec.
  3. The real robot has arms that move.
"""

from __future__ import annotations

import re
from typing import Literal

from mjlab.asset_zoo.robots.unitree_g1.g1_constants import (
  G1_ACTION_SCALE as _MJLAB_G1_ACTION_SCALE,
)
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import get_g1_robot_cfg
from mjlab.entity import EntityCfg

DofSet = Literal["full29", "legs12", "legs_waist15"]
"""Which joints the action term drives. Phase 1 uses ``full29``."""

G1_NUM_DOF = 29

G1_TORSO_BODY = "torso_link"
G1_PELVIS_BODY = "pelvis"

G1_FOOT_SITES: tuple[str, ...] = ("left_foot", "right_foot")
"""Sites at the sole of each foot, used for clearance/slip rewards and the
per-foot terrain height scan."""

G1_FOOT_GEOMS: tuple[str, ...] = tuple(
  f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 8)
)
"""The 7 collision capsules per foot, laid out fore-aft across the sole.

Relevant beyond friction randomisation: this is a ready-made discretisation of
the sole surface. When the 26-taxel tactile array is modelled, a per-geom
``ContactSensorCfg`` over these names is the natural starting point -- MuJoCo's
``mjSENS_TOUCH`` is NOT exposed by mjlab's ``BuiltinSensorCfg`` in 1.6.0, so
contact sensors are the available path.
"""

G1_ACTION_SCALE: dict[str, float] = dict(_MJLAB_G1_ACTION_SCALE)
"""Per-joint action scale, ``0.25 * effort_limit / stiffness``. Keys are regexes."""

_LEG_PATTERNS: tuple[str, ...] = (".*_hip_.*", ".*_knee_joint", ".*_ankle_.*")
_WAIST_PATTERNS: tuple[str, ...] = ("waist_.*",)

_LEG_JOINTS: tuple[str, ...] = tuple(
  f"{side}_{joint}"
  for side in ("left", "right")
  for joint in (
    "hip_pitch_joint",
    "hip_roll_joint",
    "hip_yaw_joint",
    "knee_joint",
    "ankle_pitch_joint",
    "ankle_roll_joint",
  )
)
_WAIST_JOINTS: tuple[str, ...] = (
  "waist_yaw_joint",
  "waist_roll_joint",
  "waist_pitch_joint",
)


def leg_joint_names() -> tuple[str, ...]:
  return _LEG_JOINTS


def actuator_patterns(dof_set: DofSet) -> tuple[str, ...]:
  """Regexes selecting which actuators the action term drives."""
  if dof_set == "full29":
    return (".*",)
  if dof_set == "legs12":
    return _LEG_PATTERNS
  if dof_set == "legs_waist15":
    return _LEG_PATTERNS + _WAIST_PATTERNS
  raise ValueError(f"unknown dof_set: {dof_set!r}")


def resolve_action_scale(dof_set: DofSet) -> dict[str, float]:
  """Action scale restricted to the joints ``dof_set`` actually drives.

  Necessary, not cosmetic: mjlab resolves the scale dict against the *selected*
  joints and raises if any regex key matches nothing. Subsetting
  ``actuator_names`` without also subsetting the scale dict throws at env
  construction.
  """
  if dof_set == "full29":
    return dict(G1_ACTION_SCALE)

  if dof_set == "legs12":
    selected = _LEG_JOINTS
  else:
    selected = _LEG_JOINTS + _WAIST_JOINTS

  return {
    pattern: value
    for pattern, value in G1_ACTION_SCALE.items()
    if any(re.fullmatch(pattern, joint) for joint in selected)
  }


def get_g1_cfg() -> EntityCfg:
  """A fresh 29-DoF G1 entity config (knees-bent keyframe, full collisions)."""
  return get_g1_robot_cfg()
