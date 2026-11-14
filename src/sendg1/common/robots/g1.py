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
from dataclasses import replace
from typing import Literal

import mujoco
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import (
  FULL_COLLISION,
  KNEES_BENT_KEYFRAME,
  get_g1_robot_cfg,
)
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import (
  G1_ACTION_SCALE as _MJLAB_G1_ACTION_SCALE,
)
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import get_spec as _get_g1_spec
from mjlab.entity import EntityCfg

from sendg1.common.robots.slipper import G1_SOLE_GEOMS, add_slipper
from sendg1.common.tactile.layout import G1_SOLE_Z, SOLE_Z

DofSet = Literal["full29", "legs12", "legs_waist15"]
"""Which joints the action term drives. Phase 1 uses ``full29``."""

G1_NUM_DOF = 29

G1_TORSO_BODY = "torso_link"
G1_PELVIS_BODY = "pelvis"

G1_FOOT_SITES: tuple[str, ...] = ("left_foot", "right_foot")
"""Sites at the sole of each foot, used for clearance/slip rewards and the
per-foot terrain height scan."""

G1_FOOT_GEOMS: tuple[str, ...] = G1_SOLE_GEOMS
"""One collision box per foot: the slipper's rubber sole
(``sendg1.common.robots.slipper``). It replaces mjlab's 7 capsules per foot.
On flat ground MuJoCo contacts its 4 corners, which is why the tactile model
(sendg1.common.tactile.model) takes the net wrench from the contacts rather
than binning contact points onto taxels.
"""

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
"""mjlab's FULL_COLLISION with its foot-capsule rules moved to the sole box.
Friction is the stock 0.6; ``foot_friction`` randomises it (0.3-1.2) anyway."""

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


# ---------------------------------------------------------------------------
# Ball hands
# ---------------------------------------------------------------------------
#
# The stock G1 ends each arm in Unitree's passive rubber hand: a fixed,
# five-fingered, slightly curled hand (visual mesh ``*_rubber_hand.STL``), which
# the MJCF collides as one capsule. The ball-hand variant replaces both with a
# sphere -- the model of a printed ball end-effector mounted in the rubber
# hand's place. A ball has one well-defined contact, cannot hook an edge, and
# gets its own friction, so hand grip no longer depends on what it touches.
#
# Only the hand changes. The wrist_yaw_link keeps its explicit inertial
# (0.255 kg with the rubber hand): the printed ball is ASSUMED to weigh about
# the same. Forearm and wrist capsules are untouched.

BALL_HAND_RADIUS = 0.035
"""m (70 mm diameter): just over the rubber hand's 67 mm thickness, and the
same radius as the stock hand's collision capsule."""
BALL_HAND_MOUNT_X = 0.0415
"""m: the wrist_yaw_link's front face, where the rubber hand bolts on. The
wrist's own mesh ends here."""
BALL_HAND_CENTER_X = BALL_HAND_MOUNT_X + BALL_HAND_RADIUS - 0.005
"""m along the wrist_yaw_link x axis: the ball sits ON the mounting face,
overlapping it by 5 mm so there is no gap. Its front reaches 0.112 m -- short
of the rubber hand's fingertips (~0.17 m)."""
BALL_HAND_FRICTION = 0.8
"""Sliding friction of the ball's cover (TPU / rubber). An assumption to be
measured on the printed part."""
BALL_HAND_PRIORITY = 4
"""Above the box-push box (3) and floor (2): every hand contact uses the ball's
friction and contact dimension."""

BALL_HAND_GEOMS: tuple[str, ...] = ("left_hand_collision", "right_hand_collision")


def get_g1_ball_hands_spec() -> mujoco.MjSpec:
  """The G1 MJCF with each rubber hand replaced by a ball.

  The collision geom keeps its name (``*_hand_collision``) so every existing
  collision rule and sensor pattern still finds it; only its shape changes. The
  rubber-hand visual mesh is swapped for a matching visual sphere.
  """
  spec = get_g1_spec()
  for side in ("left", "right"):
    body = spec.body(f"{side}_wrist_yaw_link")
    for geom in list(body.geoms):
      if geom.meshname == f"{side}_rubber_hand":
        spec.delete(geom)
    col = spec.geom(f"{side}_hand_collision")
    col.type = mujoco.mjtGeom.mjGEOM_SPHERE
    col.fromto = [float("nan")] * 6  # unset: position by pos, not fromto
    col.pos = (BALL_HAND_CENTER_X, 0.0, 0.0)
    col.size = (BALL_HAND_RADIUS, 0.0, 0.0)
    body.add_geom(
      name=f"{side}_hand_ball_visual",
      type=mujoco.mjtGeom.mjGEOM_SPHERE,
      pos=(BALL_HAND_CENTER_X, 0.0, 0.0),
      size=(BALL_HAND_RADIUS, 0.0, 0.0),
      rgba=(0.15, 0.15, 0.15, 1.0),
      group=2,
      contype=0,
      conaffinity=0,
      density=0.0,
    )
  return spec


BALL_HANDS_COLLISION = replace(
  SLIPPER_COLLISION,
  # Friction needs condim 3: the stock rule makes every non-foot geom
  # frictionless (condim 1), and with the ball's top priority its condim is the
  # one a hand contact uses.
  condim={
    SOLE_RE: 3,
    r"^(left|right)_hand_collision$": 3,
    ".*_collision": 1,
  },
  priority={
    SOLE_RE: 1,
    r"^(left|right)_hand_collision$": BALL_HAND_PRIORITY,
    ".*": 0,
  },
  friction={
    SOLE_RE: (0.6,),
    r"^(left|right)_hand_collision$": (BALL_HAND_FRICTION,),
  },
)
"""SLIPPER_COLLISION, plus the ball hands' contact settings."""


def get_g1_ball_hands_cfg() -> EntityCfg:
  """A fresh 29-DoF G1 with ball hands (see :func:`get_g1_ball_hands_spec`)."""
  cfg = get_g1_cfg()
  cfg.spec_fn = get_g1_ball_hands_spec
  cfg.collisions = (BALL_HANDS_COLLISION,)
  return cfg
