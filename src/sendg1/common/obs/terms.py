"""The single registry of observation terms, each tagged with its availability.

This module is the source of truth that ``tests/test_actor_allowlist.py``
enforces against. A term that is not declared here cannot appear in any
observation group -- the test fails on unknown terms as well as on
non-deployable ones, so there is no way to slip a privileged signal into a
policy by forgetting to register it.

Availability
------------
``onboard``     Measurable on the real blind G1 today. Legal in an actor group.
``oracle``      A quantity the tactile sensor WILL make measurable, but whose
                current implementation is idealised (noiseless, zero-latency,
                perfectly calibrated). Legal in an actor group ONLY for obs sets
                explicitly marked diagnostic.
``privileged``  Simulation state with no hardware counterpart, or requiring a
                perception/estimation stack we do not have. Critic only, always.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

from mjlab.envs import mdp as envs_mdp
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.tasks.velocity import mdp as vel_mdp
from mjlab.utils.noise import UniformNoiseCfg as Unoise

Availability = Literal["onboard", "oracle", "privileged"]

TERRAIN_SCAN_SENSOR = "terrain_scan"
TERRAIN_SCAN_MAX_DISTANCE = 5.0
FOOT_HEIGHT_SENSOR = "foot_height_scan"
FOOT_CONTACT_SENSOR = "feet_ground_contact"


@dataclass(frozen=True)
class TermSpec:
  """One observation term, plus the evidence for its availability tag."""

  build: Callable[[], ObservationTermCfg]
  availability: Availability
  rationale: str
  """Cite the hardware signal, or state why the quantity is unavailable."""


# ---------------------------------------------------------------------------
# Deployable (onboard) terms -- the blind G1's real sensor suite.
# ---------------------------------------------------------------------------

def _base_ang_vel() -> ObservationTermCfg:
  return ObservationTermCfg(
    func=envs_mdp.builtin_sensor,
    params={"sensor_name": "robot/imu_ang_vel"},
    noise=Unoise(n_min=-0.2, n_max=0.2),
  )


def _projected_gravity() -> ObservationTermCfg:
  # Deliberately NOT mjlab's default ``mdp.projected_gravity``, which reads
  # ground-truth root body orientation and bypasses the IMU site entirely.
  # This variant reads the ``imu_upvector`` framezaxis sensor, so it reflects
  # IMU site pose randomisation the way the real estimate would.
  return ObservationTermCfg(
    func=envs_mdp.projected_gravity_from_sensor,
    params={"sensor_name": "robot/imu_upvector"},
    noise=Unoise(n_min=-0.05, n_max=0.05),
  )


def _joint_pos() -> ObservationTermCfg:
  # biased=True routes through joint_pos_biased, which carries the +/-0.015 rad
  # encoder_bias startup randomisation. This is the encoder reading, not truth.
  return ObservationTermCfg(
    func=envs_mdp.joint_pos_rel,
    params={"biased": True},
    noise=Unoise(n_min=-0.01, n_max=0.01),
  )


def _joint_vel() -> ObservationTermCfg:
  return ObservationTermCfg(
    func=envs_mdp.joint_vel_rel,
    noise=Unoise(n_min=-1.5, n_max=1.5),
  )


def _last_action() -> ObservationTermCfg:
  return ObservationTermCfg(func=envs_mdp.last_action)


def _command() -> ObservationTermCfg:
  return ObservationTermCfg(
    func=envs_mdp.generated_commands,
    params={"command_name": "twist"},
  )


# ---------------------------------------------------------------------------
# Oracle terms -- idealised stand-ins for the tactile array, diagnostic use only.
# ---------------------------------------------------------------------------

def _foot_contact_ideal() -> ObservationTermCfg:
  return ObservationTermCfg(
    func=vel_mdp.foot_contact,
    params={"sensor_name": FOOT_CONTACT_SENSOR},
  )


def _foot_contact_forces_ideal() -> ObservationTermCfg:
  return ObservationTermCfg(
    func=vel_mdp.foot_contact_forces,
    params={"sensor_name": FOOT_CONTACT_SENSOR},
  )


# ---------------------------------------------------------------------------
# Privileged terms -- critic only, permanently.
# ---------------------------------------------------------------------------

def _base_lin_vel() -> ObservationTermCfg:
  return ObservationTermCfg(
    func=envs_mdp.builtin_sensor,
    params={"sensor_name": "robot/imu_lin_vel"},
  )


def _joint_pos_true() -> ObservationTermCfg:
  return ObservationTermCfg(func=envs_mdp.joint_pos_rel)


def _height_scan() -> ObservationTermCfg:
  return ObservationTermCfg(
    func=envs_mdp.height_scan,
    params={"sensor_name": TERRAIN_SCAN_SENSOR},
    scale=1 / TERRAIN_SCAN_MAX_DISTANCE,
  )


def _foot_height() -> ObservationTermCfg:
  return ObservationTermCfg(
    func=vel_mdp.foot_height,
    params={"sensor_name": FOOT_HEIGHT_SENSOR},
  )


def _foot_air_time() -> ObservationTermCfg:
  return ObservationTermCfg(
    func=vel_mdp.foot_air_time,
    params={"sensor_name": FOOT_CONTACT_SENSOR},
  )


TERMS: dict[str, TermSpec] = {
  # -- onboard ---------------------------------------------------------------
  "base_ang_vel": TermSpec(
    _base_ang_vel,
    "onboard",
    "IMU gyro at the pelvis site (MJCF sensor imu_ang_vel).",
  ),
  "projected_gravity": TermSpec(
    _projected_gravity,
    "onboard",
    "Gravity direction in body frame from IMU attitude fusion; read here from "
    "the imu_upvector site sensor so IMU pose randomisation applies.",
  ),
  "joint_pos": TermSpec(
    _joint_pos,
    "onboard",
    "Joint encoders, with encoder bias randomisation applied.",
  ),
  "joint_vel": TermSpec(
    _joint_vel,
    "onboard",
    "Encoder differentiation; noisy on hardware, modelled with +/-1.5 rad/s.",
  ),
  "actions": TermSpec(
    _last_action,
    "onboard",
    "The policy's own previous output.",
  ),
  "command": TermSpec(
    _command,
    "onboard",
    "Operator velocity command (vx, vy, wz).",
  ),
  # -- oracle ----------------------------------------------------------------
  "foot_contact_ideal": TermSpec(
    _foot_contact_ideal,
    "oracle",
    "Binary per-foot ground contact, noiseless and zero-latency. A real tactile "
    "array measures this, but not this cleanly. Diagnostic upper bound only.",
  ),
  "foot_contact_forces_ideal": TermSpec(
    _foot_contact_forces_ideal,
    "oracle",
    "Net 3D contact force per foot (reduce='netforce'), noiseless. Note this is "
    "a load cell, not a tactile array: no spatial distribution over the sole, "
    "so no centre of pressure and no heel-vs-toe discrimination.",
  ),
  # -- privileged ------------------------------------------------------------
  "base_lin_vel": TermSpec(
    _base_lin_vel,
    "privileged",
    "Exact body-frame linear velocity (velocimeter). Not measurable; would need "
    "a state estimator, whose drift is a known sim2real failure. mjlab treats "
    "this the same way via its has_state_estimation flag on the tracking task.",
  ),
  "joint_pos_true": TermSpec(
    _joint_pos_true,
    "privileged",
    "Unbiased joint positions -- the encoder reading without its bias.",
  ),
  "height_scan": TermSpec(
    _height_scan,
    "privileged",
    "187-dim terrain raycast from the pelvis. Would require LiDAR/depth plus an "
    "elevation-mapping stack. The policies in this study are blind.",
  ),
  "foot_height": TermSpec(
    _foot_height,
    "privileged",
    "Per-foot clearance above terrain, by raycast. No hardware counterpart.",
  ),
  "foot_air_time": TermSpec(
    _foot_air_time,
    "privileged",
    "Accumulated swing time from the sim contact sensor.",
  ),
}


def deployable_term_names() -> frozenset[str]:
  return frozenset(n for n, s in TERMS.items() if s.availability == "onboard")


def term_names_with(availability: Availability) -> frozenset[str]:
  return frozenset(n for n, s in TERMS.items() if s.availability == availability)
