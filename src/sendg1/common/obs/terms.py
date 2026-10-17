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
``perceived``   Not measured by the G1 itself, but ASSUMED provided by an
                external perception stack as part of a task's specification
                (the box-pushing task's box pose and target). Legal in an actor
                group, in every obs set -- so it is a control variable, never
                the treatment. The assumption is the task's, and is stated in
                each term's rationale.
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

from sendg1.common.box_push import mdp as box_mdp
from sendg1.common.tactile.sensing import (
  TactileParams,
  tactile_aggregate_obs,
  tactile_array_obs,
)

Availability = Literal["onboard", "oracle", "perceived", "privileged"]

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


def _tactile_array() -> ObservationTermCfg:
  # Explicit params, so a change to the sensor model moves the fingerprint.
  p = TactileParams()
  return ObservationTermCfg(
    func=tactile_array_obs,
    params={
      "layout": p.layout,
      "rays_per_axis": p.rays_per_axis,
      "compliance_depth": p.compliance_depth,
    },
  )


def _tactile_aggregate() -> ObservationTermCfg:
  p = TactileParams()
  return ObservationTermCfg(
    func=tactile_aggregate_obs,
    params={
      "layout": p.layout,
      "rays_per_axis": p.rays_per_axis,
      "compliance_depth": p.compliance_depth,
    },
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


# ---------------------------------------------------------------------------
# Box pushing. Perceived terms (actor) carry small noise; privileged ones do not.
# ---------------------------------------------------------------------------

BOX_POS_NOISE = 0.02
"""m, uniform +/-, on the perceived box position and target offset."""
BOX_YAW_NOISE = 0.03
"""On each of (sin, cos) of the perceived relative yaw (~ +/-1.7 deg)."""


def _box_term(func, noise: float | None = None) -> Callable[[], ObservationTermCfg]:
  def build() -> ObservationTermCfg:
    return ObservationTermCfg(
      func=func,
      noise=None if noise is None else Unoise(n_min=-noise, n_max=noise),
    )

  return build


_PERCEPTION_ASSUMPTION = (
  "Box-pushing task assumption: an external perception stack (e.g. a head "
  "camera with box and floor-marker tracking) reports this. Not a G1 sensor; "
  "identical in every obs set, so it cannot account for a difference between "
  "them. Noise is small and uniform, with no latency or dropout."
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
  "tactile_aggregate": TermSpec(
    _tactile_aggregate,
    "oracle",
    "Per-foot net normal force, shear, centre of pressure and shear/normal "
    "ratio, from the same sole reading as tactile_array. Implied by the array, "
    "but the ratio -- the foot's distance from its friction cone -- is the "
    "quantity a slip-aware policy needs, so it is given explicitly.",
  ),
  "tactile_array": TermSpec(
    _tactile_array,
    "oracle",
    "Per-taxel normal and 2D shear force on both soles, from the simulated "
    "contacts via sendg1.common.tactile.model. Oracle until the sensor model "
    "carries the real array's noise, latency, calibration error and "
    "saturation; for now it is what a perfect array would read.",
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
  # -- box pushing: perceived (actor) -----------------------------------------
  "pilot_command": TermSpec(
    _box_term(box_mdp.pilot_twist),
    "perceived",
    "The scripted high level's (vx, vy, wz) walking command, in the robot's "
    "yaw frame -- the same shape as the velocity tasks' operator command. It "
    "is computed from the box and target poses, which the task assumes an "
    "external perception stack provides, and it is IDENTICAL across obs sets "
    "(sendg1.common.box_push.pilot), so it cannot account for a difference "
    "between them.",
  ),
  # -- box pushing: privileged (critic) ---------------------------------------
  "box_pos_b_true": TermSpec(
    _box_term(box_mdp.box_pos_b),
    "privileged",
    "Box centre in the pelvis frame. Critic only: the actor is told where to "
    "walk by the pilot, not where the box is.",
  ),
  "box_yaw_b_true": TermSpec(
    _box_term(box_mdp.box_yaw_b),
    "privileged",
    "Box yaw relative to the pelvis heading, (sin, cos).",
  ),
  "target_rel_box_true": TermSpec(
    _box_term(box_mdp.target_rel_box),
    "privileged",
    "Target centre minus box centre, xy in the pelvis yaw frame.",
  ),
  "floor_mu_under_feet": TermSpec(
    _box_term(box_mdp.floor_mu_under_feet),
    "privileged",
    "Friction coefficient of the floor under each foot (the episode's floor "
    "mu, or a low-friction patch's). The quantity the policy must infer.",
  ),
  "box_mass": TermSpec(
    _box_term(box_mdp.box_mass),
    "privileged",
    "Sampled box mass. A property of the world, not a measurement.",
  ),
  "box_mu": TermSpec(
    _box_term(box_mdp.box_mu),
    "privileged",
    "Sampled box-floor friction coefficient.",
  ),
  "box_size": TermSpec(
    _box_term(box_mdp.box_size),
    "privileged",
    "Sampled box extents (depth, width, height).",
  ),
  "box_lin_vel": TermSpec(
    _box_term(box_mdp.box_lin_vel),
    "privileged",
    "Box linear velocity, pelvis yaw frame. Would need box tracking with "
    "reliable differentiation.",
  ),
  "box_ang_vel": TermSpec(
    _box_term(box_mdp.box_ang_vel),
    "privileged",
    "Box angular velocity, pelvis yaw frame.",
  ),
  "box_up": TermSpec(
    _box_term(box_mdp.box_up),
    "privileged",
    "Box z axis in the world: how far it is tipping.",
  ),
  "foot_slip_vel": TermSpec(
    _box_term(box_mdp.foot_slip_vel),
    "privileged",
    "Tangential sole velocity of each foot while in contact, pelvis yaw frame. "
    "Exactly the slip the reward penalises; no hardware measures it directly.",
  ),
  "push_force": TermSpec(
    _box_term(box_mdp.push_force),
    "privileged",
    "Net contact force of the robot on the box, from the simulator.",
  ),
}


def deployable_term_names() -> frozenset[str]:
  return frozenset(n for n, s in TERMS.items() if s.availability == "onboard")


def term_names_with(availability: Availability) -> frozenset[str]:
  return frozenset(n for n, s in TERMS.items() if s.availability == availability)
