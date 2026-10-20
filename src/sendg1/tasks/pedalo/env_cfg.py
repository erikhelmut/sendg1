"""Pedalo riding env configs: ``G1-Pedalo`` (Classic), ``G1-Pedalo-Sport``
and ``G1-Pedalo-Sport-Blind``.

The G1 stands on a Pedalo Classic (``common/pedalo/model.py``) or Sport
(``common/pedalo/sport.py``), one foot per board or step, and rides it
forward at a joystick speed or holds it still. Plain PPO with the beam
task's actor inputs (IMU, joints, previous action, forward command, tactile
touch features; ``blind`` drops the touch features) and an MLP critic on
that plus ground truth about the pedalo. The actor network (plain or
symmetric GRU) is chosen at registration, in ``tasks/pedalo/__init__.py``.

Every reward term, weight and termination is in this file, in one table each.
"""

from __future__ import annotations

import math

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.velocity import mdp as vel_mdp
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

from sendg1.common.beam import mdp as beam_mdp
from sendg1.common.pedalo import mdp
from sendg1.common.pedalo import model as pm
from sendg1.common.pedalo.terms import ACTOR_TERMS, BLIND_ACTOR_TERMS, PRIVILEGED_TERMS, TERMS
from sendg1.common.pedalo.variants import CLASSIC, SPORT, PedaloVariant
from sendg1.common.rewards import POSE_STD_WALKING
from sendg1.common.robots.g1 import (
  G1_FOOT_GEOMS,
  G1_TORSO_BODY,
  get_g1_cfg,
  resolve_action_scale,
)
from sendg1.common.sensors import feet_ground_contact_sensor, self_collision_sensor
from sendg1.common.tactile.sensing import tactile_scene_sensors
from sendg1.tasks.beam.env_cfg import (
  ARM_JOINTS,
  ARM_POSE_STD,
  LEG_JOINTS,
  nonfoot_contact_sensor,
)

DEFAULT_NUM_ENVS = 4096
EPISODE_S = 20.0
MAX_FORWARD_SPEED = 0.6
"""m/s. A child rides a pedalo at roughly walking pace; one crank turn is
0.69 m (22 cm wheels)."""
MIN_FORWARD_SPEED = 0.15
STANDING_SHARE = 0.2
"""Share of 'hold the pedalo still' commands: balancing on it is the first
skill, and the eval wants a robot that can stop."""

FOOT_STD_ALONG = {CLASSIC.name: 0.06, SPORT.name: 0.02}
"""``feet_on_boards`` width along the pedalo: a G1 sole (~19 cm) has room on
a 30 cm Classic board; on a 6 cm Sport step the sole centre has to sit over
the axle, or the step turns."""
COM_STD_ALONG = {CLASSIC.name: 0.06, SPORT.name: 0.03}
"""``com_over_pedalo`` width along the pedalo: the Classic's wheels touch
down 24 cm apart, the Sport's along one line under its axle."""

BALANCE_ARM_STD: dict[str, float] = {
  r".*shoulder_pitch.*": 2.0,
  r".*shoulder_roll.*": 2.0,
  r".*shoulder_yaw.*": 2.0,
  r".*elbow.*": 2.0,
}
"""Tolerances (rad) that leave the shoulders free in every direction, so the
arms can go straight out sideways to balance, as a person does. The elbow is
wide here because its symmetric pull to the bent default (0.6) is replaced by
the ``elbow_range`` window."""

ELBOW_RANGE = (1.13, 1.39)
"""Elbow window (rad). On the G1 elbow the arm is straight at ~1.39 (0 is a
79 deg bend, the default 0.6 is 45 deg); above that it hyperextends, up to the
2.09 limit. 1.13 is a ~15 deg natural bend. Earlier runs, with only a pull to
the default, folded the elbow to 0.25 or, with a one-sided fold cost,
hyperextended it to 1.79."""
WRIST_RANGE = 0.15
"""Wrist window (+/- rad, ~9 deg) on roll, pitch and yaw: a fine-tune with
free shoulders bent the wrists by ~1 rad."""

POSE_STD_PEDALO: dict[str, float] = {
  **{k: 1.5 * v for k, v in POSE_STD_WALKING.items()},
  **BALANCE_ARM_STD,
}
"""Walking tolerances x1.5 on every joint: the boards run 11 cm apart in
height at the dead centre, so the knees and hips swing far from the default
stance even when riding well. Shoulders and elbows: ``BALANCE_ARM_STD``."""
ARM_POSE_STD_PEDALO: dict[str, float] = {**ARM_POSE_STD, **BALANCE_ARM_STD}
"""The beam's light arm pull, with the same balancing freedom."""


# ---------------------------------------------------------------------------
# Reward table
# ---------------------------------------------------------------------------


def build_rewards(variant: PedaloVariant = CLASSIC) -> dict[str, RewardTermCfg]:
  rewards = {
    # -- task ------------------------------------------------------------------
    # The pedalo's speed, with slack and a symmetric clip (beam lesson: exact
    # tracking punished pauses, and a clip at 0 paid for rocking in place).
    "track_speed": RewardTermCfg(func=mdp.track_pedalo_speed, weight=2.0),
    "alive": RewardTermCfg(func=mdp.alive, weight=0.5),
    # -- on the pedalo -----------------------------------------------------------
    "feet_on_boards": RewardTermCfg(
      func=mdp.feet_on_boards, weight=1.0, params={"std_along": FOOT_STD_ALONG[variant.name]}
    ),
    "foot_off_pedalo": RewardTermCfg(func=mdp.foot_off_pedalo, weight=-1.0),
    "foot_yaw": RewardTermCfg(func=mdp.foot_yaw, weight=0.5, params={"std": 0.25}),
    "com_over_pedalo": RewardTermCfg(
      func=mdp.com_over_pedalo, weight=1.0, params={"std_along": COM_STD_ALONG[variant.name]}
    ),
    "heading_to_pedalo": RewardTermCfg(func=mdp.heading_to_pedalo, weight=0.5),
    "foot_board_slip": RewardTermCfg(func=mdp.foot_board_slip, weight=-0.5),
    "touchdown_force_rate": RewardTermCfg(func=mdp.touchdown_force_rate, weight=-2.0),
    "pedalo_lateral_vel": RewardTermCfg(func=mdp.pedalo_lateral_velocity, weight=-1.0),
    "pedalo_yaw_rate": RewardTermCfg(func=mdp.pedalo_yaw_rate, weight=-0.5),
    "base_roll": RewardTermCfg(func=beam_mdp.base_roll, weight=-2.0),
    "foot_flat": RewardTermCfg(func=mdp.stance_foot_flat, weight=0.5),
    "arm_posture": RewardTermCfg(
      func=beam_mdp.pose_l2,
      weight=-0.02,
      params={"std": ARM_POSE_STD_PEDALO, "asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINTS)},
    ),
    # Elbows straight or slightly bent, never hyperextended; wrists near neutral.
    "elbow_range": RewardTermCfg(
      func=mdp.joint_outside_range,
      weight=-5.0,
      params={
        "low": ELBOW_RANGE[0],
        "high": ELBOW_RANGE[1],
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*elbow.*",)),
      },
    ),
    "wrist_range": RewardTermCfg(
      func=mdp.joint_outside_range,
      weight=-5.0,
      params={
        "low": -WRIST_RANGE,
        "high": WRIST_RANGE,
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*wrist.*",)),
      },
    ),
    # -- posture (from the walking reward) -------------------------------------
    "upright": RewardTermCfg(
      func=vel_mdp.upright,
      weight=0.5,
      params={
        "std": math.sqrt(0.2),
        "asset_cfg": SceneEntityCfg("robot", body_names=(G1_TORSO_BODY,)),
      },
    ),
    "posture": RewardTermCfg(
      func=envs_mdp.posture,
      weight=0.5,
      params={"std": POSE_STD_PEDALO, "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    ),
    # -- regularisers (from the beam reward) -------------------------------------
    "action_rate_l2": RewardTermCfg(func=vel_mdp.action_rate_l2, weight=-0.05),
    "leg_joint_acc": RewardTermCfg(
      func=envs_mdp.joint_acc_l2,
      weight=-1e-6,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINTS)},
    ),
    "arm_joint_vel": RewardTermCfg(
      func=envs_mdp.joint_vel_l2,
      weight=-0.02,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINTS)},
    ),
    "joint_torques_l2": RewardTermCfg(func=envs_mdp.joint_torques_l2, weight=-1e-5),
    "dof_pos_limits": RewardTermCfg(func=vel_mdp.joint_pos_limits, weight=-1.0),
    "body_ang_vel": RewardTermCfg(
      func=vel_mdp.body_angular_velocity_penalty,
      weight=-0.05,
      params={"asset_cfg": SceneEntityCfg("robot", body_names=(G1_TORSO_BODY,))},
    ),
    "angular_momentum": RewardTermCfg(
      func=vel_mdp.angular_momentum_penalty,
      weight=-0.02,
      params={"sensor_name": "robot/root_angmom"},
    ),
    "self_collisions": RewardTermCfg(
      func=vel_mdp.self_collision_cost,
      weight=-1.0,
      params={"sensor_name": "self_collision", "force_threshold": 10.0},
    ),
    # One-off -2 on an early termination (beam lesson: without it the random
    # policy's regulariser costs made falling at once the best return).
    "termination": RewardTermCfg(func=envs_mdp.is_terminated, weight=-100.0),
  }
  if variant.steps_turn:
    # Sport: each step turns freely on its axle; the foot has to hold it level.
    rewards["board_tilt"] = RewardTermCfg(func=mdp.board_tilt, weight=-2.0)
  return rewards


# ---------------------------------------------------------------------------
# Terminations
# ---------------------------------------------------------------------------


def build_terminations(variant: PedaloVariant = CLASSIC) -> dict[str, TerminationTermCfg]:
  terms = {
    "time_out": TerminationTermCfg(func=vel_mdp.time_out, time_out=True),
    "foot_on_ground": TerminationTermCfg(func=mdp.foot_on_ground),
    "nonfoot_contact": TerminationTermCfg(
      func=vel_mdp.illegal_contact, params={"sensor_name": beam_mdp.NONFOOT_CONTACT_SENSOR}
    ),
    "base_too_low": TerminationTermCfg(func=mdp.base_too_low, params={"min_height": 0.6}),
    "tilt": TerminationTermCfg(
      func=vel_mdp.bad_orientation, params={"limit_angle": math.radians(40.0)}
    ),
    "pedalo_tipped": TerminationTermCfg(func=mdp.pedalo_tipped),
    "nan_detection": TerminationTermCfg(func=envs_mdp.nan_detection),
  }
  if variant.steps_turn:
    terms["board_tipped"] = TerminationTermCfg(func=mdp.board_tipped)
  return terms


# ---------------------------------------------------------------------------
# Events and domain randomisation
# ---------------------------------------------------------------------------


def build_events() -> dict[str, EventTermCfg]:
  robot = SceneEntityCfg("robot")
  foot_geoms = SceneEntityCfg("robot", geom_names=G1_FOOT_GEOMS)
  return {
    # -- reset -----------------------------------------------------------------
    "reset_on_pedalo": EventTermCfg(func=mdp.reset_on_pedalo, mode="reset"),
    "tactile_faults": EventTermCfg(func=mdp.reset_tactile_noise, mode="reset"),
    # -- startup DR --------------------------------------------------------------
    "foot_friction": EventTermCfg(
      mode="startup",
      func=dr.geom_friction,
      params={"asset_cfg": foot_geoms, "operation": "abs", "ranges": (0.6, 1.2), "shared_random": True},
    ),
    "tyre_friction": EventTermCfg(
      mode="startup",
      func=dr.geom_friction,
      params={"asset_cfg": mdp.PEDALO_GEOMS, "operation": "abs", "ranges": (0.6, 1.0), "shared_random": True},
    ),
    "foot_softness": EventTermCfg(
      mode="startup",
      func=beam_mdp.foot_contact_softness,
      params={"asset_cfg": foot_geoms, "timeconst_range": (0.02, 0.035)},
    ),
    "torso_mass": EventTermCfg(
      mode="startup",
      func=dr.body_mass,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=(G1_TORSO_BODY,)),
        "operation": "add",
        "ranges": (-1.0, 3.0),
      },
    ),
    "base_com": EventTermCfg(
      mode="startup",
      func=dr.body_com_offset,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=(G1_TORSO_BODY,)),
        "operation": "add",
        "ranges": {0: (-0.025, 0.025), 1: (-0.025, 0.025), 2: (-0.03, 0.03)},
      },
    ),
    "motor_gains": EventTermCfg(
      mode="startup",
      func=dr.pd_gains,
      params={"asset_cfg": robot, "kp_range": (0.9, 1.1), "kd_range": (0.9, 1.1)},
    ),
    "motor_strength": EventTermCfg(
      mode="startup",
      func=dr.effort_limits,
      params={"asset_cfg": robot, "effort_limit_range": (0.85, 1.0)},
    ),
    "encoder_bias": EventTermCfg(
      mode="startup",
      func=dr.encoder_bias,
      params={"asset_cfg": robot, "bias_range": (-0.015, 0.015)},
    ),
    # -- startup -----------------------------------------------------------------
    "register_actor_mirror": EventTermCfg(func=mdp.register_actor_mirror, mode="startup"),
  }


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def build_observations(blind: bool = False) -> dict[str, ObservationGroupCfg]:
  """The critic is the same for both actors, so a tactile-vs-blind pair
  differs only in what the actor sees."""
  critic_actor = BLIND_ACTOR_TERMS
  return {
    "actor": ObservationGroupCfg(
      terms={n: TERMS[n].build() for n in (BLIND_ACTOR_TERMS if blind else ACTOR_TERMS)},
      concatenate_terms=True,
      enable_corruption=True,
    ),
    # The faulted touch term is stateful (frame drops): computed once, for
    # the actor; the critic reads the clean one.
    "critic": ObservationGroupCfg(
      terms={n: TERMS[n].build() for n in critic_actor + PRIVILEGED_TERMS},
      concatenate_terms=True,
      enable_corruption=False,
    ),
  }


def pedalo_entity(variant: PedaloVariant = CLASSIC) -> EntityCfg:
  return EntityCfg(
    spec_fn=variant.spec_fn,
    init_state=EntityCfg.InitialStateCfg(pos=(variant.root_ahead, 0.0, pm.WHEEL_RADIUS)),
  )


def make_pedalo_env_cfg(
  *,
  play: bool = False,
  num_envs: int | None = None,
  variant: PedaloVariant = CLASSIC,
  blind: bool = False,
) -> ManagerBasedRlEnvCfg:
  cfg = make_velocity_env_cfg()
  cfg.episode_length_s = EPISODE_S

  cfg.scene.entities = {"robot": get_g1_cfg(), mdp.PEDALO: pedalo_entity(variant)}
  cfg.scene.num_envs = num_envs or DEFAULT_NUM_ENVS
  # Worlds do not interact; the spacing only matters for the viewer.
  cfg.scene.env_spacing = 4.0
  cfg.scene.sensors = (
    feet_ground_contact_sensor(),
    mdp.feet_pedalo_contact_sensor(variant),
    self_collision_sensor(),
    nonfoot_contact_sensor(),
    *tactile_scene_sensors(secondary=mdp.pedalo_match(variant)),
  )
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 128
  cfg.sim.nconmax = 120

  cfg.observations = build_observations(blind)

  cfg.actions = {
    "joint_pos": beam_mdp.DelayedJointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=resolve_action_scale("full29"),
      use_default_offset=True,
      max_lag_substeps=4,
    )
  }

  # Forward speed only (vy and yaw rate 0); nothing ever holds it at zero, so
  # the beam command class behaves as the plain uniform one here.
  cfg.commands = {
    "twist": beam_mdp.BeamForwardCommandCfg(
      entity_name="robot",
      resampling_time_range=(6.0, 12.0),
      rel_standing_envs=STANDING_SHARE,
      rel_heading_envs=0.0,
      heading_command=False,
      ranges=beam_mdp.BeamForwardCommandCfg.Ranges(
        lin_vel_x=(MIN_FORWARD_SPEED, MAX_FORWARD_SPEED), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0)
      ),
    )
  }

  cfg.rewards = build_rewards(variant)
  cfg.events = build_events()
  cfg.terminations = build_terminations(variant)
  cfg.curriculum = {}

  cfg.viewer.body_name = G1_TORSO_BODY
  cfg.viewer.distance = 3.0
  cfg.viewer.elevation = -15.0
  cfg.viewer.azimuth = 120.0
  del play  # no training-only terms yet
  return cfg
