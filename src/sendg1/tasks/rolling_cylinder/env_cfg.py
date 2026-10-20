"""Rolling-cylinder balance env config: ``G1-RollingCylinder-{Touch,Blind}``.

The G1 stands across a cylinder lying free on the floor
(``common/rolling_cylinder/model.py``) and has to stay on it for the whole
episode without letting it roll away. Plain PPO with a GRU actor (IMU,
joints, previous action, and, for ``Touch``, the tactile touch features) and
an MLP critic on that plus ground truth about the cylinder. The two tasks
differ only in the actor's inputs.

Every reward term, weight and termination is in this file, in one table
each; ``build_rewards`` argues the reward.
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
from sendg1.common.rewards import POSE_STD_WALKING
from sendg1.common.robots.g1 import (
  G1_FOOT_GEOMS,
  G1_TORSO_BODY,
  get_g1_cfg,
  resolve_action_scale,
)
from sendg1.common.rolling_cylinder import mdp
from sendg1.common.rolling_cylinder import model as cm
from sendg1.common.rolling_cylinder.terms import (
  BLIND_ACTOR_TERMS,
  PRIVILEGED_TERMS,
  TERMS,
  TOUCH_ACTOR_TERMS,
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

BALANCE_ARM_STD: dict[str, float] = {
  r".*shoulder_pitch.*": 2.0,
  r".*shoulder_roll.*": 2.0,
  r".*shoulder_yaw.*": 2.0,
  r".*elbow.*": 2.0,
}
"""The pedalo's: shoulders free in every direction, so the arms can go out to
balance as a person's do. The elbow's pull to its bent default is replaced by
the ``elbow_range`` window."""
ELBOW_RANGE = (1.13, 1.39)
"""Elbow window (rad), from the pedalo: on the G1 the arm is straight at
~1.39 (0 is a 79 deg bend) and hyperextends above it; 1.13 is a ~15 deg
natural bend."""
WRIST_RANGE = 0.15
"""Wrist window (+/- rad, ~9 deg) on roll, pitch and yaw, from the pedalo."""

POSE_STD_CYLINDER: dict[str, float] = {
  **{k: 1.5 * v for k, v in POSE_STD_WALKING.items()},
  **BALANCE_ARM_STD,
}
"""Walking tolerances x1.5: the hips and ankles do the balancing here, as
they swing the pelvis over a support line that moves."""
ARM_POSE_STD_CYLINDER: dict[str, float] = {**ARM_POSE_STD, **BALANCE_ARM_STD}


# ---------------------------------------------------------------------------
# Reward table
# ---------------------------------------------------------------------------


def build_rewards() -> dict[str, RewardTermCfg]:
  """Balance on a free cylinder.

  At equilibrium the CoM, the line where the soles touch the cylinder and
  the line where the cylinder touches the floor are stacked vertically and
  the cylinder is at rest. Any fore-aft CoM error rolls the cylinder, which
  carries the feet with it. So the table is, in order of weight:

  1. Survival: ``alive`` each step and a large one-off ``termination`` cost.
  2. The equilibrium itself: the cylinder at rest (``cylinder_still``), the
     CoM over the cylinder (``com_over_support``), and no slow creep away
     from the spawn point (``stay_near_spawn``).
  3. The posture with the most control authority: soles centred on the top
     line (``feet_on_top``), so the support can roll toward toe or heel by
     the same amount; soles across the axis and flat; torso upright; facing
     the rolling direction.
  4. Costs for what ends badly or looks wrong on hardware: a foot lifted or
     hanging past an end, the cylinder sliding or turning, foot slip,
     stamping, the elbow and wrist windows from the pedalo, the beam's
     regularisers.

  Every positive term is in [0, 1]; while the robot stays up the per-step
  reward is positive (5.7 at best), so surviving always pays.
  """
  return {
    # -- 1. survival -------------------------------------------------------------
    "alive": RewardTermCfg(func=mdp.alive, weight=1.0),
    # Large one-off cost of an early end (beam lesson: without it the random
    # policy's regulariser costs made falling at once the best return).
    "termination": RewardTermCfg(func=envs_mdp.is_terminated, weight=-100.0),
    # -- 2. the equilibrium ------------------------------------------------------
    "cylinder_still": RewardTermCfg(func=mdp.cylinder_still, weight=1.0, params={"std": 0.1}),
    "com_over_support": RewardTermCfg(
      func=mdp.com_over_support, weight=1.0, params={"std_fwd": 0.03, "std_lat": 0.08}
    ),
    "stay_near_spawn": RewardTermCfg(func=mdp.stay_near_spawn, weight=0.5, params={"std": 0.3}),
    # -- 3. posture with control authority ---------------------------------------
    "feet_on_top": RewardTermCfg(func=mdp.feet_on_top, weight=0.5, params={"std": 0.04}),
    "foot_yaw": RewardTermCfg(func=mdp.foot_yaw, weight=0.3, params={"std": 0.25}),
    "foot_flat": RewardTermCfg(func=mdp.stance_foot_flat, weight=0.3),
    "heading_to_cylinder": RewardTermCfg(func=mdp.heading_to_cylinder, weight=0.3),
    # -- 4. costs ------------------------------------------------------------------
    "foot_off_cylinder": RewardTermCfg(func=mdp.foot_off_cylinder, weight=-0.5),
    "foot_past_end": RewardTermCfg(func=mdp.foot_past_end, weight=-10.0),
    "cylinder_axial_vel": RewardTermCfg(func=mdp.cylinder_axial_velocity, weight=-1.0),
    "cylinder_yaw_rate": RewardTermCfg(func=mdp.cylinder_yaw_rate, weight=-0.5),
    "foot_cylinder_slip": RewardTermCfg(func=mdp.foot_cylinder_slip, weight=-0.5),
    "touchdown_force_rate": RewardTermCfg(func=mdp.touchdown_force_rate, weight=-1.0),
    "arm_posture": RewardTermCfg(
      func=beam_mdp.pose_l2,
      weight=-0.02,
      params={"std": ARM_POSE_STD_CYLINDER, "asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINTS)},
    ),
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
    # -- posture (3., from the walking reward) -----------------------------------
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
      weight=0.3,
      params={"std": POSE_STD_CYLINDER, "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    ),
    # -- regularisers (4., from the beam reward) ---------------------------------
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
  }


# ---------------------------------------------------------------------------
# Terminations
# ---------------------------------------------------------------------------


def build_terminations() -> dict[str, TerminationTermCfg]:
  return {
    "time_out": TerminationTermCfg(func=vel_mdp.time_out, time_out=True),
    "foot_on_ground": TerminationTermCfg(func=mdp.foot_on_ground),
    "nonfoot_contact": TerminationTermCfg(
      func=vel_mdp.illegal_contact, params={"sensor_name": beam_mdp.NONFOOT_CONTACT_SENSOR}
    ),
    "base_too_low": TerminationTermCfg(func=mdp.base_too_low, params={"min_height": 0.8}),
    "tilt": TerminationTermCfg(
      func=vel_mdp.bad_orientation, params={"limit_angle": math.radians(40.0)}
    ),
    "nan_detection": TerminationTermCfg(func=envs_mdp.nan_detection),
  }


# ---------------------------------------------------------------------------
# Events and domain randomisation
# ---------------------------------------------------------------------------


def build_events() -> dict[str, EventTermCfg]:
  robot = SceneEntityCfg("robot")
  foot_geoms = SceneEntityCfg("robot", geom_names=G1_FOOT_GEOMS)
  return {
    # -- reset -----------------------------------------------------------------
    "reset_on_cylinder": EventTermCfg(func=mdp.reset_on_cylinder, mode="reset"),
    "tactile_faults": EventTermCfg(func=mdp.reset_tactile_noise, mode="reset"),
    # -- startup DR --------------------------------------------------------------
    "foot_friction": EventTermCfg(
      mode="startup",
      func=dr.geom_friction,
      params={"asset_cfg": foot_geoms, "operation": "abs", "ranges": (0.6, 1.2), "shared_random": True},
    ),
    "cylinder_friction": EventTermCfg(
      mode="startup",
      func=dr.geom_friction,
      params={"asset_cfg": mdp.CYLINDER_GEOMS, "operation": "abs", "ranges": (0.6, 1.0), "shared_random": True},
    ),
    # Mass x exp(2 alpha), ~0.6-1.5x, with the inertia scaled alike (a denser
    # or lighter roller of the same size; dr.body_mass would leave it).
    "cylinder_mass": EventTermCfg(
      mode="startup",
      func=dr.pseudo_inertia,
      params={
        "asset_cfg": SceneEntityCfg(mdp.CYLINDER, body_names=(cm.ROOT_BODY,)),
        "alpha_range": (-0.25, 0.2),
      },
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
  }


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def build_observations(tactile: bool) -> dict[str, ObservationGroupCfg]:
  actor = TOUCH_ACTOR_TERMS if tactile else BLIND_ACTOR_TERMS
  return {
    "actor": ObservationGroupCfg(
      terms={n: TERMS[n].build() for n in actor},
      concatenate_terms=True,
      enable_corruption=True,
    ),
    # Identical in both tasks. The faulted touch term is stateful (frame
    # drops), so only the actor computes it; the critic reads the clean one.
    "critic": ObservationGroupCfg(
      terms={n: TERMS[n].build() for n in BLIND_ACTOR_TERMS + PRIVILEGED_TERMS},
      concatenate_terms=True,
      enable_corruption=False,
    ),
  }


def make_rolling_cylinder_env_cfg(
  *, play: bool = False, num_envs: int | None = None, tactile: bool = True
) -> ManagerBasedRlEnvCfg:
  """``tactile``: whether the actor gets the slipper's touch features. The
  scene carries the tactile sensors either way (the critic reads them), so
  the two conditions run the same simulation."""
  cfg = make_velocity_env_cfg()
  cfg.episode_length_s = EPISODE_S

  cfg.scene.entities = {
    "robot": get_g1_cfg(),
    mdp.CYLINDER: EntityCfg(
      spec_fn=cm.cylinder_spec,
      init_state=EntityCfg.InitialStateCfg(pos=(0.0, 0.0, cm.RADIUS)),
    ),
  }
  cfg.scene.num_envs = num_envs or DEFAULT_NUM_ENVS
  # Worlds do not interact; the spacing only matters for the viewer.
  cfg.scene.env_spacing = 4.0
  cfg.scene.sensors = (
    feet_ground_contact_sensor(),
    mdp.feet_cylinder_contact_sensor(),
    self_collision_sensor(),
    nonfoot_contact_sensor(),
    *tactile_scene_sensors(secondary=mdp.cylinder_match()),
  )
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 128
  cfg.sim.nconmax = 80

  cfg.observations = build_observations(tactile)

  cfg.actions = {
    "joint_pos": beam_mdp.DelayedJointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=resolve_action_scale("full29"),
      use_default_offset=True,
      max_lag_substeps=4,
    )
  }

  # Balance only: nothing to command.
  cfg.commands = {}

  cfg.rewards = build_rewards()
  cfg.events = build_events()
  cfg.terminations = build_terminations()
  cfg.curriculum = {}

  cfg.viewer.body_name = G1_TORSO_BODY
  cfg.viewer.distance = 3.0
  cfg.viewer.elevation = -15.0
  cfg.viewer.azimuth = 90.0
  del play  # no training-only terms yet
  return cfg
