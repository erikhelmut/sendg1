"""Slackline env config: ``G1-Slackline-{Touch,Blind}``.

The G1 starts in a tandem stance on a tensioned slackline
(``common/slackline/model.py``) and either balances there or walks along
it, as the forward-speed command asks. Plain PPO with a GRU actor (IMU,
joints, previous action, command, and, for ``Touch``, the tactile touch
features) and an MLP critic on that plus ground truth about the line. The
two tasks differ only in the actor's inputs.

Every reward term, weight and termination is in this file, in one table
each; ``build_rewards`` argues the reward.
"""

from __future__ import annotations

import math

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.managers.curriculum_manager import CurriculumTermCfg
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
from sendg1.common.sensors import feet_ground_contact_sensor, self_collision_sensor
from sendg1.common.slackline import mdp
from sendg1.common.slackline import model as sm
from sendg1.common.slackline.terms import (
  BLIND_ACTOR_TERMS,
  PRIVILEGED_TERMS,
  TERMS,
  TOUCH_ACTOR_TERMS,
)
from sendg1.common.tactile.sensing import tactile_scene_sensors
from sendg1.tasks.beam.env_cfg import (
  ARM_JOINTS,
  ARM_POSE_STD,
  LEG_JOINTS,
  nonfoot_contact_sensor,
)

DEFAULT_NUM_ENVS = 4096
EPISODE_S = 20.0
MAX_FORWARD_SPEED = 0.25
"""Slacklining is slow: a careful step at a time. Below the beam's 0.3."""
STANDING_SHARE = 0.5
"""Half the commands are 'stand': balance is the first skill and the one
every walking episode also needs."""

BALANCE_ARM_STD: dict[str, float] = {
  r".*shoulder_pitch.*": 2.0,
  r".*shoulder_roll.*": 2.0,
  r".*shoulder_yaw.*": 2.0,
  r".*elbow.*": 2.0,
}
"""The pedalo's and cylinder's: shoulders free in every direction, so the
arms can go up and out, as every slackliner's do. The elbow's pull to its
bent default is replaced by the ``elbow_range`` window."""
ELBOW_RANGE = (1.13, 1.39)
"""Elbow window (rad), from the pedalo: the G1 arm is straight at ~1.39 and
hyperextends above it; 1.13 is a ~15 deg natural bend."""
WRIST_RANGE = 0.15
"""Wrist window (+/- rad) on roll, pitch and yaw, from the pedalo."""

POSE_STD_SLACKLINE: dict[str, float] = {
  **{k: 1.5 * v for k, v in POSE_STD_WALKING.items()},
  r".*hip_yaw.*": 0.8,
  **BALANCE_ARM_STD,
}
"""Walking tolerances x1.5, as on the cylinder: hips and ankles do the
balancing. Hip yaw is nearly free: the stance turns the back leg out 45 deg
(``mdp.BACK_FOOT_YAW``), which a pull to the parallel-feet default would fight."""
ARM_POSE_STD_SLACKLINE: dict[str, float] = {**ARM_POSE_STD, **BALANCE_ARM_STD}


# ---------------------------------------------------------------------------
# Reward table
# ---------------------------------------------------------------------------


def build_rewards() -> dict[str, RewardTermCfg]:
  """Stand and walk on a slackline.

  The line is the whole support sideways, and it is not fixed: push it and
  it swings and rolls. Balance means the CoM over the webbing beneath it,
  and the webbing quiet. The table, in order of weight:

  1. Survival: ``alive`` each step and a large one-off ``termination`` cost.
  2. The command: forward progress along the line with the beam's slack
     (a pause to rebalance only forgoes reward), or, standing, no drift;
     walking, a reward per step of new ground with alternating feet.
  3. Balance: the CoM over the line beneath it (``com_over_line``), the line
     not swinging under the robot (``line_still``).
  4. The stance slackliners teach: soles centred on the webbing, the front
     foot pointing along the line and the back foot turned out 45 deg
     (``stance_foot_yaw``), soles flat, torso upright, facing along the line.
  5. Costs: a foot lifted while standing, foot slip on the webbing,
     stamping (it throws the line), the elbow and wrist windows, and the
     beam's regularisers.

  Every positive term but ``step_progress`` is in [0, 1]; while the robot
  stays up the per-step reward is positive, so surviving always pays.

  The reward is the same on every curriculum level: the curriculum changes
  the line, never what is asked of the robot, so returns are comparable
  across levels and between the two conditions.
  """
  return {
    # -- 1. survival -------------------------------------------------------------
    "alive": RewardTermCfg(func=mdp.alive, weight=1.0),
    # Beam lesson: without a large one-off cost of an early end, the random
    # policy's regulariser costs made falling at once the best return.
    "termination": RewardTermCfg(func=envs_mdp.is_terminated, weight=-100.0),
    # -- 2. the command ---------------------------------------------------------
    "track_forward_progress": RewardTermCfg(func=mdp.track_forward_progress, weight=2.0),
    # Per alternating step, paid per metre of new ground (the beam's
    # ``feet_in_line`` without its lateral-gap factor: on a line every step
    # is in line or it is off). Replaces ``feet_air_time``, which tapping farms.
    "step_progress": RewardTermCfg(func=mdp.step_progress, weight=1.0),
    # -- 3. balance --------------------------------------------------------------
    "com_over_line": RewardTermCfg(func=mdp.com_over_line, weight=1.0, params={"std": 0.04}),
    "line_still": RewardTermCfg(func=mdp.line_still, weight=0.5, params={"std": 0.15}),
    # -- 4. stance -----------------------------------------------------------------
    "feet_on_line": RewardTermCfg(func=mdp.feet_on_line, weight=1.0, params={"std": 0.025}),
    "foot_yaw": RewardTermCfg(func=mdp.stance_foot_yaw, weight=0.5, params={"std": 0.25}),
    "foot_flat": RewardTermCfg(func=mdp.stance_foot_flat, weight=0.3),
    "heading_along_line": RewardTermCfg(func=mdp.heading_along_line, weight=0.3),
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
      params={"std": POSE_STD_SLACKLINE, "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    ),
    # -- 5. costs ------------------------------------------------------------------
    "foot_off_line": RewardTermCfg(func=mdp.foot_off_line_standing, weight=-0.5),
    "foot_line_slip": RewardTermCfg(func=mdp.foot_line_slip, weight=-0.5),
    "touchdown_force_rate": RewardTermCfg(func=mdp.touchdown_force_rate, weight=-2.0),
    "arm_posture": RewardTermCfg(
      func=beam_mdp.pose_l2,
      weight=-0.02,
      params={"std": ARM_POSE_STD_SLACKLINE, "asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINTS)},
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
    # -- regularisers (5., from the beam reward) ---------------------------------
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
    "nonfoot_on_line": TerminationTermCfg(func=mdp.nonfoot_on_line),
    "base_too_low": TerminationTermCfg(func=mdp.base_too_low, params={"min_height": 0.55}),
    "tilt": TerminationTermCfg(
      func=vel_mdp.bad_orientation, params={"limit_angle": math.radians(45.0)}
    ),
    "nan_detection": TerminationTermCfg(func=envs_mdp.nan_detection),
  }


# ---------------------------------------------------------------------------
# Events and domain randomisation
# ---------------------------------------------------------------------------


def build_events() -> dict[str, EventTermCfg]:
  robot = SceneEntityCfg("robot")
  foot_geoms = SceneEntityCfg("robot", geom_names=G1_FOOT_GEOMS)
  lo, hi = sm.TENSION_RANGE
  return {
    # -- startup DR (the tension first: the reset reads it to pre-sag the line)
    # Tension: every string tendon scaled alike, per world.
    "line_tension": EventTermCfg(
      mode="startup",
      func=dr.tendon_stiffness,
      params={
        "asset_cfg": mdp.LINE_STRING_TENDONS,
        "operation": "scale",
        "ranges": (lo / sm.TENSION, hi / sm.TENSION),
        "shared_random": True,
      },
    ),
    "foot_friction": EventTermCfg(
      mode="startup",
      func=dr.geom_friction,
      params={"asset_cfg": foot_geoms, "operation": "abs", "ranges": (0.6, 1.2), "shared_random": True},
    ),
    "line_friction": EventTermCfg(
      mode="startup",
      func=dr.geom_friction,
      params={"asset_cfg": mdp.LINE_GEOMS, "operation": "abs", "ranges": (0.6, 1.0), "shared_random": True},
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
    # -- reset -----------------------------------------------------------------
    # The level first (``line_levels`` picked it): the spawn sags the line
    # under the robot with the level's helper springs.
    "line_level": EventTermCfg(func=mdp.apply_line_level, mode="reset"),
    "reset_on_line": EventTermCfg(func=mdp.reset_on_line, mode="reset"),
    "tactile_faults": EventTermCfg(func=mdp.reset_tactile_noise, mode="reset"),
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


def make_slackline_env_cfg(
  *, play: bool = False, num_envs: int | None = None, tactile: bool = True
) -> ManagerBasedRlEnvCfg:
  """``tactile``: whether the actor gets the slipper's touch features. The
  scene carries the tactile sensors either way (the critic reads them), so
  the two conditions run the same simulation."""
  cfg = make_velocity_env_cfg()
  cfg.episode_length_s = EPISODE_S

  cfg.scene.entities = {
    "robot": get_g1_cfg(),
    # Fixed base: mjlab wraps it in a mocap body, placed per env on reset.
    mdp.LINE: EntityCfg(spec_fn=sm.slackline_spec),
  }
  cfg.scene.num_envs = num_envs or DEFAULT_NUM_ENVS
  # Worlds do not interact; the spacing only matters for the viewer.
  cfg.scene.env_spacing = 5.0
  cfg.scene.sensors = (
    feet_ground_contact_sensor(),
    mdp.feet_line_contact_sensor(),
    mdp.nonfoot_line_contact_sensor(),
    self_collision_sensor(),
    nonfoot_contact_sensor(),
    *tactile_scene_sensors(secondary=mdp.line_match()),
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

  cfg.commands = {
    "twist": mdp.SlacklineForwardCommandCfg(
      entity_name="robot",
      resampling_time_range=(6.0, 12.0),
      rel_standing_envs=STANDING_SHARE,
      rel_heading_envs=0.0,
      heading_command=False,
      ranges=mdp.SlacklineForwardCommandCfg.Ranges(
        lin_vel_x=(0.1, MAX_FORWARD_SPEED), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0)
      ),
    )
  }

  cfg.rewards = build_rewards()
  cfg.events = build_events()
  cfg.terminations = build_terminations()
  cfg.curriculum = {"line_levels": CurriculumTermCfg(func=mdp.line_levels)}

  cfg.viewer.body_name = G1_TORSO_BODY
  cfg.viewer.distance = 3.5
  cfg.viewer.elevation = -15.0
  cfg.viewer.azimuth = 90.0
  if play:
    # Evaluation and videos: the real slackline unless a script says otherwise.
    cfg.curriculum = {}
    cfg.events["line_level"].params = {"fixed_level": mdp.TOP_LEVEL}
  return cfg
