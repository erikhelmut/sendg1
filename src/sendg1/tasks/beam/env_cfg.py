"""Beam walking env config: ``G1-Beam-E2E`` (straight beams) and
``G1-Beam-E2E-Curved`` (straight and bent beams).

Plain PPO: a GRU actor on what the robot measures (``actor`` group: IMU,
joints, previous action, forward command, tactile contact map) and an MLP
critic on that plus simulation ground truth (``critic`` group).
``tasks/beam/__init__.py`` refuses to register an actor group with anything
but ``onboard`` / ``sensor`` terms.

Every reward term, weight and termination is in this file, in one table each.
"""

from __future__ import annotations

import math
from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp as vel_mdp
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

from sendg1.common.beam import mdp
from sendg1.common.beam.terms import ACTOR_TERMS, PRIVILEGED_TERMS, TERMS
from sendg1.common.beam.terrain import make_beam_terrain_cfg
from sendg1.common.rewards import POSE_STD_WALKING
from sendg1.common.robots.g1 import (
  G1_FOOT_GEOMS,
  G1_FOOT_SITES,
  G1_TORSO_BODY,
  get_g1_cfg,
  resolve_action_scale,
)
from sendg1.common.sensors import feet_ground_contact_sensor, self_collision_sensor
from sendg1.common.tactile.sensing import tactile_scene_sensors

DEFAULT_NUM_ENVS = 4096
EPISODE_S = 20.0
MAX_FORWARD_SPEED = 0.3
"""A careful, probing step on an unseen beam is easier to learn at walking
pace, and evaluation runs at 0.3 m/s anyway."""

SPAWN = {
  "platform_from_level": 0,
  "platform_arc_range": (-0.3, 0.2),
  "platform_lateral_range": (-0.15, 0.15),
  "yaw_range": (-0.25, 0.25),
}
"""Every episode starts on the start platform, 0.3-0.8 m before the beam, up to
15 cm off the centreline and 14 deg off heading. The actor never sees where the
beam is, so it has to find and follow it by touch."""

CENTRELINE_SIGMA = 0.035
"""Fixed stance-foot / CoM centreline reward width (m), all levels. Run 2
stalled at mean level ~2.2 and shuffled with its feet ~10-17 cm apart: the
level-coupled widths (``mdp.CENTRELINE_SIGMA``, 12 cm at level 0) never asked
for more. A foot (and the CoM) now pays only over the beam, which the actor can
only find by touch."""
SUCCESS_BEAM_M = 1.0
"""Metres of beam that promote a walking episode at any pace."""
STATE_BANK = Path(__file__).resolve().parents[4] / "results" / "beam" / "v4_beam_states.pt"
"""Mid-beam states recorded from an earlier privileged policy that walked
heel-to-toe on 0.05-0.10 m beams (``mdp.spawn_from_bank``)."""
BANK_SHARE = 0.25
"""Share of resets restarted mid-beam from the bank."""
SHOULDER_ROLL_STD = 1.2
"""Posture tolerance (rad) of shoulder roll: arms may go out sideways, as a
person balances. Shoulder pitch / yaw, elbows and wrists keep their pull, which
rules out odd arm configurations; self-collisions are still penalised."""

POSE_STD_BEAM: dict[str, float] = {
  **{
    k: (v if any(j in k for j in ("hip_pitch", "knee", "ankle_pitch")) else 1.5 * v)
    for k, v in POSE_STD_WALKING.items()
  },
  r".*shoulder_roll.*": SHOULDER_ROLL_STD,
}
"""Walking tolerances x1.5 for the lateral/yaw joints a tandem stance needs,
for standing and walking alike (mjlab's tight standing std, 0.05 rad, would
punish exactly the posture the task requires). Sagittal leg joints keep the
walking tolerance: at x1.5 an early policy settled into a deep crouch."""
ARM_JOINTS = (".*shoulder.*", ".*elbow.*", ".*wrist.*")
LEG_JOINTS = (".*hip.*", ".*knee.*", ".*ankle.*")
ARM_POSE_STD: dict[str, float] = {
  r".*shoulder_pitch.*": 0.3,
  r".*shoulder_roll.*": SHOULDER_ROLL_STD,
  r".*shoulder_yaw.*": 0.15,
  r".*elbow.*": 0.3,
  r".*wrist.*": 0.1,
}
"""Light pull toward the default arm pose: wrists held near neutral (an early
policy bent them by ~60 deg for nothing), shoulder yaw moderately, elbows
freer, shoulder roll nearly free so the arms can work as a balance pole."""


def nonfoot_contact_sensor() -> ContactSensorCfg:
  """Any robot body other than the two feet touching the terrain."""
  return ContactSensorCfg(
    name=mdp.NONFOOT_CONTACT_SENSOR,
    primary=ContactMatch(
      mode="body", pattern=r".*", entity="robot", exclude=(r".*_ankle_roll_link",)
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found",),
    reduce="maxforce",
    num_slots=1,
  )


# ---------------------------------------------------------------------------
# Reward table
# ---------------------------------------------------------------------------


def build_rewards() -> dict[str, RewardTermCfg]:
  feet = SceneEntityCfg("robot", site_names=G1_FOOT_SITES)
  return {
    # -- task ------------------------------------------------------------------
    # Forward progress with slack: a pause to feel for the beam only forgoes
    # reward (exact speed tracking punished it).
    "track_forward_progress": RewardTermCfg(func=mdp.track_forward_progress, weight=2.0),
    "alive": RewardTermCfg(func=mdp.alive, weight=0.5),
    "reach_end": RewardTermCfg(func=mdp.reach_end_bonus, weight=10.0),
    # -- beam shaping ------------------------------------------------------------
    "foot_centreline": RewardTermCfg(
      func=mdp.stance_foot_centreline,
      weight=1.0,
      params={"sigmas": (CENTRELINE_SIGMA,)},
    ),
    "foot_yaw": RewardTermCfg(func=mdp.stance_foot_yaw, weight=0.5, params={"std": 0.25}),
    "cop_centred": RewardTermCfg(func=mdp.cop_centred, weight=0.5, params={"std": 0.015}),
    "com_over_beam": RewardTermCfg(
      func=mdp.com_over_beam, weight=1.0, params={"sigmas": (CENTRELINE_SIGMA,)}
    ),
    "base_lateral_vel": RewardTermCfg(func=mdp.base_lateral_velocity, weight=-1.0),
    "base_roll": RewardTermCfg(func=mdp.base_roll, weight=-2.0),
    # -- gait shape -------------------------------------------------------------
    "passing_step": RewardTermCfg(func=mdp.passing_step, weight=1.0),
    "foot_flat": RewardTermCfg(func=mdp.stance_foot_flat, weight=1.0),
    "swing_foot_speed": RewardTermCfg(func=mdp.swing_foot_speed, weight=-1.0),
    "touchdown_force_rate": RewardTermCfg(func=mdp.touchdown_force_rate, weight=-5.0),
    "arm_posture": RewardTermCfg(
      func=mdp.pose_l2,
      weight=-0.02,
      params={"std": ARM_POSE_STD, "asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINTS)},
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
      params={"std": POSE_STD_BEAM, "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    ),
    "feet_air_time": RewardTermCfg(
      func=vel_mdp.feet_air_time,
      weight=0.25,
      params={
        "sensor_name": mdp.FOOT_CONTACT_SENSOR,
        "threshold_min": 0.05,
        "threshold_max": 0.4,
        "command_name": "twist",
        "command_threshold": 0.05,
      },
    ),
    # -- regularisers (from the walking reward) ----------------------------------
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
    "foot_slip": RewardTermCfg(
      func=vel_mdp.feet_slip,
      weight=-0.1,
      params={
        "sensor_name": mdp.FOOT_CONTACT_SENSOR,
        "command_name": "twist",
        "command_threshold": 0.05,
        "asset_cfg": feet,
      },
    ),
    "self_collisions": RewardTermCfg(
      func=vel_mdp.self_collision_cost,
      weight=-1.0,
      params={"sensor_name": "self_collision", "force_threshold": 10.0},
    ),
    # -- touch: step onto an unseen beam lightly, feel, then commit weight -------
    "load_off_support": RewardTermCfg(func=mdp.load_off_support, weight=-1.0),
    "early_contact_overload": RewardTermCfg(func=mdp.early_contact_overload, weight=-3.0),
    # One-off -2 on an early termination (weights are scaled by the 0.02 s
    # step). Without it, the random initial policy's regulariser costs made
    # falling over at once the best return and episodes collapsed to 7 steps.
    "termination": RewardTermCfg(func=envs_mdp.is_terminated, weight=-100.0),
    # Heel-to-toe, paid per metre gained; cheaper forms were farmed (see
    # mdp.feet_in_line).
    "feet_in_line": RewardTermCfg(func=mdp.feet_in_line, weight=3.0),
  }


# ---------------------------------------------------------------------------
# Terminations
# ---------------------------------------------------------------------------


def build_terminations() -> dict[str, TerminationTermCfg]:
  return {
    "time_out": TerminationTermCfg(func=vel_mdp.time_out, time_out=True),
    "nonfoot_contact": TerminationTermCfg(
      func=vel_mdp.illegal_contact, params={"sensor_name": mdp.NONFOOT_CONTACT_SENSOR}
    ),
    "foot_off_beam": TerminationTermCfg(func=mdp.foot_off_support),
    "base_too_low": TerminationTermCfg(func=mdp.base_too_low, params={"min_height": 0.5}),
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
    "spawn": EventTermCfg(func=mdp.spawn_on_beam, mode="reset", params=dict(SPAWN)),
    "reset_joints": EventTermCfg(
      func=envs_mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (-0.03, 0.03),
        "velocity_range": (0.0, 0.0),
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
    "tactile_faults": EventTermCfg(func=mdp.reset_tactile_noise, mode="reset"),
    # Reference-state initialisation: a share of episodes restarts mid-beam in
    # a heel-to-toe stance that fits a narrow beam. Training only.
    "spawn_from_bank": EventTermCfg(
      func=mdp.spawn_from_bank,
      mode="reset",
      params={"bank_path": str(STATE_BANK), "share": BANK_SHARE},
    ),
    # -- interval --------------------------------------------------------------
    "push": EventTermCfg(
      func=mdp.push_by_level,
      mode="interval",
      interval_range_s=(2.0, 4.0),
      params={"max_velocity": {"x": 0.3, "y": 0.25, "roll": 0.3, "pitch": 0.3, "yaw": 0.3}},
    ),
    # -- startup DR --------------------------------------------------------------
    "foot_friction": EventTermCfg(
      mode="startup",
      func=dr.geom_friction,
      params={"asset_cfg": foot_geoms, "operation": "abs", "ranges": (0.4, 1.2), "shared_random": True},
    ),
    "foot_softness": EventTermCfg(
      mode="startup",
      func=mdp.foot_contact_softness,
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


def build_observations(curves: bool = False) -> dict[str, ObservationGroupCfg]:
  critic_actor = tuple(n for n in ACTOR_TERMS if n != "tactile_map")
  critic_priv = PRIVILEGED_TERMS + (("beam_ahead",) if curves else ())
  return {
    "actor": ObservationGroupCfg(
      terms={n: TERMS[n].build() for n in ACTOR_TERMS},
      concatenate_terms=True,
      enable_corruption=True,
    ),
    # The faulted tactile term is stateful (frame drops), so it is computed
    # once, for the actor; the critic reads the clean map.
    "critic": ObservationGroupCfg(
      terms={n: TERMS[n].build() for n in critic_actor + critic_priv},
      concatenate_terms=True,
      enable_corruption=False,
    ),
  }


def make_beam_env_cfg(
  *, play: bool = False, num_envs: int | None = None, curves: bool = False
) -> ManagerBasedRlEnvCfg:
  """``curves``: two thirds of the terrain columns are bent beams
  (``terrain.BEND_MAX_TURN_DEG``), and the critic also sees how the beam
  turns ahead (``beam_ahead``)."""
  cfg = make_velocity_env_cfg()
  cfg.episode_length_s = EPISODE_S

  cfg.scene.entities = {"robot": get_g1_cfg()}
  cfg.scene.num_envs = num_envs or DEFAULT_NUM_ENVS
  cfg.scene.sensors = (
    feet_ground_contact_sensor(),
    self_collision_sensor(),
    nonfoot_contact_sensor(),
    *tactile_scene_sensors(),
  )

  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "generator"
  cfg.scene.terrain.terrain_generator = make_beam_terrain_cfg(curves=curves)
  cfg.scene.terrain.max_init_terrain_level = 0  # everyone starts on the widest beam

  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 128
  cfg.sim.nconmax = 70

  cfg.observations = build_observations(curves)

  cfg.actions = {
    "joint_pos": mdp.DelayedJointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=resolve_action_scale("full29"),
      use_default_offset=True,
      max_lag_substeps=4,
    )
  }

  cfg.commands = {
    "twist": mdp.BeamForwardCommandCfg(
      entity_name="robot",
      resampling_time_range=(6.0, 12.0),
      rel_standing_envs=mdp.STANDING_SHARE[0],
      rel_heading_envs=0.0,
      heading_command=False,
      ranges=mdp.BeamForwardCommandCfg.Ranges(
        lin_vel_x=(0.1, MAX_FORWARD_SPEED), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0)
      ),
    )
  }

  cfg.rewards = build_rewards()
  cfg.events = build_events()
  cfg.terminations = build_terminations()
  # Only walking episodes promote (the first run rose to level 3 on standing
  # commands alone), and a walking episode also promotes once it has covered
  # SUCCESS_BEAM_M of beam without falling, at any pace.
  cfg.curriculum = {
    "beam_levels": CurriculumTermCfg(
      func=mdp.beam_levels,
      params={"standing_promotes": False, "success_beam_m": SUCCESS_BEAM_M},
    )
  }

  cfg.viewer.body_name = G1_TORSO_BODY
  if play:
    cfg.curriculum = {}
    cfg.events.pop("spawn_from_bank")
    cfg.scene.terrain.max_init_terrain_level = None
  return cfg
