"""Beam walking env config (``tasks/beam/__init__.py`` registers the tasks).

Plain PPO: an actor on what the robot measures (``actor`` group: IMU, joints,
previous action, forward command, and for touch tasks the tactile slipper's
touch features) and an MLP
critic on that plus simulation ground truth (``critic`` group).
``tasks/beam/__init__.py`` refuses to register an actor group with anything
but ``onboard`` / ``sensor`` terms.

Every reward term, weight and termination is in this file, in one table each.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

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
from sendg1.common.beam.terms import (
  BLIND_ACTOR_TERMS,
  TERMS,
  TOUCH_ACTOR_TERMS,
  TOUCH_PRIVILEGED_TERMS,
)
from sendg1.common.beam.terrain import EVAL_TERRAIN_SEED, make_beam_terrain_cfg
from sendg1.common.ppo import NUM_ENVS
from sendg1.common.rewards import POSE_STD_RUNNING, POSE_STD_STANDING, POSE_STD_WALKING
from sendg1.common.robots.g1 import (
  G1_FOOT_GEOMS,
  G1_FOOT_SITES,
  G1_TORSO_BODY,
  get_g1_cfg,
  resolve_action_scale,
)
from sendg1.common.sensors import feet_ground_contact_sensor, self_collision_sensor
from sendg1.common.tactile.sensing import tactile_scene_sensors

DEFAULT_NUM_ENVS = NUM_ENVS
"""The shared default (``sendg1.common.ppo.NUM_ENVS``)."""
EPISODE_S = 35.0
"""Episode length, training and ``-Eval`` alike. A crossing from the start
platform is ~4.65 m, 15.5 s at a perfect 0.3 m/s; 35 s lets a policy that
keeps walking at half that pace finish, so a time-out means it stalled, not
that it was careful. Training used 20 s before, so a recurrent actor ran
longer in evaluation than it ever had in training. An episode ends 2 s after
reaching the end platform (``mdp.finished``), so the long episodes add no idle
standing."""
EVAL_EPISODE_S = EPISODE_S
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

SUCCESS_BEAM_M = 2.0
"""Metres of beam (of 4) that make a walking episode a curriculum success at
any pace. 1 m promoted envs that could not yet cross the beam."""
STATE_BANK = Path(__file__).resolve().parents[4] / "results" / "beam" / "v4_beam_states.pt"
"""Mid-beam states recorded from an earlier privileged policy that walked
heel-to-toe on 0.05-0.10 m beams (``mdp.spawn_from_bank``)."""
BANK_SHARE = 0.25
"""Share of resets restarted mid-beam from the bank."""
SHOULDER_ROLL_STD = 1.2
"""Posture tolerance (rad) of shoulder roll: arms may go out sideways, as a
person balances. Shoulder pitch / yaw, elbows and wrists keep their pull, which
rules out odd arm configurations; self-collisions are still penalised."""

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


POSE_STD_BEAM_STANDING: dict[str, float] = {
  r"^(?!.*shoulder_roll).*$": POSE_STD_STANDING[".*"],
  r".*shoulder_roll.*": SHOULDER_ROLL_STD,
}
"""The walking task's standing tolerance (0.05 rad), arms free to go out
sideways."""
POSE_STD_BEAM_WALKING: dict[str, float] = {
  **POSE_STD_WALKING,
  r".*shoulder_roll.*": SHOULDER_ROLL_STD,
}
"""The walking task's walking tolerances unchanged (hip roll / yaw 0.15 rad:
v3's x1.5 left room for the legs to cross), shoulder roll free."""
POSE_STD_BEAM_RUNNING: dict[str, float] = {
  **POSE_STD_RUNNING,
  r".*shoulder_roll.*": SHOULDER_ROLL_STD,
}


def build_beam_reward() -> dict[str, RewardTermCfg]:
  """``beam_reward``: the one reward of every beam task.

  v4 (2026-11-22): the walking task's reward (``sendg1.common.rewards``) with
  the fewest beam additions. Every v1-v3 policy walked unnaturally (a shuffle,
  then a straight-leg kick and legs crossed onto the centreline); v3 had 31
  terms, most of them shaping individual footsteps. v4 keeps the walking
  terms and their weights: smooth speed tracking (here along the beam), upright,
  posture with the walking tolerances, swing-foot clearance and swing height
  (10 cm over the beam top), and the regularisers. The beam adds heading
  tracking along the bend, the CoM over the beam with a width that scales with
  the beam, a cost for load on a foot hanging over an edge, and a small
  termination cost. Nothing places individual footsteps.

  v4b (2026-11-22): speed is paid by v1-v3's ``track_forward_progress``
  (linear in speed up to the command) instead of the smooth kernel: the v4
  runs learned to stand still, the kernel paying nothing below ~2/3 speed.

  v4b is the reward of every beam task. v4c (``foot_slip`` -2) was tried on
  2026-11-22 and not kept."""
  feet = SceneEntityCfg("robot", site_names=G1_FOOT_SITES)
  return {
    # -- task (walking: track_linear_velocity / track_angular_velocity) --------
    # v4b: v1-v3's progress term, not the walking task's smooth kernel. With
    # std 0.15 m/s half speed earned ~0, and the 224700/1 runs stood still.
    "track_forward_progress": RewardTermCfg(func=mdp.track_forward_progress, weight=2.0),
    "track_heading": RewardTermCfg(func=mdp.track_beam_heading, weight=2.0),
    # -- beam --------------------------------------------------------------------
    "com_over_support": RewardTermCfg(func=mdp.com_over_support, weight=1.0),
    "load_off_support": RewardTermCfg(func=mdp.load_off_support, weight=-1.0),
    # Falling forfeits the dense positive terms already; -100 (v1-v3) also
    # bought a crouched, over-cautious gait.
    "termination": RewardTermCfg(func=envs_mdp.is_terminated, weight=-10.0),
    # -- walking reward, unchanged -----------------------------------------------
    "upright": RewardTermCfg(
      func=vel_mdp.upright,
      weight=1.0,
      params={
        "std": math.sqrt(0.2),
        "asset_cfg": SceneEntityCfg("robot", body_names=(G1_TORSO_BODY,)),
      },
    ),
    "pose": RewardTermCfg(
      func=vel_mdp.variable_posture,
      weight=1.0,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
        "command_name": "twist",
        "std_standing": POSE_STD_BEAM_STANDING,
        "std_walking": POSE_STD_BEAM_WALKING,
        "std_running": POSE_STD_BEAM_RUNNING,
        "walking_threshold": 0.05,
        "running_threshold": 1.5,
      },
    ),
    "foot_clearance": RewardTermCfg(
      func=mdp.feet_clearance, weight=-2.0, params={"target_height": 0.1}
    ),
    "foot_swing_height": RewardTermCfg(
      func=mdp.feet_swing_height, weight=-0.25, params={"target_height": 0.1}
    ),
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
    "dof_pos_limits": RewardTermCfg(func=vel_mdp.joint_pos_limits, weight=-1.0),
    "action_rate_l2": RewardTermCfg(func=vel_mdp.action_rate_l2, weight=-0.1),
    # The walking task's weight; v4c's -2 was not kept.
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
    "finished": TerminationTermCfg(func=mdp.finished, time_out=True),
    "foot_off_beam": TerminationTermCfg(
      func=mdp.foot_off_support, params={"tol": mdp.FOOT_OFF_TERMINATE_TOL}
    ),
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
  }


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


Tactile = Literal["touch", "none"]
"""Actor tactile input. ``touch``: the shared touch features (``tactile_touch``:
26 taxel bits, load, CoP; 29 per foot). ``none``: blind, no tactile actor term.
Both share one critic (clean touch features, per-taxel normal/shear forces,
foot slip velocity)."""


def build_observations(
  curves: bool = False, tactile: Tactile = "touch"
) -> dict[str, ObservationGroupCfg]:
  actor_terms, priv_terms, noisy = {
    "touch": (TOUCH_ACTOR_TERMS, TOUCH_PRIVILEGED_TERMS, "tactile_touch"),
    "none": (BLIND_ACTOR_TERMS, TOUCH_PRIVILEGED_TERMS, None),
  }[tactile]
  critic_actor = tuple(n for n in actor_terms if n != noisy)
  critic_priv = priv_terms + (("beam_ahead",) if curves else ())
  return {
    "actor": ObservationGroupCfg(
      terms={n: TERMS[n].build() for n in actor_terms},
      concatenate_terms=True,
      enable_corruption=True,
    ),
    # The faulted tactile term is stateful (frame drops), so it is computed
    # once, for the actor; the critic reads the clean features.
    "critic": ObservationGroupCfg(
      terms={n: TERMS[n].build() for n in critic_actor + critic_priv},
      concatenate_terms=True,
      enable_corruption=False,
    ),
  }


def make_beam_env_cfg(
  *,
  play: bool = False,
  num_envs: int | None = None,
  curves: bool = False,
  tactile: Tactile = "touch",
  evaluation: bool = False,
) -> ManagerBasedRlEnvCfg:
  """``curves``: two thirds of the terrain columns are bent beams
  (``terrain.BEND_MAX_TURN_DEG``), and the critic also sees how the beam
  turns ahead (``beam_ahead``). ``tactile``: see ``Tactile``. ``evaluation``:
  held-out terrain (``EVAL_TERRAIN_SEED``) and ``EVAL_EPISODE_S`` episodes,
  observations unchanged so training checkpoints load."""
  cfg = make_velocity_env_cfg()
  cfg.episode_length_s = EVAL_EPISODE_S if evaluation else EPISODE_S

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
  cfg.scene.terrain.terrain_generator = make_beam_terrain_cfg(
    curves=curves,
    **({"seed": EVAL_TERRAIN_SEED} if evaluation else {}),
  )
  cfg.scene.terrain.max_init_terrain_level = 0  # everyone starts on the widest beam

  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 128
  cfg.sim.nconmax = 70

  cfg.observations = build_observations(curves, tactile)

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

  cfg.rewards = build_beam_reward()
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
