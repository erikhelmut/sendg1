"""Reward wiring for the G1 velocity tasks.

FROZEN FOR THE STUDY. The reward function is held byte-identical across every
observation set: it is the control variable, not the treatment. Changing a
weight invalidates every prior run (tests/test_config_fingerprint.py will say
so).

On contact-dependent rewards
----------------------------
Five terms consume sim-only foot sensors: foot_clearance, foot_swing_height,
foot_slip, soft_landing, air_time. They are kept deliberately.

Rewards are consumed only by the training loop, so they cannot leak into a
deployed policy -- there is no sim2real issue. The subtler effect is on the
BASELINE: these terms push the blind policy toward contact-appropriate gait
without giving it contact observations, so it must infer contact state from
proprioception. That is privileged-information distillation via reward shaping,
and it makes the baseline stronger than a contact-naive one would be. Any
tactile gain measured against it is therefore conservative, which is the honest
comparison. Removing them to weaken the baseline would inflate the effect size
without making it more true.

Two terms are inert at their shipped weights and are kept only for parity with
upstream: ``air_time`` (weight 0.0 for the G1) and ``soft_landing`` (-1e-5).
"""

from __future__ import annotations

import math

from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity import mdp as vel_mdp

from sendg1.common.robots.g1 import G1_FOOT_SITES, G1_TORSO_BODY

# Per-joint posture standard deviations. Looser std = more freedom.
# Knees/hip_pitch loosest for natural stride; hip roll/yaw tighter to limit
# lateral sway; ankle roll very tight for balance; waist roll/pitch tight to
# keep the torso upright; shoulders/elbows moderate for arm swing; wrists loose.
POSE_STD_STANDING: dict[str, float] = {".*": 0.05}

POSE_STD_WALKING: dict[str, float] = {
  r".*hip_pitch.*": 0.3,
  r".*hip_roll.*": 0.15,
  r".*hip_yaw.*": 0.15,
  r".*knee.*": 0.35,
  r".*ankle_pitch.*": 0.25,
  r".*ankle_roll.*": 0.1,
  r".*waist_yaw.*": 0.2,
  r".*waist_roll.*": 0.08,
  r".*waist_pitch.*": 0.1,
  r".*shoulder_pitch.*": 0.15,
  r".*shoulder_roll.*": 0.15,
  r".*shoulder_yaw.*": 0.1,
  r".*elbow.*": 0.15,
  r".*wrist.*": 0.3,
}

POSE_STD_RUNNING: dict[str, float] = {
  r".*hip_pitch.*": 0.5,
  r".*hip_roll.*": 0.2,
  r".*hip_yaw.*": 0.2,
  r".*knee.*": 0.6,
  r".*ankle_pitch.*": 0.35,
  r".*ankle_roll.*": 0.15,
  r".*waist_yaw.*": 0.3,
  r".*waist_roll.*": 0.08,
  r".*waist_pitch.*": 0.2,
  r".*shoulder_pitch.*": 0.5,
  r".*shoulder_roll.*": 0.2,
  r".*shoulder_yaw.*": 0.15,
  r".*elbow.*": 0.35,
  r".*wrist.*": 0.3,
}


def build_rewards() -> dict[str, RewardTermCfg]:
  """The full G1 velocity reward set. Identical for every observation set."""
  return {
    "track_linear_velocity": RewardTermCfg(
      func=vel_mdp.track_linear_velocity,
      weight=2.0,
      params={"command_name": "twist", "std": math.sqrt(0.25)},
    ),
    "track_angular_velocity": RewardTermCfg(
      func=vel_mdp.track_angular_velocity,
      weight=2.0,
      params={"command_name": "twist", "std": math.sqrt(0.5)},
    ),
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
        "std_standing": POSE_STD_STANDING,
        "std_walking": POSE_STD_WALKING,
        "std_running": POSE_STD_RUNNING,
        "walking_threshold": 0.05,
        "running_threshold": 1.5,
      },
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
    "air_time": RewardTermCfg(
      func=vel_mdp.feet_air_time,
      weight=0.0,  # Inert for the G1; kept for upstream parity.
      params={
        "sensor_name": "feet_ground_contact",
        "threshold_min": 0.05,
        "threshold_max": 0.5,
        "command_name": "twist",
        "command_threshold": 0.5,
      },
    ),
    "foot_clearance": RewardTermCfg(
      func=vel_mdp.feet_clearance,
      weight=-2.0,
      params={
        "target_height": 0.1,
        "height_sensor_name": "foot_height_scan",
        "command_name": "twist",
        "command_threshold": 0.05,
        "asset_cfg": SceneEntityCfg("robot", site_names=G1_FOOT_SITES),
      },
    ),
    "foot_swing_height": RewardTermCfg(
      func=vel_mdp.feet_swing_height,
      weight=-0.25,
      params={
        "sensor_name": "feet_ground_contact",
        "height_sensor_name": "foot_height_scan",
        "target_height": 0.1,
        "command_name": "twist",
        "command_threshold": 0.05,
      },
    ),
    "foot_slip": RewardTermCfg(
      func=vel_mdp.feet_slip,
      weight=-0.1,
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": "twist",
        "command_threshold": 0.05,
        "asset_cfg": SceneEntityCfg("robot", site_names=G1_FOOT_SITES),
      },
    ),
    "soft_landing": RewardTermCfg(
      func=vel_mdp.soft_landing,
      weight=-1e-5,  # Effectively inert; kept for upstream parity.
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": "twist",
        "command_threshold": 0.05,
      },
    ),
    "self_collisions": RewardTermCfg(
      func=vel_mdp.self_collision_cost,
      weight=-1.0,
      params={"sensor_name": "self_collision", "force_threshold": 10.0},
    ),
  }
