"""Heavy-box pushing under unknown floor friction, Unitree G1.

The robot starts behind a box on a flat floor and must push it into a 0.5 m
target disc 2-4 m away. Each episode samples the floor friction, the box's
mass, size and box-floor friction, and the target direction
(``sendg1/common/box_push/params.py``). Floor friction is never observed by the
actor: pushing a heavy box on a slippery floor without slipping means sensing
the traction the feet actually have, which is what the tactile study is about.

The actor gets proprioception plus the box pose and the target, which the task
ASSUMES an external perception stack provides (availability ``perceived``, see
``sendg1/common/obs/terms.py``). Everything else about the world -- floor mu,
box mass and mu, slip velocities, forces -- reaches only the critic, the
rewards and the logs.
"""

from __future__ import annotations

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.velocity import mdp as vel_mdp
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

from sendg1.common.box_push import mdp as box_mdp
from sendg1.common.box_push.params import BoxPushRanges, CurriculumCfg, PatchCfg
from sendg1.common.box_push.scene import (
  box_entities,
  box_push_sensors,
  edit_scene_spec,
  edit_scene_spec_with_patches,
)
from sendg1.common.events import build_events
from sendg1.common.obs.groups import build_groups
from sendg1.common.obs.obs_sets import uses_tactile_array
from mjlab.managers.scene_entity_config import SceneEntityCfg

from sendg1.common.rewards import POSE_STD_WALKING, build_rewards
from sendg1.common.robots.g1 import (
  DofSet,
  G1_TORSO_BODY,
  actuator_patterns,
  get_g1_ball_hands_cfg,
  get_g1_cfg,
  resolve_action_scale,
)
from sendg1.common.sensors import (
  feet_ground_contact_sensor,
  foot_height_sensor,
  self_collision_sensor,
)
from sendg1.common.tactile.sensing import tactile_scene_sensors

DEFAULT_NUM_ENVS = 4096
"""Measured on the RTX 5070; see the README's box-pushing section."""

EPISODE_LENGTH_S = 20.0

LOW_FRICTION_PATCHES = False
"""Default for the optional 1-3 low-friction floor patches on the path."""
CURRICULUM = False
"""Default for the light-box / high-friction curriculum."""
BALL_HANDS = True
"""Ball end-effectors instead of the stock rubber hands (see
``sendg1.common.robots.g1``): one contact per hand, with its own friction."""

FALL_TILT = math.radians(70.0)
"""Torso tilt that ends the episode as a fall (the velocity task's value)."""
FALL_HEIGHT = 0.40
"""Pelvis height (m) below which the episode ends as a fall. Standing ~0.76."""
BOX_TIP_ANGLE = math.radians(45.0)
"""Box tilt that ends the episode: past 45 degrees an unaided box goes over."""

# ---------------------------------------------------------------------------
# Rewards: every term and weight, in one place
# ---------------------------------------------------------------------------
#
# The reward manager multiplies each term by its weight and by step_dt (0.02 s),
# so a term returning 1 every step with weight w pays w per second. One-off
# terms (success, fall) divide by step_dt, so their weight IS the one-off amount.
#
# | term              | weight | value                                       |
# |-------------------|--------|---------------------------------------------|
# | upright           |  +1.0  | the velocity task's term, unchanged          |
# | posture           |  +1.0  | the velocity task's walking tolerances       |
# | box_progress      |  +4.0  | box speed toward the target, /0.4 m/s        |
# | track_lin_vel     |  +1.0  | exp(-|v_cmd - v|^2 / 0.5^2), pilot command   |
# | track_ang_vel     |  +1.0  | exp(-(wz_cmd - wz)^2 / 0.7^2)                |
# | box_to_target     |  +2.5  | exp(-d_box_target^2 / 2.5^2)                 |
# | success           | +20.0  | once, box in the disc for 2 s                |
# | hand_contact      |  +0.5  | 1 while a hand/forearm is on the push face   |
# | foot_slip         |  -2.0  | sum_feet in_contact * |v_sole_xy|^2 (m2/s2)  |
# | fall              | -10.0  | once, on a fall termination                  |
# | box_tipped        | -10.0  | once, when the box tips over                 |
# | box_tilt          |  -5.0  | sin^2(box tilt)                              |
# | feet_box_contact  |  -1.0  | number of feet touching the box              |
# | joint_torques     | -1e-5  | sum of squared actuator torques (N2m2)       |
# | action_rate_l2    |  -0.1  | the velocity task's term, unchanged          |
#
# The three positive dense terms come from VOFA (arXiv:2605.01518), whose
# push/reach/alignment rewards are all exponential kernels: the robot earns
# reward every step it is doing roughly the right thing. The first version of
# this task had no positive per-step term, so the return was negative
# everywhere and both 2,000-iteration checks converged to ENDING the episode --
# first by tipping the box over, then by falling immediately. Tracking rewards
# replace VOFA's reach and alignment terms, which the scripted pilot
# (common/box_push/pilot.py) now handles geometrically.
#
# upright and posture are the velocity task's positive terms, and they are what
# make staying alive worth more than the -10 fall penalty REGARDLESS of how the
# task is going. Without them the only per-step income was command tracking,
# whose exp kernel pays ~0 while the error is large, so early in training a
# policy that dove at the floor in 0.3 s beat one that kept walking: a run
# without them collapsed to 99.97% falls-before-contact by iteration 1,000.
#
# Command tracking is SHAPING, not the objective, and is weighted below the
# box terms on purpose: the pilot's command is a function of the robot's own
# position, so tracking it is gameable -- a run with tracking at 2.0 and no
# progress term circled the box forever (reward 33, success 0%, box untouched
# at iteration 5,000). Progress is measured on the box and cannot be faked.
#
# Weights are not tuned. Per second, roughly: perfect tracking pays 2, a box
# at the target 2.5, a foot sliding at 0.3 m/s costs 0.18, a 10-degree box tilt
# 0.15.

REWARD_WEIGHTS: dict[str, float] = {
  "upright": 1.0,
  "posture": 1.0,
  "box_progress": 4.0,
  "track_lin_vel": 1.0,
  "track_ang_vel": 1.0,
  "box_to_target": 2.5,
  "success": 20.0,
  "hand_contact": 0.5,
  "foot_slip": -2.0,
  "fall": -10.0,
  "box_tipped": -10.0,
  "box_tilt": -5.0,
  "feet_box_contact": -1.0,
  "joint_torques": -1e-5,
  "action_rate_l2": -0.1,
}

TRACK_LIN_STD = 0.5
"""m/s. The velocity task uses sqrt(0.25); the same tolerance."""
TRACK_ANG_STD = 0.7
"""rad/s. The velocity task uses sqrt(0.5)."""
BOX_PROGRESS_MAX_SPEED = 0.4
"""m/s of box motion toward the target above which box_progress pays no more.
Below the pilot's 0.5 m/s push command: a heavy box does not move that fast,
and rewarding speed the floor cannot deliver just asks the feet to slip."""
BOX_TO_TARGET_STD = 2.5
"""m. VOFA uses 10 m for a 3 m task, i.e. an almost flat potential. 2.5 m over
our 2-4 m keeps a usable gradient toward the target."""

BOX_PROGRESS_MAX_SPEED = 0.5
"""m/s. Box speed toward the target above which box_progress pays no more."""
PUSH_FACE_ANGLE = 45.0
"""deg. A side face counts as the push face when its outward normal is within
this of the direction pointing away from the target."""


def build_box_push_rewards() -> dict[str, RewardTermCfg]:
  w = REWARD_WEIGHTS
  shared = build_rewards()
  action_rate = shared["action_rate_l2"]
  assert action_rate.weight == w["action_rate_l2"]
  assert shared["upright"].weight == w["upright"]
  return {
    "upright": shared["upright"],
    "posture": RewardTermCfg(
      func=envs_mdp.posture,
      weight=w["posture"],
      params={
        "std": POSE_STD_WALKING,
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
    "box_progress": RewardTermCfg(
      func=box_mdp.box_progress,
      weight=w["box_progress"],
      params={"max_speed": BOX_PROGRESS_MAX_SPEED},
    ),
    "track_lin_vel": RewardTermCfg(
      func=box_mdp.track_linear_velocity,
      weight=w["track_lin_vel"],
      params={"std": TRACK_LIN_STD},
    ),
    "track_ang_vel": RewardTermCfg(
      func=box_mdp.track_angular_velocity,
      weight=w["track_ang_vel"],
      params={"std": TRACK_ANG_STD},
    ),
    "box_to_target": RewardTermCfg(
      func=box_mdp.box_to_target,
      weight=w["box_to_target"],
      params={"std": BOX_TO_TARGET_STD},
    ),
    "success": RewardTermCfg(func=box_mdp.box_success_bonus, weight=w["success"]),
    "hand_contact": RewardTermCfg(
      func=box_mdp.hand_contact_on_push_face,
      weight=w["hand_contact"],
      params={"max_angle_deg": PUSH_FACE_ANGLE},
    ),
    "foot_slip": RewardTermCfg(func=box_mdp.feet_slip, weight=w["foot_slip"]),
    "fall": RewardTermCfg(func=box_mdp.fell, weight=w["fall"]),
    "joint_torques": RewardTermCfg(
      func=envs_mdp.joint_torques_l2, weight=w["joint_torques"]
    ),
    "action_rate_l2": action_rate,
    "feet_box_contact": RewardTermCfg(
      func=box_mdp.feet_box_contact, weight=w["feet_box_contact"]
    ),
    "box_tilt": RewardTermCfg(func=box_mdp.box_tilt, weight=w["box_tilt"]),
    "box_tipped": RewardTermCfg(
      func=box_mdp.terminated_by,
      weight=w["box_tipped"],
      params={"term_names": (box_mdp.BOX_TIPPED_TERM,)},
    ),
  }


def build_box_push_events(
  ranges: BoxPushRanges, patches: PatchCfg | None, curriculum: CurriculumCfg | None
) -> dict[str, EventTermCfg]:
  events = build_events()
  # The robot stands at the env origin facing +x, the box ahead of it; only the
  # heading is noisy. The spawn gap to the box is sampled by box_push_reset.
  yaw = ranges.robot_yaw_noise
  events["reset_base"].params["pose_range"] = {
    "z": (0.01, 0.05),
    "yaw": (-yaw, yaw),
  }
  # No pushes: the box is the disturbance, and a shove while leaning into a
  # 40 kg box is a different task.
  events.pop("push_robot")
  # The foot capsules' own friction never reaches a contact the task cares
  # about: the floor and patches outrank them (see params.py). Randomising it
  # would be inert, so it is removed rather than left to mislead.
  events.pop("foot_friction")
  events[box_mdp.RESET_EVENT] = EventTermCfg(
    func=box_mdp.box_push_reset,
    mode="reset",
    params={"ranges": ranges, "patches": patches, "curriculum": curriculum},
  )
  return events


def build_box_push_metrics() -> dict[str, MetricsTermCfg]:
  """Per-episode statistics for the training logs (Episode_Metrics/*).

  Each is the value at the episode's final step, averaged over the episodes
  that ended in that iteration. The evaluation script records the same
  quantities per episode, with NaN where a quantity is undefined.
  """
  fns = {
    "success": box_mdp.metric_success,
    "time_to_target": box_mdp.metric_time_to_target,
    "slip_events": box_mdp.metric_slip_events,
    "fell": box_mdp.metric_fell,
    "fell_before_contact": box_mdp.metric_fell_before_contact,
    "fell_after_contact": box_mdp.metric_fell_after_contact,
    "command_error": box_mdp.metric_command_error,
    "traction_margin": box_mdp.metric_traction_margin,
    "box_tipped": box_mdp.metric_box_tipped,
    "mean_push_force": box_mdp.metric_mean_push_force,
    "double_support_impulse_frac": box_mdp.metric_double_support_impulse_frac,
    "floor_mu": box_mdp.metric_floor_mu,
    "box_mass": box_mdp.metric_box_mass,
    "box_mu": box_mdp.metric_box_mu,
    "box_to_target": box_mdp.metric_box_to_target,
  }
  return {name: MetricsTermCfg(func=fn, reduce="last") for name, fn in fns.items()}


def make_g1_box_push_env_cfg(
  obs_set: str,
  *,
  dof_set: DofSet = "full29",
  play: bool = False,
  num_envs: int | None = None,
  low_friction_patches: bool = LOW_FRICTION_PATCHES,
  curriculum: bool = CURRICULUM,
  ball_hands: bool = BALL_HANDS,
  ranges: BoxPushRanges = BoxPushRanges(),
) -> ManagerBasedRlEnvCfg:
  """Build a G1 box-pushing config for one observation set."""
  cfg = make_velocity_env_cfg()
  patch_cfg = PatchCfg() if low_friction_patches else None
  curriculum_cfg = CurriculumCfg() if curriculum else None

  # -- scene ---------------------------------------------------------------
  robot = get_g1_ball_hands_cfg() if ball_hands else get_g1_cfg()
  cfg.scene.entities = {"robot": robot, **box_entities()}
  cfg.scene.num_envs = num_envs or DEFAULT_NUM_ENVS
  # Worlds do not interact; the spacing only matters for the viewer.
  cfg.scene.env_spacing = 8.0
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None
  cfg.scene.spec_fn = (
    edit_scene_spec_with_patches if low_friction_patches else edit_scene_spec
  )

  sensors = [
    foot_height_sensor(),
    feet_ground_contact_sensor(),
    self_collision_sensor(),
    *box_push_sensors(),
  ]
  if uses_tactile_array(obs_set):
    sensors.extend(tactile_scene_sensors())
  cfg.scene.sensors = tuple(sensors)

  # -- solver budgets (flat-ground values plus room for the box) -----------
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None
  cfg.sim.njmax = 400

  # -- observations --------------------------------------------------------
  # The actor's task input is the pilot's twist, exactly like the velocity
  # tasks' operator command: the box pose it would need to plan the approach is
  # the scripted high level's business, and stays with the critic.
  cfg.observations = build_groups(
    obs_set, has_terrain_scan=False, with_command=False, box_push=True
  )

  # -- actions / commands --------------------------------------------------
  cfg.actions = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=actuator_patterns(dof_set),
      scale=resolve_action_scale(dof_set),
      use_default_offset=True,
    )
  }
  cfg.commands = {}

  # -- rewards / events / metrics ------------------------------------------
  cfg.rewards = build_box_push_rewards()
  cfg.events = build_box_push_events(ranges, patch_cfg, curriculum_cfg)
  cfg.metrics = build_box_push_metrics()

  # -- terminations --------------------------------------------------------
  cfg.episode_length_s = EPISODE_LENGTH_S
  cfg.terminations = {
    "time_out": TerminationTermCfg(func=vel_mdp.time_out, time_out=True),
    box_mdp.SUCCESS_TERM: TerminationTermCfg(func=box_mdp.box_reached_target),
    "fell_over": TerminationTermCfg(
      func=vel_mdp.bad_orientation, params={"limit_angle": FALL_TILT}
    ),
    "base_too_low": TerminationTermCfg(
      func=envs_mdp.root_height_below_minimum, params={"minimum_height": FALL_HEIGHT}
    ),
    box_mdp.BOX_TIPPED_TERM: TerminationTermCfg(
      func=box_mdp.box_tipped_over, params={"limit_angle": BOX_TIP_ANGLE}
    ),
    "nan_detection": TerminationTermCfg(func=envs_mdp.nan_detection),
  }

  # -- curriculum ----------------------------------------------------------
  cfg.curriculum = {}
  if curriculum_cfg is not None:
    cfg.curriculum["box_push_levels"] = CurriculumTermCfg(func=box_mdp.box_push_levels)

  cfg.viewer.body_name = G1_TORSO_BODY
  # From behind the robot's right shoulder, looking along the push: robot, box
  # and target disc all in frame.
  cfg.viewer.distance = 5.5
  cfg.viewer.elevation = -25.0
  cfg.viewer.azimuth = 30.0

  if play:
    cfg.observations["proprio"].enable_corruption = False

  return cfg
