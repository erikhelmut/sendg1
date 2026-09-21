"""Beam-walking environment config for the blind Unitree G1.

The robot starts on a 0.6 m platform facing a narrow beam over a pit and must
get as far along it as it can. The beam bends at random joints (how sharply is
set by the terrain column) and narrows with the curriculum (by row); see
``sendg1/common/beam/terrain.py``.

There is NO command. Nothing tells the actor where the beam goes, so the only
way to follow a bend is to sense where the support is -- the capability the
tactile study is about. Everything the task knows about the beam (progress,
offset, width) is ground truth and reaches only rewards, terminations, the
curriculum and the critic.
"""

from __future__ import annotations

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.velocity import mdp as vel_mdp
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

from sendg1.common.beam import mdp as beam_mdp
from sendg1.common.beam.terrain import make_beam_terrain_cfg
from sendg1.common.events import build_events
from sendg1.common.obs.groups import build_groups
from sendg1.common.rewards import POSE_STD_WALKING, build_rewards
from sendg1.common.robots.g1 import (
  DofSet,
  G1_FOOT_SITES,
  G1_TORSO_BODY,
  actuator_patterns,
  get_g1_cfg,
  resolve_action_scale,
)
from sendg1.common.sensors import (
  feet_ground_contact_sensor,
  foot_height_sensor,
  self_collision_sensor,
  terrain_scan_sensor,
)

DEFAULT_NUM_ENVS = 2048
"""Measured on an RTX 5070 (12 GB): 2,048 envs use 2.8 GiB at 86k steps/s;
4,096 use 6.3 GiB at 97k steps/s -- 13% more throughput for twice the batch and
the memory. Retune with scripts/tune_num_envs.py."""

POSE_STD_SCALE = 3.0
"""Posture tolerances are the flat-ground walking ones times this.

mjlab's posture reward is exp(-mean(err^2 / std^2)), which is flat zero once a
gait strays far from the default pose -- it then shapes nothing. A 3,000-iteration
run learned a deep lunging gait (pelvis 0.47 m above the beam against ~0.73 m
standing) on which the walking-std posture reward had median 0.000. Measured on
that gait, x2 still gives only 0.08; x3 gives 0.32, a usable gradient back
toward the default pose. It also covers the hip adduction a narrow stance needs,
which previously had its own override."""

POSE_STD_BEAM: dict[str, float] = {
  pattern: POSE_STD_SCALE * std for pattern, std in POSE_STD_WALKING.items()
}

BASE_HEIGHT_TARGET = 0.70
"""Base height above the beam top below which the base_height penalty applies.
Just under the ~0.73 m of a relaxed stance, so normal knee bend costs nothing."""

STEPS_PER_ITERATION = 48
"""PPO rollout length (``num_steps_per_env``). 0.96 s at 50 Hz, about one gait
cycle: truncated backpropagation through time has to span a probing step and
what it revealed. Defined here because the penalty ramp is scheduled in env
steps."""

PENALTY_TERMS: tuple[str, ...] = (
  "body_ang_vel",
  "angular_momentum",
  "dof_pos_limits",
  "action_rate_l2",
  "foot_clearance",
  "self_collisions",
  "base_height",
)

PENALTY_RAMP: tuple[tuple[int, float], ...] = tuple(
  (300 * k, round(0.1 * (k + 1), 1)) for k in range(10)
)
"""(PPO iteration, fraction of full weight) for every term in PENALTY_TERMS:
+10% every 300 iterations, full weight at iteration 2,700.

At full weight a random policy (action std 1.0) pays ~8.7 per second alive in
penalties -- action_rate_l2 alone ~7 -- against ~1.7 from the positive terms. On
a 0.6 m platform the cheapest way to stop paying is to step off it, and two
smoke runs learned exactly that: mean episode length fell from 51 to 34 steps
in 25 iterations. So penalties start at 10%.

The ramp is gradual because an earlier one jumped 10% -> 40% -> 100% at
iterations 1,000 and 2,000, and in two runs each jump collapsed training (reward
to -6, episodes from ~100 to 15 steps) before the policy recovered."""


def build_beam_rewards() -> dict[str, RewardTermCfg]:
  """Progress along the beam, plus the velocity task's regularisers.

  Velocity tracking is gone (there is no command), and so are the terms that
  require a command to gate them (foot_swing_height, foot_slip) or were inert
  anyway (air_time, soft_landing).

  upright and posture keep their velocity-task weights, and the penalties are
  ramped in (see PENALTY_RAMP). Both address one failure: before the robot has
  made any progress these are its only positive rewards, and with the penalties
  at full weight every living step was net-negative, so smoke runs learned to
  step off the platform to end the episode. Standing still is countered by the
  curriculum instead, which demotes an env that makes no progress to a wider,
  nearly riskless beam.

  Three terms counter the deep lunging gait a 3,000-iteration run converged to:
  posture tolerances widened so the term keeps a gradient far from the default
  pose (POSE_STD_SCALE), a base_height penalty against the crouch, and a lower
  progress speed cap (0.4 m/s, was 0.6) so lunging forward pays less.
  """
  shared = build_rewards()
  return {
    "beam_progress": RewardTermCfg(
      func=beam_mdp.beam_progress, weight=2.0, params={"max_speed": 0.4}
    ),
    "beam_success": RewardTermCfg(func=beam_mdp.beam_success_bonus, weight=20.0),
    "beam_centreline": RewardTermCfg(func=beam_mdp.beam_centreline, weight=0.5),
    "upright": RewardTermCfg(
      func=vel_mdp.upright,
      weight=1.0,
      params={
        "std": math.sqrt(0.2),
        "asset_cfg": SceneEntityCfg("robot", body_names=(G1_TORSO_BODY,)),
      },
    ),
    "posture": RewardTermCfg(
      func=envs_mdp.posture,
      weight=1.0,
      params={
        "std": POSE_STD_BEAM,
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
    "body_ang_vel": shared["body_ang_vel"],
    "angular_momentum": shared["angular_momentum"],
    "dof_pos_limits": shared["dof_pos_limits"],
    "action_rate_l2": shared["action_rate_l2"],
    "foot_clearance": RewardTermCfg(
      func=vel_mdp.feet_clearance,
      weight=-2.0,
      params={
        "target_height": 0.1,
        "height_sensor_name": "foot_height_scan",
        # Ungated: with no command there is no "standing" to exempt. The cost is
        # weighted by foot speed, so a still foot costs nothing anyway.
        "command_name": None,
        "asset_cfg": SceneEntityCfg("robot", site_names=G1_FOOT_SITES),
      },
    ),
    "self_collisions": shared["self_collisions"],
    "base_height": RewardTermCfg(
      func=beam_mdp.base_height_shortfall,
      # At the measured lunge (0.47 m) this costs ~1 per second at full weight;
      # a relaxed stance costs nothing.
      weight=-20.0,
      params={"target": BASE_HEIGHT_TARGET},
    ),
  }


def build_beam_events() -> dict[str, EventTermCfg]:
  events = build_events()
  # Spawn at the platform centre facing the beam, with just enough noise that the
  # first steps cannot be replayed open-loop.
  events["reset_base"].params["pose_range"] = {
    "x": (-0.05, 0.05),
    "y": (-0.05, 0.05),
    "z": (0.01, 0.05),
    "yaw": (-math.radians(10.0), math.radians(10.0)),
  }
  # No pushes. The beam is the disturbance: on a 0.15 m beam a 0.5 m/s shove is
  # an unrecoverable fall for any policy, which would floor the narrow rows for
  # both conditions and hide exactly the effect being measured.
  events.pop("push_robot")
  events["reset_beam_tracker"] = EventTermCfg(
    func=beam_mdp.reset_beam_tracker, mode="reset"
  )
  return events


def build_penalty_ramp(rewards: dict[str, RewardTermCfg]) -> dict[str, CurriculumTermCfg]:
  """Scale each penalty in from a fraction of its full weight; see PENALTY_RAMP."""
  return {
    f"ramp_{name}": CurriculumTermCfg(
      func=envs_mdp.reward_curriculum,
      params={
        "reward_name": name,
        "stages": [
          {"step": iteration * STEPS_PER_ITERATION, "weight": frac * rewards[name].weight}
          for iteration, frac in PENALTY_RAMP
        ],
      },
    )
    for name in PENALTY_TERMS
  }


def make_g1_beam_env_cfg(
  obs_set: str,
  *,
  dof_set: DofSet = "full29",
  play: bool = False,
  num_envs: int | None = None,
) -> ManagerBasedRlEnvCfg:
  """Build a G1 beam-walking config for one observation set."""
  cfg = make_velocity_env_cfg()

  # -- robot ---------------------------------------------------------------
  cfg.scene.entities = {"robot": get_g1_cfg()}
  cfg.scene.num_envs = num_envs or DEFAULT_NUM_ENVS

  # -- sensors -------------------------------------------------------------
  # The terrain scan serves the critic (it sees the beam); the foot height scan
  # feeds foot_clearance. Neither reaches the actor.
  cfg.scene.sensors = (
    terrain_scan_sensor(),
    foot_height_sensor(),
    feet_ground_contact_sensor(),
    self_collision_sensor(),
  )

  # -- terrain -------------------------------------------------------------
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "generator"
  cfg.scene.terrain.terrain_generator = make_beam_terrain_cfg()
  cfg.scene.terrain.max_init_terrain_level = 0  # everyone starts on the widest beam

  # -- solver budgets ------------------------------------------------------
  # Contact budgets as on rough: at a bend a foot can touch a box and a cylinder
  # at once. CCD iterations as on flat, NOT rough's 500: measured on the 5070,
  # 500 made this scene memory-bound at 1,024 envs (5.6 GiB, 67k steps/s; 2,048
  # failed to allocate), while 50 runs 2,048 envs in 2.8 GiB at 86k steps/s.
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 500
  cfg.sim.nconmax = 70

  # -- observations --------------------------------------------------------
  cfg.observations = build_groups(
    obs_set, has_terrain_scan=True, with_command=False, beam_privileged=True
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

  # -- rewards / events ----------------------------------------------------
  cfg.rewards = build_beam_rewards()
  cfg.events = build_beam_events()

  # -- terminations --------------------------------------------------------
  cfg.terminations = {
    "time_out": TerminationTermCfg(func=vel_mdp.time_out, time_out=True),
    "fell_over": TerminationTermCfg(
      func=vel_mdp.bad_orientation,
      params={"limit_angle": math.radians(70.0)},
    ),
    "fell_off": TerminationTermCfg(
      func=beam_mdp.fell_off_beam, params={"margin": beam_mdp.FALL_MARGIN}
    ),
    # Resets an env whose physics state went NaN/Inf instead of letting it
    # poison the batch: a full run died at iteration 1,916 when NaN reached the
    # critic's observations. A real termination, not a time-out, so the critic
    # never bootstraps from the corrupted state.
    "nan_detection": TerminationTermCfg(func=envs_mdp.nan_detection),
  }
  # Crossing the beam does NOT end the episode. The robot keeps collecting
  # upright and posture at the far end, so crossing early forfeits nothing that
  # standing on the platform would have earned.

  # -- curriculum ----------------------------------------------------------
  cfg.curriculum = {
    "beam_levels": CurriculumTermCfg(
      func=beam_mdp.beam_levels,
      params={"promote_frac": 0.8, "demote_frac": 0.3},
    ),
    **build_penalty_ramp(cfg.rewards),
  }

  cfg.viewer.body_name = G1_TORSO_BODY

  if play:
    _apply_play_overrides(cfg)

  return cfg


def _apply_play_overrides(cfg: ManagerBasedRlEnvCfg) -> None:
  """For looking at a policy, not measuring it -- see sendg1.eval."""
  cfg.observations["proprio"].enable_corruption = False
  cfg.curriculum = {}
  assert cfg.scene.terrain is not None
  # Spread viewing envs over every width, not just the widest row.
  cfg.scene.terrain.max_init_terrain_level = None
