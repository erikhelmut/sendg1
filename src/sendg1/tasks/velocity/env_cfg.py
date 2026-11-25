"""Velocity-tracking environment configs for the blind Unitree G1.

We own this assembly outright. It starts from mjlab's robot-agnostic base
factory (``make_velocity_env_cfg``) for sim settings, command term and viewer,
then applies our own G1 wiring -- rather than calling mjlab's
``unitree_g1_flat_env_cfg`` and mutating its result.

The reason is comparability: the numbers that define our baseline (which
observations sit in which group, reward weights, DR ranges) must not move when
mjlab releases a new version. The MDP term *library* underneath -- rewards,
sensors, terrain generation, entity/actuator machinery -- is imported and we do
want upstream fixes to it.

Both terrains produce a 96-dim actor observation. Removing the height scan makes
flat and rough share one policy input space, so a single architecture serves
both and flat->rough transfer is a legitimate experiment later.
"""

from __future__ import annotations

import math
from typing import Literal

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.velocity import mdp as vel_mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

from sendg1.common.events import build_events
from sendg1.common.hand_forces import (
  HandForceCfg,
  apply_hand_forces,
  hand_force_curriculum,
  reset_hand_forces,
)
from sendg1.common.obs.groups import build_groups
from sendg1.common.obs.obs_sets import uses_tactile
from sendg1.common.ppo import NUM_ENVS
from sendg1.common.rewards import build_rewards
from sendg1.common.robots.g1 import (
  G1_TORSO_BODY,
  DofSet,
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
from sendg1.common.tactile.sensing import tactile_scene_sensors
from sendg1.common.terrains import make_rough_terrain_cfg

Terrain = Literal["flat", "rough"]

DEFAULT_NUM_ENVS: dict[Terrain, int] = {"flat": NUM_ENVS, "rough": NUM_ENVS}
"""The shared default (``sendg1.common.ppo.NUM_ENVS``), flat and rough alike.
Until 2026-11-25: flat 8192, rough 1024. Rough is memory-bound: its curriculum
terrain compiles to ~3,100 static geoms and MuJoCo-Warp's broadphase scales
with geoms x envs, so 4096 does not build on a 12 GB card (2048 needs ~11 GiB);
the 24 GB cluster cards ran 3072. Where 4096 does not fit, train both
conditions with the same smaller ``--env.scene.num-envs``."""


def make_g1_velocity_env_cfg(
  terrain: Terrain,
  obs_set: str,
  *,
  dof_set: DofSet = "full29",
  play: bool = False,
  num_envs: int | None = None,
  hand_forces: HandForceCfg | None = None,
) -> ManagerBasedRlEnvCfg:
  """Build a G1 velocity config for one (terrain, observation set).

  ``hand_forces`` adds VOFA's force curriculum at the end effectors
  (``sendg1.common.hand_forces``) and changes NOTHING else: same rewards, same
  command, same observations. The policy it produces is a drop-in low level for
  any task with the same actor, which is the point.
  """
  cfg = make_velocity_env_cfg()

  # -- robot ---------------------------------------------------------------
  cfg.scene.entities = {"robot": get_g1_cfg()}
  cfg.scene.num_envs = num_envs or DEFAULT_NUM_ENVS[terrain]

  # -- sensors -------------------------------------------------------------
  has_terrain_scan = terrain == "rough"
  sensors = [foot_height_sensor(), feet_ground_contact_sensor(), self_collision_sensor()]
  if has_terrain_scan:
    sensors.insert(0, terrain_scan_sensor())
  if uses_tactile(obs_set):
    sensors.extend(tactile_scene_sensors())
  cfg.scene.sensors = tuple(sensors)

  # -- terrain -------------------------------------------------------------
  assert cfg.scene.terrain is not None
  if terrain == "flat":
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None
  else:
    cfg.scene.terrain.terrain_type = "generator"
    # Our own curriculum mix, not mjlab's ROUGH_TERRAINS_CFG: that default
    # spends ~40% of its columns on smooth terrain (slopes, waves) where foot
    # tactile can add nothing by construction, which would dilute the ablation.
    # See sendg1/common/terrains.py for the selection argument.
    cfg.scene.terrain.terrain_generator = make_rough_terrain_cfg()
    cfg.scene.terrain.max_init_terrain_level = 5

  # -- solver budgets ------------------------------------------------------
  cfg.sim.mujoco.ccd_iterations = 500 if terrain == "rough" else 50
  cfg.sim.contact_sensor_maxmatch = 500 if terrain == "rough" else 64
  cfg.sim.nconmax = 70 if terrain == "rough" else None
  if terrain == "flat":
    cfg.sim.njmax = 300

  # -- observations --------------------------------------------------------
  # The actor is blind and deployable-only; the critic is identical across
  # every observation set. Enforced by tests/test_actor_allowlist.py.
  # A low level trained for box pushing observes what the box-push policy
  # observes (same proprio size, same tactile term), so it drops into that task.
  cfg.observations = build_groups(obs_set, has_terrain_scan=has_terrain_scan)

  # -- actions -------------------------------------------------------------
  cfg.actions = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=actuator_patterns(dof_set),
      scale=resolve_action_scale(dof_set),
      use_default_offset=True,
    )
  }

  # -- rewards / events ----------------------------------------------------
  cfg.rewards = build_rewards()
  cfg.events = build_events()
  if hand_forces is not None:
    cfg.events["hand_forces_reset"] = EventTermCfg(
      func=reset_hand_forces, mode="reset", params={"cfg": hand_forces}
    )
    cfg.events["hand_forces"] = EventTermCfg(
      func=apply_hand_forces, mode="step", params={"cfg": hand_forces}
    )

  # -- terminations --------------------------------------------------------
  cfg.terminations = {
    "time_out": TerminationTermCfg(func=vel_mdp.time_out, time_out=True),
    "fell_over": TerminationTermCfg(
      func=vel_mdp.bad_orientation,
      params={"limit_angle": math.radians(70.0)},
    ),
  }
  if terrain == "rough":
    cfg.terminations["out_of_terrain_bounds"] = TerminationTermCfg(
      func=vel_mdp.out_of_terrain_bounds,
      time_out=True,
    )

  # -- curriculum ----------------------------------------------------------
  cfg.curriculum = {
    "command_vel": CurriculumTermCfg(
      func=vel_mdp.commands_vel,
      params={
        "command_name": "twist",
        "velocity_stages": [
          {"step": 0, "lin_vel_x": (-1.0, 1.0), "ang_vel_z": (-0.5, 0.5)},
          {"step": 5000 * 24, "lin_vel_x": (-1.5, 2.0), "ang_vel_z": (-0.7, 0.7)},
          {"step": 10000 * 24, "lin_vel_x": (-2.0, 3.0)},
        ],
      },
    )
  }
  if terrain == "rough":
    cfg.curriculum["terrain_levels"] = CurriculumTermCfg(
      func=vel_mdp.terrain_levels_vel,
      params={"command_name": "twist"},
    )
  if hand_forces is not None:
    cfg.curriculum["hand_forces"] = CurriculumTermCfg(
      func=hand_force_curriculum, params={"cfg": hand_forces}
    )

  # -- viewer / command viz ------------------------------------------------
  cfg.viewer.body_name = G1_TORSO_BODY
  twist = cfg.commands["twist"]
  assert isinstance(twist, UniformVelocityCommandCfg)
  twist.viz.z_offset = 1.15

  if play:
    _apply_play_overrides(cfg, terrain)

  return cfg


def _apply_play_overrides(cfg: ManagerBasedRlEnvCfg, terrain: Terrain) -> None:
  """Deterministic-ish viewing config. Not the eval protocol -- see sendg1.eval."""
  cfg.episode_length_s = int(1e9)
  cfg.observations["proprio"].enable_corruption = False
  cfg.events.pop("push_robot", None)
  cfg.terminations.pop("out_of_terrain_bounds", None)
  cfg.curriculum = {}
  cfg.events["randomize_terrain"] = EventTermCfg(
    func=envs_mdp.randomize_terrain, mode="reset", params={}
  )

  twist = cfg.commands["twist"]
  assert isinstance(twist, UniformVelocityCommandCfg)
  twist.ranges.lin_vel_x = (-1.5, 2.0)
  twist.ranges.ang_vel_z = (-0.7, 0.7)

  if terrain == "rough" and cfg.scene.terrain is not None:
    gen = cfg.scene.terrain.terrain_generator
    if gen is not None:
      gen.curriculum = False
      gen.num_cols = 5
      gen.num_rows = 5
      gen.border_width = 10.0
