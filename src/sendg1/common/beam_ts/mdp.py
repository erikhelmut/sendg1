"""MDP terms for straight-beam walking (teacher-student).

Episode logic
-------------
Every episode is judged by :class:`BeamEpisode`, updated once per env step:

- ``progress``  arc length the base has advanced along the beam since spawn.
- ``cmd_dist``  integral of the commanded forward speed: how far the operator
                asked it to go.
- ``reached_end`` both feet and the base past the end of the beam, on the end
                platform. From then on the command is held at zero
                (:class:`BeamForwardCommand`) so the robot stops and stands
                instead of walking off the far side -- the old beam task's
                episodes all ended that way.

Success = no early termination AND (reached the end platform, OR covered at
least ``TRACK_FRAC`` of the commanded distance; a commanded distance under
``WALK_MIN_DIST`` counts as standing and only has to survive).

Curriculum (per env, :func:`beam_levels`)
-----------------------------------------
Promote one width level on success, demote one on an early termination, stay
otherwise (time-out without tracking). Levels from ``PLATFORM_FROM_LEVEL`` on
spawn on the start platform; below it, directly on the beam. Difficulty is
coupled to the level: the centreline reward's sigma tightens
(``CENTRELINE_SIGMA``), pushes switch on from ``PUSH_FROM_LEVEL`` and grow,
tactile faults grow (``tactile.noise_strength``), and the share of walking
commands rises with the population's mean level.

Replay: each env keeps its own curriculum ("home") level, but a share
``REPLAY_SHARE`` of resets is placed on a uniformly drawn level at or below it.
A replay episode does not move the home level. Without it the population drains
out of the wide levels and the teacher forgets them (first full run: 100 %
success at 0.05 m, 36 % at 0.40 m, with the width obs out of distribution).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.envs.mdp.actions.actions import JointPositionAction, JointPositionActionCfg
from mjlab.managers.event_manager import requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity.mdp.velocity_command import (
  UniformVelocityCommand,
  UniformVelocityCommandCfg,
)
from mjlab.utils.lab_api.math import quat_from_euler_xyz

from sendg1.common.beam_ts import tactile as tac
from sendg1.common.beam_ts.geometry import BeamGeometry
from sendg1.common.robots.g1 import G1_FOOT_GEOMS, G1_FOOT_SITES
from sendg1.common.tactile.layout import LAYOUTS
from sendg1.common.tactile.sensing import TactileParams, read_tactile

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

# -- episode / curriculum constants ------------------------------------------

PLATFORM_FROM_LEVEL = 2
"""First level (0.20 m) that starts on the start platform. The default stance is
0.31 m wide at the outer sole edges, so a spawn straight onto 0.20 m would put
both feet half off the beam; from here on the robot steps onto the beam itself."""

PUSH_FROM_LEVEL = 3
TRACK_FRAC = 0.7
WALK_MIN_DIST = 0.5
"""Commanded distance (m) below which an episode counts as standing."""

CENTRELINE_SIGMA: tuple[float, ...] = (0.12, 0.10, 0.07, 0.05, 0.04, 0.03, 0.025)
"""Stance-foot centreline reward width per level (m)."""

REPLAY_SHARE = 0.2
"""Share of resets placed on a random easier level (does not move the home level)."""

STANDING_SHARE = (0.6, 0.15)
"""Share of standing commands at mean level 0 and at the top level."""

FOOT_CONTACT_SENSOR = "feet_ground_contact"
SOLE_CONTACT_SENSOR = "sole_contacts"
NONFOOT_CONTACT_SENSOR = "nonfoot_contact"

FOOT_BELOW_TOP_TOL = 0.03
"""A sole contact this far below the top surface is on a beam side or the ground."""

TACTILE = TactileParams()


def _robot(env: ManagerBasedRlEnv):
  return env.scene["robot"]


def _forward_cmd(env: ManagerBasedRlEnv) -> torch.Tensor:
  return env.command_manager.get_command("twist")[:, 0]


# ---------------------------------------------------------------------------
# Per-episode state
# ---------------------------------------------------------------------------


@dataclass
class EpisodeStats:
  success: torch.Tensor
  progress: torch.Tensor
  cmd_dist: torch.Tensor
  reached_end: torch.Tensor
  vel_err: torch.Tensor
  level: torch.Tensor
  terminated: torch.Tensor


class BeamEpisode:
  _ATTR = "_sendg1_beam_ts_episode"

  def __init__(self, env: ManagerBasedRlEnv) -> None:
    self._env = env
    self.geo = BeamGeometry.get(env)
    n, dev = env.num_envs, env.device
    z = lambda: torch.zeros(n, device=dev)  # noqa: E731
    b = lambda: torch.zeros(n, device=dev, dtype=torch.bool)  # noqa: E731
    self.spawn_arc = z()
    self.progress = z()
    self.max_progress = z()
    self.cmd_dist = z()
    self.vel_err_sum = z()
    self.steps = z()
    self.reached_end = b()
    self.new_reach = b()
    self.platform_mode = b()
    self._stamp = -1
    robot = _robot(env)
    self._foot_sites, _ = robot.find_sites(G1_FOOT_SITES, preserve_order=True)

  @classmethod
  def get(cls, env: ManagerBasedRlEnv) -> BeamEpisode:
    ep = getattr(env, cls._ATTR, None)
    if ep is None:
      ep = cls(env)
      setattr(env, cls._ATTR, ep)
    return ep

  def base_proj(self):
    pos = _robot(self._env).data.root_link_pos_w[:, None, :2]
    return self.geo.project(pos)

  def feet_proj(self):
    feet = _robot(self._env).data.site_pos_w[:, self._foot_sites, :2]
    return self.geo.project(feet)

  def along_velocity(self) -> tuple[torch.Tensor, torch.Tensor]:
    """Base velocity along and across the local beam direction (m/s)."""
    yaw = self.base_proj().tangent_yaw[:, 0]
    v = _robot(self._env).data.root_link_lin_vel_w[:, :2]
    c, s = torch.cos(yaw), torch.sin(yaw)
    return c * v[:, 0] + s * v[:, 1], -s * v[:, 0] + c * v[:, 1]

  def update(self) -> None:
    env = self._env
    if env.common_step_counter == self._stamp:
      return
    self._stamp = env.common_step_counter
    arc = self.base_proj().arc[:, 0]
    self.progress = arc - self.spawn_arc
    self.max_progress = torch.maximum(self.max_progress, self.progress)
    cmd = _forward_cmd(env)
    self.cmd_dist += cmd * env.step_dt
    v_along, _ = self.along_velocity()
    self.vel_err_sum += (cmd - v_along).abs()
    self.steps += 1
    beam_end = self.geo.beam_arc()[:, 1]
    feet_arc = self.feet_proj().arc
    reached = (feet_arc.min(dim=1).values > beam_end + 0.02) & (arc > beam_end + 0.15)
    self.new_reach = reached & ~self.reached_end
    self.reached_end |= reached

  def stats(self, env_ids: torch.Tensor) -> EpisodeStats:
    env = self._env
    terminated = env.termination_manager.terminated[env_ids]
    cmd = self.cmd_dist[env_ids]
    prog = self.max_progress[env_ids]
    tracked = (cmd < WALK_MIN_DIST) | (prog >= TRACK_FRAC * cmd)
    reached = self.reached_end[env_ids]
    success = ~terminated & (reached | tracked)
    return EpisodeStats(
      success=success,
      progress=prog.clamp(min=0.0),
      cmd_dist=cmd,
      reached_end=reached,
      vel_err=self.vel_err_sum[env_ids] / self.steps[env_ids].clamp(min=1),
      level=self.geo.levels[env_ids],
      terminated=terminated,
    )

  def on_spawn(self, env_ids: torch.Tensor, spawn_arc: torch.Tensor, platform: torch.Tensor) -> None:
    self.spawn_arc[env_ids] = spawn_arc
    self.platform_mode[env_ids] = platform
    for t in (self.progress, self.max_progress, self.cmd_dist, self.vel_err_sum, self.steps):
      t[env_ids] = 0.0
    self.reached_end[env_ids] = False
    self.new_reach[env_ids] = False


def _sigma_for_level(env: ManagerBasedRlEnv, sigmas: tuple[float, ...]) -> torch.Tensor:
  geo = BeamGeometry.get(env)
  table = torch.tensor(sigmas, device=env.device)
  return table[geo.levels.clamp(max=len(sigmas) - 1)]


def _feet_in_contact(env: ManagerBasedRlEnv) -> torch.Tensor:
  found = env.scene[FOOT_CONTACT_SENSOR].data.found
  return found.view(env.num_envs, 2) > 0


def _foot_yaw(env: ManagerBasedRlEnv, site_ids) -> torch.Tensor:
  q = _robot(env).data.site_quat_w[:, site_ids]  # [B, 2, 4] wxyz
  w, x, y, z = q.unbind(-1)
  return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _wrap(a: torch.Tensor) -> torch.Tensor:
  return torch.atan2(torch.sin(a), torch.cos(a))


def whole_body_com_xy(env: ManagerBasedRlEnv) -> torch.Tensor:
  robot = _robot(env)
  root = robot.indexing.root_body_id
  return robot.data.data.subtree_com[:, root, :2]


# ---------------------------------------------------------------------------
# Command: forward speed only, held at zero once the end platform is reached
# ---------------------------------------------------------------------------


class BeamForwardCommand(UniformVelocityCommand):
  cfg: BeamForwardCommandCfg

  def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
    super()._update_command(env_ids)
    ep = getattr(self._env, BeamEpisode._ATTR, None)
    if ep is not None:
      done = ep.reached_end
      self.vel_command_b[done] = 0.0


@dataclass(kw_only=True)
class BeamForwardCommandCfg(UniformVelocityCommandCfg):
  def build(self, env: ManagerBasedRlEnv) -> BeamForwardCommand:
    return BeamForwardCommand(self, env)


def command_vx(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Observation: the commanded forward speed only (vy and yaw rate are 0)."""
  return _forward_cmd(env)[:, None]


# ---------------------------------------------------------------------------
# Action latency
# ---------------------------------------------------------------------------


class DelayedJointPositionAction(JointPositionAction):
  """Joint position targets that arrive ``lag`` physics substeps late.

  Each env draws a lag in [0, max_lag_substeps] at reset; during the first
  ``lag`` substeps of a policy step the previous target is still applied. At
  200 Hz physics, 4 substeps = 20 ms.
  """

  cfg: DelayedJointPositionActionCfg

  def __init__(self, cfg: DelayedJointPositionActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self._prev_target = self._entity.data.default_joint_pos[:, self._target_ids].clone()
    self._new_target = self._prev_target.clone()
    self.lag = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
    self._substep = 0

  def process_actions(self, actions: torch.Tensor) -> None:
    self._prev_target = self._new_target.clone()
    super().process_actions(actions)
    self._new_target = self._processed_actions.clone()
    self._substep = 0

  def apply_actions(self) -> None:
    use_prev = (self._substep < self.lag)[:, None]
    self._processed_actions = torch.where(use_prev, self._prev_target, self._new_target)
    self._substep += 1
    super().apply_actions()

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    super().reset(env_ids)
    ids = slice(None) if env_ids is None else env_ids
    default = self._entity.data.default_joint_pos[:, self._target_ids]
    self._prev_target[ids] = default[ids]
    self._new_target[ids] = default[ids]
    n = self.lag[ids].numel()
    self.lag[ids] = torch.randint(
      0, self.cfg.max_lag_substeps + 1, (n,), device=self.device
    )


@dataclass(kw_only=True)
class DelayedJointPositionActionCfg(JointPositionActionCfg):
  max_lag_substeps: int = 4

  def build(self, env: ManagerBasedRlEnv) -> DelayedJointPositionAction:
    return DelayedJointPositionAction(self, env)


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------


def tactile_features_noisy(env: ManagerBasedRlEnv) -> torch.Tensor:
  """The student's tactile input: canonical features of the faulted sensor."""
  layout = LAYOUTS[TACTILE.layout]
  normal = tac.canonical_normals(env, TACTILE)
  noisy = tac.TactileNoiseState.get(env, layout).apply(normal)
  return tac.features_from_normals(noisy, layout)


def tactile_features_clean(env: ManagerBasedRlEnv) -> torch.Tensor:
  layout = LAYOUTS[TACTILE.layout]
  return tac.features_from_normals(tac.canonical_normals(env, TACTILE), layout)


def tactile_map_noisy(env: ManagerBasedRlEnv) -> torch.Tensor:
  """End-to-end actor's tactile input: faulted sensor, per-taxel contact map
  instead of the load shape (``tactile.features_from_normals``)."""
  layout = LAYOUTS[TACTILE.layout]
  normal = tac.canonical_normals(env, TACTILE)
  noisy = tac.TactileNoiseState.get(env, layout).apply(normal)
  return tac.features_from_normals(noisy, layout, contact_map=True)


def tactile_map_clean(env: ManagerBasedRlEnv) -> torch.Tensor:
  layout = LAYOUTS[TACTILE.layout]
  return tac.features_from_normals(tac.canonical_normals(env, TACTILE), layout, contact_map=True)


def feet_beam_pose(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Per foot: lateral offset from the centreline, edge margin (half width minus
  |offset|), height above the top surface, yaw error vs. the beam (sin, cos).
  [B, 10]."""
  ep = BeamEpisode.get(env)
  geo = ep.geo
  proj = ep.feet_proj()
  feet_z = _robot(env).data.site_pos_w[:, ep._foot_sites, 2]
  half = geo.support_half_width(proj.arc)
  yaw_err = _wrap(_foot_yaw(env, ep._foot_sites) - proj.tangent_yaw)
  per_foot = torch.stack(
    [
      proj.lateral,
      half - proj.lateral.abs(),
      feet_z - geo.top_z()[:, None],
      torch.sin(yaw_err),
      torch.cos(yaw_err),
    ],
    dim=-1,
  )
  return per_foot.flatten(start_dim=1)


def base_beam_pose(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Base lateral offset, heading error (sin, cos), height above the top, CoM
  lateral offset. [B, 5]."""
  ep = BeamEpisode.get(env)
  geo = ep.geo
  proj = ep.base_proj()
  robot = _robot(env)
  err = _wrap(robot.data.heading_w - proj.tangent_yaw[:, 0])
  com = geo.project(whole_body_com_xy(env)[:, None])
  return torch.stack(
    [
      proj.lateral[:, 0],
      torch.sin(err),
      torch.cos(err),
      robot.data.root_link_pos_w[:, 2] - geo.top_z(),
      com.lateral[:, 0],
    ],
    dim=-1,
  )


def beam_progress_obs(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Distance from the base to the beam start and to the beam end along the
  path, clipped to +/- 3 m, and whether the end platform was reached. [B, 3]."""
  ep = BeamEpisode.get(env)
  arc = ep.base_proj().arc[:, 0]
  beam = ep.geo.beam_arc()
  return torch.stack(
    [
      (arc - beam[:, 0]).clamp(-3, 3),
      (beam[:, 1] - arc).clamp(-3, 3),
      ep.reached_end.float(),
    ],
    dim=-1,
  )


def beam_params_obs(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Beam width, beam height above the ground, contact friction. [B, 3].

  Contact friction is the foot geoms' friction: they carry contact priority 1,
  so MuJoCo uses their friction for every foot-terrain contact."""
  geo = BeamGeometry.get(env)
  return torch.stack([geo.width(), geo.top_z() - _ground_z(env), _foot_friction(env)], -1)


def _ground_z(env: ManagerBasedRlEnv) -> torch.Tensor:
  return torch.zeros(env.num_envs, device=env.device)


def _foot_friction(env: ManagerBasedRlEnv) -> torch.Tensor:
  robot = _robot(env)
  attr = "_sendg1_beam_ts_foot_geom"
  gid = getattr(env, attr, None)
  if gid is None:
    ids, _ = robot.find_geoms(G1_FOOT_GEOMS[:1])
    gid = int(robot.indexing.geom_ids[ids[0]])
    setattr(env, attr, gid)
  fr = env.sim.model.geom_friction
  if fr.dim() == 3:
    return fr[:, gid, 0]
  return fr[gid, 0].expand(env.num_envs)


def action_lag_obs(env: ManagerBasedRlEnv) -> torch.Tensor:
  term = env.action_manager.get_term("joint_pos")
  lag = getattr(term, "lag", None)
  if lag is None:
    return torch.zeros(env.num_envs, 1, device=env.device)
  return lag.float()[:, None] / 4.0


# ---------------------------------------------------------------------------
# Touch-revealed beam observations (probing teacher)
# ---------------------------------------------------------------------------
#
# The plain teacher sees where the beam is from the first step on, so it never
# needs to look for it, and DAgger can only copy what the teacher does. The
# probing teacher's actor gets the beam geometry only once touch could have
# told it: from the first step on which a loaded foot has a beam edge under its
# taxels (the tactile pattern then shows the edge). From then on the beam stays
# revealed for the episode -- it is straight, so one felt edge plus
# proprioception fixes it. On a wide beam an edge is felt only when a foot is
# placed near it; at 0.05-0.07 m every touchdown on the beam feels both edges.
# Before that the robot has to find the beam with its feet. The critic keeps
# the full ground truth.

SOLE_HALF_WIDTH = 0.035
"""Lateral half extent of the slipper's taxel grid (slipper_v1): a beam edge
closer than this to the foot's centre lies under the taxels."""


def edge_felt(env: ManagerBasedRlEnv) -> torch.Tensor:
  """[B, 2] bool: the foot is loaded and a beam edge lies under its taxels."""
  ep = BeamEpisode.get(env)
  proj = ep.feet_proj()
  margin = ep.geo.support_half_width(proj.arc) - proj.lateral.abs()
  return _feet_in_contact(env) & (margin < SOLE_HALF_WIDTH) & (margin > -SOLE_HALF_WIDTH)


def beam_revealed(env: ManagerBasedRlEnv) -> torch.Tensor:
  """[B] bool: a loaded foot has felt a beam edge this episode. Idempotent
  within a step, so every term may call it."""
  rev = getattr(env, "_beam_revealed", None)
  if rev is None:
    rev = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
  rev = rev & (env.episode_length_buf > 0)
  rev = rev | edge_felt(env).any(dim=-1)
  env._beam_revealed = rev  # type: ignore[attr-defined]
  return rev


def beam_revealed_obs(env: ManagerBasedRlEnv) -> torch.Tensor:
  return beam_revealed(env).float()[:, None]


def _touch_revealed(func):
  def term(env: ManagerBasedRlEnv) -> torch.Tensor:
    return func(env) * beam_revealed(env).float()[:, None]

  term.__name__ = f"{func.__name__}_touch"
  term.__doc__ = f"``{func.__name__}``, zero until the beam is revealed by touch."
  return term


feet_beam_pose_touch = _touch_revealed(feet_beam_pose)
base_beam_pose_touch = _touch_revealed(base_beam_pose)
beam_progress_touch = _touch_revealed(beam_progress_obs)
beam_params_touch = _touch_revealed(beam_params_obs)


# The ``*_touch`` poses above reveal the whole beam-relative pose, live, after
# the first felt edge. A teacher trained on them balanced on the CoM's lateral
# offset from the centreline -- which no onboard sensor measures -- and could
# not be distilled (the student fell within 0.2 s). The ``*_felt`` versions
# keep only what the tactile image shows: a foot's lateral offset and edge
# margin while that foot is loaded with an edge under its taxels. Base and CoM
# lateral offsets are always zero; heading and heights once revealed. Same
# shapes as the originals, so a ``*_touch`` teacher can be fine-tuned on them.


def feet_beam_pose_felt(env: ManagerBasedRlEnv) -> torch.Tensor:
  pose = feet_beam_pose(env).view(env.num_envs, 2, 5)
  felt = edge_felt(env).float()[..., None]
  rev = beam_revealed(env).float()[:, None, None]
  pose = torch.cat([pose[..., :2] * felt, pose[..., 2:] * rev], dim=-1)
  return pose.flatten(start_dim=1)


def base_beam_pose_felt(env: ManagerBasedRlEnv) -> torch.Tensor:
  pose = base_beam_pose(env) * beam_revealed(env).float()[:, None]
  keep = torch.tensor([0.0, 1.0, 1.0, 1.0, 0.0], device=env.device)
  return pose * keep


# ---------------------------------------------------------------------------
# Rewards
# ---------------------------------------------------------------------------


def track_forward_velocity(env: ManagerBasedRlEnv, std: float = 0.1) -> torch.Tensor:
  """exp(-(v_cmd - v_along)^2 / std^2), v_along along the local beam direction."""
  ep = BeamEpisode.get(env)
  ep.update()
  v_along, _ = ep.along_velocity()
  return torch.exp(-torch.square(_forward_cmd(env) - v_along) / std**2)


def alive(env: ManagerBasedRlEnv) -> torch.Tensor:
  return torch.ones(env.num_envs, device=env.device)


def stance_foot_centreline(
  env: ManagerBasedRlEnv, sigmas: tuple[float, ...] = CENTRELINE_SIGMA
) -> torch.Tensor:
  """Mean over feet in contact of exp(-d^2 / sigma^2), d the foot's lateral
  offset from the centreline and sigma the level's. Zero in flight."""
  ep = BeamEpisode.get(env)
  d = ep.feet_proj().lateral
  sigma = _sigma_for_level(env, sigmas)[:, None]
  contact = _feet_in_contact(env).float()
  r = torch.exp(-torch.square(d / sigma)) * contact
  return r.sum(-1) / contact.sum(-1).clamp(min=1.0)


def stance_foot_yaw(env: ManagerBasedRlEnv, std: float = 0.25) -> torch.Tensor:
  """Mean over feet in contact of exp(-yaw_err^2 / std^2) against the beam."""
  ep = BeamEpisode.get(env)
  err = _wrap(_foot_yaw(env, ep._foot_sites) - ep.feet_proj().tangent_yaw)
  contact = _feet_in_contact(env).float()
  r = torch.exp(-torch.square(err / std)) * contact
  return r.sum(-1) / contact.sum(-1).clamp(min=1.0)


def cop_centred(env: ManagerBasedRlEnv, std: float = 0.015) -> torch.Tensor:
  """Mean over loaded feet of exp(-off^2 / std^2), off the lateral distance of
  the (true) centre of pressure from the centroid of the sole area that has
  support beneath it. Rewards loading the part of the foot that is on the beam,
  not its overhang."""
  off = tac.cop_offset_from_support(env, TACTILE)
  contact = _feet_in_contact(env).float()
  r = torch.exp(-torch.square(off / std)) * contact
  return r.sum(-1) / contact.sum(-1).clamp(min=1.0)


def com_over_beam(env: ManagerBasedRlEnv, sigmas: tuple[float, ...] = CENTRELINE_SIGMA) -> torch.Tensor:
  """exp(-d^2 / sigma^2), d the lateral offset of the whole-body CoM's ground
  projection from the centreline."""
  geo = BeamGeometry.get(env)
  d = geo.project(whole_body_com_xy(env)[:, None]).lateral[:, 0]
  sigma = _sigma_for_level(env, sigmas)
  return torch.exp(-torch.square(d / sigma))


def base_lateral_velocity(env: ManagerBasedRlEnv) -> torch.Tensor:
  _, v_lat = BeamEpisode.get(env).along_velocity()
  return torch.square(v_lat)


def base_roll(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Squared sideways component of gravity in the base frame (~ roll^2) plus
  the squared roll rate scaled down."""
  robot = _robot(env)
  g = robot.data.projected_gravity_b
  wx = robot.data.root_link_ang_vel_b[:, 0]
  return torch.square(g[:, 1]) + 0.05 * torch.square(wx)


def passing_step(
  env: ManagerBasedRlEnv,
  scale: float = 0.5,
  min_air_time: float = 0.1,
  min_advance: float = 0.02,
  speed_std: float = 0.15,
  speed_ema: float = 0.04,
  command_threshold: float = 0.05,
) -> torch.Tensor:
  """At each ALTERNATING touchdown (the other foot made the previous one):
  clip(lead / scale, -1, 1), lead the landing foot's arc minus the other
  foot's. A step-to shuffle (one foot always leading, the other closing up
  behind it) scores +1 and -1 per cycle, i.e. nothing; an alternating gait
  scores on every step. Re-planting the same foot scores nothing (without the
  alternation gate the first fine-tune tapped the lead foot to farm +1). Only
  when walking is commanded. Divided by step_dt so the weight is the reward per
  touchdown.

  ``scale`` must exceed the shuffle's foot gap (~0.3 m): at 0.25 m the trailing
  foot's landing saturated at -1, so landing it further forward paid nothing
  and the second run kept shuffling.

  The positive part is capped by the new ground the step gains: min(lead,
  arc - furthest touchdown of the episode so far). Without it the v3 run
  learned to scissor its feet back and forth on the start platform and collect
  the reward without moving. With it the positive total is bounded by the
  distance walked / ``scale``; a shuffle's trailing-foot penalty cancels its
  gains, an alternating gait keeps them. The positive part is also scaled by
  how well the (smoothed) speed matches the command, else paying per metre
  teaches rushing."""
  data = env.scene[FOOT_CONTACT_SENSOR].data
  assert data.current_contact_time is not None and data.last_air_time is not None
  first = (data.current_contact_time > 0) & (data.current_contact_time <= env.step_dt + 1e-6)
  first &= data.last_air_time > min_air_time
  last = getattr(env, "_beam_last_touchdown", None)
  if last is None:
    last = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)
    env._beam_last_touchdown = last
  last[env.episode_length_buf <= 1] = -1
  foot = torch.arange(2, device=env.device)[None, :]
  alternating = first & (last[:, None] != foot)
  arc = BeamEpisode.get(env).feet_proj().arc
  frontier = getattr(env, "_beam_step_frontier", None)
  if frontier is None:
    frontier = arc.max(-1).values.clone()
    env._beam_step_frontier = frontier
  fresh = env.episode_length_buf <= 1
  frontier[fresh] = arc[fresh].max(-1).values
  lead = torch.stack([arc[:, 0] - arc[:, 1], arc[:, 1] - arc[:, 0]], dim=-1)
  gained = (arc - frontier[:, None]).clamp(min=0.0)
  gained = torch.where(gained > min_advance, gained, torch.zeros_like(gained))
  r = torch.where(lead > 0, torch.minimum(lead, gained), lead)
  r = torch.clamp(r / scale, -1.0, 1.0)
  # Gate the positive part on the speed matching the command, smoothed over
  # ~0.5 s: paying per metre otherwise teaches rushing (1.3 m/s at 0.3 m/s).
  ema = getattr(env, "_beam_speed_ema", None)
  if ema is None:
    ema = torch.zeros(env.num_envs, device=env.device)
    env._beam_speed_ema = ema
  ema[fresh] = 0.0
  v_along, _ = BeamEpisode.get(env).along_velocity()
  ema.mul_(1.0 - speed_ema).add_(speed_ema * v_along)
  match = torch.exp(-torch.square(_forward_cmd(env) - ema) / speed_std**2)
  r = torch.where(r > 0, r * match[:, None], r)
  r = (r * alternating.float()).sum(-1)
  landed = torch.where(first, arc, torch.full_like(arc, -1e9)).max(-1).values
  frontier[:] = torch.maximum(frontier, landed)
  # Latest touchdown wins; if both feet land on the same step, keep it unset.
  both = first.all(-1)
  last[:] = torch.where(both, -1, torch.where(first[:, 0], 0, torch.where(first[:, 1], 1, last)))
  cmd = env.command_manager.get_command("twist")[:, 0]
  return r * (cmd > command_threshold).float() / env.step_dt


def stance_foot_flat(env: ManagerBasedRlEnv, std: float = 0.15) -> torch.Tensor:
  """Mean over feet in contact of exp(-tilt^2 / std^2), tilt ~ |xy of gravity
  in the sole frame| (rad). A flat sole puts load on many taxels instead of one
  edge, which is what the tactile features need to say anything."""
  q = _robot(env).data.site_quat_w[:, BeamEpisode.get(env)._foot_sites]
  w, x, y, z = q.unbind(-1)
  # Rows 3 (z) of the rotation matrix = world z expressed in the sole frame.
  gx = 2 * (x * z - w * y)
  gy = 2 * (y * z + w * x)
  tilt2 = gx * gx + gy * gy
  contact = _feet_in_contact(env).float()
  r = torch.exp(-tilt2 / std**2) * contact
  return r.sum(-1) / contact.sum(-1).clamp(min=1.0)


def swing_foot_speed(env: ManagerBasedRlEnv, max_speed: float = 1.2) -> torch.Tensor:
  """Sum over feet in the air of relu(|v_foot| - max_speed)^2. At 0.3 m/s a
  human swing foot peaks near 1 m/s; the lopsided teacher whipped its right
  foot through at ~4 m/s right after each left touchdown."""
  robot = _robot(env)
  ids = getattr(env, "_beam_foot_body_ids", None)
  if ids is None:
    ids, _ = robot.find_bodies(("left_ankle_roll_link", "right_ankle_roll_link"), preserve_order=True)
    env._beam_foot_body_ids = ids
  speed = robot.data.body_link_lin_vel_w[:, ids].norm(dim=-1)
  air = ~_feet_in_contact(env)
  return (torch.relu(speed - max_speed).square() * air.float()).sum(-1)


def touchdown_force_rate(env: ManagerBasedRlEnv, body_weight_n: float = 343.0) -> torch.Tensor:
  """Sum over feet of (relu(rise of the vertical contact force in one step) /
  body weight)^2. Penalises slamming a foot down, i.e. favours a light first
  touch with weight shifted on gradually -- the physical shape of probing, and
  cleaner tactile readings without impact spikes."""
  force = env.scene[FOOT_CONTACT_SENSOR].data.force
  assert force is not None
  fz = force.view(env.num_envs, 2, 3)[..., 2].abs()
  prev = getattr(env, "_beam_prev_fz", None)
  if prev is None:
    prev = fz.clone()
    env._beam_prev_fz = prev
  prev[env.episode_length_buf <= 1] = 0.0
  rise = torch.relu(fz - prev) / body_weight_n
  prev.copy_(fz)
  return rise.square().sum(-1)


def track_forward_progress(env: ManagerBasedRlEnv, std: float = 0.1, overshoot: float = 2.0) -> torch.Tensor:
  """Forward progress with slack, for the end-to-end policy. Walking command:
  clip(v / v_cmd, -1, 1) - overshoot * relu(v / v_cmd - 1), so moving at half
  the command earns half (exp tracking pays ~0.1) and a pause to feel for the
  beam only forgoes reward. The lower clip is symmetric: clipping at 0 paid
  for rocking in place (backward sway cost nothing), and the first full run
  stood still and swayed. Standing command: exp(-v^2 / std^2)."""
  ep = BeamEpisode.get(env)
  ep.update()
  v, _ = ep.along_velocity()
  cmd = _forward_cmd(env)
  walking = cmd > 0.05
  ratio = v / cmd.clamp(min=0.05)
  walk = ratio.clamp(-1.0, 1.0) - overshoot * torch.relu(ratio - 1.0)
  stand = torch.exp(-torch.square(v) / std**2)
  return torch.where(walking, walk, stand)


def load_off_support(env: ManagerBasedRlEnv, body_weight_n: float = 343.0) -> torch.Tensor:
  """Sum over feet of (vertical load / body weight) x (share of the sole's
  taxels with no support beneath them). Weight on a foot that hangs half off
  the beam costs; a light touch does not. The cheap strategy it leaves is the
  probing one: touch, feel, then commit weight."""
  reading = read_tactile(env, TACTILE)
  unsupported = 1.0 - reading.support.mean(-1)  # [B, 2]
  load = reading.net_force[..., 2].abs() / body_weight_n
  return (load * unsupported).sum(-1)


def early_contact_overload(
  env: ManagerBasedRlEnv,
  window_s: float = 0.1,
  max_load: float = 0.3,
  body_weight_n: float = 343.0,
  settle_s: float = 0.5,
) -> torch.Tensor:
  """Sum over feet of relu(load / body weight - max_load)^2 during the first
  ``window_s`` of each contact: the first touch of a step should be light (it
  is the probe), weight goes on afterwards. A penalty, so tapping cannot farm it.
  Off for the first ``settle_s`` of an episode, while the robot lands from spawn."""
  contact = _feet_in_contact(env)
  t = getattr(env, "_beam_contact_time", None)
  if t is None:
    t = torch.zeros(env.num_envs, 2, device=env.device)
    env._beam_contact_time = t
  t.mul_(contact.float()).add_(contact.float() * env.step_dt)
  t[env.episode_length_buf <= 1] = 0.0
  force = env.scene[FOOT_CONTACT_SENSOR].data.force
  assert force is not None
  load = force.view(env.num_envs, 2, 3)[..., 2].abs() / body_weight_n
  early = contact & (t <= window_s) & (env.episode_length_buf[:, None] * env.step_dt > settle_s)
  return (torch.relu(load - max_load).square() * early.float()).sum(-1)


class pose_l2:
  """Mean over the selected joints of ((q - q_default) / std)^2: a quadratic
  pull toward the default pose. Unlike mjlab's exp-kernel ``posture`` it does
  not flatten out far from the target (the lopsided teacher's wrists sat ~1 rad
  off, where the exp kernel gave no gradient at all)."""

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    from mjlab.utils.lab_api.string import resolve_matching_names_values

    asset = env.scene[cfg.params["asset_cfg"].name]
    _, names = asset.find_joints(cfg.params["asset_cfg"].joint_names)
    _, _, std = resolve_matching_names_values(data=cfg.params["std"], list_of_strings=names)
    self.std = torch.tensor(std, device=env.device, dtype=torch.float32)

  def __call__(self, env: ManagerBasedRlEnv, std, asset_cfg) -> torch.Tensor:
    del std
    asset = env.scene[asset_cfg.name]
    err = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.mean(torch.square(err / self.std), dim=1)


def reach_end_bonus(env: ManagerBasedRlEnv) -> torch.Tensor:
  """One-off, on the step the end platform is reached. Divided by step_dt so
  the weight is the bonus itself."""
  ep = BeamEpisode.get(env)
  ep.update()
  return ep.new_reach.float() / env.step_dt


# ---------------------------------------------------------------------------
# Terminations
# ---------------------------------------------------------------------------


def foot_off_support(env: ManagerBasedRlEnv, tol: float = FOOT_BELOW_TOP_TOL) -> torch.Tensor:
  """A sole touches the ground or a side face of the beam/platform: any
  sole-terrain contact more than ``tol`` below the top surface."""
  data = env.scene[SOLE_CONTACT_SENSOR].data
  assert data.pos is not None and data.normal is not None
  b = env.num_envs
  pos = data.pos.view(b, -1, 3)
  valid = data.normal.view(b, -1, 3).norm(dim=-1) > 1e-8
  low = pos[..., 2] < BeamGeometry.get(env).top_z()[:, None] - tol
  return (low & valid).any(dim=-1)


def base_too_low(env: ManagerBasedRlEnv, min_height: float = 0.5) -> torch.Tensor:
  z = _robot(env).data.root_link_pos_w[:, 2]
  return z < BeamGeometry.get(env).top_z() + min_height


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


def _point_at_arc(path_xy: torch.Tensor, arc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
  """xy [B, 2] and tangent yaw [B] at arc length ``arc`` [B] along ``path_xy``."""
  seg = path_xy[:, 1:] - path_xy[:, :-1]
  seg_len = seg.norm(dim=-1).clamp_min(1e-9)
  cum = torch.cumsum(seg_len, dim=1)
  idx = torch.searchsorted(cum, arc[:, None]).clamp(max=seg.shape[1] - 1)[:, 0]
  rows = torch.arange(path_xy.shape[0], device=path_xy.device)
  start = cum[rows, idx] - seg_len[rows, idx]
  t = ((arc - start) / seg_len[rows, idx]).clamp(0, 1)
  xy = path_xy[rows, idx] + t[:, None] * seg[rows, idx]
  yaw = torch.atan2(seg[rows, idx, 1], seg[rows, idx, 0])
  return xy, yaw


def spawn_on_beam(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  beam_arc_range: tuple[float, float] = (0.3, 1.0),
  lateral_range: tuple[float, float] = (-0.02, 0.02),
  platform_arc_range: tuple[float, float] = (-0.1, 0.1),
  platform_lateral_range: tuple[float, float] = (-0.08, 0.08),
  yaw_range: tuple[float, float] = (-0.08, 0.08),
  platform_from_level: int = PLATFORM_FROM_LEVEL,
) -> None:
  """Reset event: place the robot on the beam (low levels) or on the start
  platform (from ``platform_from_level``), facing along the beam."""
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  robot = _robot(env)
  geo = BeamGeometry.get(env)
  ep = BeamEpisode.get(env)
  n = len(env_ids)
  dev = env.device
  u = lambda lo_hi: torch.empty(n, device=dev).uniform_(*lo_hi)  # noqa: E731

  levels = geo.levels[env_ids]
  platform = levels >= platform_from_level
  path = geo.path()[env_ids]
  beam = geo.beam_arc()[env_ids]
  arc_beam = beam[:, 0] + u(beam_arc_range)
  arc_plat = 0.5 * geo.platform_length + u(platform_arc_range)
  arc = torch.where(platform, arc_plat, arc_beam)
  lateral = torch.where(platform, u(platform_lateral_range), u(lateral_range))
  xy, yaw = _point_at_arc(path[..., :2], arc)
  nrm = torch.stack([-torch.sin(yaw), torch.cos(yaw)], dim=-1)
  xy = xy + lateral[:, None] * nrm
  yaw = yaw + u(yaw_range)

  default = robot.data.default_root_state[env_ids]
  z = path[:, 0, 2] + default[:, 2] + 0.01
  pos = torch.cat([xy, z[:, None]], dim=-1)
  zeros = torch.zeros_like(yaw)
  quat = quat_from_euler_xyz(zeros, zeros, yaw)
  robot.write_root_link_pose_to_sim(torch.cat([pos, quat], dim=-1), env_ids=env_ids)
  robot.write_root_link_velocity_to_sim(torch.zeros(n, 6, device=dev), env_ids=env_ids)
  ep.on_spawn(env_ids, arc, platform)


def register_actor_mirror(env: ManagerBasedRlEnv, env_ids: torch.Tensor | None) -> None:
  """Startup event: hand the actor group's mirror map to the symmetric GRU
  actor (``sendg1.rl.symmetric_rnn``), which only ever sees tensors."""
  from sendg1.common.beam_ts.symmetry import _maps
  from sendg1.rl import symmetric_rnn

  maps = _maps(env)
  symmetric_rnn.set_mirrors(obs={g: m for g, m in maps.items() if g != "__actions__"}, actions=maps["__actions__"])


def reset_tactile_noise(env: ManagerBasedRlEnv, env_ids: torch.Tensor | None) -> None:
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  geo = BeamGeometry.get(env)
  st = tac.TactileNoiseState.get(env, LAYOUTS[TACTILE.layout])
  st.resample(env_ids, tac.noise_strength(geo.levels[env_ids], geo.num_levels))


def push_by_level(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  max_velocity: dict[str, float],
  from_level: int = PUSH_FROM_LEVEL,
) -> None:
  """Interval event: add a random root velocity, zero below ``from_level`` and
  growing linearly to ``max_velocity`` at the top level."""
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  geo = BeamGeometry.get(env)
  lv = geo.levels[env_ids].float()
  top = geo.num_levels - 1
  scale = ((lv - from_level + 1) / max(top - from_level + 1, 1)).clamp(0, 1)
  scale = torch.where(lv >= from_level, scale, torch.zeros_like(scale))
  keys = ("x", "y", "z", "roll", "pitch", "yaw")
  mag = torch.tensor([max_velocity.get(k, 0.0) for k in keys], device=env.device)
  dv = (2 * torch.rand(len(env_ids), 6, device=env.device) - 1) * mag * scale[:, None]
  robot = _robot(env)
  vel = robot.data.root_link_vel_w[env_ids] + dv
  robot.write_root_link_velocity_to_sim(vel, env_ids=env_ids)


@requires_model_fields("geom_solref")
def foot_contact_softness(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  timeconst_range: tuple[float, float],
  asset_cfg: SceneEntityCfg,
) -> None:
  """Randomise the contact time constant of the foot geoms. The feet carry
  contact priority 1, so their solref governs every foot-terrain contact: this
  is the (small) compliance of the beam as the robot feels it."""
  from mjlab.envs.mdp.dr._core import _randomize_model_field

  _randomize_model_field(
    env,
    env_ids,
    "geom_solref",
    entity_type="geom",
    ranges=timeconst_range,
    operation="abs",
    asset_cfg=asset_cfg,
    axes=[0],
    shared_random=True,
    default_axes=[0],
    valid_axes=[0, 1],
  )


# ---------------------------------------------------------------------------
# Curriculum
# ---------------------------------------------------------------------------


def beam_levels(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  standing_share: tuple[float, float] = STANDING_SHARE,
  replay_share: float = REPLAY_SHARE,
  standing_promotes: bool = True,
) -> dict[str, torch.Tensor]:
  """Per-env width curriculum plus the coupled command mix. See module doc.
  ``standing_promotes=False``: surviving a standing command does not promote,
  so the level only rises on walking successes."""
  terrain = env.scene.terrain
  assert terrain is not None
  geo = BeamGeometry.get(env)
  ep = BeamEpisode.get(env)
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  out: dict[str, torch.Tensor] = {}

  home = getattr(env, "_beam_home_level", None)
  if home is None:
    home = geo.levels.clone()
    env._beam_home_level = home
    env._beam_replay = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  replay = env._beam_replay

  if env.common_step_counter > 0 and len(env_ids) > 0:
    st = ep.stats(env_ids)
    up = st.success
    if not standing_promotes:
      up = up & (st.cmd_dist >= WALK_MIN_DIST)
    up = up.long()
    down = st.terminated.long()
    moved = (home[env_ids] + up - down).clamp(0, geo.num_levels - 1)
    home[env_ids] = torch.where(replay[env_ids], home[env_ids], moved)

    out["success"] = st.success.float().mean()
    out["terminated"] = st.terminated.float().mean()
    out["progress_m"] = st.progress.mean()
    out["reached_end"] = st.reached_end.float().mean()
    out["vel_err"] = st.vel_err.mean()
    walking = st.cmd_dist >= WALK_MIN_DIST
    if walking.any():
      ratio = (st.progress[walking] / st.cmd_dist[walking]).clamp(0, 2)
      out["track_ratio_walking"] = ratio.mean()

  if len(env_ids) > 0:
    n = len(env_ids)
    rep = (torch.rand(n, device=env.device) < replay_share) & (home[env_ids] > 0)
    easier = (torch.rand(n, device=env.device) * (home[env_ids] + 1).float()).long()
    lv = torch.where(rep, easier.clamp(max=geo.num_levels - 1), home[env_ids])
    replay[env_ids] = rep & (lv < home[env_ids])
    terrain.terrain_levels[env_ids] = lv
    terrain.env_origins[env_ids] = terrain.terrain_origins[lv, terrain.terrain_types[env_ids]]

  levels = home.float()
  frac = levels.mean() / max(geo.num_levels - 1, 1)
  share = standing_share[0] + (standing_share[1] - standing_share[0]) * frac
  cmd = env.command_manager.get_term("twist")
  cmd.cfg.rel_standing_envs = float(share)
  out["mean"] = levels.mean()
  out["max"] = levels.max()
  out["standing_share"] = torch.tensor(float(share))
  for i in range(geo.num_levels):
    out[f"frac_L{i}"] = (home == i).float().mean()
  out["replay_share"] = replay.float().mean()
  return out



def set_levels(env: ManagerBasedRlEnv, levels: torch.Tensor) -> None:
  """Place every env on ``levels`` (home and current level), e.g. to start a
  run from a level distribution. Takes effect at the next reset."""
  terrain = env.scene.terrain
  assert terrain is not None
  terrain.terrain_levels[:] = levels
  terrain.env_origins[:] = terrain.terrain_origins[levels, terrain.terrain_types]
  env._beam_home_level = levels.clone()
  env._beam_replay = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
