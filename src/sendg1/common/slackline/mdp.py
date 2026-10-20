"""MDP terms for standing and walking on a slackline (``model``).

The robot starts in a tandem stance on the line between the anchors, front
foot pointing along it, back foot turned out. A forward-speed command asks
it either to stand (balance) or to walk along the line; the command drops
to zero once the robot is ``END_MARGIN`` short of the anchor ahead.

Frames (``LineState``):

- line frame   the env's line, x along it from mid-span, y to its left, z up.
- direction    each episode faces the robot one way along the line
               (``+1``: towards the ``hi`` anchor). Every along/lateral
               quantity is signed in the robot's sense: forward is forward,
               left is left, whichever way it faces.
- the line at x  webbing top centre at line coordinate x, interpolated
               between nodes (the anchors are fixed points of zero offset).

Curriculum (per env, :func:`line_levels`)
-----------------------------------------
``LEVELS`` runs from a stiff, wide plank to the real slackline. Each level
sets the webbing's width and two helper springs per piece, against the world:
one on its sideways and vertical motion (a shorter, tauter line: less sag,
less sway) and one on its roll (webbing that will not twist under a sole).
The tension itself cannot carry the curriculum: above ~2.4 kN the string's
springs outrun the 5 ms step. The top level has no helpers at all. Promote on
success (survived the episode and, walking, covered ``TRACK_FRAC`` of the
commanded distance or reached the end), demote on an early termination; a
share ``REPLAY_SHARE`` of resets replays an easier level without moving the
env's own one (the beam's lesson: without it the population forgets the easy
levels). The share of standing commands falls as the mean level rises.

The reward is laid out and argued in ``tasks/slackline/env_cfg.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.managers.event_manager import requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity.mdp.velocity_command import (
  UniformVelocityCommand,
  UniformVelocityCommandCfg,
)
from mjlab.utils.lab_api.math import quat_from_euler_xyz

from sendg1.common.beam import tactile as tac
from sendg1.common.beam.mdp import TACTILE, whole_body_com_xy
from sendg1.common.robots.g1 import G1_FOOT_SITES
from sendg1.common.slackline import model as sm
from sendg1.common.tactile.layout import LAYOUTS

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

LINE = "slackline"
FOOT_LINE_SENSOR = "feet_line_contact"
"""Net force of each foot on the webbing."""
FOOT_GROUND_SENSOR = "feet_ground_contact"
"""Each foot on the floor: an early termination."""
NONFOOT_LINE_SENSOR = "nonfoot_line_contact"
"""Any other body on the webbing (a knee, a hand grabbing it): a fall."""

STANCE_LENGTH = 0.28
"""Front sole centre to back sole centre along the line (m). With the back
foot turned 45 deg its toe still clears the front heel by ~10 cm."""
BACK_FOOT_YAW = math.radians(45.0)
"""The back foot turned out (toes away from the body's midline): the
slackliner's stance. Its arch, not its length, crosses the line."""
SETTLE_S = 0.3
"""The robot is spawned a few mm above the line; contact terms wait."""
END_MARGIN = 0.6
"""Command held at zero this far (m, base to anchor) short of the anchor ahead."""
SPAWN_ALONG = (-1.2, 0.3)
"""Spawn point along the line (m from mid-span, in the robot's direction):
from 0.8 m after the first anchor to just past mid-span, so a walking episode
has 1-2.5 m of line ahead before ``END_MARGIN``."""
TACTILE_NOISE_STRENGTH = 0.5
"""Sensor faults at half the beam task's top level, as on the cylinder and
the pedalo: no curriculum yet to grow them."""
BODY_WEIGHT_N = 343.0


@dataclass(frozen=True)
class LineLevel:
  width: float
  """Webbing width (m); the G1 sole is ~7 cm across."""
  stiffness: float
  """Helper spring per piece on its sideways and vertical motion (N/m)."""
  roll_stiffness: float
  """Helper spring per piece on its roll (N m/rad)."""


LEVELS: tuple[LineLevel, ...] = (
  LineLevel(0.20, 8000.0, 400.0),
  LineLevel(0.14, 8000.0, 400.0),
  LineLevel(0.10, 4000.0, 150.0),
  LineLevel(0.07, 2000.0, 50.0),
  LineLevel(0.05, 800.0, 15.0),
  LineLevel(0.05, 250.0, 4.0),
  LineLevel(0.05, 0.0, 0.0),
)
"""Easiest first. Level 0 is a 20 cm plank that sags ~2 cm and barely rolls:
a sole fits on it, the tandem stance can be learnt. Then the webbing narrows
to its real 5 cm under the sole (the beam task never got past 10 cm, so this
is the step to watch), and the helpers fade: from level 4 the line sways,
sags and rolls more each level, until level 6 is the free line. Helper limits
at the 5 ms step: ~12 kN/m and well over 800 N m/rad per piece."""
TOP_LEVEL = len(LEVELS) - 1
TRACK_FRAC = 0.7
WALK_MIN_DIST = 0.5
"""Commanded distance (m) below which an episode counts as standing."""
REPLAY_SHARE = 0.2
STANDING_SHARE = (0.7, 0.4)
"""Share of standing commands at mean level 0 and at the top level: balance
first, then more walking."""


def _robot(env: ManagerBasedRlEnv):
  return env.scene["robot"]


def _line(env: ManagerBasedRlEnv):
  return env.scene[LINE]


def _wrap(a: torch.Tensor) -> torch.Tensor:
  return torch.atan2(torch.sin(a), torch.cos(a))


def _settled(env: ManagerBasedRlEnv) -> torch.Tensor:
  return env.episode_length_buf * env.step_dt > SETTLE_S


def _forward_cmd(env: ManagerBasedRlEnv) -> torch.Tensor:
  return env.command_manager.get_command("twist")[:, 0]


# ---------------------------------------------------------------------------
# Sensors
# ---------------------------------------------------------------------------


def line_match() -> ContactMatch:
  return ContactMatch(mode="subtree", pattern=sm.ROOT_BODY, entity=LINE)


def feet_line_contact_sensor() -> ContactSensorCfg:
  """Net contact force per foot against the webbing."""
  return ContactSensorCfg(
    name=FOOT_LINE_SENSOR,
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
      entity="robot",
    ),
    secondary=line_match(),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )


def nonfoot_line_contact_sensor() -> ContactSensorCfg:
  """Any robot collision geom but the soles and the ankle-linkage braces on
  the webbing. The braces (capsules on the knee link, just above the heel)
  graze the webbing in a sound tandem stance, where it runs out behind the
  heel and rises away from the soles; a shin, knee or hand on it is a fall."""
  return ContactSensorCfg(
    name=NONFOOT_LINE_SENSOR,
    primary=ContactMatch(
      mode="geom",
      pattern=r".*_collision",
      entity="robot",
      exclude=(r".*_foot\d_collision", r".*_linkage_brace_collision"),
    ),
    secondary=line_match(),
    fields=("found",),
    reduce="maxforce",
    num_slots=1,
  )


# ---------------------------------------------------------------------------
# Line state
# ---------------------------------------------------------------------------


@dataclass
class LinePoint:
  """The webbing top centre at some line coordinates [B, ...]: offsets from
  the unloaded line (lateral ``y``, height ``z`` above the floor), roll and
  velocities. ``y`` and ``vy`` are in the line frame, unsigned."""

  y: torch.Tensor
  z: torch.Tensor
  roll: torch.Tensor
  vy: torch.Tensor
  vz: torch.Tensor


class LineState:
  """Index lookups, the line's per-env direction and the queries every term reads."""

  _ATTR = "_sendg1_slackline_state"

  def __init__(self, env: ManagerBasedRlEnv) -> None:
    self._env = env
    line = _line(env)
    n = sm.NUM_NODES
    self.y_ids = line.find_joints(tuple(sm.SLIDE_Y.format(i) for i in range(n)), preserve_order=True)[0]
    self.z_ids = line.find_joints(tuple(sm.SLIDE_Z.format(i) for i in range(n)), preserve_order=True)[0]
    self.roll_ids = line.find_joints(tuple(sm.ROLL.format(i) for i in range(n)), preserve_order=True)[0]
    self.pitch_ids = line.find_joints(tuple(sm.PITCH.format(i) for i in range(n)), preserve_order=True)[0]
    self.foot_sites, _ = _robot(env).find_sites(G1_FOOT_SITES, preserve_order=True)
    dev = env.device
    self.direction = torch.ones(env.num_envs, device=dev)
    """+1: the robot faces the ``hi`` anchor (line +x)."""
    self.front_left = torch.ones(env.num_envs, dtype=torch.bool, device=dev)
    """Which foot this episode's stance put in front."""
    self.spawn_along = torch.zeros(env.num_envs, device=dev)
    self.level = torch.zeros(env.num_envs, dtype=torch.long, device=dev)
    """The level this episode runs on (``LEVELS``)."""
    self.home = torch.zeros(env.num_envs, dtype=torch.long, device=dev)
    """Each env's own curriculum level; a replay episode runs below it."""
    self.replay = torch.zeros(env.num_envs, dtype=torch.bool, device=dev)
    # Per-episode bookkeeping for the curriculum (``update``).
    self.cmd_dist = torch.zeros(env.num_envs, device=dev)
    self.max_progress = torch.zeros(env.num_envs, device=dev)
    self.reached_end = torch.zeros(env.num_envs, dtype=torch.bool, device=dev)
    self._stamp = -1

  def update(self) -> None:
    """Once per env step: commanded distance, progress along the line from
    the spawn point, and whether the robot reached the end margin."""
    env = self._env
    if env.common_step_counter == self._stamp:
      return
    self._stamp = env.common_step_counter
    self.cmd_dist += _forward_cmd(env).clamp(min=0.0) * env.step_dt
    along = self.base()[:, 0]
    self.max_progress = torch.maximum(self.max_progress, along - self.spawn_along)
    self.reached_end |= along > sm.SPAN / 2 - END_MARGIN

  def on_spawn(self, env_ids: torch.Tensor) -> None:
    for t in (self.cmd_dist, self.max_progress):
      t[env_ids] = 0.0
    self.reached_end[env_ids] = False

  @classmethod
  def get(cls, env: ManagerBasedRlEnv) -> LineState:
    st = getattr(env, cls._ATTR, None)
    if st is None:
      st = cls(env)
      setattr(env, cls._ATTR, st)
    return st

  # -- the line ---------------------------------------------------------------

  def origin(self) -> torch.Tensor:
    return self._env.scene.env_origins

  def _knots(self, ids) -> tuple[torch.Tensor, torch.Tensor]:
    """[B, N + 2] joint positions and velocities with the anchors' zeros."""
    data = _line(self._env).data
    q, v = data.joint_pos[:, ids], data.joint_vel[:, ids]
    pad = torch.zeros_like(q[:, :1])
    return torch.cat([pad, q, pad], -1), torch.cat([pad, v, pad], -1)

  def at(self, x: torch.Tensor) -> LinePoint:
    """The line at line-frame x [B, K] (m from mid-span, line +x)."""
    # nan_to_num: a world that has blown up must reach ``nan_detection``, not
    # crash the whole batch on an out-of-range gather.
    s = torch.nan_to_num((x + sm.SPAN / 2) / sm.NODE_SPACING).clamp(0.0, sm.NUM_NODES + 1 - 1e-4)
    i0 = s.floor().long()
    t = s - i0
    i1 = i0 + 1

    def lerp(k: torch.Tensor) -> torch.Tensor:
      return (1 - t) * k.gather(1, i0) + t * k.gather(1, i1)

    qy, vy = self._knots(self.y_ids)
    qz, vz = self._knots(self.z_ids)
    qr, _ = self._knots(self.roll_ids)
    z = sm.HEIGHT + lerp(qz) + self.origin()[:, 2:3]
    return LinePoint(y=lerp(qy), z=z, roll=lerp(qr), vy=lerp(vy), vz=lerp(vz))

  def to_line(self, pos_w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """World xy [B, K, 2] -> line-frame (x, y) [B, K]."""
    d = pos_w - self.origin()[:, None, :2]
    return d[..., 0], d[..., 1]

  def heading(self) -> torch.Tensor:
    """The robot's forward direction along the line, world yaw."""
    return torch.where(self.direction > 0, 0.0, math.pi)

  # -- the robot on it --------------------------------------------------------

  def along(self, pos_w: torch.Tensor) -> torch.Tensor:
    """Distance along the line in the robot's direction, from mid-span."""
    x, _ = self.to_line(pos_w)
    return self.direction[:, None] * x

  def lateral(self, pos_w: torch.Tensor) -> torch.Tensor:
    """Offset to the robot's left of the (moving) line beneath [B, K]."""
    x, y = self.to_line(pos_w)
    return self.direction[:, None] * (y - self.at(x).y)

  def feet(self) -> torch.Tensor:
    """[B, 2, 3] each sole centre: along, lateral (robot frame, from the
    line under it), height above the webbing top."""
    p = _robot(self._env).data.site_pos_w[:, self.foot_sites]
    x, y = self.to_line(p[..., :2])
    pt = self.at(x)
    return torch.stack(
      [self.direction[:, None] * x, self.direction[:, None] * (y - pt.y), p[..., 2] - pt.z], -1
    )

  def feet_yaw(self) -> torch.Tensor:
    """[B, 2] each foot's yaw relative to the robot's line direction."""
    q = _robot(self._env).data.site_quat_w[:, self.foot_sites]
    w, x, y, z = q.unbind(-1)
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return _wrap(yaw - self.heading()[:, None])

  def back_is_left(self) -> torch.Tensor:
    """[B] the left foot is currently the rear one (along the line)."""
    a = self.feet()[..., 0]
    return a[:, 0] < a[:, 1]

  def base(self) -> torch.Tensor:
    """[B, 3] the base: along, lateral, height above the webbing beneath."""
    p = _robot(self._env).data.root_link_pos_w[:, None]
    x, y = self.to_line(p[..., :2])
    pt = self.at(x)
    d = self.direction[:, None]
    return torch.stack([d * x, d * (y - pt.y), p[..., 2] - pt.z], -1)[:, 0]

  def com_lateral(self) -> torch.Tensor:
    """[B] whole-body CoM ground projection to the left of the line beneath."""
    return self.lateral(whole_body_com_xy(self._env)[:, None])[:, 0]

  def base_velocity(self) -> tuple[torch.Tensor, torch.Tensor]:
    """Base velocity along and across the line, robot's sense (m/s)."""
    v = _robot(self._env).data.root_link_lin_vel_w[:, :2]
    return self.direction * v[:, 0], self.direction * v[:, 1]

  def line_under_base(self) -> LinePoint:
    x, _ = self.to_line(_robot(self._env).data.root_link_pos_w[:, None, :2])
    return self.at(x)

  def heading_error(self) -> torch.Tensor:
    return _wrap(_robot(self._env).data.heading_w - self.heading())

  def foot_slip(self) -> torch.Tensor:
    """[B, 2, 2] each sole's velocity relative to the webbing under it
    (along, lateral; robot's sense)."""
    robot = _robot(self._env)
    v = robot.data.site_lin_vel_w[:, self.foot_sites]
    x, _ = self.to_line(robot.data.site_pos_w[:, self.foot_sites, :2])
    pt = self.at(x)
    d = self.direction[:, None]
    return torch.stack([d * v[..., 0], d * (v[..., 1] - pt.vy)], -1)


def _feet_on_line(env: ManagerBasedRlEnv) -> torch.Tensor:
  found = env.scene[FOOT_LINE_SENSOR].data.found
  return found.view(env.num_envs, 2) > 0


def _feet_on_ground(env: ManagerBasedRlEnv) -> torch.Tensor:
  found = env.scene[FOOT_GROUND_SENSOR].data.found
  return found.view(env.num_envs, 2) > 0


def _foot_load(env: ManagerBasedRlEnv) -> torch.Tensor:
  """[B, 2] vertical force of each foot on the webbing (N)."""
  force = env.scene[FOOT_LINE_SENSOR].data.force
  assert force is not None
  return force.view(env.num_envs, 2, 3)[..., 2].abs()


def _standing(env: ManagerBasedRlEnv, threshold: float = 0.05) -> torch.Tensor:
  return _forward_cmd(env) <= threshold


# ---------------------------------------------------------------------------
# Command: forward speed along the line, held at zero near the anchor ahead
# ---------------------------------------------------------------------------


class SlacklineForwardCommand(UniformVelocityCommand):
  cfg: SlacklineForwardCommandCfg

  def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
    super()._update_command(env_ids)
    st = getattr(self._env, LineState._ATTR, None)
    if st is not None:
      near_end = st.base()[:, 0] > sm.SPAN / 2 - END_MARGIN
      self.vel_command_b[near_end] = 0.0


@dataclass(kw_only=True)
class SlacklineForwardCommandCfg(UniformVelocityCommandCfg):
  def build(self, env: ManagerBasedRlEnv) -> SlacklineForwardCommand:
    return SlacklineForwardCommand(self, env)


# ---------------------------------------------------------------------------
# Observations (privileged: simulation ground truth, critic only)
# ---------------------------------------------------------------------------


def line_under_robot(env: ManagerBasedRlEnv) -> torch.Tensor:
  """The webbing under the base and under each sole: lateral offset (robot's
  left) and sag below the anchors, roll, lateral and vertical velocity;
  then the base's distance to the anchor ahead and the line's tension
  (kN). [B, 17]"""
  st = LineState.get(env)
  robot = _robot(env)
  pts = torch.cat(
    [robot.data.root_link_pos_w[:, None, :2], robot.data.site_pos_w[:, st.foot_sites, :2]], 1
  )
  x, _ = st.to_line(pts)
  pt = st.at(x)
  d = st.direction[:, None]
  sag = pt.z - st.origin()[:, 2:3] - sm.HEIGHT
  per_point = torch.stack([d * pt.y, sag, d * pt.roll, d * pt.vy, pt.vz], -1).flatten(1)
  ahead = sm.SPAN / 2 - st.base()[:, 0:1]
  return torch.cat([per_point, ahead, line_tension(env)[:, None] / 1000.0], -1)


def line_level_obs(env: ManagerBasedRlEnv) -> torch.Tensor:
  """This episode's curriculum level: level / top, webbing width (m), the
  helper springs scaled to level 0's. [B, 4]"""
  st = LineState.get(env)
  table = _level_table(env)[st.level]
  base = _level_table(env)[0]
  return torch.cat(
    [(st.level.float() / max(TOP_LEVEL, 1))[:, None], table[:, :1], table[:, 1:] / base[1:]], -1
  )


def feet_line_pose(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Per foot: along (from the base), lateral, height above the webbing, and
  sin/cos of its yaw against the line. [B, 10]"""
  st = LineState.get(env)
  f = st.feet()
  f = torch.stack([f[..., 0] - st.base()[:, 0:1], f[..., 1], f[..., 2]], -1)
  yaw = st.feet_yaw()
  return torch.cat([f, torch.sin(yaw)[..., None], torch.cos(yaw)[..., None]], -1).flatten(1)


def base_line_pose(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Base lateral offset and height over the line, sin/cos heading error,
  CoM lateral offset. [B, 5]"""
  st = LineState.get(env)
  b = st.base()
  err = st.heading_error()
  return torch.stack([b[:, 1], b[:, 2], torch.sin(err), torch.cos(err), st.com_lateral()], -1)


def foot_slip_on_line(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Per foot: velocity relative to the webbing under it while on it. [B, 4]"""
  on = _feet_on_line(env).float()
  return (LineState.get(env).foot_slip() * on[..., None]).flatten(1)


def foot_line_forces(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Per foot: net force on the webbing (along, lateral, up; robot's sense),
  ``sign * log1p(|f|)``. [B, 6]"""
  st = LineState.get(env)
  force = env.scene[FOOT_LINE_SENSOR].data.force
  assert force is not None
  f = force.view(env.num_envs, 2, 3)
  d = st.direction[:, None]
  out = torch.stack([d * f[..., 0], d * f[..., 1], f[..., 2]], -1).flatten(1)
  return torch.sign(out) * torch.log1p(out.abs())


def line_tension(env: ManagerBasedRlEnv) -> torch.Tensor:
  """[B] this world's line tension (N), from the string tendons' stiffness."""
  tid = getattr(env, "_slackline_tendon_id", None)
  if tid is None:
    tid = env.sim.mj_model.tendon(f"{LINE}/seg_00_y").id
    env._slackline_tendon_id = tid
  k = env.sim.model.tendon_stiffness
  k = k[:, tid] if k.dim() == 2 else k[tid].expand(env.num_envs)
  return k * sm.NODE_SPACING


# ---------------------------------------------------------------------------
# Rewards
# ---------------------------------------------------------------------------


def alive(env: ManagerBasedRlEnv) -> torch.Tensor:
  return torch.ones(env.num_envs, device=env.device)


def track_forward_progress(
  env: ManagerBasedRlEnv, std: float = 0.1, overshoot: float = 2.0
) -> torch.Tensor:
  """The beam's: walking, clip(v / v_cmd, -1, 1) - overshoot * relu(v / v_cmd
  - 1), so a pause to rebalance only forgoes reward; standing,
  exp(-v^2 / std^2), v the base's speed along the line."""
  st = LineState.get(env)
  st.update()
  v, _ = st.base_velocity()
  cmd = _forward_cmd(env)
  ratio = v / cmd.clamp(min=0.05)
  walk = ratio.clamp(-1.0, 1.0) - overshoot * torch.relu(ratio - 1.0)
  stand = torch.exp(-torch.square(v) / std**2)
  return torch.where(_standing(env), stand, walk)


def com_over_line(env: ManagerBasedRlEnv, std: float = 0.04) -> torch.Tensor:
  """exp(-d^2 / std^2), d the CoM's lateral offset from the webbing beneath
  it. The line is the whole support sideways: the CoM has to stay over it,
  and it moves."""
  return torch.exp(-torch.square(LineState.get(env).com_lateral() / std))


def line_still(env: ManagerBasedRlEnv, std: float = 0.15) -> torch.Tensor:
  """exp(-vy^2 / std^2), vy the webbing's sideways speed under the base. A
  balanced slackliner keeps the line quiet; one who stays up by letting it
  swing is chasing the support (the cylinder's ``cylinder_still``)."""
  return torch.exp(-torch.square(LineState.get(env).line_under_base().vy[:, 0] / std))


def feet_on_line(env: ManagerBasedRlEnv, std: float = 0.025) -> torch.Tensor:
  """Mean over feet on the webbing of exp(-(lat / std)^2), lat the sole
  centre's offset from its centreline: the webbing (5 cm) is narrower than
  the sole (7 cm), and a sole off-centre rolls it."""
  on = _feet_on_line(env).float()
  lat = LineState.get(env).feet()[..., 1]
  return (torch.exp(-torch.square(lat / std)) * on).sum(-1) / on.sum(-1).clamp(min=1.0)


def stance_foot_yaw(
  env: ManagerBasedRlEnv, std: float = 0.25, back_yaw: float = BACK_FOOT_YAW
) -> torch.Tensor:
  """Mean over feet on the webbing of exp(-err^2 / std^2): the front foot
  points along the line, the back foot is turned out by ``back_yaw`` (left
  foot's toes to the left, right foot's to the right). Front and back by
  position along the line, so the roles swap with each step."""
  st = LineState.get(env)
  yaw = st.feet_yaw()
  back_left = st.back_is_left()
  target = torch.zeros_like(yaw)
  target[:, 0] = torch.where(back_left, back_yaw, 0.0)
  target[:, 1] = torch.where(~back_left, -back_yaw, 0.0)
  on = _feet_on_line(env).float()
  r = torch.exp(-torch.square(_wrap(yaw - target) / std))
  return (r * on).sum(-1) / on.sum(-1).clamp(min=1.0)


def stance_foot_flat(env: ManagerBasedRlEnv, std: float = 0.15) -> torch.Tensor:
  """Mean over feet on the webbing of exp(-tilt^2 / std^2): a sole held
  level keeps the webbing under it from rolling."""
  q = _robot(env).data.site_quat_w[:, LineState.get(env).foot_sites]
  w, x, y, z = q.unbind(-1)
  gx = 2 * (x * z - w * y)
  gy = 2 * (y * z + w * x)
  on = _feet_on_line(env).float()
  return (torch.exp(-(gx * gx + gy * gy) / std**2) * on).sum(-1) / on.sum(-1).clamp(min=1.0)


def heading_along_line(env: ManagerBasedRlEnv, std: float = 0.3) -> torch.Tensor:
  return torch.exp(-torch.square(LineState.get(env).heading_error() / std))


def step_progress(
  env: ManagerBasedRlEnv,
  scale: float = 0.3,
  min_air_time: float = 0.1,
  command_threshold: float = 0.05,
) -> torch.Tensor:
  """Metres stepped along the line, from the beam's ``feet_in_line``: at
  each ALTERNATING touchdown on the webbing, clip(gained / scale, 0, 1),
  gained the new ground the landing foot takes beyond the furthest touchdown
  of the episode so far. Divided by step_dt, so the weight is the reward per
  full step. Only while walking is commanded.

  Paid per metre gained and only on alternation, so tapping a foot, shuffling
  (one foot always leading) and scissoring in place pay nothing: the beam's
  lessons. It replaces ``feet_air_time``, which pays for any lift-off."""
  data = env.scene[FOOT_LINE_SENSOR].data
  assert data.current_contact_time is not None and data.last_air_time is not None
  first = (data.current_contact_time > 0) & (data.current_contact_time <= env.step_dt + 1e-6)
  first &= data.last_air_time > min_air_time
  fresh = env.episode_length_buf <= 1
  last = getattr(env, "_slackline_last_touchdown", None)
  if last is None:
    last = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)
    env._slackline_last_touchdown = last
  last[fresh] = -1
  foot = torch.arange(2, device=env.device)[None, :]
  alternating = first & (last[:, None] != foot)

  along = LineState.get(env).feet()[..., 0]
  frontier = getattr(env, "_slackline_step_frontier", None)
  if frontier is None:
    frontier = along.max(-1).values.clone()
    env._slackline_step_frontier = frontier
  frontier[fresh] = along[fresh].max(-1).values
  gained = (along - frontier[:, None]).clamp(0.0, scale) / scale
  r = (gained * alternating.float()).sum(-1)

  landed = torch.where(first, along, torch.full_like(along, -1e9)).max(-1).values
  frontier[:] = torch.maximum(frontier, landed)
  both = first.all(-1)
  last[:] = torch.where(both, -1, torch.where(first[:, 0], 0, torch.where(first[:, 1], 1, last)))
  walking = _forward_cmd(env) > command_threshold
  return r * walking.float() / env.step_dt


def foot_off_line_standing(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Number of feet off the webbing while standing is commanded, after the
  spawn settles: the stance keeps both feet down."""
  off = (~_feet_on_line(env)).float().sum(-1)
  return off * (_settled(env) & _standing(env)).float()


def foot_line_slip(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Sum over feet on the webbing of their squared speed relative to it."""
  return foot_slip_on_line(env).square().sum(-1)


def touchdown_force_rate(env: ManagerBasedRlEnv, body_weight_n: float = BODY_WEIGHT_N) -> torch.Tensor:
  """Sum over feet of (relu(rise of the foot's load in one step) / body
  weight)^2: place a foot, then shift weight onto it; a stamp throws the
  line. Zero while the spawn settles."""
  fz = _foot_load(env)
  prev = getattr(env, "_slackline_prev_fz", None)
  if prev is None:
    prev = fz.clone()
    env._slackline_prev_fz = prev
  prev[~_settled(env)] = fz[~_settled(env)]
  rise = torch.relu(fz - prev) / body_weight_n
  prev.copy_(fz)
  return rise.square().sum(-1)


def joint_outside_range(
  env: ManagerBasedRlEnv, low: float, high: float, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
  """Sum over joints of the squared distance (rad^2) to [low, high]: free
  inside the window, a quadratic wall either side (the pedalo's)."""
  q = _robot(env).data.joint_pos[:, asset_cfg.joint_ids]
  return (torch.relu(low - q) + torch.relu(q - high)).square().sum(-1)


# ---------------------------------------------------------------------------
# Terminations
# ---------------------------------------------------------------------------


def foot_on_ground(env: ManagerBasedRlEnv) -> torch.Tensor:
  """A foot touches the floor: the robot has stepped or fallen off."""
  return _feet_on_ground(env).any(dim=-1)


def nonfoot_on_line(env: ManagerBasedRlEnv) -> torch.Tensor:
  found = env.scene[NONFOOT_LINE_SENSOR].data.found
  return found.view(env.num_envs, -1).any(-1)


def base_too_low(env: ManagerBasedRlEnv, min_height: float = 0.55) -> torch.Tensor:
  """Base below ``min_height`` over the webbing beneath it (standing puts it
  at ~0.75 m)."""
  return LineState.get(env).base()[:, 2] < min_height


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

SPAWN_CLEARANCE = 0.01


def _stance_tensors(env: ManagerBasedRlEnv) -> dict[str, torch.Tensor]:
  """Per front foot: [nj] joint positions of the solved stance, base x/z."""
  cache = getattr(env, "_slackline_stances", None)
  if cache is not None:
    return cache
  from sendg1.common.slackline.stance import STANCES

  robot = _robot(env)
  cache = {}
  for front, joints in STANCES.items():
    q = robot.data.default_joint_pos[0].clone()
    for name, v in joints.items():
      if name.startswith("__"):
        continue
      ids, _ = robot.find_joints((name,))
      q[ids[0]] = v
    cache[front] = q
    cache[f"{front}_base"] = torch.tensor(
      [joints["__base_x__"], joints["__base_z__"]], device=env.device
    )
  env._slackline_stances = cache
  return cache


def reset_on_line(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  along_range: tuple[float, float] = SPAWN_ALONG,
  heading_noise: float = 0.05,
  joint_noise: float = 0.02,
) -> None:
  """Put the line at the env origin, at rest and already sagged under the
  robot's weight, and the robot on it in the tandem stance (``stance``):
  facing a random way along the line, a random foot in front."""
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  n, dev = len(env_ids), env.device
  line, robot = _line(env), _robot(env)
  st = LineState.get(env)
  stances = _stance_tensors(env)

  def uniform(lo: float, hi: float) -> torch.Tensor:
    return lo + (hi - lo) * torch.rand(n, device=dev)

  origin = env.scene.env_origins[env_ids]
  pose = torch.zeros(n, 7, device=dev)
  pose[:, :3] = origin
  pose[:, 3] = 1.0
  line.write_mocap_pose_to_sim(pose, env_ids=env_ids)

  direction = torch.where(torch.rand(n, device=dev) < 0.5, 1.0, -1.0)
  front_left = torch.rand(n, device=dev) < 0.5
  along = uniform(*along_range)
  st.direction[env_ids] = direction
  st.front_left[env_ids] = front_left
  st.spawn_along[env_ids] = along

  # The line sagged under half the weight at each foot, at this world's
  # tension and level: the string's stiffness (``T / dx`` between
  # neighbours, ``model``) plus the level's helper springs, solved exactly.
  feet_x = torch.stack([direction * (along + f * STANCE_LENGTH / 2) for f in (-1.0, 1.0)], -1)
  sag = static_sag(line_tension(env)[env_ids], _level_table(env)[st.level[env_ids], 1], feet_x)
  st.on_spawn(env_ids)
  q = torch.zeros(n, line.data.joint_pos.shape[1], device=dev)
  q[:, st.z_ids] = sag
  # Pieces already pitched to that shape (``model``'s slope equality).
  padded = torch.nn.functional.pad(sag, (1, 1))
  q[:, st.pitch_ids] = -(padded[:, 2:] - padded[:, :-2]) / (2 * sm.NODE_SPACING)
  line.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=env_ids)

  # The robot: the solved stance, centred on ``along``, soles on the sagged line.
  centre = (direction * along)[:, None] + sm.SPAN / 2
  knots = torch.cat([torch.zeros(n, 1, device=dev), sag, torch.zeros(n, 1, device=dev)], -1)
  k = (centre / sm.NODE_SPACING).clamp(0.0, sm.NUM_NODES + 1 - 1e-4)
  i0 = k.floor().long()
  t = k - i0
  sag_c = (1 - t) * knots.gather(1, i0) + t * knots.gather(1, i0 + 1)

  base_xz = torch.where(front_left[:, None], stances["left_base"], stances["right_base"])
  heading = torch.where(direction > 0, 0.0, math.pi) + heading_noise * (2 * torch.rand(n, device=dev) - 1)
  base = torch.zeros(n, 13, device=dev)
  base[:, 0] = origin[:, 0] + direction * (along + base_xz[:, 0])
  base[:, 1] = origin[:, 1]
  base[:, 2] = origin[:, 2] + sm.HEIGHT + sag_c[:, 0] + base_xz[:, 1] + SPAWN_CLEARANCE
  zero = torch.zeros(n, device=dev)
  base[:, 3:7] = quat_from_euler_xyz(zero, zero, heading)
  robot.write_root_state_to_sim(base, env_ids=env_ids)

  qj = torch.where(front_left[:, None], stances["left"], stances["right"])
  qj = qj + joint_noise * (2 * torch.rand_like(qj) - 1)
  robot.write_joint_state_to_sim(qj, torch.zeros_like(qj), env_ids=env_ids)


def static_sag(tension: torch.Tensor, helper: torch.Tensor, feet_x: torch.Tensor) -> torch.Tensor:
  """[B, N] node heights (m, negative down) of the line resting under the
  robot's weight split over ``feet_x`` [B, F] (line frame), each load shared
  by its two nearest nodes; ``tension`` (N) and ``helper`` (N/m) per env."""
  b, n = len(tension), sm.NUM_NODES
  dev = tension.device
  lap = 2 * torch.eye(n, device=dev) - torch.diag(torch.ones(n - 1, device=dev), 1)
  lap = lap - torch.diag(torch.ones(n - 1, device=dev), -1)
  k = (tension / sm.NODE_SPACING)[:, None, None] * lap + helper[:, None, None] * torch.eye(n, device=dev)
  f = torch.zeros(b, n + 2, device=dev)  # knots with the anchors at either end
  s = ((feet_x + sm.SPAN / 2) / sm.NODE_SPACING).clamp(0.0, n + 1 - 1e-4)
  i0 = s.floor().long()
  t = s - i0
  share = BODY_WEIGHT_N / feet_x.shape[1]
  f.scatter_add_(1, i0, -share * (1 - t))
  f.scatter_add_(1, i0 + 1, -share * t)
  return torch.linalg.solve(k, f[:, 1:-1])


def _level_table(env: ManagerBasedRlEnv) -> torch.Tensor:
  """[levels, 3]: width, helper stiffness, helper roll stiffness."""
  table = getattr(env, "_slackline_levels", None)
  if table is None:
    table = torch.tensor(
      [[lv.width, lv.stiffness, lv.roll_stiffness] for lv in LEVELS], device=env.device
    )
    env._slackline_levels = table
  return table


@requires_model_fields("jnt_stiffness", "geom_size", "geom_rbound", "geom_aabb")
def apply_line_level(
  env: ManagerBasedRlEnv, env_ids: torch.Tensor | None, fixed_level: int | None = None
) -> None:
  """Reset event, before ``reset_on_line``: give each world its level's
  webbing width and helper springs (``LEVELS``). ``fixed_level``: put every
  env there instead (evaluation, play)."""
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  st = LineState.get(env)
  if fixed_level is not None:
    st.level[env_ids] = fixed_level
    st.home[env_ids] = fixed_level
  ids = getattr(env, "_slackline_model_ids", None)
  if ids is None:
    m = env.sim.mj_model

    def jid(fmt: str) -> torch.Tensor:
      return torch.tensor([m.joint(f"{LINE}/{fmt.format(i)}").id for i in range(sm.NUM_NODES)], device=env.device)

    ids = {
      "slide": torch.cat([jid(sm.SLIDE_Y), jid(sm.SLIDE_Z)]),
      "roll": jid(sm.ROLL),
      "geoms": torch.tensor(
        [m.geom(f"{LINE}/{fmt.format(i)}").id for fmt in (sm.NODE_GEOM, sm.LOOK_GEOM) for i in range(sm.NUM_NODES)],
        device=env.device,
      ),
    }
    env._slackline_model_ids = ids
  lv = _level_table(env)[st.level[env_ids]]
  model = env.sim.model
  e = env_ids[:, None]
  model.jnt_stiffness[e, ids["slide"][None]] = lv[:, 1:2].expand(-1, len(ids["slide"]))
  model.jnt_stiffness[e, ids["roll"][None]] = lv[:, 2:3].expand(-1, len(ids["roll"]))
  g = ids["geoms"][None]
  size = model.geom_size[e, g]
  size[..., 1] = lv[:, :1] / 2
  model.geom_size[e, g] = size
  # Boxes: bounding sphere and box from the new half-sizes.
  model.geom_rbound[e, g] = size.norm(dim=-1)
  model.geom_aabb[e, g, 1] = size


def line_levels(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  replay_share: float = REPLAY_SHARE,
  standing_share: tuple[float, float] = STANDING_SHARE,
) -> dict[str, torch.Tensor]:
  """Per-env stiff-to-free curriculum and the coupled command mix (module
  doc). Runs before the reset events, which build the new level."""
  st = LineState.get(env)
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  out: dict[str, torch.Tensor] = {}
  if env.common_step_counter > 0 and len(env_ids) > 0:
    terminated = env.termination_manager.terminated[env_ids]
    cmd = st.cmd_dist[env_ids]
    tracked = (cmd < WALK_MIN_DIST) | (st.max_progress[env_ids] >= TRACK_FRAC * cmd) | st.reached_end[env_ids]
    success = ~terminated & tracked
    moved = (st.home[env_ids] + success.long() - terminated.long()).clamp(0, TOP_LEVEL)
    frozen = st.replay[env_ids]
    st.home[env_ids] = torch.where(frozen, st.home[env_ids], moved)
    out["success"] = success.float().mean()
    out["terminated"] = terminated.float().mean()
    out["progress_m"] = st.max_progress[env_ids].clamp(min=0.0).mean()
    walking = cmd >= WALK_MIN_DIST
    if walking.any():
      out["track_ratio_walking"] = (st.max_progress[env_ids][walking] / cmd[walking]).clamp(0, 2).mean()

  if len(env_ids) > 0:
    n = len(env_ids)
    home = st.home[env_ids]
    rep = (torch.rand(n, device=env.device) < replay_share) & (home > 0)
    easier = (torch.rand(n, device=env.device) * (home + 1).float()).long().clamp(max=TOP_LEVEL)
    st.level[env_ids] = torch.where(rep, easier, home)
    st.replay[env_ids] = rep & (st.level[env_ids] < home)

  levels = st.home.float()
  frac = levels.mean() / max(TOP_LEVEL, 1)
  share = standing_share[0] + (standing_share[1] - standing_share[0]) * frac
  env.command_manager.get_term("twist").cfg.rel_standing_envs = float(share)
  out["mean"] = levels.mean()
  out["max"] = levels.max()
  out["standing_share"] = torch.tensor(float(share))
  for i in range(len(LEVELS)):
    out[f"frac_L{i}"] = (st.home == i).float().mean()
  return out


def set_levels(env: ManagerBasedRlEnv, levels: torch.Tensor) -> None:
  """Place every env on ``levels`` (own and current level), e.g. to resume a
  run at a level distribution. Takes effect at the next reset."""
  st = LineState.get(env)
  st.home[:] = levels
  st.level[:] = levels
  st.replay[:] = False


def reset_tactile_noise(env: ManagerBasedRlEnv, env_ids: torch.Tensor | None) -> None:
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  st = tac.TactileNoiseState.get(env, LAYOUTS[TACTILE.layout])
  st.resample(env_ids, torch.full((len(env_ids),), TACTILE_NOISE_STRENGTH, device=env.device))


LINE_GEOMS = SceneEntityCfg(LINE, geom_names=(r"webbing_\d+",))
LINE_STRING_TENDONS = SceneEntityCfg(
  LINE, tendon_names=(r"anchor_.*_[yz]", r"seg_.*_[yz]")
)
"""The tendons that carry the tension (not the roll couplings)."""
