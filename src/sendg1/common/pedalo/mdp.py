"""MDP terms for riding a pedalo: the Classic or the Sport
(``variants``; ``PedaloState`` finds out which from the scene).

The robot starts standing on the pedalo, one foot per board (Classic) or
step (Sport), and is asked to ride it forward at a joystick speed (or hold it
still). It drives the pedalo the way a person does: weight onto the step that
is ahead and coming down, off the one that is rising, carried through the
dead centre (one step on top, one at the bottom) by momentum. On the Sport it
also balances fore-aft over the single wheel axis and keeps each freely
turning step level under its foot. Below, "board" means either.

Geometry and frames (``PedaloState``):

- heading     yaw of the root wheelset's forward axis (its lateral axis turned
              by -90 deg: pitching about it never changes it).
- centre      mean of the wheel centres (the Sport's single axle, the Classic's
              support polygon centre).
- crank phi   root wheelset pitch, 0 with the boards level and the left board
              ahead; forward rolling is phi increasing.
- board top   centre of each board's top face, in the world.

The reward reuses the beam task's lessons (``tasks/beam/env_cfg.py``):
progress with slack and a symmetric clip instead of exact speed tracking (the
beam policy rocked in place while backward motion cost nothing), a large
one-off termination penalty (without it the random policy fell at once), feet
and CoM centred on the support with fixed Gaussian widths, flat stance feet,
walking regularisers, a symmetric actor.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.utils.lab_api.math import quat_apply, quat_from_euler_xyz, quat_mul

from sendg1.common.beam import tactile as tac
from sendg1.common.beam.mdp import TACTILE, whole_body_com_xy
from sendg1.common.pedalo import model as pm
from sendg1.common.pedalo.variants import CLASSIC, VARIANTS, PedaloVariant
from sendg1.common.robots.g1 import G1_FOOT_SITES
from sendg1.common.tactile.layout import LAYOUTS

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

PEDALO = "pedalo"
FOOT_PEDALO_SENSOR = "feet_pedalo_contact"
"""Net force of each foot on the pedalo (any of its bodies)."""
FOOT_GROUND_SENSOR = "feet_ground_contact"
"""Each foot on the floor: an early termination."""

SETTLE_S = 0.3
"""The robot is spawned a few cm above the higher board; contact terms wait."""
TACTILE_NOISE_STRENGTH = 0.5
"""Sensor faults (``beam.tactile.TactileNoiseCfg``) at half the beam task's
top-level strength: no curriculum yet to grow them."""


def _robot(env: ManagerBasedRlEnv):
  return env.scene["robot"]


def _pedalo(env: ManagerBasedRlEnv):
  return env.scene[PEDALO]


def _forward_cmd(env: ManagerBasedRlEnv) -> torch.Tensor:
  return env.command_manager.get_command("twist")[:, 0]


def _wrap(a: torch.Tensor) -> torch.Tensor:
  return torch.atan2(torch.sin(a), torch.cos(a))


def _settled(env: ManagerBasedRlEnv) -> torch.Tensor:
  return env.episode_length_buf * env.step_dt > SETTLE_S


# ---------------------------------------------------------------------------
# Sensors
# ---------------------------------------------------------------------------


def pedalo_match(variant: PedaloVariant = CLASSIC) -> ContactMatch:
  """The whole pedalo as a contact secondary (its root's subtree)."""
  return ContactMatch(mode="subtree", pattern=variant.root_body, entity=PEDALO)


def feet_pedalo_contact_sensor(variant: PedaloVariant = CLASSIC) -> ContactSensorCfg:
  """Net contact force per foot against the pedalo."""
  return ContactSensorCfg(
    name=FOOT_PEDALO_SENSOR,
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
      entity="robot",
    ),
    secondary=pedalo_match(variant),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )


# ---------------------------------------------------------------------------
# Pedalo state
# ---------------------------------------------------------------------------


class PedaloState:
  """Index lookups, and the pedalo/feet geometry every term reads."""

  _ATTR = "_sendg1_pedalo_state"

  def __init__(self, env: ManagerBasedRlEnv) -> None:
    self._env = env
    ped = _pedalo(env)
    self.variant = v = variant_of(env)
    self.root_id = ped.find_bodies((v.root_body,), preserve_order=True)[0][0]
    self.wheel_ids, _ = ped.find_bodies(v.wheel_bodies, preserve_order=True)
    self.board_ids, _ = ped.find_bodies(v.step_bodies, preserve_order=True)
    robot = _robot(env)
    self.foot_sites, _ = robot.find_sites(G1_FOOT_SITES, preserve_order=True)
    self.board_top_local = torch.tensor(v.step_top_local, device=env.device)

  @classmethod
  def get(cls, env: ManagerBasedRlEnv) -> PedaloState:
    st = getattr(env, cls._ATTR, None)
    if st is None:
      st = cls(env)
      setattr(env, cls._ATTR, st)
    return st

  # -- the pedalo -------------------------------------------------------------

  def wheel_centres(self) -> torch.Tensor:
    """[B, 2, 3] front, rear."""
    return _pedalo(self._env).data.body_link_pos_w[:, self.wheel_ids]

  def centre(self) -> torch.Tensor:
    return self.wheel_centres().mean(dim=1)

  def _root_axis(self, axis: int) -> torch.Tensor:
    """[B, 3] one axis of the root wheelset's frame, in the world."""
    quat = _pedalo(self._env).data.body_link_quat_w[:, self.root_id]
    e = torch.zeros_like(quat[:, :3])
    e[:, axis] = 1.0
    return quat_apply(quat, e)

  def heading(self) -> torch.Tensor:
    lat = self._root_axis(1)
    return torch.atan2(-lat[:, 0], lat[:, 1])

  def roll(self) -> torch.Tensor:
    """[B] wheelset roll (rad): the axle line's tilt from horizontal."""
    return torch.asin(self._root_axis(1)[:, 2].clamp(-1.0, 1.0))

  def to_frame(self, vec_xy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """World xy vectors [B, ..., 2] into the pedalo heading frame (fwd, left)."""
    yaw = self.heading()
    for _ in range(vec_xy.dim() - 2):
      yaw = yaw[:, None]
    c, s = torch.cos(yaw), torch.sin(yaw)
    return c * vec_xy[..., 0] + s * vec_xy[..., 1], -s * vec_xy[..., 0] + c * vec_xy[..., 1]

  def velocity(self) -> tuple[torch.Tensor, torch.Tensor]:
    """Wheel-centre velocity along and across the pedalo heading (m/s)."""
    v = _pedalo(self._env).data.body_link_lin_vel_w[:, self.wheel_ids, :2].mean(dim=1)
    return self.to_frame(v)

  def yaw_rate(self) -> torch.Tensor:
    return _pedalo(self._env).data.body_link_ang_vel_w[:, self.root_id, 2]

  def crank(self) -> tuple[torch.Tensor, torch.Tensor]:
    """Crank angle phi (rad) and rate (rad/s): the root wheelset's pitch about
    its own lateral axis. A turn by phi about +y takes its x axis to
    (cos phi, 0, -sin phi)."""
    x = self._root_axis(0)
    fwd, _ = self.to_frame(x[:, :2])
    omega = _pedalo(self._env).data.body_link_ang_vel_w[:, self.root_id]
    return torch.atan2(-x[:, 2], fwd), (omega * self._root_axis(1)).sum(-1)

  def board_tops(self) -> torch.Tensor:
    """[B, 2, 3] centre of each board's top face (left, right)."""
    data = _pedalo(self._env).data
    pos = data.body_link_pos_w[:, self.board_ids]
    quat = data.body_link_quat_w[:, self.board_ids]
    return pos + quat_apply(quat, self.board_top_local.expand_as(pos))

  def board_velocity(self) -> torch.Tensor:
    """[B, 2, 3] velocity of each board's top centre (Classic boards only
    translate; Sport steps also turn)."""
    data = _pedalo(self._env).data
    v = data.body_link_lin_vel_w[:, self.board_ids]
    w = data.body_link_ang_vel_w[:, self.board_ids]
    r = self.board_tops() - data.body_link_pos_w[:, self.board_ids]
    return v + torch.linalg.cross(w, r)

  def board_tilt(self) -> torch.Tensor:
    """[B, 2] each board's tilt from level (rad), from its z axis."""
    quat = _pedalo(self._env).data.body_link_quat_w[:, self.board_ids]
    z = torch.zeros_like(quat[..., :3])
    z[..., 2] = 1.0
    up = quat_apply(quat, z)
    return torch.acos(up[..., 2].clamp(-1.0, 1.0))

  # -- the robot on it ----------------------------------------------------------

  def feet_on_boards(self) -> torch.Tensor:
    """[B, 2, 3] each foot's sole centre relative to its own board's top centre
    (left foot - left board), in the pedalo heading frame."""
    feet = _robot(self._env).data.site_pos_w[:, self.foot_sites]
    d = feet - self.board_tops()
    fwd, lat = self.to_frame(d[..., :2])
    return torch.stack([fwd, lat, d[..., 2]], dim=-1)

  def feet_yaw_error(self) -> torch.Tensor:
    q = _robot(self._env).data.site_quat_w[:, self.foot_sites]
    w, x, y, z = q.unbind(-1)
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return _wrap(yaw - self.heading()[:, None])

  def base_offset(self) -> torch.Tensor:
    """[B, 3] robot base relative to the pedalo centre, pedalo frame."""
    d = _robot(self._env).data.root_link_pos_w - self.centre()
    fwd, lat = self.to_frame(d[:, :2])
    return torch.stack([fwd, lat, d[:, 2]], dim=-1)

  def com_offset(self) -> torch.Tensor:
    """[B, 2] whole-body CoM ground projection relative to the pedalo centre."""
    d = whole_body_com_xy(self._env) - self.centre()[:, :2]
    fwd, lat = self.to_frame(d)
    return torch.stack([fwd, lat], dim=-1)

  def heading_error(self) -> torch.Tensor:
    return _wrap(_robot(self._env).data.heading_w - self.heading())


def variant_of(env: ManagerBasedRlEnv) -> PedaloVariant:
  """Which pedalo is in the scene, by its root body's name."""
  names = set(_pedalo(env).body_names)
  for v in VARIANTS:
    if v.root_body in names:
      return v
  raise ValueError(f"no known pedalo among bodies {sorted(names)}")


def _feet_on_pedalo(env: ManagerBasedRlEnv) -> torch.Tensor:
  found = env.scene[FOOT_PEDALO_SENSOR].data.found
  return found.view(env.num_envs, 2) > 0


def _feet_on_ground(env: ManagerBasedRlEnv) -> torch.Tensor:
  found = env.scene[FOOT_GROUND_SENSOR].data.found
  return found.view(env.num_envs, 2) > 0


def _foot_load(env: ManagerBasedRlEnv) -> torch.Tensor:
  """[B, 2] vertical force of each foot on the pedalo (N)."""
  force = env.scene[FOOT_PEDALO_SENSOR].data.force
  assert force is not None
  return force.view(env.num_envs, 2, 3)[..., 2].abs()


# ---------------------------------------------------------------------------
# Observations (privileged: simulation ground truth, critic only)
# ---------------------------------------------------------------------------


def pedalo_motion(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Forward and lateral speed, yaw rate, sin/cos crank angle, crank rate. [B, 6]"""
  st = PedaloState.get(env)
  v_fwd, v_lat = st.velocity()
  phi, dphi = st.crank()
  return torch.stack(
    [v_fwd, v_lat, st.yaw_rate(), torch.sin(phi), torch.cos(phi), dphi], dim=-1
  )


def feet_board_pose(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Per foot, relative to its own board: along, across, height above the top,
  sin/cos of the yaw error to the pedalo heading. [B, 10]"""
  st = PedaloState.get(env)
  rel = st.feet_on_boards()
  yaw = st.feet_yaw_error()
  return torch.cat([rel, torch.sin(yaw)[..., None], torch.cos(yaw)[..., None]], dim=-1).flatten(1)


def base_pedalo_pose(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Base offset from the pedalo centre (fwd, lat, up), sin/cos heading error,
  CoM offset (fwd, lat). [B, 7]"""
  st = PedaloState.get(env)
  err = st.heading_error()
  return torch.cat(
    [st.base_offset(), torch.sin(err)[:, None], torch.cos(err)[:, None], st.com_offset()], dim=-1
  )


def foot_slip_on_board(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Per foot: planar velocity relative to its board (pedalo frame) while in
  contact, 0 otherwise. [B, 4]"""
  st = PedaloState.get(env)
  v = _robot(env).data.site_lin_vel_w[:, st.foot_sites, :2] - st.board_velocity()[..., :2]
  fwd, lat = st.to_frame(v)
  on = _feet_on_pedalo(env).float()
  return (torch.stack([fwd, lat], dim=-1) * on[..., None]).flatten(1)


def foot_pedalo_forces(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Per foot: net force on the pedalo (fwd, lat, up in the pedalo frame),
  ``sign * log1p(|f|)``. [B, 6]"""
  st = PedaloState.get(env)
  force = env.scene[FOOT_PEDALO_SENSOR].data.force
  assert force is not None
  f = force.view(env.num_envs, 2, 3)
  fwd, lat = st.to_frame(f[..., :2])
  out = torch.stack([fwd, lat, f[..., 2]], dim=-1).flatten(1)
  return torch.sign(out) * torch.log1p(out.abs())


# ---------------------------------------------------------------------------
# Rewards
# ---------------------------------------------------------------------------


def alive(env: ManagerBasedRlEnv) -> torch.Tensor:
  return torch.ones(env.num_envs, device=env.device)


def track_pedalo_speed(
  env: ManagerBasedRlEnv, std: float = 0.1, overshoot: float = 2.0
) -> torch.Tensor:
  """The beam task's progress-with-slack, on the pedalo's own speed. Riding:
  clip(v / v_cmd, -1, 1) - overshoot * relu(v / v_cmd - 1); rolling backward
  costs as much as forward earns, so rocking the cranks back and forth pays
  nothing. Holding still: exp(-v^2 / std^2)."""
  v, _ = PedaloState.get(env).velocity()
  cmd = _forward_cmd(env)
  riding = cmd > 0.05
  ratio = v / cmd.clamp(min=0.05)
  ride = ratio.clamp(-1.0, 1.0) - overshoot * torch.relu(ratio - 1.0)
  hold = torch.exp(-torch.square(v) / std**2)
  return torch.where(riding, ride, hold)


def feet_on_boards(
  env: ManagerBasedRlEnv, std_along: float = 0.06, std_across: float = 0.03
) -> torch.Tensor:
  """Mean over feet of contact x exp(-(along/std_along)^2 - (across/std_across)^2),
  the sole centre's offset from its own board's centre. The beam's stance-foot
  centreline, in two axes: a foot centred on its board has the most margin
  before toes or heel overhang (board 30 x 14.5 cm, G1 sole ~20 x 7 cm)."""
  rel = PedaloState.get(env).feet_on_boards()
  r = torch.exp(-torch.square(rel[..., 0] / std_along) - torch.square(rel[..., 1] / std_across))
  return (r * _feet_on_pedalo(env).float()).mean(-1)


def foot_off_pedalo(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Number of feet not touching the pedalo, after the spawn settles. Riding
  is done with both feet down; lifting one is a step off."""
  off = (~_feet_on_pedalo(env)).float().sum(-1)
  return off * _settled(env).float()


def foot_yaw(env: ManagerBasedRlEnv, std: float = 0.25) -> torch.Tensor:
  """Mean over feet on the pedalo of exp(-yaw_err^2 / std^2), feet along the boards."""
  err = PedaloState.get(env).feet_yaw_error()
  on = _feet_on_pedalo(env).float()
  return (torch.exp(-torch.square(err / std)) * on).sum(-1) / on.sum(-1).clamp(min=1.0)


def com_over_pedalo(
  env: ManagerBasedRlEnv, std_along: float = 0.06, std_across: float = 0.035
) -> torch.Tensor:
  """exp of the squared CoM offset from the pedalo centre, scaled per axis.
  The wheels touch down 24 cm apart fore-aft and 35 cm apart across, but the
  pedalo only stays put under the rider while the load is shared."""
  off = PedaloState.get(env).com_offset()
  return torch.exp(-torch.square(off[:, 0] / std_along) - torch.square(off[:, 1] / std_across))


def heading_to_pedalo(env: ManagerBasedRlEnv, std: float = 0.3) -> torch.Tensor:
  """exp(-err^2 / std^2), err the robot's heading minus the pedalo's."""
  return torch.exp(-torch.square(PedaloState.get(env).heading_error() / std))


def pedalo_lateral_velocity(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Squared sideways speed of the wheels: skidding."""
  _, v_lat = PedaloState.get(env).velocity()
  return torch.square(v_lat)


def pedalo_yaw_rate(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Squared yaw rate. A pedalo has no steering; turning means scrubbing the tyres."""
  return torch.square(PedaloState.get(env).yaw_rate())


def stance_foot_flat(env: ManagerBasedRlEnv, std: float = 0.15) -> torch.Tensor:
  """Mean over feet on the pedalo of exp(-tilt^2 / std^2). The boards stay
  level, so a flat sole is a sole flat on its board."""
  q = _robot(env).data.site_quat_w[:, PedaloState.get(env).foot_sites]
  w, x, y, z = q.unbind(-1)
  gx = 2 * (x * z - w * y)
  gy = 2 * (y * z + w * x)
  tilt2 = gx * gx + gy * gy
  on = _feet_on_pedalo(env).float()
  return (torch.exp(-tilt2 / std**2) * on).sum(-1) / on.sum(-1).clamp(min=1.0)


def foot_board_slip(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Sum over feet on the pedalo of their squared speed relative to the board."""
  return foot_slip_on_board(env).square().sum(-1)


def board_tilt(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Sum over boards of their squared tilt (rad^2). Only the Sport's steps can
  tilt: each turns freely on its axle, so this is a foot failing to hold it."""
  return PedaloState.get(env).board_tilt().square().sum(-1)


def joint_outside_range(
  env: ManagerBasedRlEnv, low: float, high: float, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
  """Sum over joints of the squared distance (rad^2) to [low, high]: free
  inside the window, a quadratic wall either side."""
  q = _robot(env).data.joint_pos[:, asset_cfg.joint_ids]
  return (torch.relu(low - q) + torch.relu(q - high)).square().sum(-1)


def touchdown_force_rate(env: ManagerBasedRlEnv, body_weight_n: float = 343.0) -> torch.Tensor:
  """Sum over feet of (relu(rise of the foot's load in one step) / body
  weight)^2: weight goes onto a board smoothly, it is not stamped on. Zero
  while the spawn settles."""
  fz = _foot_load(env)
  prev = getattr(env, "_pedalo_prev_fz", None)
  if prev is None:
    prev = fz.clone()
    env._pedalo_prev_fz = prev
  prev[~_settled(env)] = fz[~_settled(env)]
  rise = torch.relu(fz - prev) / body_weight_n
  prev.copy_(fz)
  return rise.square().sum(-1)


# ---------------------------------------------------------------------------
# Terminations
# ---------------------------------------------------------------------------


def foot_on_ground(env: ManagerBasedRlEnv) -> torch.Tensor:
  """A foot touches the floor: the rider has stepped off."""
  return _feet_on_ground(env).any(dim=-1)


def base_too_low(env: ManagerBasedRlEnv, min_height: float = 0.6) -> torch.Tensor:
  """Base below ``min_height`` over the floor (standing on the boards puts
  it at ~0.9 m)."""
  return _robot(env).data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2] < min_height


def pedalo_tipped(env: ManagerBasedRlEnv, limit_angle: float = math.radians(25.0)) -> torch.Tensor:
  """The wheelset rolls past ``limit_angle``: the pedalo is up on one side."""
  return PedaloState.get(env).roll().abs() > limit_angle


def board_tipped(env: ManagerBasedRlEnv, limit_angle: float = math.radians(45.0)) -> torch.Tensor:
  """A Sport step turns past ``limit_angle``: its foot has rolled off it."""
  return (PedaloState.get(env).board_tilt() > limit_angle).any(dim=-1)


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

SPAWN_STANCE_Y = 0.058
"""Hip roll (rad) that brings the default 0.237 m stance in to the boards'
0.175 m spacing (ankle roll undoes it, so the soles stay flat)."""
LEG_LENGTH = 0.6
"""Hip-to-sole, for the fore-aft split: d(hip pitch) = -dx / LEG_LENGTH puts
a foot dx ahead (FK: -0.09 rad moves it +5.4 cm)."""
SPAWN_CLEARANCE = 0.01
FOOT_SITE_X = 0.0388
"""Sole-centre site ahead of the base in the default stance."""
STANDING_BASE_Z = 0.76
"""G1 init_state base height with the soles on the floor."""


def reset_on_pedalo(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  crank_range: tuple[float, float] = (-0.3, 0.3),
  yaw_range: tuple[float, float] = (-math.pi, math.pi),
  heading_noise: float = 0.1,
  joint_noise: float = 0.03,
) -> None:
  """Place the pedalo at the env origin, at a random heading and a crank angle
  near level boards (``crank_range`` about 0 or pi, either board ahead), and the
  robot standing on it: a split stance, each sole over its board's centre,
  dropped from just above the higher board."""
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  n, dev = len(env_ids), env.device
  ped, robot = _pedalo(env), _robot(env)
  variant = PedaloState.get(env).variant

  def uniform(lo: float, hi: float) -> torch.Tensor:
    return lo + (hi - lo) * torch.rand(n, device=dev)

  yaw = uniform(*yaw_range)
  flip = torch.rand(n, device=dev) < 0.5
  phi = uniform(*crank_range) + flip.float() * math.pi
  c, s = torch.cos(yaw), torch.sin(yaw)
  origin = env.scene.env_origins[env_ids]

  # -- pedalo: the root wheelset in its place ahead of the centre (Classic:
  # half a wheelbase; Sport: on it), pitched by phi; hinges counter-turn so
  # the boards are level.
  root = torch.zeros(n, 13, device=dev)
  root[:, 0] = origin[:, 0] + c * variant.root_ahead
  root[:, 1] = origin[:, 1] + s * variant.root_ahead
  root[:, 2] = origin[:, 2] + pm.WHEEL_RADIUS
  zero = torch.zeros(n, device=dev)
  root[:, 3:7] = quat_mul(quat_from_euler_xyz(zero, zero, yaw), quat_from_euler_xyz(zero, phi, zero))
  ped.write_root_state_to_sim(root, env_ids=env_ids)
  per_phi = dict(variant.hinge_phi)
  sign = torch.tensor([per_phi[j] for j in ped.joint_names], device=dev)
  hinge = phi[:, None] * sign
  ped.write_joint_state_to_sim(hinge, torch.zeros_like(hinge), env_ids=env_ids)

  # -- robot: feet over the board centres. Board centres sit +-e cos(phi)
  # along the pedalo from its centre (left ahead at phi = 0); tops at
  # base -+ e sin(phi).
  ahead = pm.ECCENTRIC * torch.cos(phi)
  top = torch.maximum(*variant.step_top_height(phi))
  heading = yaw + heading_noise * (2 * torch.rand(n, device=dev) - 1)
  base = torch.zeros(n, 13, device=dev)
  base[:, 0] = origin[:, 0] - FOOT_SITE_X * torch.cos(heading)
  base[:, 1] = origin[:, 1] - FOOT_SITE_X * torch.sin(heading)
  base[:, 2] = origin[:, 2] + top + STANDING_BASE_Z + SPAWN_CLEARANCE
  base[:, 3:7] = quat_from_euler_xyz(zero, zero, heading)
  robot.write_root_state_to_sim(base, env_ids=env_ids)

  q = robot.data.default_joint_pos[env_ids].clone()
  q += joint_noise * (2 * torch.rand_like(q) - 1)
  for side, sign, dx in (("left", 1.0, ahead), ("right", -1.0, -ahead)):
    j = _joint_index(robot)
    dpitch = -dx / LEG_LENGTH
    q[:, j[f"{side}_hip_pitch_joint"]] += dpitch
    q[:, j[f"{side}_ankle_pitch_joint"]] -= dpitch
    q[:, j[f"{side}_hip_roll_joint"]] -= sign * SPAWN_STANCE_Y
    q[:, j[f"{side}_ankle_roll_joint"]] += sign * SPAWN_STANCE_Y
  robot.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=env_ids)


def _joint_index(robot) -> dict[str, int]:
  return {n: i for i, n in enumerate(robot.joint_names)}


def reset_tactile_noise(env: ManagerBasedRlEnv, env_ids: torch.Tensor | None) -> None:
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  st = tac.TactileNoiseState.get(env, LAYOUTS[TACTILE.layout])
  st.resample(env_ids, torch.full((len(env_ids),), TACTILE_NOISE_STRENGTH, device=env.device))


def register_actor_mirror(env: ManagerBasedRlEnv, env_ids: torch.Tensor | None) -> None:
  """Startup event: the symmetric GRU actor's mirror maps (``pedalo.symmetry``)."""
  from sendg1.common.pedalo.symmetry import maps
  from sendg1.rl import symmetric_rnn

  m = maps(env)
  symmetric_rnn.set_mirrors(obs={g: v for g, v in m.items() if g != "__actions__"}, actions=m["__actions__"])


PEDALO_GEOMS = SceneEntityCfg(PEDALO, geom_names=(r".*_wheel$",))
"""The tyres, six on the Classic, three on the Sport (wheel-face and stripe
geoms are visual)."""
