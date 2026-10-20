"""MDP terms for balancing on a rolling cylinder (``model``).

The robot starts standing across the top of a cylinder lying free on the
floor and has to stay up there: keep the cylinder under its feet, the feet on
its top line, the CoM over the support line, and the cylinder itself from
rolling away. There is no command: the task is balance.

Geometry and frames (``CylinderState``):

- axis      the cylinder's axis, signed to point to the robot's left. The
            cylinder is the same either way round; the sign keeps every
            cylinder-frame quantity in the robot's sense (forward is
            forward), whichever way round it was spawned.
- heading   the axis turned by -90 deg: the direction the cylinder rolls.
- centre    the point on the axis midway between the ends.
- top line  the axis lifted by the radius: where the soles stand.
- drift     the centre's distance from where the episode put it.

The reward is laid out and argued in ``tasks/rolling_cylinder/env_cfg.py``.
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
from sendg1.common.robots.g1 import G1_FOOT_SITES
from sendg1.common.rolling_cylinder import model as cm
from sendg1.common.tactile.layout import LAYOUTS

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

CYLINDER = "cylinder"
FOOT_CYLINDER_SENSOR = "feet_cylinder_contact"
"""Net force of each foot on the cylinder."""
FOOT_GROUND_SENSOR = "feet_ground_contact"
"""Each foot on the floor: an early termination."""

SETTLE_S = 0.3
"""The robot is spawned a few mm above the top line; contact terms wait."""
TACTILE_NOISE_STRENGTH = 0.5
"""Sensor faults (``beam.tactile.TactileNoiseCfg``) at half the beam task's
top-level strength, as on the pedalo: no curriculum yet to grow them."""
SOLE_HALF_WIDTH = 0.035
"""Half a G1 sole across (~7 cm): how far a sole centre may go towards an end
before the sole hangs over it."""


def _robot(env: ManagerBasedRlEnv):
  return env.scene["robot"]


def _cylinder(env: ManagerBasedRlEnv):
  return env.scene[CYLINDER]


def _wrap(a: torch.Tensor) -> torch.Tensor:
  return torch.atan2(torch.sin(a), torch.cos(a))


def _settled(env: ManagerBasedRlEnv) -> torch.Tensor:
  return env.episode_length_buf * env.step_dt > SETTLE_S


# ---------------------------------------------------------------------------
# Sensors
# ---------------------------------------------------------------------------


def cylinder_match() -> ContactMatch:
  return ContactMatch(mode="body", pattern=cm.ROOT_BODY, entity=CYLINDER)


def feet_cylinder_contact_sensor() -> ContactSensorCfg:
  """Net contact force per foot against the cylinder."""
  return ContactSensorCfg(
    name=FOOT_CYLINDER_SENSOR,
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
      entity="robot",
    ),
    secondary=cylinder_match(),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )


# ---------------------------------------------------------------------------
# Cylinder state
# ---------------------------------------------------------------------------


class CylinderState:
  """Index lookups, the cylinder's geometry and the spawn point every term reads."""

  _ATTR = "_sendg1_cylinder_state"

  def __init__(self, env: ManagerBasedRlEnv) -> None:
    self._env = env
    cyl = _cylinder(env)
    self.root_id = cyl.find_bodies((cm.ROOT_BODY,), preserve_order=True)[0][0]
    self.foot_sites, _ = _robot(env).find_sites(G1_FOOT_SITES, preserve_order=True)
    gid = env.sim.mj_model.geom(f"{CYLINDER}/{cm.GEOM}").id
    size = env.sim.mj_model.geom_size[gid]
    self.radius = float(size[0])
    self.length = 2.0 * float(size[1])
    self.spawn_xy = env.scene.env_origins[:, :2].clone()
    """Where each episode put the cylinder centre (``reset_on_cylinder``)."""

  @classmethod
  def get(cls, env: ManagerBasedRlEnv) -> CylinderState:
    st = getattr(env, cls._ATTR, None)
    if st is None:
      st = cls(env)
      setattr(env, cls._ATTR, st)
    return st

  # -- the cylinder -----------------------------------------------------------

  def centre(self) -> torch.Tensor:
    return _cylinder(self._env).data.body_link_pos_w[:, self.root_id]

  def axis(self) -> torch.Tensor:
    """[B, 3] unit axis, signed towards the robot's left (module docstring)."""
    quat = _cylinder(self._env).data.body_link_quat_w[:, self.root_id]
    e = torch.zeros_like(quat[:, :3])
    e[:, 1] = 1.0
    a = quat_apply(quat, e)
    h = _robot(self._env).data.heading_w
    left = -torch.sin(h) * a[:, 0] + torch.cos(h) * a[:, 1]
    return torch.where(left[:, None] < 0, -a, a)

  def heading(self) -> torch.Tensor:
    a = self.axis()
    return torch.atan2(-a[:, 0], a[:, 1])

  def to_frame(self, vec_xy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """World xy vectors [B, ..., 2] into the cylinder frame (fwd, left)."""
    yaw = self.heading()
    for _ in range(vec_xy.dim() - 2):
      yaw = yaw[:, None]
    c, s = torch.cos(yaw), torch.sin(yaw)
    return c * vec_xy[..., 0] + s * vec_xy[..., 1], -s * vec_xy[..., 0] + c * vec_xy[..., 1]

  def velocity(self) -> tuple[torch.Tensor, torch.Tensor]:
    """Centre velocity across (rolling) and along (sliding) the axis, m/s."""
    v = _cylinder(self._env).data.body_link_lin_vel_w[:, self.root_id, :2]
    return self.to_frame(v)

  def angular_velocity(self) -> torch.Tensor:
    return _cylinder(self._env).data.body_link_ang_vel_w[:, self.root_id]

  def spin_rate(self) -> torch.Tensor:
    """Rolling rate about the signed axis (rad/s). Rolling forward (along the
    heading) turns it positively about a left-pointing axis: v = omega * R."""
    return (self.angular_velocity() * self.axis()).sum(-1)

  def yaw_rate(self) -> torch.Tensor:
    return self.angular_velocity()[:, 2]

  def drift(self) -> torch.Tensor:
    """[B, 2] centre offset from its spawn point, in the cylinder frame."""
    fwd, lat = self.to_frame(self.centre()[:, :2] - self.spawn_xy)
    return torch.stack([fwd, lat], dim=-1)

  # -- the robot on it --------------------------------------------------------

  def feet_rel(self) -> torch.Tensor:
    """[B, 2, 3] each sole centre relative to the top line, cylinder frame:
    ahead of it, along the axis from the centre, above the top."""
    feet = _robot(self._env).data.site_pos_w[:, self.foot_sites]
    d = feet - self.centre()[:, None]
    fwd, lat = self.to_frame(d[..., :2])
    return torch.stack([fwd, lat, d[..., 2] - self.radius], dim=-1)

  def surface_velocity_under_feet(self) -> torch.Tensor:
    """[B, 2, 3] velocity of the cylinder's material point under each sole
    centre (projected onto the surface), world frame."""
    c = self.centre()
    feet = _robot(self._env).data.site_pos_w[:, self.foot_sites]
    a = self.axis()[:, None]
    d = feet - c[:, None]
    radial = d - (d * a).sum(-1, keepdim=True) * a
    point = (d - radial) + self.radius * radial / radial.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    data = _cylinder(self._env).data
    v = data.body_link_lin_vel_w[:, self.root_id, None]
    w = data.body_link_ang_vel_w[:, self.root_id, None].expand_as(point)
    return v + torch.linalg.cross(w, point)

  def feet_yaw_error(self) -> torch.Tensor:
    q = _robot(self._env).data.site_quat_w[:, self.foot_sites]
    w, x, y, z = q.unbind(-1)
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return _wrap(yaw - self.heading()[:, None])

  def base_offset(self) -> torch.Tensor:
    """[B, 3] robot base relative to the top line's centre, cylinder frame."""
    d = _robot(self._env).data.root_link_pos_w - self.centre()
    fwd, lat = self.to_frame(d[:, :2])
    return torch.stack([fwd, lat, d[:, 2] - self.radius], dim=-1)

  def com_offset(self) -> torch.Tensor:
    """[B, 2] whole-body CoM ground projection relative to the centre."""
    d = whole_body_com_xy(self._env) - self.centre()[:, :2]
    fwd, lat = self.to_frame(d)
    return torch.stack([fwd, lat], dim=-1)

  def heading_error(self) -> torch.Tensor:
    return _wrap(_robot(self._env).data.heading_w - self.heading())


def _feet_on_cylinder(env: ManagerBasedRlEnv) -> torch.Tensor:
  found = env.scene[FOOT_CYLINDER_SENSOR].data.found
  return found.view(env.num_envs, 2) > 0


def _feet_on_ground(env: ManagerBasedRlEnv) -> torch.Tensor:
  found = env.scene[FOOT_GROUND_SENSOR].data.found
  return found.view(env.num_envs, 2) > 0


def _foot_load(env: ManagerBasedRlEnv) -> torch.Tensor:
  """[B, 2] vertical force of each foot on the cylinder (N)."""
  force = env.scene[FOOT_CYLINDER_SENSOR].data.force
  assert force is not None
  return force.view(env.num_envs, 2, 3)[..., 2].abs()


# ---------------------------------------------------------------------------
# Observations (privileged: simulation ground truth, critic only)
# ---------------------------------------------------------------------------


def cylinder_motion(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Rolling and sliding speed, yaw rate, spin rate, drift (fwd, lat). [B, 6]"""
  st = CylinderState.get(env)
  v_fwd, v_lat = st.velocity()
  return torch.cat(
    [torch.stack([v_fwd, v_lat, st.yaw_rate(), st.spin_rate()], dim=-1), st.drift()], dim=-1
  )


def feet_cylinder_pose(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Per foot, relative to the top line: ahead, along the axis, above, sin/cos
  of the yaw error to the cylinder heading. [B, 10]"""
  st = CylinderState.get(env)
  yaw = st.feet_yaw_error()
  return torch.cat(
    [st.feet_rel(), torch.sin(yaw)[..., None], torch.cos(yaw)[..., None]], dim=-1
  ).flatten(1)


def base_cylinder_pose(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Base offset from the top line's centre (fwd, lat, up), sin/cos heading
  error, CoM offset (fwd, lat). [B, 7]"""
  st = CylinderState.get(env)
  err = st.heading_error()
  return torch.cat(
    [st.base_offset(), torch.sin(err)[:, None], torch.cos(err)[:, None], st.com_offset()], dim=-1
  )


def foot_slip_on_cylinder(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Per foot: planar velocity relative to the cylinder surface under it
  (cylinder frame) while in contact, 0 otherwise. [B, 4]"""
  st = CylinderState.get(env)
  v = _robot(env).data.site_lin_vel_w[:, st.foot_sites, :2] - st.surface_velocity_under_feet()[..., :2]
  fwd, lat = st.to_frame(v)
  on = _feet_on_cylinder(env).float()
  return (torch.stack([fwd, lat], dim=-1) * on[..., None]).flatten(1)


def foot_cylinder_forces(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Per foot: net force on the cylinder (fwd, lat, up in its frame),
  ``sign * log1p(|f|)``. [B, 6]"""
  st = CylinderState.get(env)
  force = env.scene[FOOT_CYLINDER_SENSOR].data.force
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


def cylinder_still(env: ManagerBasedRlEnv, std: float = 0.1) -> torch.Tensor:
  """exp(-v^2 / std^2), v the rolling speed. Balanced means the cylinder is
  at rest too: a robot that stays up by letting the cylinder run back and
  forth under it is chasing its support, not holding it."""
  v, _ = CylinderState.get(env).velocity()
  return torch.exp(-torch.square(v) / std**2)


def stay_near_spawn(env: ManagerBasedRlEnv, std: float = 0.3) -> torch.Tensor:
  """exp(-|drift|^2 / std^2). Stops a slow creep that ``cylinder_still``, at
  a few cm/s, barely sees: rolling the cylinder 'walks' the robot away."""
  d = CylinderState.get(env).drift()
  return torch.exp(-d.square().sum(-1) / std**2)


def feet_on_top(env: ManagerBasedRlEnv, std: float = 0.03) -> torch.Tensor:
  """Mean over feet of contact x exp(-(ahead/std)^2), the sole centre's
  fore-aft offset from the top line: a sole centred on the line can roll
  the support either way under it, toward the heel or the toes."""
  rel = CylinderState.get(env).feet_rel()
  return (torch.exp(-torch.square(rel[..., 0] / std)) * _feet_on_cylinder(env).float()).mean(-1)


def foot_past_end(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Sum over feet of how far (m) the sole edge hangs past an end."""
  st = CylinderState.get(env)
  lat = st.feet_rel()[..., 1].abs()
  return torch.relu(lat + SOLE_HALF_WIDTH - st.length / 2).sum(-1)


def foot_off_cylinder(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Number of feet not touching the cylinder, after the spawn settles."""
  off = (~_feet_on_cylinder(env)).float().sum(-1)
  return off * _settled(env).float()


def foot_yaw(env: ManagerBasedRlEnv, std: float = 0.25) -> torch.Tensor:
  """Mean over feet on the cylinder of exp(-yaw_err^2 / std^2): soles across
  the axis, where the sole's full length meets the curve."""
  err = CylinderState.get(env).feet_yaw_error()
  on = _feet_on_cylinder(env).float()
  return (torch.exp(-torch.square(err / std)) * on).sum(-1) / on.sum(-1).clamp(min=1.0)


def com_over_support(
  env: ManagerBasedRlEnv, std_fwd: float = 0.03, std_lat: float = 0.08
) -> torch.Tensor:
  """exp of the squared CoM offset from the cylinder centre, scaled per axis.
  Fore-aft the support is a line, so the width is tight; along the axis it is
  as wide as the stance."""
  off = CylinderState.get(env).com_offset()
  return torch.exp(-torch.square(off[:, 0] / std_fwd) - torch.square(off[:, 1] / std_lat))


def heading_to_cylinder(env: ManagerBasedRlEnv, std: float = 0.3) -> torch.Tensor:
  """exp(-err^2 / std^2), err the robot's heading minus the rolling direction."""
  return torch.exp(-torch.square(CylinderState.get(env).heading_error() / std))


def cylinder_axial_velocity(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Squared speed along the axis: the cylinder sliding sideways."""
  _, v_lat = CylinderState.get(env).velocity()
  return torch.square(v_lat)


def cylinder_yaw_rate(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Squared yaw rate: turning a cylinder on the floor scrubs it."""
  return torch.square(CylinderState.get(env).yaw_rate())


def stance_foot_flat(env: ManagerBasedRlEnv, std: float = 0.15) -> torch.Tensor:
  """Mean over feet on the cylinder of exp(-tilt^2 / std^2). The tangent at
  the top line is level, so a level sole is one centred on it."""
  q = _robot(env).data.site_quat_w[:, CylinderState.get(env).foot_sites]
  w, x, y, z = q.unbind(-1)
  gx = 2 * (x * z - w * y)
  gy = 2 * (y * z + w * x)
  tilt2 = gx * gx + gy * gy
  on = _feet_on_cylinder(env).float()
  return (torch.exp(-tilt2 / std**2) * on).sum(-1) / on.sum(-1).clamp(min=1.0)


def foot_cylinder_slip(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Sum over feet on the cylinder of their squared speed relative to its surface."""
  return foot_slip_on_cylinder(env).square().sum(-1)


def joint_outside_range(
  env: ManagerBasedRlEnv, low: float, high: float, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
  """Sum over joints of the squared distance (rad^2) to [low, high]: free
  inside the window, a quadratic wall either side."""
  q = _robot(env).data.joint_pos[:, asset_cfg.joint_ids]
  return (torch.relu(low - q) + torch.relu(q - high)).square().sum(-1)


def touchdown_force_rate(env: ManagerBasedRlEnv, body_weight_n: float = 343.0) -> torch.Tensor:
  """Sum over feet of (relu(rise of the foot's load in one step) / body
  weight)^2: no stamping on the cylinder. Zero while the spawn settles."""
  fz = _foot_load(env)
  prev = getattr(env, "_cylinder_prev_fz", None)
  if prev is None:
    prev = fz.clone()
    env._cylinder_prev_fz = prev
  prev[~_settled(env)] = fz[~_settled(env)]
  rise = torch.relu(fz - prev) / body_weight_n
  prev.copy_(fz)
  return rise.square().sum(-1)


# ---------------------------------------------------------------------------
# Terminations
# ---------------------------------------------------------------------------


def foot_on_ground(env: ManagerBasedRlEnv) -> torch.Tensor:
  """A foot touches the floor: the robot has stepped or fallen off."""
  return _feet_on_ground(env).any(dim=-1)


def base_too_low(env: ManagerBasedRlEnv, min_height: float = 0.8) -> torch.Tensor:
  """Base below ``min_height`` over the floor (standing on a 30 cm cylinder
  puts it at ~1.06 m)."""
  return _robot(env).data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2] < min_height


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

FOOT_SITE_X = 0.0388
"""Sole-centre site ahead of the base in the default stance."""
STANDING_BASE_Z = 0.76
"""G1 init_state base height with the soles on the floor."""
SPAWN_CLEARANCE = 0.01


def reset_on_cylinder(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  yaw_range: tuple[float, float] = (-math.pi, math.pi),
  heading_noise: float = 0.1,
  fore_aft_noise: float = 0.02,
  joint_noise: float = 0.03,
) -> None:
  """Put the cylinder at rest at the env origin, its axis at a random yaw,
  and the robot standing across it in the default stance: soles centred on
  the top line to within ``fore_aft_noise`` (m), so the first thing a policy
  meets is a cylinder starting to roll."""
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  n, dev = len(env_ids), env.device
  cyl, robot = _cylinder(env), _robot(env)
  st = CylinderState.get(env)

  def uniform(lo: float, hi: float) -> torch.Tensor:
    return lo + (hi - lo) * torch.rand(n, device=dev)

  yaw = uniform(*yaw_range)
  # Random spin about the axis: the stripes start anywhere.
  spin = uniform(-math.pi, math.pi)
  origin = env.scene.env_origins[env_ids]
  zero = torch.zeros(n, device=dev)

  root = torch.zeros(n, 13, device=dev)
  root[:, :2] = origin[:, :2]
  root[:, 2] = origin[:, 2] + st.radius
  root[:, 3:7] = quat_mul(quat_from_euler_xyz(zero, zero, yaw), quat_from_euler_xyz(zero, spin, zero))
  cyl.write_root_state_to_sim(root, env_ids=env_ids)
  st.spawn_xy[env_ids] = origin[:, :2]

  # The robot faces the rolling direction (either way round: the cylinder
  # is symmetric), sole centres over the top line.
  heading = yaw + math.pi * (torch.rand(n, device=dev) < 0.5).float()
  heading = heading + heading_noise * (2 * torch.rand(n, device=dev) - 1)
  ahead = uniform(-fore_aft_noise, fore_aft_noise) - FOOT_SITE_X
  base = torch.zeros(n, 13, device=dev)
  base[:, 0] = origin[:, 0] + ahead * torch.cos(heading)
  base[:, 1] = origin[:, 1] + ahead * torch.sin(heading)
  base[:, 2] = origin[:, 2] + 2 * st.radius + STANDING_BASE_Z + SPAWN_CLEARANCE
  base[:, 3:7] = quat_from_euler_xyz(zero, zero, heading)
  robot.write_root_state_to_sim(base, env_ids=env_ids)

  q = robot.data.default_joint_pos[env_ids].clone()
  q += joint_noise * (2 * torch.rand_like(q) - 1)
  robot.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=env_ids)


def reset_tactile_noise(env: ManagerBasedRlEnv, env_ids: torch.Tensor | None) -> None:
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  st = tac.TactileNoiseState.get(env, LAYOUTS[TACTILE.layout])
  st.resample(env_ids, torch.full((len(env_ids),), TACTILE_NOISE_STRENGTH, device=env.device))


CYLINDER_GEOMS = SceneEntityCfg(CYLINDER, geom_names=(cm.GEOM,))
