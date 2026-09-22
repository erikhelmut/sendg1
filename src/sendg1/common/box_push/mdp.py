"""MDP terms for the box-pushing task.

One :class:`BoxPushState` per env instance holds everything an episode needs to
remember: the sampled world (floor mu, box mass/size/mu, target, patches), the
box-to-target distance of the previous step, and the per-episode statistics the
paper reports. It is advanced at most once per env step (``update``) by
whichever term asks first, and archived then re-sampled by the
``box_push_reset`` event, so ``last_episode`` always holds the episode that just
ended -- which is what ``scripts/evaluate_box_push.py`` reads.

Frames: "yaw frame" means world axes rotated by the pelvis heading only, so it
does not tilt when the robot leans into the box.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import mujoco
import torch
from mjlab.managers.event_manager import RecomputeLevel, requires_model_fields
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply,
  quat_apply_inverse,
  yaw_quat,
)

from sendg1.common.box_push.params import (
  MAX_PATCHES,
  PATCH_LIFT,
  BoxPushRanges,
  CurriculumCfg,
  PatchCfg,
  curriculum_ranges,
)
from sendg1.common.box_push.scene import (
  BOX_ENTITY,
  FEET_BOX_SENSOR,
  HAND_BOX_SENSOR,
  PUSH_FORCE_SENSOR,
  TARGET_ENTITY,
  TERRAIN_GEOM,
  patch_geom_name,
)
from sendg1.common.robots.g1 import G1_FOOT_SITES

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

RESET_EVENT = "box_push_reset"
"""Name of the reset event. The state reads its configuration from that
event's params, so there is exactly one copy of it in the env config."""

FOOT_CONTACT_SENSOR = "feet_ground_contact"
FOOT_BODIES: tuple[str, ...] = ("left_ankle_roll_link", "right_ankle_roll_link")

SOLE_Z = -0.035
"""Sole plane in the ankle_roll_link frame (capsule centres at -0.025, radius
0.01)."""
HEEL_TOE_X: tuple[float, float] = (-0.04, 0.12)
"""Heel and toe points on the sole, x in the ankle_roll_link frame. The
collision capsules span about -0.05..0.13."""

SLIP_SPEED = 0.10
"""m/s. A foot is sliding when it is in contact and its heel AND toe both move
tangentially faster than this. Requiring both separates a sliding foot from one
pivoting on its toe or heel, where the pivot point is still."""
SLIP_MIN_STEPS = 3
"""A full-foot slip EVENT is a slide sustained for this many consecutive control
steps (60 ms), counted once when it reaches that length. Without it, 56% of
counted events lasted a single 20 ms step -- touchdown and lift-off transients,
not slips (measured on the first 2,000-iteration check policy)."""

PINNABLE: tuple[str, ...] = (
  "floor_mu",
  "box_mass",
  "box_mu",
  "box_depth",
  "box_width",
  "box_height",
  "robot_gap",
  "target_distance",
  "target_angle",
)
"""World parameters an evaluation can fix per env (see ``BoxPushState.pins``)."""

FALL_TERMS: tuple[str, ...] = ("fell_over", "base_too_low")
BOX_TIPPED_TERM = "box_tipped"
SUCCESS_TERM = "success"


@dataclass
class EpisodeRecord:
  """One finished episode per env. All tensors are [B]."""

  valid: torch.Tensor
  success: torch.Tensor
  time_to_target: torch.Tensor
  """s; NaN when the box never reached the target."""
  length_s: torch.Tensor
  slip_events: torch.Tensor
  fell: torch.Tensor
  box_tipped: torch.Tensor
  mean_push_force: torch.Tensor
  """N, mean horizontal robot-on-box force over steps with contact; NaN if none."""
  double_support_impulse_frac: torch.Tensor
  """Share of the horizontal push impulse applied with both feet on the ground;
  NaN if the robot never touched the box."""
  floor_mu: torch.Tensor
  box_mass: torch.Tensor
  box_mu: torch.Tensor
  final_distance: torch.Tensor
  """m, box centre to target centre when the episode ended."""


class BoxPushState:
  _ATTR = "_sendg1_box_push_state"

  def __init__(self, env: ManagerBasedRlEnv) -> None:
    params = env.cfg.events[RESET_EVENT].params
    self.ranges: BoxPushRanges = params["ranges"]
    self.patches: PatchCfg | None = params["patches"]
    self.curriculum: CurriculumCfg | None = params["curriculum"]

    self._env = env
    self.robot = env.scene["robot"]
    self.box = env.scene[BOX_ENTITY]
    self.target = env.scene[TARGET_ENTITY]
    self.foot_contact = env.scene[FOOT_CONTACT_SENSOR]
    self.push_sensor = env.scene[PUSH_FORCE_SENSOR]

    mj_model = env.sim.mj_model
    self.box_body = int(self.box.indexing.body_ids[0])
    self.box_geom = int(self.box.indexing.geom_ids[0])
    self.box_dofs = self.box.indexing.free_joint_v_adr.to(env.device, dtype=torch.long)
    self.floor_geom = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, TERRAIN_GEOM)
    self.patch_geoms: list[int] = []
    if self.patches is not None:
      self.patch_geoms = [
        mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, patch_geom_name(i))
        for i in range(MAX_PATCHES)
      ]
      if min(self.patch_geoms) < 0:
        raise RuntimeError("low_friction_patches is on but the patch geoms are missing")
    self.foot_body_ids, _ = self.robot.find_bodies(FOOT_BODIES, preserve_order=True)
    self.foot_site_ids, _ = self.robot.find_sites(G1_FOOT_SITES, preserve_order=True)

    n, dev = env.num_envs, env.device
    z = lambda: torch.zeros(n, device=dev)  # noqa: E731
    b = lambda: torch.zeros(n, device=dev, dtype=torch.bool)  # noqa: E731

    # -- the sampled world ---------------------------------------------------
    self.floor_mu = z()
    self.box_mass = z()
    self.box_mu = z()
    self.box_half = torch.zeros(n, 3, device=dev)
    self.target_xy = torch.zeros(n, 2, device=dev)
    self.patch_rect = torch.zeros(n, MAX_PATCHES, 4, device=dev)
    """(cx, cy, half_x, half_y) in world; half 0 for an unused patch."""
    self.patch_mu = z()[:, None].repeat(1, MAX_PATCHES)

    self.pins: dict[str, torch.Tensor] = {
      name: torch.full((n,), float("nan"), device=dev) for name in PINNABLE
    }
    """Per-env values the next reset uses instead of sampling; NaN = sample.
    Evaluation pins every episode's world so all conditions see the same ones."""

    self.alpha = 0.0 if self.curriculum is not None else 1.0
    self._window_success = 0
    self._window_count = 0
    self.window_success_rate = float("nan")

    # -- per-episode progress and statistics ---------------------------------
    self.prev_dist = z()
    self.dist = z()
    self.delta = z()
    """Decrease in box-to-target distance over the latest step (m)."""
    self.succeeded = b()
    self.new_success = b()
    self.success_step = z()
    self.steps = z()
    self.slip_run = torch.zeros(n, 2, device=dev)
    """Consecutive control steps each foot has been sliding."""
    self.slip_events = z()
    self.push_force_sum = z()
    self.push_steps = z()
    self.impulse = z()
    self.impulse_double_support = z()

    self.last_episode = EpisodeRecord(**{
      name: z() for name in EpisodeRecord.__dataclass_fields__
    })
    self.last_episode.valid = b()
    self._stamp = -1

  @classmethod
  def get(cls, env: ManagerBasedRlEnv) -> BoxPushState:
    state = getattr(env, cls._ATTR, None)
    if state is None:
      state = cls(env)
      setattr(env, cls._ATTR, state)
    return state

  # -- geometry --------------------------------------------------------------

  @property
  def box_pos(self) -> torch.Tensor:
    return self.box.data.root_link_pos_w

  @property
  def box_quat(self) -> torch.Tensor:
    return self.box.data.root_link_quat_w

  def pelvis_yaw_quat(self) -> torch.Tensor:
    return yaw_quat(self.robot.data.root_link_quat_w)

  def to_yaw_frame(self, vec_w: torch.Tensor) -> torch.Tensor:
    """World vector(s) [B, 3] into the pelvis yaw frame."""
    return quat_apply_inverse(self.pelvis_yaw_quat(), vec_w)

  def target_dist(self) -> torch.Tensor:
    return torch.norm(self.target_xy - self.box_pos[:, :2], dim=-1)

  def box_up_w(self) -> torch.Tensor:
    ez = torch.zeros_like(self.box_pos)
    ez[:, 2] = 1.0
    return quat_apply(self.box_quat, ez)

  def sole_point_speeds(self) -> torch.Tensor:
    """[B, 2 feet, 2 (heel, toe)] tangential (xy) speed of sole points."""
    rot = matrix_from_quat(self.robot.data.body_link_quat_w[:, self.foot_body_ids])
    v = self.robot.data.body_link_lin_vel_w[:, self.foot_body_ids]  # [B, F, 3]
    w = self.robot.data.body_link_ang_vel_w[:, self.foot_body_ids]
    pts = torch.tensor(
      [[x, 0.0, SOLE_Z] for x in HEEL_TOE_X], device=v.device, dtype=v.dtype
    )  # [2, 3]
    r = torch.einsum("bfij,pj->bfpi", rot, pts)  # [B, F, P, 3]
    vp = v[:, :, None, :] + torch.cross(w[:, :, None, :].expand_as(r), r, dim=-1)
    return torch.norm(vp[..., :2], dim=-1)

  def feet_in_contact(self) -> torch.Tensor:
    found = self.foot_contact.data.found
    assert found is not None
    return found > 0  # [B, 2]

  def horizontal_push_force(self) -> torch.Tensor:
    """[B] |F_xy| of the robot on the box (N)."""
    f = self.push_sensor.data.force
    assert f is not None
    return torch.norm(f.view(f.shape[0], -1, 3).sum(dim=1)[:, :2], dim=-1)

  def mu_under_feet(self) -> torch.Tensor:
    """[B, 2] friction of the floor surface under each foot site."""
    mu = self.floor_mu[:, None].repeat(1, 2)
    if self.patches is None:
      return mu
    feet = self.robot.data.site_pos_w[:, self.foot_site_ids, :2]  # [B, 2, 2]
    r = self.patch_rect[:, None]  # [B, 1, P, 4]
    inside = (
      ((feet[:, :, None, 0] - r[..., 0]).abs() <= r[..., 2])
      & ((feet[:, :, None, 1] - r[..., 1]).abs() <= r[..., 3])
    )  # [B, 2, P]
    patch_mu = self.patch_mu[:, None, :].expand_as(inside)
    lowest = torch.where(inside, patch_mu, torch.full_like(patch_mu, float("inf")))
    return torch.minimum(mu, lowest.amin(dim=-1))

  # -- per-step update -------------------------------------------------------

  def update(self) -> None:
    """Advance progress and statistics. Idempotent within one env step."""
    stamp = self._env.common_step_counter
    if stamp == self._stamp:
      return
    self._stamp = stamp
    dt = self._env.step_dt

    self.steps += 1
    self.dist = self.target_dist()
    self.delta = self.prev_dist - self.dist
    self.prev_dist = self.dist.clone()

    reached = self.dist <= self.ranges.target_radius
    self.new_success = reached & ~self.succeeded
    self.success_step = torch.where(self.new_success, self.steps, self.success_step)
    self.succeeded |= reached

    in_contact = self.feet_in_contact()
    sliding = in_contact & (self.sole_point_speeds().amin(dim=-1) > SLIP_SPEED)
    self.slip_run = torch.where(sliding, self.slip_run + 1, torch.zeros_like(self.slip_run))
    self.slip_events += (self.slip_run == SLIP_MIN_STEPS).sum(dim=1).float()

    force = self.horizontal_push_force()
    touching = force > 1e-3
    self.push_force_sum += force
    self.push_steps += touching.float()
    self.impulse += force * dt
    self.impulse_double_support += force * dt * in_contact.all(dim=1).float()

  # -- episode bookkeeping ---------------------------------------------------

  def _term(self, name: str, env_ids: torch.Tensor) -> torch.Tensor:
    tm = self._env.termination_manager
    if name not in tm.active_terms:
      return torch.zeros(len(env_ids), device=self._env.device, dtype=torch.bool)
    return tm.get_term(name)[env_ids].bool()

  def archive(self, env_ids: torch.Tensor) -> None:
    """Store the episode that is ending in ``last_episode``."""
    rec = self.last_episode
    dt = self._env.step_dt
    nan = torch.full((len(env_ids),), float("nan"), device=self._env.device)
    steps = self.steps[env_ids]
    rec.valid[env_ids] = steps > 0
    rec.success[env_ids] = self.succeeded[env_ids].float()
    rec.time_to_target[env_ids] = torch.where(
      self.succeeded[env_ids], self.success_step[env_ids] * dt, nan
    )
    rec.length_s[env_ids] = steps * dt
    rec.slip_events[env_ids] = self.slip_events[env_ids]
    fell = torch.zeros_like(steps, dtype=torch.bool)
    for name in FALL_TERMS:
      fell |= self._term(name, env_ids)
    rec.fell[env_ids] = fell.float()
    rec.box_tipped[env_ids] = self._term(BOX_TIPPED_TERM, env_ids).float()
    ps = self.push_steps[env_ids]
    rec.mean_push_force[env_ids] = torch.where(
      ps > 0, self.push_force_sum[env_ids] / ps.clamp_min(1), nan
    )
    imp = self.impulse[env_ids]
    rec.double_support_impulse_frac[env_ids] = torch.where(
      imp > 0, self.impulse_double_support[env_ids] / imp.clamp_min(1e-9), nan
    )
    rec.floor_mu[env_ids] = self.floor_mu[env_ids]
    rec.box_mass[env_ids] = self.box_mass[env_ids]
    rec.box_mu[env_ids] = self.box_mu[env_ids]
    rec.final_distance[env_ids] = self.dist[env_ids]

  def clear(self, env_ids: torch.Tensor) -> None:
    for buf in (
      self.delta, self.success_step, self.steps, self.slip_events,
      self.push_force_sum, self.push_steps, self.impulse, self.impulse_double_support,
    ):
      buf[env_ids] = 0.0
    self.succeeded[env_ids] = False
    self.new_success[env_ids] = False
    self.slip_run[env_ids] = 0.0


# ---------------------------------------------------------------------------
# Reset event: sample the world, write it into the model, place everything
# ---------------------------------------------------------------------------


def _uniform(lo_hi: tuple[float, float], n: int, device) -> torch.Tensor:
  lo, hi = lo_hi
  return lo + (hi - lo) * torch.rand(n, device=device)


@requires_model_fields(
  "geom_size",
  "geom_rbound",
  "geom_aabb",
  "geom_friction",
  "geom_pos",
  "body_mass",
  "body_inertia",
  "body_invweight0",
  "dof_invweight0",
  "body_subtreemass",
  recompute=RecomputeLevel.none,
)
def box_push_reset(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  ranges: BoxPushRanges,
  patches: PatchCfg | None = None,
  curriculum: CurriculumCfg | None = None,
) -> None:
  """Event (mode="reset"): archive the finished episode, sample a new world.

  The robot is placed by ``reset_base`` at the env origin facing +x; the box is
  placed ahead of it, and the target ahead of the box. ``ranges``, ``patches``
  and ``curriculum`` are read back by :class:`BoxPushState`.
  """
  del patches, curriculum  # read through the state, from these same params
  st = BoxPushState.get(env)
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  env_ids = env_ids.to(env.device, dtype=torch.long)
  st.archive(env_ids)
  st.clear(env_ids)

  n, dev = len(env_ids), env.device
  r = ranges
  cur = curriculum_ranges(r, st.alpha)

  def draw(name: str, lo_hi: tuple[float, float]) -> torch.Tensor:
    pin = st.pins[name][env_ids]
    return torch.where(pin.isnan(), _uniform(lo_hi, n, dev), pin)

  floor_mu = draw("floor_mu", cur["floor_mu"])
  box_mass = draw("box_mass", cur["box_mass"])
  box_mu = draw("box_mu", cur["box_mu"])
  half = 0.5 * torch.stack(
    [
      draw("box_depth", r.box_depth),
      draw("box_width", r.box_width),
      draw("box_height", r.box_height),
    ],
    dim=-1,
  )

  st.floor_mu[env_ids] = floor_mu
  st.box_mass[env_ids] = box_mass
  st.box_mu[env_ids] = box_mu
  st.box_half[env_ids] = half

  # -- model fields ----------------------------------------------------------
  model = env.sim.model
  g, bid = st.box_geom, st.box_body
  model.geom_size[env_ids, g] = half
  model.geom_rbound[env_ids, g] = torch.norm(half, dim=-1)
  model.geom_aabb[env_ids, g, 1] = half
  model.geom_friction[env_ids, g, 0] = box_mu
  model.geom_friction[env_ids, st.floor_geom, 0] = floor_mu
  model.body_mass[env_ids, bid] = box_mass
  sq = half.square()
  inertia = (box_mass[:, None] / 3.0) * torch.stack(
    [sq[:, 1] + sq[:, 2], sq[:, 0] + sq[:, 2], sq[:, 0] + sq[:, 1]], dim=-1
  )
  model.body_inertia[env_ids, bid] = inertia
  # The constants set_const would derive, written directly for the resetting
  # envs only. mjlab's set_const recomputes every world whenever any env
  # resets -- every step -- which measured 2.4x slower collection. For a lone
  # free body they have a closed form (checked against mujoco_warp.set_const in
  # tests/test_box_push.py): 1/m for translation, mean(1/I) for rotation.
  inv_rot = (1.0 / inertia).mean(dim=-1)
  model.body_invweight0[env_ids, bid] = torch.stack([1.0 / box_mass, inv_rot], dim=-1)
  model.dof_invweight0[env_ids[:, None], st.box_dofs[None]] = torch.cat(
    [(1.0 / box_mass)[:, None].expand(-1, 3), inv_rot[:, None].expand(-1, 3)], dim=-1
  )
  model.body_subtreemass[env_ids, bid] = box_mass
  model.body_subtreemass[env_ids, 0] = model.body_mass[env_ids, 1:].sum(dim=-1)

  # -- poses -------------------------------------------------------------------
  origin = env.scene.env_origins[env_ids]
  gap = draw("robot_gap", r.robot_gap)
  box_xy = origin[:, :2].clone()
  box_xy[:, 0] += gap + half[:, 0]
  root = torch.zeros(n, 13, device=dev)
  root[:, :2] = box_xy
  root[:, 2] = origin[:, 2] + half[:, 2] + 0.002
  root[:, 3] = 1.0
  st.box.write_root_state_to_sim(root, env_ids=env_ids)

  dist = draw("target_distance", r.target_distance)
  ang = draw("target_angle", r.target_angle)
  target_xy = box_xy + dist[:, None] * torch.stack([ang.cos(), ang.sin()], dim=-1)
  st.target_xy[env_ids] = target_xy
  st.prev_dist[env_ids] = dist
  st.dist[env_ids] = dist
  mocap = torch.zeros(n, 7, device=dev)
  mocap[:, :2] = target_xy
  mocap[:, 2] = origin[:, 2]
  mocap[:, 3] = 1.0
  st.target.write_mocap_pose_to_sim(mocap, env_ids=env_ids)

  if st.patches is not None:
    _place_patches(env, st, env_ids, origin, target_xy)


def _place_patches(
  env: ManagerBasedRlEnv,
  st: BoxPushState,
  env_ids: torch.Tensor,
  origin: torch.Tensor,
  target_xy: torch.Tensor,
) -> None:
  p = st.patches
  assert p is not None
  n, dev = len(env_ids), env.device
  model = env.sim.model
  count = torch.randint(p.count[0], p.count[1] + 1, (n,), device=dev)
  start = origin[:, :2]
  seg = target_xy - start
  seg_len = torch.norm(seg, dim=-1, keepdim=True).clamp_min(1e-6)
  along, across = seg / seg_len, torch.stack([-seg[:, 1], seg[:, 0]], -1) / seg_len
  for i, gid in enumerate(st.patch_geoms):
    used = i < count
    t = _uniform(p.path_fraction, n, dev)[:, None]
    lat = _uniform((-p.lateral, p.lateral), n, dev)[:, None]
    centre = start + t * seg_len * along + lat * across
    half_xy = 0.5 * torch.stack(
      [_uniform(p.size, n, dev), _uniform(p.size, n, dev)], dim=-1
    )
    half_z = torch.full((n,), 0.5 * p.thickness, device=dev)
    mu = _uniform(p.mu, n, dev)
    pos = torch.zeros(n, 3, device=dev)
    pos[:, :2] = centre
    pos[:, 2] = torch.where(
      used, origin[:, 2] + PATCH_LIFT - half_z, torch.full_like(half_z, -5.0)
    )
    half = torch.cat([half_xy, half_z[:, None]], dim=-1)
    model.geom_pos[env_ids, gid] = pos
    # Patches are static (world-body) geoms, whose world pose mujoco_warp
    # computes once and never again: geom_pos alone would not move them. Their
    # pose is written straight into the data, where nothing overwrites it
    # (tests/test_box_push.py checks a foot on a moved patch gets its friction).
    env.sim.data.geom_xpos[env_ids, gid] = pos
    model.geom_size[env_ids, gid] = half
    model.geom_rbound[env_ids, gid] = torch.norm(half, dim=-1)
    model.geom_aabb[env_ids, gid, 1] = half
    model.geom_friction[env_ids, gid, 0] = mu
    st.patch_rect[env_ids, i] = torch.cat(
      [centre, half_xy * used[:, None].float()], dim=-1
    )
    st.patch_mu[env_ids, i] = mu


# ---------------------------------------------------------------------------
# Rewards
# ---------------------------------------------------------------------------


def box_progress(env: ManagerBasedRlEnv, max_speed: float = 0.5) -> torch.Tensor:
  """Decrease in box-to-target distance, as a speed normalised to [-1, 1].

  Signed: pushing the box away from the target costs as much as pushing it
  toward pays. The cap removes any incentive to shove or kick the box fast.
  """
  st = BoxPushState.get(env)
  st.update()
  return (st.delta / env.step_dt).clamp(-max_speed, max_speed) / max_speed


def box_success_bonus(env: ManagerBasedRlEnv) -> torch.Tensor:
  """One-off, on the step the box centre enters the target disc. Divided by
  ``step_dt`` so the term's weight is the bonus itself."""
  st = BoxPushState.get(env)
  st.update()
  return st.new_success.float() / env.step_dt


def hand_contact_on_push_face(
  env: ManagerBasedRlEnv, max_angle_deg: float = 45.0
) -> torch.Tensor:
  """1 while any hand/forearm touches the box's push face, else 0.

  The push face is the side face whose outward normal points AWAY from the
  target: pushing on it drives the box toward the target. A side face qualifies
  when its outward normal is within ``max_angle_deg`` of -(target - box).
  Contacts on the top face never count.
  """
  st = BoxPushState.get(env)
  data = env.scene[HAND_BOX_SENSOR].data
  assert data.found is not None and data.pos is not None
  found = data.found > 0  # [B, H]
  pos = data.pos.view(found.shape[0], found.shape[1], 3)
  rel_b = quat_apply_inverse(
    st.box_quat[:, None].expand(-1, found.shape[1], -1),
    pos - st.box_pos[:, None],
  )
  scaled = rel_b / st.box_half[:, None].clamp_min(1e-6)
  axis = scaled.abs().argmax(dim=-1)  # [B, H]
  side = axis < 2
  normal_b = torch.zeros_like(rel_b)
  normal_b.scatter_(-1, axis[..., None], torch.gather(scaled, -1, axis[..., None]).sign())
  normal_w = quat_apply(
    st.box_quat[:, None].expand(-1, found.shape[1], -1), normal_b
  )[..., :2]
  away = st.box_pos[:, :2] - st.target_xy
  away = away / torch.norm(away, dim=-1, keepdim=True).clamp_min(1e-6)
  facing = (normal_w * away[:, None]).sum(dim=-1) >= math.cos(math.radians(max_angle_deg))
  return (found & side & facing).any(dim=1).float()


def stay_near_box(env: ManagerBasedRlEnv, max_distance: float = 1.5) -> torch.Tensor:
  """Metres the pelvis is beyond ``max_distance`` (xy) from the box centre."""
  st = BoxPushState.get(env)
  d = torch.norm(st.robot.data.root_link_pos_w[:, :2] - st.box_pos[:, :2], dim=-1)
  return (d - max_distance).clamp_min(0.0)


def feet_slip(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Sum over feet in contact of the squared tangential sole-site speed.

  The velocity task's ``feet_slip`` without its command gate (there is no
  command here): same sensor, same sites, same squared-speed kernel.
  """
  st = BoxPushState.get(env)
  in_contact = st.feet_in_contact().float()
  v = st.robot.data.site_lin_vel_w[:, st.foot_site_ids, :2]
  return (torch.square(v).sum(dim=-1) * in_contact).sum(dim=1)


def terminated_by(
  env: ManagerBasedRlEnv, term_names: tuple[str, ...] = FALL_TERMS
) -> torch.Tensor:
  """One-off, on the step any of ``term_names`` ends the episode (1 / step_dt)."""
  tm = env.termination_manager
  out = torch.zeros(env.num_envs, device=env.device)
  for name in term_names:
    if name in tm.active_terms:
      out = torch.maximum(out, tm.get_term(name).float())
  return out / env.step_dt


def fell(env: ManagerBasedRlEnv) -> torch.Tensor:
  """One-off, on the step a fall terminates the episode (1 / step_dt)."""
  return terminated_by(env, FALL_TERMS)


def feet_box_contact(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Number of feet touching the box."""
  found = env.scene[FEET_BOX_SENSOR].data.found
  assert found is not None
  return (found > 0).float().sum(dim=1)


def box_tilt(env: ManagerBasedRlEnv) -> torch.Tensor:
  """sin^2 of the box's tilt from upright (0 upright, 1 on its side)."""
  up = BoxPushState.get(env).box_up_w()
  return 1.0 - up[:, 2].clamp(-1.0, 1.0).square()


# ---------------------------------------------------------------------------
# Terminations
# ---------------------------------------------------------------------------


def box_reached_target(env: ManagerBasedRlEnv) -> torch.Tensor:
  st = BoxPushState.get(env)
  st.update()
  return st.succeeded.clone()


def box_tipped_over(env: ManagerBasedRlEnv, limit_angle: float) -> torch.Tensor:
  up = BoxPushState.get(env).box_up_w()
  return up[:, 2] < math.cos(limit_angle)


# ---------------------------------------------------------------------------
# Curriculum
# ---------------------------------------------------------------------------


def box_push_levels(
  env: ManagerBasedRlEnv, env_ids: torch.Tensor | None
) -> dict[str, torch.Tensor | float]:
  """Move the global difficulty ``alpha`` by the success rate of finished
  episodes, pooled over ``CurriculumCfg.window`` episodes at a time.

  Runs before the reset events, so ``succeeded`` still describes the episode
  that is ending.
  """
  st = BoxPushState.get(env)
  cfg = st.curriculum
  if cfg is not None and env_ids is not None and env.common_step_counter > 0:
    st._window_success += int(st.succeeded[env_ids].sum().item())
    st._window_count += int(len(env_ids))
    if st._window_count >= cfg.window:
      rate = st._window_success / st._window_count
      st.window_success_rate = rate
      if rate >= cfg.promote:
        st.alpha = min(1.0, st.alpha + cfg.step)
      elif rate < cfg.demote:
        st.alpha = max(0.0, st.alpha - cfg.step)
      st._window_success = st._window_count = 0
  return {"alpha": st.alpha, "window_success_rate": st.window_success_rate}


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------
#
# Actor ("perceived"): assumed to come from an external perception stack.


def box_pos_b(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Box centre in the pelvis frame (m) [3]."""
  st = BoxPushState.get(env)
  return quat_apply_inverse(
    st.robot.data.root_link_quat_w, st.box_pos - st.robot.data.root_link_pos_w
  )


def _yaw(q: torch.Tensor) -> torch.Tensor:
  w, x, y, z = q.unbind(-1)
  return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def box_yaw_b(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Box yaw relative to the pelvis heading, as (sin, cos) [2]."""
  st = BoxPushState.get(env)
  err = _yaw(st.box_quat) - _yaw(st.robot.data.root_link_quat_w)
  return torch.stack([err.sin(), err.cos()], dim=-1)


def target_rel_box(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Target centre minus box centre, xy in the pelvis yaw frame (m) [2]."""
  st = BoxPushState.get(env)
  rel = torch.zeros_like(st.box_pos)
  rel[:, :2] = st.target_xy - st.box_pos[:, :2]
  return st.to_yaw_frame(rel)[:, :2]


# Critic (privileged).


def floor_mu_under_feet(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Floor friction under each foot [2]: the episode's floor mu, or a patch's."""
  return BoxPushState.get(env).mu_under_feet()


def box_mass(env: ManagerBasedRlEnv) -> torch.Tensor:
  return BoxPushState.get(env).box_mass[:, None]


def box_mu(env: ManagerBasedRlEnv) -> torch.Tensor:
  return BoxPushState.get(env).box_mu[:, None]


def box_size(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Full box extents (depth, width, height) in m [3]."""
  return 2.0 * BoxPushState.get(env).box_half


def box_lin_vel(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Box linear velocity in the pelvis yaw frame [3]."""
  st = BoxPushState.get(env)
  return st.to_yaw_frame(st.box.data.root_link_lin_vel_w)


def box_ang_vel(env: ManagerBasedRlEnv) -> torch.Tensor:
  st = BoxPushState.get(env)
  return st.to_yaw_frame(st.box.data.root_link_ang_vel_w)


def box_up(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Box z axis in the world [3]: tipping state."""
  return BoxPushState.get(env).box_up_w()


def foot_slip_vel(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Per-foot sole-site xy velocity while in contact, pelvis yaw frame [4]."""
  st = BoxPushState.get(env)
  v = st.robot.data.site_lin_vel_w[:, st.foot_site_ids]  # [B, 2, 3]
  q = st.pelvis_yaw_quat()[:, None].expand(-1, 2, -1)
  v_y = quat_apply_inverse(q, v)[..., :2] * st.feet_in_contact()[..., None].float()
  return v_y.reshape(v.shape[0], 4)


def push_force(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Net robot-on-box contact force, pelvis yaw frame, in units of 100 N [3]."""
  st = BoxPushState.get(env)
  f = st.push_sensor.data.force
  assert f is not None
  return st.to_yaw_frame(f.view(f.shape[0], -1, 3).sum(dim=1)) / 100.0


# ---------------------------------------------------------------------------
# Training-time episode metrics (reduce="last": the value at the final step)
# ---------------------------------------------------------------------------


def metric_success(env: ManagerBasedRlEnv) -> torch.Tensor:
  return BoxPushState.get(env).succeeded.float()


def metric_time_to_target(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Time to target, or the episode length so far if not (yet) reached."""
  st = BoxPushState.get(env)
  t = torch.where(st.succeeded, st.success_step, st.steps)
  return t * env.step_dt


def metric_slip_events(env: ManagerBasedRlEnv) -> torch.Tensor:
  return BoxPushState.get(env).slip_events.clone()


def metric_fell(env: ManagerBasedRlEnv) -> torch.Tensor:
  return fell(env) * env.step_dt


def metric_box_tipped(env: ManagerBasedRlEnv) -> torch.Tensor:
  tm = env.termination_manager
  return tm.get_term(BOX_TIPPED_TERM).float()


def metric_mean_push_force(env: ManagerBasedRlEnv) -> torch.Tensor:
  st = BoxPushState.get(env)
  return st.push_force_sum / st.push_steps.clamp_min(1.0)


def metric_double_support_impulse_frac(env: ManagerBasedRlEnv) -> torch.Tensor:
  """0 when the robot never touched the box (the eval CSV records NaN)."""
  st = BoxPushState.get(env)
  return st.impulse_double_support / st.impulse.clamp_min(1e-9)


def metric_floor_mu(env: ManagerBasedRlEnv) -> torch.Tensor:
  return BoxPushState.get(env).floor_mu.clone()


def metric_box_mass(env: ManagerBasedRlEnv) -> torch.Tensor:
  return BoxPushState.get(env).box_mass.clone()


def metric_box_mu(env: ManagerBasedRlEnv) -> torch.Tensor:
  return BoxPushState.get(env).box_mu.clone()


def metric_box_to_target(env: ManagerBasedRlEnv) -> torch.Tensor:
  return BoxPushState.get(env).dist.clone()
