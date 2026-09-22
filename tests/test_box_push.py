"""The box-pushing task's physics claims, checked against the simulator.

Each slow test builds a small env and checks one claim the task's design leans
on: the friction each contact pair actually gets, the analytic stand-in for
set_const, the box resting on the floor at every sampled size, and pinning.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest
import torch

from sendg1.common.box_push.params import (
  G1_MASS_KG,
  BoxPushRanges,
  curriculum_ranges,
  tips_before_sliding,
  traction_margin,
)

TASK = "Sendg1-BoxPush-G1-Blind"


def test_g1_mass_constant_matches_model() -> None:
  from mjlab.entity import Entity

  from sendg1.common.robots.g1 import get_g1_cfg

  model = Entity(get_g1_cfg()).spec.compile()
  assert float(np.sum(model.body_mass)) == pytest.approx(G1_MASS_KG, abs=0.01)


def test_feasibility_helpers() -> None:
  # 0.5 * 33.34 = 16.67 kg-equivalent of traction; a 0.5-mu 20 kg box needs 10.
  assert traction_margin(np.array(0.5), np.array(20.0), np.array(0.5)) == pytest.approx(
    1 - 10 / 16.67, abs=1e-3
  )
  assert traction_margin(np.array(0.2), np.array(40.0), np.array(0.8)) < 0
  # d / (2 mu) = 0.5 / 1.0: a push above 0.5 m tips first.
  assert tips_before_sliding(np.array(0.6), np.array(0.5), np.array(0.5))
  assert not tips_before_sliding(np.array(0.4), np.array(0.5), np.array(0.5))


def test_curriculum_ranges_span_easy_corner_to_full() -> None:
  """alpha 0: a comfortable margin on the grippiest floor. alpha 1: everything."""
  r = BoxPushRanges()
  easy = curriculum_ranges(r, 0.0)
  assert easy["traction_margin"] == (r.traction_margin[1], r.traction_margin[1])
  assert easy["floor_mu"] == (r.floor_mu[1], r.floor_mu[1])
  full = curriculum_ranges(r, 1.0)
  expected = {
    "traction_margin": r.traction_margin,
    "floor_mu": r.floor_mu,
    "box_mu": r.box_mu,
  }
  assert full.keys() == expected.keys()
  for key, (lo, hi) in expected.items():  # the arithmetic leaves float dust
    assert full[key] == (pytest.approx(lo), pytest.approx(hi))


def test_mass_follows_the_traction_margin() -> None:
  """A zero-margin episode needs exactly the robot's available traction."""
  import numpy as np

  from sendg1.common.box_push.params import mass_from_margin

  mu_f, mu_b = np.array(0.5), np.array(0.5)
  m = mass_from_margin(np.array(0.0), mu_f, mu_b, (1.0, 1e6))
  assert float(mu_b * m) == pytest.approx(float(mu_f * G1_MASS_KG), rel=1e-6)
  assert traction_margin(mu_f, m, mu_b) == pytest.approx(0.0, abs=1e-9)
  # Half the margin, half the box.
  m2 = mass_from_margin(np.array(0.5), mu_f, mu_b, (1.0, 1e6))
  assert float(m2) == pytest.approx(float(m) * 0.5, rel=1e-6)


# ---------------------------------------------------------------------------
# Simulation checks
# ---------------------------------------------------------------------------


def _make_env(num_envs: int = 4, pop_terminations: tuple[str, ...] = (), **kwargs):
  from mjlab.envs import ManagerBasedRlEnv

  import sendg1.tasks  # noqa: F401
  from sendg1.tasks.box_push.env_cfg import make_g1_box_push_env_cfg

  cfg = make_g1_box_push_env_cfg("blind", num_envs=num_envs, **kwargs)
  for name in pop_terminations:
    cfg.terminations.pop(name)
  cfg.seed = 0
  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  env.reset()
  return env


def _zero(env) -> torch.Tensor:
  return torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)


def _contact_friction(env) -> dict[tuple[int, str, str], float]:
  """(world, geom a, geom b) -> sliding friction the solver used."""
  wd = env.sim.wp_data
  n = int(wd.nacon.numpy()[0])
  geom = wd.contact.geom.numpy()[:n]
  fr = wd.contact.friction.numpy()[:n, 0]
  wid = wd.contact.worldid.numpy()[:n]
  name = lambda g: mujoco.mj_id2name(env.sim.mj_model, mujoco.mjtObj.mjOBJ_GEOM, int(g))  # noqa: E731
  return {(int(w), name(a), name(b)): float(f) for (a, b), f, w in zip(geom, fr, wid)}


@pytest.mark.slow
def test_contact_friction_takes_the_sampled_values() -> None:
  from sendg1.common.box_push.mdp import BoxPushState

  env = _make_env()
  try:
    st = BoxPushState.get(env)
    for _ in range(5):
      env.step(_zero(env))
    seen_foot = seen_box = 0
    for (w, a, b), mu in _contact_friction(env).items():
      pair = {a, b}
      if "terrain" in pair and any("foot" in g for g in pair):
        assert mu == pytest.approx(float(st.floor_mu[w]), abs=1e-5), (a, b)
        seen_foot += 1
      if pair == {"terrain", "box/box_geom"}:
        assert mu == pytest.approx(float(st.box_mu[w]), abs=1e-5)
        seen_box += 1
    assert seen_foot > 0 and seen_box == env.num_envs
  finally:
    env.close()


@pytest.mark.slow
def test_analytic_constants_match_set_const() -> None:
  from mjlab.managers.event_manager import RecomputeLevel

  from sendg1.common.box_push.mdp import BoxPushState

  env = _make_env()
  try:
    st = BoxPushState.get(env)
    m = env.sim.model
    fields = ("body_invweight0", "dof_invweight0", "body_subtreemass")
    ours = {f: getattr(m, f).clone() for f in fields}
    env.sim.recompute_constants(RecomputeLevel.set_const)
    for f in fields:
      theirs = torch.as_tensor(getattr(m, f)[:])
      torch.testing.assert_close(ours[f], theirs, rtol=1e-5, atol=1e-5)
    # And the box's values really were re-derived for the sampled mass.
    assert torch.allclose(
      m.body_invweight0[:, st.box_body, 0], 1.0 / st.box_mass, rtol=1e-5
    )
  finally:
    env.close()


@pytest.mark.slow
def test_box_rests_on_the_floor_at_its_sampled_size() -> None:
  from sendg1.common.box_push.mdp import BoxPushState

  env = _make_env(num_envs=8)
  try:
    st = BoxPushState.get(env)
    for _ in range(10):
      env.step(_zero(env))
    box = env.scene["box"]
    z = box.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
    torch.testing.assert_close(z, st.box_half[:, 2], atol=2e-3, rtol=0)
    assert (box.data.root_link_lin_vel_w.norm(dim=-1) < 0.05).all()
  finally:
    env.close()


@pytest.mark.slow
def test_pins_override_sampling() -> None:
  from sendg1.common.box_push.mdp import BoxPushState

  env = _make_env()
  try:
    st = BoxPushState.get(env)
    st.pins["floor_mu"][:] = torch.tensor([0.2, 0.4, 0.6, 1.0], device=env.device)
    st.pins["box_mass"][:] = 30.0
    st.pins["target_angle"][:] = 0.0
    st.pins["target_distance"][:] = 3.0
    env.reset()
    torch.testing.assert_close(st.floor_mu, st.pins["floor_mu"])
    assert torch.all(st.box_mass == 30.0)
    assert torch.all(env.sim.model.body_mass[:, st.box_body] == 30.0)
    box_xy = env.scene["box"].data.root_link_pos_w[:, :2]
    torch.testing.assert_close(st.target_xy[:, 1], box_xy[:, 1], atol=1e-5, rtol=0)
    torch.testing.assert_close(st.target_xy[:, 0] - box_xy[:, 0], torch.full_like(box_xy[:, 0], 3.0))
  finally:
    env.close()


@pytest.mark.slow
def test_patch_friction_applies_to_feet_on_a_patch() -> None:
  from sendg1.common.box_push.mdp import BoxPushState
  from sendg1.common.box_push.scene import patch_geom_name

  env = _make_env(low_friction_patches=True)
  try:
    st = BoxPushState.get(env)
    m = env.sim.model
    # Slide patch 0 under world 0's robot.
    g0 = st.patch_geoms[0]
    origin = env.scene.env_origins[0]
    pos = torch.tensor([origin[0], origin[1], origin[2] + 0.001 - 0.005], device=env.device)
    m.geom_pos[0, g0] = pos
    env.sim.data.geom_xpos[0, g0] = pos  # as box_push_reset does
    m.geom_size[0, g0] = torch.tensor([0.5, 0.5, 0.005], device=env.device)
    m.geom_aabb[0, g0, 1] = m.geom_size[0, g0]
    m.geom_rbound[0, g0] = m.geom_size[0, g0].norm()
    m.geom_friction[0, g0, 0] = 0.123
    for _ in range(5):
      env.step(_zero(env))
    on_patch = [
      mu for (w, a, b), mu in _contact_friction(env).items()
      if w == 0 and patch_geom_name(0) in (a, b) and ("foot" in a or "foot" in b)
    ]
    assert on_patch and all(mu == pytest.approx(0.123, abs=1e-5) for mu in on_patch)
    # The foot contact sensor, which matches the terrain body, sees the patch.
    assert bool(st.feet_in_contact()[0].any())
    # And the pose survives steps and a reset of another world.
    env.reset(env_ids=torch.tensor([1], device=env.device))
    for _ in range(3):
      env.step(_zero(env))
    torch.testing.assert_close(env.sim.data.geom_xpos[0, g0], pos)
  finally:
    env.close()


@pytest.mark.slow
def test_reset_places_patches_on_the_path() -> None:
  from sendg1.common.box_push.mdp import BoxPushState

  env = _make_env(num_envs=8, low_friction_patches=True)
  try:
    st = BoxPushState.get(env)
    xpos = env.sim.data.geom_xpos
    for i, gid in enumerate(st.patch_geoms):
      used = st.patch_rect[:, i, 2] > 0
      torch.testing.assert_close(xpos[used, gid, :2], st.patch_rect[used, i, :2])
      assert torch.all(xpos[~used, gid, 2] < -1.0)
    assert torch.all((st.patch_rect[:, :, 2] > 0).sum(dim=1) >= 1)
  finally:
    env.close()


def test_ball_hands_model() -> None:
  from mjlab.entity import Entity

  from sendg1.common.robots.g1 import (
    BALL_HAND_FRICTION,
    BALL_HAND_PRIORITY,
    BALL_HAND_RADIUS,
    get_g1_ball_hands_cfg,
    get_g1_cfg,
  )

  stock = Entity(get_g1_cfg()).spec.compile()
  ball = Entity(get_g1_ball_hands_cfg()).spec.compile()
  for side in ("left", "right"):
    g = ball.geom(f"{side}_hand_collision")
    assert g.type[0] == mujoco.mjtGeom.mjGEOM_SPHERE
    assert g.size[0] == pytest.approx(BALL_HAND_RADIUS)
    assert g.priority[0] == BALL_HAND_PRIORITY
    assert g.condim[0] == 3
    assert g.friction[0] == pytest.approx(BALL_HAND_FRICTION)
  meshes = lambda m: {  # noqa: E731
    mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, int(m.geom_dataid[i]))
    for i in range(m.ngeom)
    if m.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH
  }
  assert not {n for n in meshes(ball) if "rubber_hand" in n}
  # Only the hands changed: same mass, same number of collision geoms, and the
  # stock model still has its rubber hands.
  assert float(ball.body_mass.sum()) == pytest.approx(float(stock.body_mass.sum()))
  assert {n for n in meshes(stock) if "rubber_hand" in n}
  assert ball.nbody == stock.nbody


@pytest.mark.slow
def test_ball_hand_contact_uses_ball_friction() -> None:
  from sendg1.common.box_push.mdp import BoxPushState
  from sendg1.common.robots.g1 import BALL_HAND_FRICTION, BALL_HAND_RADIUS

  env = _make_env(num_envs=2)
  try:
    st = BoxPushState.get(env)
    robot, box = env.scene["robot"], env.scene["box"]
    # Put the box's near face 5 mm into the left ball hand, then step once.
    gid = int(robot.indexing.geom_ids[robot.geom_names.index("left_hand_collision")])
    ball = env.sim.data.geom_xpos[:, gid]
    root = torch.zeros(2, 13, device=env.device)
    root[:, 0] = ball[:, 0] + BALL_HAND_RADIUS - 0.005 + st.box_half[:, 0]
    root[:, 1] = ball[:, 1]
    root[:, 2] = st.box_half[:, 2] + 0.002
    root[:, 3] = 1.0
    box.write_root_state_to_sim(root)
    env.step(_zero(env))
    hand = [
      mu for (w, a, b), mu in _contact_friction(env).items()
      if {a, b} == {"robot/left_hand_collision", "box/box_geom"}
    ]
    assert hand and all(mu == pytest.approx(BALL_HAND_FRICTION, abs=1e-5) for mu in hand)
  finally:
    env.close()


@pytest.mark.slow
def test_pilot_pushes_when_behind_and_circles_when_not() -> None:
  """The scripted high level's two phases, checked geometrically.

  Env 0 stands behind the box (the spawn), env 1 is teleported past the box,
  between it and the target -- the wrong side. Behind: drive forward. Wrong
  side: do not drive at the box, move around it.
  """
  from sendg1.common.box_push.mdp import BoxPushState

  env = _make_env(num_envs=2)
  try:
    st = BoxPushState.get(env)
    st.pins["target_angle"][:] = 0.0
    st.pins["target_distance"][:] = 3.0
    env.reset()
    robot = env.scene["robot"]
    root = robot.data.root_link_pos_w.clone()
    quat = robot.data.root_link_quat_w.clone()
    # Env 1: 0.8 m beyond the box, on the far side, still facing +x.
    root[1, 0] = st.box_pos[1, 0] + st.box_half[1, 0] + 0.8
    state = torch.cat([root, quat, torch.zeros(2, 6, device=env.device)], dim=-1)
    robot.write_root_state_to_sim(state)
    env.step(_zero(env))

    cmd = st.commanded()
    yaw = robot.data.heading_w
    # Command back into the world frame.
    c, s = torch.cos(yaw), torch.sin(yaw)
    vx_w = c * cmd[:, 0] - s * cmd[:, 1]
    vy_w = s * cmd[:, 0] + c * cmd[:, 1]
    vel_w = torch.stack([vx_w, vy_w], dim=-1)

    # Behind the box: push toward the target (+x).
    assert float(cmd[0, 0]) > 0.2, cmd[0]
    assert float(vel_w[0, 0]) > 0.2

    # Wrong side: circle the box rather than walk at it. The arc also closes
    # distance, so some inward component is expected -- but the sideways
    # component must dominate.
    to_box = st.box_pos[1, :2] - robot.data.root_link_pos_w[1, :2]
    to_box = to_box / to_box.norm()
    direction = vel_w[1] / vel_w[1].norm()
    inward = float(direction @ to_box)
    sideways = float((direction - inward * to_box).norm())
    assert inward < 0.75 and sideways > inward, (vel_w[1], inward, sideways)
  finally:
    env.close()


@pytest.mark.slow
def test_success_needs_the_dwell() -> None:
  """A box inside the disc for one step is not yet a success."""
  from sendg1.common.box_push.mdp import BoxPushState

  # The robot falls over doing nothing, which would reset the episode and the
  # dwell; this test is about the dwell alone.
  env = _make_env(num_envs=2, pop_terminations=("fell_over", "base_too_low"))
  try:
    st = BoxPushState.get(env)
    env.reset()
    box = env.scene["box"]

    def park_box_on_target() -> None:
      root = torch.cat(
        [box.data.root_link_pos_w, box.data.root_link_quat_w,
         torch.zeros(2, 6, device=env.device)], dim=-1
      )
      root[:, :2] = st.target_xy
      box.write_root_state_to_sim(root)

    park_box_on_target()
    env.step(_zero(env))
    assert not bool(st.succeeded.any()), "one step inside the disc must not count"
    dwell_steps = round(st.ranges.success_dwell_s / env.step_dt)
    for _ in range(dwell_steps + 1):
      park_box_on_target()
      env.step(_zero(env))
    # Completing the dwell fires the success termination, which resets the env
    # and archives the episode -- so the record, not the live flag, is the proof.
    assert bool(st.last_episode.success.all()), "should succeed once the dwell elapses"
    expected_s = st.ranges.success_dwell_s
    assert float(st.last_episode.time_to_target[0]) == pytest.approx(
      expected_s, abs=2 * env.step_dt
    )
  finally:
    env.close()
