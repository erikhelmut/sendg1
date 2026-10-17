"""Beam task: no privileged leak into the actor, terrain and geometry (straight
and bent beams), tactile canonicalisation, mirror symmetry, and envs that
build and step."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

import sendg1.tasks  # noqa: F401
from sendg1.common.beam import tactile as tac
from sendg1.common.beam.geometry import project_points
from sendg1.common.beam.terms import DEPLOYABLE, TERMS
from sendg1.common.beam.terrain import (
  CURVED_TILE_SIZE,
  BeamTileCfg,
  make_beam_terrain_cfg,
  resample_polyline,
)
from sendg1.common.tactile.layout import LAYOUTS
from sendg1.tasks.beam import (
  CURVED_TASK,
  TASK,
  TOUCH_PLAIN_TASK,
  TOUCH_TASK,
  assert_actor_is_deployable,
)

ACTOR_EXPECTED = {
  "base_ang_vel",
  "projected_gravity",
  "joint_pos",
  "joint_vel",
  "actions",
  "command_vx",
  "tactile_map",
}
ACTOR_DIM = 3 + 3 + 29 + 29 + 29 + 1 + 2 * 33
TOUCH_ACTOR_DIM = ACTOR_DIM - 2 * 33 + 2 * 31
ALL_TASKS = [TASK, CURVED_TASK, TOUCH_TASK, TOUCH_PLAIN_TASK]
TOUCH_TASKS = (TOUCH_TASK, TOUCH_PLAIN_TASK)


def _actor_expected(task: str) -> tuple[set[str], int]:
  if task in TOUCH_TASKS:
    return (ACTOR_EXPECTED - {"tactile_map"}) | {"tactile_touch"}, TOUCH_ACTOR_DIM
  return ACTOR_EXPECTED, ACTOR_DIM


@pytest.mark.parametrize("task", ALL_TASKS)
def test_actor_sees_only_deployable_signals(task: str) -> None:
  env_cfg = load_env_cfg(task)
  rl_cfg = load_rl_cfg(task)
  terms = {t for g in rl_cfg.obs_groups["actor"] for t in env_cfg.observations[g].terms}
  assert terms == _actor_expected(task)[0]
  assert terms <= DEPLOYABLE
  critic = {t for g in rl_cfg.obs_groups["critic"] for t in env_cfg.observations[g].terms}
  assert {"base_beam_pose", "feet_beam_pose", "base_lin_vel"} <= critic
  assert ("beam_ahead" in critic) == (task != TASK)
  touch_critic = {"tactile_touch_clean", "taxel_forces_clean", "foot_slip_vel", "beam_params"}
  assert (touch_critic <= critic) == (task in TOUCH_TASKS)


def test_touch_tasks_differ_only_in_the_actor_model() -> None:
  sym, plain = load_rl_cfg(TOUCH_TASK), load_rl_cfg(TOUCH_PLAIN_TASK)
  assert sym.actor.class_name.endswith("SymmetricRNNModel")
  assert plain.actor.class_name == "RNNModel"
  plain.actor.class_name = sym.actor.class_name
  assert plain.actor == sym.actor and plain.algorithm == sym.algorithm and plain.critic == sym.critic
  assert load_env_cfg(TOUCH_TASK).rewards.keys() == load_env_cfg(CURVED_TASK).rewards.keys()


def test_touch_features() -> None:
  """Bits are binary, area is their mean, CoP is zero on an unloaded foot and
  lateral (+y) on both feet for a lateral-edge load."""
  layout = LAYOUTS["slipper_v1"]
  t = layout.num_taxels
  lateral = torch.tensor([float(y > 0.01) for _, y in layout.centers])
  normal = torch.zeros(2, 2, t)
  normal[0, 0] = 40.0 * lateral
  normal[0, 1] = 40.0 * lateral
  normal[1, 0] = 3.0  # below the bit threshold everywhere, 78 N in total
  f = tac.touch_features(normal, layout).view(2, 2, tac.TOUCH_PER_FOOT)
  bits, load, cop, area, contact = f[..., :t], f[..., t], f[..., t + 1 : t + 3], f[..., t + 3], f[..., t + 4]
  assert set(bits.unique().tolist()) <= {0.0, 1.0}
  torch.testing.assert_close(area, bits.mean(-1))
  torch.testing.assert_close(f[0, 0], f[0, 1])
  assert cop[0, 0, 1] > 0.5 and contact[0, 0] == 1
  assert torch.all(cop[1, 1] == 0) and contact[1, 1] == 0 and load[1, 1] == 0
  assert bits[1, 0].sum() == 0 and contact[1, 0] == 1


def test_leak_guard_fires() -> None:
  env_cfg = load_env_cfg(TASK)
  with pytest.raises(ValueError, match="privileged"):
    assert_actor_is_deployable(env_cfg.observations, ("critic",))
  env_cfg.observations["actor"].terms["feet_beam_pose"] = TERMS["feet_beam_pose"].build()
  with pytest.raises(ValueError, match="privileged"):
    assert_actor_is_deployable(env_cfg.observations)


def test_curved_task_keeps_the_straight_actor() -> None:
  """Same actor inputs and network, so a straight-beam actor loads into the
  curved task."""
  a, b = load_env_cfg(TASK), load_env_cfg(CURVED_TASK)
  assert list(a.observations["actor"].terms) == list(b.observations["actor"].terms)
  assert load_rl_cfg(TASK).actor == load_rl_cfg(CURVED_TASK).actor


def test_play_cfg_has_no_curriculum_or_state_bank() -> None:
  cfg = load_env_cfg(TASK, play=True)
  assert cfg.curriculum == {}
  assert "spawn_from_bank" not in cfg.events
  assert "spawn_from_bank" in load_env_cfg(TASK).events


# -- terrain -----------------------------------------------------------------


def test_resample_keeps_every_vertex() -> None:
  pts = np.array([[0.0, 0.0], [1.0, 0.0], [1.5, 0.8], [3.0, 1.0]])
  out = resample_polyline(pts, 20)
  assert out.shape == (20, 2)
  for p in pts:
    assert np.min(np.linalg.norm(out - p, axis=1)) < 1e-9


@pytest.mark.parametrize("turn", [25.0, 60.0])
def test_bent_beam_bends_stays_in_tile_and_never_doubles_back(turn: float) -> None:
  cfg = BeamTileCfg(size=CURVED_TILE_SIZE, path_kind="bent", max_turn_deg=turn)
  rng = np.random.default_rng(0)
  bent = 0
  for _ in range(50):
    pts = cfg.beam_polyline(rng)
    seg = np.diff(pts, axis=0)
    assert math.isclose(np.linalg.norm(seg, axis=1).sum(), cfg.beam_length, rel_tol=1e-6)
    heading = np.degrees(np.arctan2(seg[:, 1], seg[:, 0]))
    assert np.all(np.abs(heading) <= cfg.max_heading_deg + 1e-6)
    assert np.all(np.abs(np.diff(heading)) <= turn + 1e-6)
    assert np.all(pts[:, 1] >= cfg.side_margin)
    assert np.all(pts[:, 1] <= CURVED_TILE_SIZE[1] - cfg.side_margin)
    bent += int(np.abs(heading).max() > 1.0)
  assert bent >= 40


def test_curved_terrain_columns() -> None:
  straight = make_beam_terrain_cfg()
  curved = make_beam_terrain_cfg(curves=True)
  assert len(curved.sub_terrains) == 3 * len(straight.sub_terrains)
  assert curved.size == CURVED_TILE_SIZE
  kinds = [c.path_kind for c in curved.sub_terrains.values()]
  assert kinds.count("straight") == len(straight.sub_terrains)
  turns = {c.max_turn_deg for c in curved.sub_terrains.values() if c.path_kind == "bent"}
  assert turns == {25.0, 60.0}


def test_projection_on_straight_path() -> None:
  path = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]])
  pts = torch.tensor([[[0.5, 0.1], [1.5, -0.2], [-1.0, 0.0]]])
  p = project_points(path, pts)
  assert torch.allclose(p.arc, torch.tensor([[0.5, 1.5, 0.0]]))
  assert torch.allclose(p.lateral[:, :2], torch.tensor([[0.1, -0.2]]))
  assert torch.allclose(p.tangent_yaw, torch.zeros(1, 3))


def test_mirror_permutation_is_an_involution() -> None:
  layout = LAYOUTS["slipper_v1"]
  perm = tac.mirror_permutation(layout)
  assert sorted(perm) == list(range(layout.num_taxels))
  assert all(perm[perm[i]] == i for i in range(layout.num_taxels))


def test_features_are_mirror_consistent() -> None:
  """A load on the left foot's lateral edge and the same load on the right
  foot's lateral edge give identical canonical features."""
  layout = LAYOUTS["slipper_v1"]
  t = layout.num_taxels
  lateral_left = torch.tensor([float(y > 0.01) for _, y in layout.centers])
  normal = torch.zeros(1, 2, t)
  normal[0, 0] = 50.0 * lateral_left
  # Canonical right foot uses the mirrored indexing already, so feed the same.
  normal[0, 1] = 50.0 * lateral_left
  f = tac.features_from_normals(normal, layout).view(2, -1)
  assert torch.allclose(f[0], f[1])
  edge_lat = f[0, t + 4]
  assert edge_lat > 0.9


def test_projection_on_bent_path() -> None:
  # Right-angle bend at (1, 0): a point inside the corner projects onto the
  # second leg, with the second leg's heading.
  path = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]])
  pts = torch.tensor([[[0.9, 0.5], [0.5, -0.1]]])
  p = project_points(path, pts)
  torch.testing.assert_close(p.arc, torch.tensor([[1.5, 0.5]]))
  torch.testing.assert_close(p.lateral, torch.tensor([[0.1, -0.1]]))
  torch.testing.assert_close(p.tangent_yaw, torch.tensor([[math.pi / 2, 0.0]]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("task", ALL_TASKS)
def test_env_builds_steps_and_obs_are_finite(task: str) -> None:
  from mjlab.envs import ManagerBasedRlEnv

  from sendg1.common.beam.mdp import beam_ahead_obs

  cfg = load_env_cfg(task)
  cfg.scene.num_envs = 8
  env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
  try:
    obs, _ = env.reset()
    assert obs["actor"].shape == (8, _actor_expected(task)[1])
    for _ in range(10):
      obs, rew, *_ = env.step(torch.zeros(8, env.action_manager.total_action_dim, device=env.device))
      assert torch.isfinite(rew).all()
      assert all(torch.isfinite(v).all() for v in obs.values())
    if task == TASK:
      assert torch.all(beam_ahead_obs(env) == 0)
  finally:
    env.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_beam_ahead_sees_bends() -> None:
  """With the base mid-beam, the look-ahead turn is zero on straight columns
  and mostly non-zero on bent ones."""
  from mjlab.envs import ManagerBasedRlEnv

  from sendg1.common.beam.geometry import BeamGeometry
  from sendg1.common.beam.mdp import _point_at_arc, beam_ahead_obs

  cfg = load_env_cfg(CURVED_TASK)
  cfg.scene.num_envs = 48
  env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
  try:
    env.reset()
    geo = BeamGeometry.get(env)
    robot = env.scene["robot"]
    xy, _ = _point_at_arc(geo.path()[..., :2], geo.beam_arc()[:, 0] + 1.0)
    pose = robot.data.root_link_pose_w.clone()
    pose[:, :2] = xy
    robot.write_root_link_pose_to_sim(pose)
    env.sim.forward()
    kinds = [c.path_kind for c in cfg.scene.terrain.terrain_generator.sub_terrains.values()]
    bent = torch.tensor([kinds[t] == "bent" for t in geo.types.tolist()], device=env.device)
    ahead = beam_ahead_obs(env).abs().max(-1).values
    assert torch.all(ahead[~bent] < 1e-5)
    assert (ahead[bent] > 0.05).float().mean() > 0.7
  finally:
    env.close()


def _swap_side(name: str) -> str:
  if name.startswith("left_"):
    return "right_" + name[len("left_") :]
  if name.startswith("right_"):
    return "left_" + name[len("right_") :]
  return name



@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("task", [TASK, CURVED_TASK, TOUCH_TASK])
def test_mirror_maps_match_g1_kinematics_and_are_involutions(task: str) -> None:
  """Mirrored joint angles must give the mirror-image robot (every body at
  (x, -y, z) of its counterpart), the default pose must be its own mirror,
  and mirroring every observation group and the actions twice must be the identity."""
  from mjlab.envs import ManagerBasedRlEnv

  from sendg1.common.beam.symmetry import _maps, joint_mirror

  cfg = load_env_cfg(task)
  cfg.scene.num_envs = 2
  env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
  try:
    obs, _ = env.reset()
    robot = env.scene["robot"]
    names = list(robot.joint_names)
    src = torch.tensor([s for s, _ in joint_mirror(names)], device=env.device)
    sign = torch.tensor([g for _, g in joint_mirror(names)], device=env.device)

    default = robot.data.default_joint_pos[0]
    torch.testing.assert_close(default[src] * sign, default)

    gen = torch.Generator(device=env.device).manual_seed(0)
    q0 = default + (torch.rand(len(names), generator=gen, device=env.device) - 0.5) * 0.8
    q = torch.stack([q0, q0[src] * sign])
    robot.write_joint_state_to_sim(q, torch.zeros_like(q))
    pose = torch.zeros(2, 7, device=env.device)
    pose[:, :3] = env.scene.env_origins + torch.tensor([0.0, 0.0, 1.0], device=env.device)
    pose[:, 3] = 1.0
    robot.write_root_link_pose_to_sim(pose)
    env.sim.forward()
    rel = robot.data.body_link_pos_w - pose[:, None, :3]
    bodies = list(robot.body_names)
    mirrored = rel[0, [bodies.index(_swap_side(b)) for b in bodies]] * torch.tensor(
      [1.0, -1.0, 1.0], device=env.device
    )
    torch.testing.assert_close(rel[1], mirrored, atol=1e-4, rtol=0)

    for name, (src, sign) in _maps(env).items():
      x = torch.randn(3, len(src), device=env.device)
      torch.testing.assert_close(((x[:, src] * sign)[:, src]) * sign, x, msg=f"{name} is not an involution")
  finally:
    env.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("task", [TASK, TOUCH_TASK])
def test_actor_is_mirror_equivariant(task: str) -> None:
  """pi(M_o o_1..t) = M_a pi(o_1..t) over a whole observation sequence."""
  from dataclasses import asdict

  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper

  from sendg1.common.beam.symmetry import _maps

  cfg = load_env_cfg(task)
  cfg.scene.num_envs = 4
  env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
  try:
    wrapped = RslRlVecEnvWrapper(env)
    rl = asdict(load_rl_cfg(task))
    rl["logger"] = "tensorboard"
    runner = MjlabOnPolicyRunner(wrapped, rl, log_dir=None, device="cuda:0")
    actor = runner.alg.actor
    maps = _maps(wrapped)
    obs, _ = wrapped.reset()
    seq = []
    for _ in range(15):
      seq.append(obs.clone())
      obs, *_ = wrapped.step(0.3 * torch.randn(4, env.action_manager.total_action_dim, device=env.device))
    src, sign = maps["actor"]
    asrc, asign = maps["__actions__"]
    with torch.no_grad():
      actor.reset()
      a = [actor(o) for o in seq]
      actor.reset()
      am = []
      for o in seq:
        om = o.clone()
        om["actor"] = o["actor"][:, src] * sign
        am.append(actor(om))
    for x, y in zip(a, am, strict=True):
      torch.testing.assert_close(y, x[:, asrc] * asign, atol=1e-5, rtol=1e-4)
  finally:
    env.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_plain_actor_trains_one_iteration(tmp_path) -> None:
  """The plain-GRU experiment builds rsl_rl's RNNModel and runs a PPO update."""
  from dataclasses import asdict

  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from rsl_rl.models.rnn_model import RNNModel

  from sendg1.rl.symmetric_rnn import SymmetricRNNModel

  cfg = load_env_cfg(TOUCH_PLAIN_TASK)
  cfg.scene.num_envs = 8
  env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
  try:
    rl = asdict(load_rl_cfg(TOUCH_PLAIN_TASK))
    rl["logger"] = "tensorboard"
    runner = MjlabOnPolicyRunner(RslRlVecEnvWrapper(env), rl, log_dir=str(tmp_path), device="cuda:0")
    actor = runner.alg.actor
    assert isinstance(actor, RNNModel) and not isinstance(actor, SymmetricRNNModel)
    runner.learn(num_learning_iterations=1)
    assert all(torch.isfinite(p).all() for p in actor.parameters())
  finally:
    env.close()
