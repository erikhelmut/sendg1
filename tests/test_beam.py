"""Beam task: no privileged leak into the actor, terrain and geometry (straight
and bent beams), tactile canonicalisation, and envs that
build and step."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg

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
  BLIND_MLP_PLAIN_TASK,
  BLIND_LSTM_TASK,
  BLIND_TASK,
  TOUCH_LSTM_TASK,
  TOUCH_MLP_PLAIN_TASK,
  TOUCH_PLAIN_TASK,
  assert_actor_is_deployable,
)

BLIND_ACTOR_EXPECTED = {
  "base_ang_vel",
  "projected_gravity",
  "joint_pos",
  "joint_vel",
  "actions",
  "command_vx",
}
BLIND_ACTOR_DIM = 3 + 3 + 29 + 29 + 29 + 1
TOUCH_ACTOR_DIM = BLIND_ACTOR_DIM + 2 * tac.TOUCH_PER_FOOT
BLIND_TASKS = (BLIND_TASK, BLIND_MLP_PLAIN_TASK, BLIND_LSTM_TASK)
TOUCH_TASKS = (TOUCH_PLAIN_TASK, TOUCH_MLP_PLAIN_TASK, TOUCH_LSTM_TASK)
ALL_TASKS = [*TOUCH_TASKS, *BLIND_TASKS]


def _actor_expected(task: str) -> tuple[set[str], int]:
  if task in BLIND_TASKS:
    return BLIND_ACTOR_EXPECTED, BLIND_ACTOR_DIM
  return BLIND_ACTOR_EXPECTED | {"tactile_touch"}, TOUCH_ACTOR_DIM


@pytest.mark.parametrize("task", ALL_TASKS)
def test_actor_sees_only_deployable_signals(task: str) -> None:
  env_cfg = load_env_cfg(task)
  rl_cfg = load_rl_cfg(task)
  terms = {t for g in rl_cfg.obs_groups["actor"] for t in env_cfg.observations[g].terms}
  assert terms == _actor_expected(task)[0]
  assert terms <= DEPLOYABLE
  critic = {t for g in rl_cfg.obs_groups["critic"] for t in env_cfg.observations[g].terms}
  assert {"base_beam_pose", "feet_beam_pose", "base_lin_vel", "beam_ahead"} <= critic
  assert {"tactile_touch_clean", "taxel_forces_clean", "foot_slip_vel", "beam_params"} <= critic


def test_every_actor_is_an_rsl_rl_model() -> None:
  """Only rsl_rl's own GRU, LSTM and MLP actors: no custom model classes."""
  from mjlab.tasks.registry import list_tasks

  for task in list_tasks():
    if task.startswith(("G1-Beam", "G1-Pedalo", "G1-RollingCylinder", "G1-Slackline")):
      assert load_rl_cfg(task).actor.class_name in ("RNNModel", "MLPModel"), task


def test_touch_features() -> None:
  """Bits are binary, CoP is zero on an unloaded foot (and under the 15 N
  contact threshold) and lateral (+y) on both feet for a lateral-edge load."""
  layout = LAYOUTS["slipper_v1"]
  t = layout.num_taxels
  lateral = torch.tensor([float(y > 0.01) for _, y in layout.centers])
  normal = torch.zeros(2, 2, t)
  normal[0, 0] = 40.0 * lateral
  normal[0, 1] = 40.0 * lateral
  normal[1, 0] = 3.0  # below the bit threshold everywhere, 78 N in total
  f = tac.touch_features(normal, layout).view(2, 2, tac.TOUCH_PER_FOOT)
  assert tac.TOUCH_PER_FOOT == t + 3
  bits, load, cop = f[..., :t], f[..., t], f[..., t + 1 : t + 3]
  assert set(bits.unique().tolist()) <= {0.0, 1.0}
  torch.testing.assert_close(f[0, 0], f[0, 1])
  assert cop[0, 0, 1] > 0.5
  assert torch.all(cop[1, 1] == 0) and load[1, 1] == 0
  assert bits[1, 0].sum() == 0 and load[1, 0] > 0


def test_leak_guard_fires() -> None:
  env_cfg = load_env_cfg(TOUCH_PLAIN_TASK)
  with pytest.raises(ValueError, match="privileged"):
    assert_actor_is_deployable(env_cfg.observations, ("critic",))
  env_cfg.observations["actor"].terms["feet_beam_pose"] = TERMS["feet_beam_pose"].build()
  with pytest.raises(ValueError, match="privileged"):
    assert_actor_is_deployable(env_cfg.observations)


def test_play_cfg_has_no_curriculum_or_state_bank() -> None:
  cfg = load_env_cfg(TOUCH_PLAIN_TASK, play=True)
  assert cfg.curriculum == {}
  assert "spawn_from_bank" not in cfg.events
  assert "spawn_from_bank" in load_env_cfg(TOUCH_PLAIN_TASK).events


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
  finally:
    env.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_beam_ahead_sees_bends() -> None:
  """With the base mid-beam, the look-ahead turn is zero on straight columns
  and mostly non-zero on bent ones."""
  from mjlab.envs import ManagerBasedRlEnv

  from sendg1.common.beam.geometry import BeamGeometry
  from sendg1.common.beam.mdp import _point_at_arc, beam_ahead_obs

  cfg = load_env_cfg(TOUCH_PLAIN_TASK)
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_plain_actor_trains_one_iteration(tmp_path) -> None:
  """The GRU experiment builds rsl_rl's RNNModel, runs a PPO update with
  ``BeamRunner``, and its checkpoint carries the curriculum."""
  from dataclasses import asdict

  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from rsl_rl.models.rnn_model import RNNModel

  from sendg1.common.beam import mdp
  from sendg1.tasks.beam.runner import BeamRunner

  cfg = load_env_cfg(TOUCH_PLAIN_TASK)
  cfg.scene.num_envs = 8
  env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
  try:
    rl = asdict(load_rl_cfg(TOUCH_PLAIN_TASK))
    rl["logger"] = "tensorboard"
    runner = BeamRunner(RslRlVecEnvWrapper(env), rl, log_dir=str(tmp_path), device="cuda:0")
    actor = runner.alg.actor
    assert type(actor) is RNNModel
    runner.learn(num_learning_iterations=1)
    assert all(torch.isfinite(p).all() for p in actor.parameters())
    # Restoring the curriculum from the checkpoint works (BeamRunner.load does
    # the same in a resumed run's fresh process).
    env._beam_home_level[:] = 3
    runner.save(str(tmp_path / "m.pt"))
    env._beam_home_level[:] = 0
    infos = torch.load(tmp_path / "m.pt", weights_only=False)["infos"]
    mdp.load_curriculum_state(env, infos["beam_curriculum"])
    assert bool((env._beam_home_level == 3).all())
  finally:
    env.close()


@pytest.mark.parametrize(
  ("touch", "blind"),
  [
    (TOUCH_PLAIN_TASK, BLIND_TASK),
    (TOUCH_MLP_PLAIN_TASK, BLIND_MLP_PLAIN_TASK),
    (TOUCH_LSTM_TASK, BLIND_LSTM_TASK),
  ],
)
def test_blind_twin_differs_only_in_the_tactile_input(touch: str, blind: str) -> None:
  t, b = load_env_cfg(touch), load_env_cfg(blind)
  assert set(t.observations["actor"].terms) - set(b.observations["actor"].terms) == {"tactile_touch"}
  assert t.observations["critic"].terms.keys() == b.observations["critic"].terms.keys()
  assert t.rewards.keys() == b.rewards.keys()
  t_rl, b_rl = load_rl_cfg(touch), load_rl_cfg(blind)
  assert t_rl.actor == b_rl.actor and t_rl.critic == b_rl.critic and t_rl.algorithm == b_rl.algorithm


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_mlp_actor_trains_one_iteration(tmp_path) -> None:
  """The memoryless experiments build their MLP actor and run a PPO update."""
  from dataclasses import asdict

  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper

  from rsl_rl.models.mlp_model import MLPModel

  task = TOUCH_MLP_PLAIN_TASK
  cfg = load_env_cfg(task)
  cfg.scene.num_envs = 8
  env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
  try:
    rl = asdict(load_rl_cfg(task))
    rl["logger"] = "tensorboard"
    runner = MjlabOnPolicyRunner(RslRlVecEnvWrapper(env), rl, log_dir=str(tmp_path), device="cuda:0")
    actor = runner.alg.actor
    assert type(actor) is MLPModel and not actor.is_recurrent
    runner.learn(num_learning_iterations=1)
    assert all(torch.isfinite(p).all() for p in actor.parameters())
  finally:
    env.close()


def test_every_beam_task_uses_beam_reward() -> None:
  """The reward study was removed: every beam task trains on beam_reward."""
  from sendg1.fingerprint import fingerprint
  from sendg1.tasks.beam.env_cfg import build_beam_reward

  ref = fingerprint(build_beam_reward())
  tasks = [t for t in list_tasks() if t.startswith("G1-Beam")]
  assert tasks and not [t for t in tasks if "-R-" in t]
  for t in tasks:
    assert fingerprint(load_env_cfg(t).rewards) == ref, t
  assert "foot_side_contact" not in build_beam_reward()


def test_actor_ablations_differ_only_in_the_actor() -> None:
  """The GRU and MLP beam tasks are the LSTM pair with another actor model:
  same env, critic and PPO settings."""
  from sendg1.fingerprint import fingerprint

  for lstm, others in (
    (TOUCH_LSTM_TASK, (TOUCH_PLAIN_TASK, TOUCH_MLP_PLAIN_TASK)),
    (BLIND_LSTM_TASK, (BLIND_TASK, BLIND_MLP_PLAIN_TASK)),
  ):
    ref = load_rl_cfg(lstm)
    assert ref.actor.rnn_type == "lstm"
    for task in others:
      rl = load_rl_cfg(task)
      assert rl.algorithm == ref.algorithm and rl.critic == ref.critic, task
      assert rl.num_steps_per_env == ref.num_steps_per_env, task
      assert fingerprint(load_env_cfg(task)) == fingerprint(load_env_cfg(lstm)), task


def test_beam_reward_is_v4b() -> None:
  """v4b: progress paid linearly, foot_slip at the walking task's -0.1; v4c
  (foot_slip -2) is a record only."""
  from sendg1.tasks.beam.env_cfg import build_beam_reward, build_beam_reward_v4c

  r, c = build_beam_reward(), build_beam_reward_v4c()
  assert "track_forward_progress" in r and r["foot_slip"].weight == -0.1
  assert c["foot_slip"].weight == -2.0 and r.keys() == c.keys()
