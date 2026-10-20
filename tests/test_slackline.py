"""Slackline: the line's physics, the actor contract, the paired blind/touch
design, and the spawn."""

from __future__ import annotations

import itertools
import math

import mujoco
import numpy as np
import pytest
import torch
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

import sendg1.tasks  # noqa: F401
from sendg1.common.slackline import model as sm
from sendg1.common.slackline.terms import (
  BLIND_ACTOR_TERMS,
  DEPLOYABLE,
  TERMS,
  TOUCH_ACTOR_TERMS,
)
from sendg1.fingerprint import fingerprint
from sendg1.tasks.slackline import BLIND_TASK, TOUCH_TASK, assert_actor_is_deployable

TASKS = (TOUCH_TASK, BLIND_TASK)
BLIND_ACTOR_DIM = 3 + 3 + 29 + 29 + 29 + 1
TOUCH_ACTOR_DIM = BLIND_ACTOR_DIM + 2 * 31


def _line_with_load(mass: float, x: float = 0.0) -> tuple[mujoco.MjModel, mujoco.MjData]:
  """The line at the task's 5 ms implicitfast step, a block resting on it at
  ``x``, its rotation locked (a block balanced on 5 cm webbing would tip)."""
  spec = sm.slackline_spec()
  spec.option.timestep = 0.005
  spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
  body = spec.worldbody.add_body(name="load", pos=(x, 0.0, sm.HEIGHT + 0.051))
  for axis in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
    body.add_joint(type=mujoco.mjtJoint.mjJNT_SLIDE, axis=axis)
  body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=(0.1, 0.035, 0.05), mass=mass)
  m = spec.compile()
  return m, mujoco.MjData(m)


def test_line_sags_like_a_taut_string_and_stays_stable() -> None:
  """35 kg at mid-span sinks the line by ~F L / 4T, and after a sideways
  shove it sways at about the taut string's rate, without blowing up."""
  m, d = _line_with_load(35.0)
  for _ in range(800):
    mujoco.mj_step(m, d)
  block_z = d.qpos[-1]
  expected = -35.0 * 9.81 * sm.SPAN / (4 * sm.TENSION)
  # The block spans 20 cm and the line's armature adds weight: within 20 %.
  assert block_z == pytest.approx(expected, rel=0.2)
  load = m.body("load").id
  d.xfrc_applied[load, 1] = 50.0
  for _ in range(20):
    mujoco.mj_step(m, d)
  d.xfrc_applied[load, 1] = 0.0
  y = []
  for _ in range(1200):
    mujoco.mj_step(m, d)
    y.append(d.qpos[-2])
  assert np.isfinite(d.qpos).all() and np.abs(y).max() < 0.1
  crossings = np.flatnonzero(np.diff(np.sign(np.asarray(y) - np.mean(y))))
  period = 2 * np.mean(np.diff(crossings)) * 0.005
  ideal = 2 * math.pi * math.sqrt(35.0 / (4 * sm.TENSION / sm.SPAN))
  # Slower than the massless ideal: the webbing's armature rides along.
  assert ideal < period < 1.6 * ideal


@pytest.mark.parametrize("task", TASKS)
def test_actor_sees_only_deployable_signals(task: str) -> None:
  env_cfg = load_env_cfg(task)
  terms = set(env_cfg.observations["actor"].terms)
  assert terms <= DEPLOYABLE
  assert terms == set(TOUCH_ACTOR_TERMS if task == TOUCH_TASK else BLIND_ACTOR_TERMS)


def test_tasks_differ_only_in_the_actor() -> None:
  """The study's rule 2: same critic, rewards, events, terminations, scene and
  network; only the actor's observations change."""
  touch, blind = load_env_cfg(TOUCH_TASK), load_env_cfg(BLIND_TASK)
  assert set(touch.observations["actor"].terms) - set(blind.observations["actor"].terms) == {
    "tactile_touch"
  }
  touch.observations.pop("actor")
  blind.observations.pop("actor")
  assert fingerprint(touch) == fingerprint(blind)

  rl_touch, rl_blind = load_rl_cfg(TOUCH_TASK), load_rl_cfg(BLIND_TASK)
  assert rl_touch.actor.class_name == rl_blind.actor.class_name == "RNNModel"
  assert rl_touch.experiment_name != rl_blind.experiment_name
  rl_blind.experiment_name = rl_touch.experiment_name
  assert fingerprint(rl_touch) == fingerprint(rl_blind)


def test_leak_guard_fires() -> None:
  env_cfg = load_env_cfg(BLIND_TASK)
  with pytest.raises(ValueError, match="privileged"):
    assert_actor_is_deployable(env_cfg.observations, ("critic",))
  env_cfg.observations["actor"].terms["line_under_robot"] = TERMS["line_under_robot"].build()
  with pytest.raises(ValueError, match="privileged"):
    assert_actor_is_deployable(env_cfg.observations)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("task", TASKS)
def test_env_spawns_in_the_stance_on_the_line(task: str) -> None:
  """Holding the solved stance, both soles land on the webbing, centred on
  it, carrying the robot's weight between them; the front foot points along
  the line and the back one is turned out ~45 deg; obs are finite."""
  from mjlab.envs import ManagerBasedRlEnv

  from sendg1.common.slackline import mdp

  cfg = load_env_cfg(task)
  cfg.scene.num_envs = 8
  cfg.events["reset_on_line"].params = {"joint_noise": 0.0, "heading_noise": 0.0}
  # The real line (training starts on the level-0 plank).
  cfg.curriculum = {}
  cfg.events["line_level"].params = {"fixed_level": mdp.TOP_LEVEL}
  env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
  try:
    obs, _ = env.reset()
    assert obs["actor"].shape == (8, TOUCH_ACTOR_DIM if task == TOUCH_TASK else BLIND_ACTOR_DIM)
    st = mdp.LineState.get(env)
    stances = mdp._stance_tensors(env)
    robot = env.scene["robot"]
    scale = env.action_manager.get_term("joint_pos")._scale
    touched = torch.zeros(8, 2, dtype=torch.bool, device=env.device)
    load = torch.zeros(8, device=env.device)
    for _ in range(3):
      q = torch.where(st.front_left[:, None], stances["left"], stances["right"])
      obs, rew, *_ = env.step((q - robot.data.default_joint_pos[0]) / scale)
      assert torch.isfinite(rew).all()
      assert all(torch.isfinite(v).all() for v in obs.values())
      touched |= mdp._feet_on_line(env)
      load = torch.maximum(load, mdp._foot_load(env).sum(-1))
    # Landing from 1 cm, a foot may be momentarily light on one step.
    assert touched.all()
    assert not mdp._feet_on_ground(env).any()
    feet = st.feet()
    # Sole centres over the 5 cm webbing (it rolls a little as they land).
    assert (feet[..., 1].abs() < 0.03).all()
    # Peak over the landing (it bounces): the line takes the robot's weight.
    assert ((load > 0.6 * mdp.BODY_WEIGHT_N) & (load < 2.5 * mdp.BODY_WEIGHT_N)).all()
    yaw = st.feet_yaw().abs()
    front = torch.where(st.back_is_left(), 1, 0)
    rows = torch.arange(8, device=env.device)
    assert (yaw[rows, front] < 0.15).all()
    # The back foot twists a little as it takes weight.
    assert ((yaw[rows, 1 - front] - mdp.BACK_FOOT_YAW).abs() < 0.2).all()
  finally:
    env.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("task", TASKS)
def test_plain_gru_trains_one_iteration(task: str, tmp_path) -> None:
  """Each condition builds rsl_rl's plain RNNModel and runs a PPO update."""
  from dataclasses import asdict

  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from rsl_rl.models.rnn_model import RNNModel

  from sendg1.rl.symmetric_rnn import SymmetricRNNModel

  cfg = load_env_cfg(task)
  cfg.scene.num_envs = 8
  env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
  try:
    rl = asdict(load_rl_cfg(task))
    rl["logger"] = "tensorboard"
    runner = MjlabOnPolicyRunner(RslRlVecEnvWrapper(env), rl, log_dir=str(tmp_path), device="cuda:0")
    actor = runner.alg.actor
    assert isinstance(actor, RNNModel) and not isinstance(actor, SymmetricRNNModel)
    runner.learn(num_learning_iterations=1)
    assert all(torch.isfinite(p).all() for p in actor.parameters())
  finally:
    env.close()


def test_static_sag_is_the_taut_string_without_helpers() -> None:
  """With no helper springs the solved sag under a point load matches the
  string's triangle, F a (L - a) / (T L), at the load; helpers stiffen it."""
  from sendg1.common.slackline import mdp

  tension = torch.tensor([1500.0, 1500.0])
  load_x = torch.tensor([[0.0], [-1.0]])  # one load each: mid-span, 1 m in
  sag = mdp.static_sag(tension, torch.zeros(2), load_x)
  for row, x in enumerate((0.0, -1.0)):
    i = round((x + sm.SPAN / 2) / sm.NODE_SPACING) - 1
    a = x + sm.SPAN / 2
    expected = -mdp.BODY_WEIGHT_N * a * (sm.SPAN - a) / (1500.0 * sm.SPAN)
    assert float(sag[row, i]) == pytest.approx(expected, rel=1e-4)
  stiff = mdp.static_sag(tension, torch.full((2,), mdp.LEVELS[0].stiffness), load_x)
  assert (stiff.abs() < 0.2 * sag.abs() + 1e-6).all()


def test_levels_run_from_stiff_and_wide_to_the_free_line() -> None:
  from sendg1.common.slackline import mdp

  widths = [lv.width for lv in mdp.LEVELS]
  stiff = [lv.stiffness for lv in mdp.LEVELS]
  roll = [lv.roll_stiffness for lv in mdp.LEVELS]
  for seq in (widths, stiff, roll):
    assert all(a >= b for a, b in itertools.pairwise(seq))
  assert widths[-1] == sm.WIDTH and stiff[-1] == 0.0 and roll[-1] == 0.0
  play = load_env_cfg(TOUCH_TASK, play=True)
  assert play.curriculum == {}
  assert play.events["line_level"].params["fixed_level"] == mdp.TOP_LEVEL
  assert "line_levels" in load_env_cfg(TOUCH_TASK).curriculum


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_curriculum_builds_each_level_and_moves_envs(monkeypatch) -> None:
  """Each world gets its level's width and helper springs; a survived
  episode promotes, an early termination demotes, a replay moves nothing."""
  from mjlab.envs import ManagerBasedRlEnv

  from sendg1.common.slackline import mdp

  cfg = load_env_cfg(BLIND_TASK)
  n = len(mdp.LEVELS)
  cfg.scene.num_envs = n
  cfg.curriculum["line_levels"].params = {"replay_share": 0.0}
  env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
  try:
    st = mdp.LineState.get(env)
    mdp.set_levels(env, torch.arange(n, device=env.device))
    env.reset()
    m = env.sim.mj_model
    gid = m.geom(f"slackline/{sm.NODE_GEOM.format(3)}").id
    jy = m.joint(f"slackline/{sm.SLIDE_Y.format(3)}").id
    jr = m.joint(f"slackline/{sm.ROLL.format(3)}").id
    for i, lv in enumerate(mdp.LEVELS):
      assert float(env.sim.model.geom_size[i, gid, 1]) == pytest.approx(lv.width / 2)
      assert float(env.sim.model.jnt_stiffness[i, jy]) == pytest.approx(lv.stiffness)
      assert float(env.sim.model.jnt_stiffness[i, jr]) == pytest.approx(lv.roll_stiffness)

    env.step(torch.zeros(n, env.action_manager.total_action_dim, device=env.device))
    ids = torch.arange(n, device=env.device)
    st.home[:] = 3
    st.level[:] = 3
    st.replay[:] = False
    st.replay[2] = True
    st.cmd_dist[:] = 0.0  # standing: surviving is success
    terminated = torch.zeros(n, dtype=torch.bool, device=env.device)
    terminated[1] = True
    monkeypatch.setattr(type(env.termination_manager), "terminated", property(lambda self: terminated))
    mdp.line_levels(env, ids, replay_share=0.0)
    assert st.home.tolist() == [4, 2, 3, 4, 4, 4, 4][:n]
  finally:
    env.close()
