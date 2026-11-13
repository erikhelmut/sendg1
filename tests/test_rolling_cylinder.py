"""Rolling-cylinder balance: the actor contract, the paired blind/touch
design, mirror maps, and the scene's physics."""

from __future__ import annotations

import pytest
import torch
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

import sendg1.tasks  # noqa: F401
from sendg1.common.rolling_cylinder.terms import (
  BLIND_ACTOR_TERMS,
  DEPLOYABLE,
  TERMS,
  TOUCH_ACTOR_TERMS,
)
from sendg1.fingerprint import fingerprint
from sendg1.tasks.rolling_cylinder import (
  BLIND_TASK,
  TOUCH_TASK,
  assert_actor_is_deployable,
)

TASKS = (TOUCH_TASK, BLIND_TASK)
BLIND_ACTOR_DIM = 3 + 3 + 29 + 29 + 29
TOUCH_ACTOR_DIM = BLIND_ACTOR_DIM + 2 * 31


@pytest.mark.parametrize("task", TASKS)
def test_actor_sees_only_deployable_signals(task: str) -> None:
  env_cfg = load_env_cfg(task)
  terms = set(env_cfg.observations["actor"].terms)
  assert terms <= DEPLOYABLE
  assert terms == set(TOUCH_ACTOR_TERMS if task == TOUCH_TASK else BLIND_ACTOR_TERMS)
  assert ("tactile_touch" in terms) == (task == TOUCH_TASK)


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
  env_cfg.observations["actor"].terms["feet_cylinder_pose"] = TERMS["feet_cylinder_pose"].build()
  with pytest.raises(ValueError, match="privileged"):
    assert_actor_is_deployable(env_cfg.observations)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("task", TASKS)
def test_env_spawns_on_the_cylinder_and_rolls(task: str) -> None:
  """After the spawn settles both soles are on the cylinder, near its top
  line; the cylinder rolls without slipping (v = omega R); obs are finite."""
  from mjlab.envs import ManagerBasedRlEnv

  from sendg1.common.rolling_cylinder import mdp

  cfg = load_env_cfg(task)
  cfg.scene.num_envs = 8
  env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
  try:
    obs, _ = env.reset()
    dim = TOUCH_ACTOR_DIM if task == TOUCH_TASK else BLIND_ACTOR_DIM
    assert obs["actor"].shape == (8, dim)
    zero = torch.zeros(8, env.action_manager.total_action_dim, device=env.device)
    st = mdp.CylinderState.get(env)
    touched = torch.zeros(8, 2, dtype=torch.bool, device=env.device)
    for i in range(15):
      obs, rew, *_ = env.step(zero)
      assert torch.isfinite(rew).all()
      assert all(torch.isfinite(v).all() for v in obs.values())
      touched |= mdp._feet_on_cylinder(env)
      assert not mdp._feet_on_ground(env).any()
      if i == 2:
        # Just landed: soles on the top line to within the spawn noise (2 cm)
        # and the first roll as the weight comes on.
        assert (st.feet_rel()[..., 0].abs() < 0.06).all()
    # A zero-action robot is soon rocking, so a foot may be momentarily light.
    assert touched.all()
    v, _ = st.velocity()
    torch.testing.assert_close(st.radius * st.spin_rate(), v, atol=0.01, rtol=0.0)
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
