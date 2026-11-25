"""Every task is comparable with every other one.

The same robot (the G1 in the tactile slipper), the same tactile sensor model
and touch features, and the same actor, critic and PPO settings
(``sendg1.common.ppo``). What may differ per task is listed here, so a new
difference has to be added on purpose.
"""

from __future__ import annotations

import pytest
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg

import sendg1.tasks  # noqa: F401
from sendg1.common import ppo
from sendg1.common.robots.g1 import get_g1_ball_hands_spec, get_g1_spec
from sendg1.common.sensors import SOLE_CONTACT_SENSOR, taxel_ray_sensor_name
from sendg1.common.tactile.layout import SIDES

TASKS = sorted(t for t in list_tasks() if t.startswith(("G1-", "Sendg1-")))
ALLOWED_ENTROPY = {0.005, 0.01}
"""Per task: 0.01 (walking) or 0.005 (tasks whose action std drifted up and
collapsed training at 0.01: beam, box push and the beam-style tasks)."""


def _is_tactile(env_cfg, rl_cfg) -> bool:
  terms = {
    t for g in rl_cfg.obs_groups["actor"] for t in env_cfg.observations[g].terms
  }
  return "tactile_touch" in terms


def test_task_list_is_not_empty() -> None:
  assert len(TASKS) > 20


@pytest.mark.parametrize("task", TASKS)
def test_robot_wears_the_slipper(task: str) -> None:
  robot = load_env_cfg(task).scene.entities["robot"]
  assert robot.spec_fn in (get_g1_spec, get_g1_ball_hands_spec), task


@pytest.mark.parametrize("task", TASKS)
def test_tactile_actor_uses_the_shared_sensor(task: str) -> None:
  env_cfg, rl_cfg = load_env_cfg(task), load_rl_cfg(task)
  if not _is_tactile(env_cfg, rl_cfg):
    return
  term = next(
    env_cfg.observations[g].terms["tactile_touch"]
    for g in rl_cfg.obs_groups["actor"]
    if "tactile_touch" in env_cfg.observations[g].terms
  )
  assert term.func.__name__ == "tactile_touch_noisy", task
  assert (term.delay_min_lag, term.delay_max_lag) == (0, 1), task
  sensors = {s.name: s for s in env_cfg.scene.sensors}
  assert SOLE_CONTACT_SENSOR in sensors, task
  rays = [sensors[taxel_ray_sensor_name(s)] for s in SIDES]
  # slipper_v1 (26 taxels) at the default 2x2 rays per taxel.
  assert [len(r.pattern.points) for r in rays] == [104, 104], task


@pytest.mark.parametrize("task", TASKS)
def test_standard_actor_and_ppo(task: str) -> None:
  rl = load_rl_cfg(task)
  assert rl.actor == ppo.actor_cfg(), task
  ref = ppo.ppo_cfg("x", rl.obs_groups, entropy_coef=0.0, max_iterations=1)
  assert rl.critic == ref.critic, task
  assert rl.num_steps_per_env == ppo.ROLLOUT_STEPS == 24, task
  assert rl.algorithm.learning_rate == ppo.LEARNING_RATE == 4.0e-4, task
  assert rl.algorithm.entropy_coef in ALLOWED_ENTROPY, task
  ref.algorithm.entropy_coef = rl.algorithm.entropy_coef
  assert rl.algorithm == ref.algorithm, task


@pytest.mark.parametrize("task", TASKS)
def test_default_training_seed_is_the_first_study_seed(task: str) -> None:
  from sendg1.seeds import FIRST_SEED

  assert load_rl_cfg(task).seed == FIRST_SEED == 1, task


def test_eval_protocols_use_the_eval_seed() -> None:
  from sendg1.eval.box_push import BoxPushProtocol
  from sendg1.eval.protocol import EvalProtocol
  from sendg1.seeds import EVAL_SEED, TRAIN_SEEDS, check_train_seed

  assert EvalProtocol().seed == BoxPushProtocol().seed == EVAL_SEED == 0
  assert EVAL_SEED not in TRAIN_SEEDS
  for bad in (0, 6):
    with pytest.raises(ValueError):
      check_train_seed(bad)


@pytest.mark.parametrize("task", [t for t in TASKS if not t.endswith("-Eval")])
def test_default_num_envs(task: str) -> None:
  """4096 everywhere; a GPU that cannot fit it gets a smaller count at launch."""
  assert load_env_cfg(task).scene.num_envs == ppo.NUM_ENVS == 4096, task
