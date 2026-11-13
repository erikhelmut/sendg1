"""Rolling-cylinder balance, plain PPO: ``G1-RollingCylinder-Touch`` and
``G1-RollingCylinder-Blind``.

The G1 stands across a cylinder lying free on the floor and keeps it still
under its feet. A paired comparison: the two tasks are the same env, critic,
reward and PPO settings, and differ only in whether the actor gets the
tactile slipper's touch features. Actor: rsl_rl's plain GRU ``RNNModel``
with the beam task's sizes and PPO settings (``tasks/beam/rl_cfg.py``); no
mirror symmetry.
"""

from __future__ import annotations

from mjlab.tasks.registry import register_mjlab_task

from sendg1.common.rolling_cylinder.terms import DEPLOYABLE, TERMS
from sendg1.tasks.beam.rl_cfg import beam_ppo_cfg
from sendg1.tasks.rolling_cylinder.env_cfg import make_rolling_cylinder_env_cfg

TOUCH_TASK = "G1-RollingCylinder-Touch"
BLIND_TASK = "G1-RollingCylinder-Blind"
MAX_ITERATIONS = 10_000
"""Half the beam's 20k: one skill on one terrain, no curriculum."""


def assert_actor_is_deployable(observations: dict, groups: tuple[str, ...] = ("actor",)) -> None:
  """Refuse to register an actor that can see privileged state."""
  for g in groups:
    for term in observations[g].terms:
      if term not in TERMS:
        raise KeyError(f"actor group {g!r} has undeclared term {term!r}")
      if term not in DEPLOYABLE:
        raise ValueError(
          f"actor group {g!r} contains {TERMS[term].availability} term {term!r}: "
          f"{TERMS[term].rationale}"
        )


for _task, _tactile, _experiment in (
  (TOUCH_TASK, True, "rolling_cylinder_touch"),
  (BLIND_TASK, False, "rolling_cylinder_blind"),
):
  _env = make_rolling_cylinder_env_cfg(tactile=_tactile)
  assert_actor_is_deployable(_env.observations)
  register_mjlab_task(
    task_id=_task,
    env_cfg=_env,
    play_env_cfg=make_rolling_cylinder_env_cfg(play=True, tactile=_tactile),
    rl_cfg=beam_ppo_cfg(
      experiment_name=_experiment, symmetric=False, max_iterations=MAX_ITERATIONS
    ),
  )
