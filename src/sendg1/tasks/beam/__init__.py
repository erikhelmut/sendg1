"""Beam walking on straight and bent beams: ``G1-Beam-Touch-Curved-LSTM`` and
``G1-Beam-Curved-Blind-LSTM``.

A paired comparison: the standard actor and PPO every task shares
(``sendg1.common.ppo``) on the robot's own signals, plus for touch the tactile
slipper's touch features (``tactile_touch``). One reward, ``beam_reward``
(``env_cfg.build_beam_reward``), and a privileged critic.

Every task has an evaluation-only twin, ``<task>-Eval``: held-out beams
(``make_beam_env_cfg(evaluation=True)``), same observations and episode
length, so its training checkpoints load. Not for training. Every task trains
with ``runner.BeamRunner``, whose checkpoints carry the curriculum. Evaluated
with ``scripts/beam_eval.py``.
"""

from __future__ import annotations

from mjlab.tasks.registry import register_mjlab_task

from sendg1.common.beam.terms import DEPLOYABLE, TERMS
from sendg1.tasks.beam.env_cfg import make_beam_env_cfg
from sendg1.tasks.beam.rl_cfg import beam_ppo_cfg
from sendg1.tasks.beam.runner import BeamRunner

TOUCH_LSTM_TASK = "G1-Beam-Touch-Curved-LSTM"
BLIND_LSTM_TASK = "G1-Beam-Curved-Blind-LSTM"


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


_TASKS = (
  # task, tactile, experiment
  (TOUCH_LSTM_TASK, "touch", "beam_touch_curved_lstm"),
  (BLIND_LSTM_TASK, "none", "beam_curved_blind_lstm"),
)
for _task, _tactile, _experiment in _TASKS:
  _kw = {"curves": True, "tactile": _tactile}
  for _suffix, _evaluation in (("", False), ("-Eval", True)):
    _env = make_beam_env_cfg(evaluation=_evaluation, **_kw)
    assert_actor_is_deployable(_env.observations)
    register_mjlab_task(
      task_id=_task + _suffix,
      env_cfg=_env,
      play_env_cfg=make_beam_env_cfg(play=True, evaluation=_evaluation, **_kw),
      rl_cfg=beam_ppo_cfg(experiment_name=_experiment),
      runner_cls=BeamRunner,
    )
