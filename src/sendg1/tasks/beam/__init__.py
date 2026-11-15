"""Beam walking, plain PPO, on straight and bent beams.

``G1-Beam-Touch-Curved-Plain``: a GRU actor on the robot's own signals and
the tactile slipper's touch features (``tactile_touch``). Its blind baseline,
``G1-Beam-Curved-Blind``, is the same task without the tactile actor input.

Memoryless actors, an MLP on the current observation only (``rl_cfg.ActorKind``),
touch or blind: ``G1-Beam-Touch-Curved-MLP-Plain``, ``G1-Beam-Curved-Blind-MLP-Plain``.

LSTM(256) actors with 64-step rollouts (gradients over 1.28 s, about one gait
cycle) and a 4e-4 starting learning rate (adaptive schedule), touch or blind:
``G1-Beam-Touch-Curved-LSTM``, ``G1-Beam-Curved-Blind-LSTM``.

One reward for every task, ``beam_reward`` (``env_cfg.build_beam_reward``),
first trained in the ``touch_plain_s0`` runs (2026-10-17/20). The reward study
that followed (``-R-<Variant>`` tasks, ``foot_side_contact``) was removed on
2026-11-15.

Every task has an evaluation-only twin, ``<task>-Eval``: held-out beams
(``make_beam_env_cfg(evaluation=True)``), same observations and episode
length, so its training checkpoints load. Not for training. Every task trains
with ``runner.BeamRunner``, whose checkpoints carry the curriculum.

Removed, with their checkpoints no longer loading:
- 2026-11-14: the symmetric (mirror-equivariant) actors and the transformer
  (``G1-Beam-Touch-Curved``, ``-Blind-Sym``, ``-MLP-Sym``, ``-Transformer``).
- 2026-11-15: ``G1-Beam-E2E{,-Curved}`` (the old soft contact-map tactile
  input), and the round-three experiments ``-Plain-R64``, ``-Plain-R2-35s``,
  ``-Plain-R2-35s-R64``, ``-LSTM-R2-35s``, ``-LSTM-R2-35s-R64``.

A privileged critic. Trained with mjlab's ``train``; evaluated with
``scripts/beam_eval.py``.
"""

from __future__ import annotations

from mjlab.tasks.registry import register_mjlab_task

from sendg1.common.beam.terms import DEPLOYABLE, TERMS
from sendg1.tasks.beam.env_cfg import make_beam_env_cfg
from sendg1.tasks.beam.rl_cfg import beam_ppo_cfg
from sendg1.tasks.beam.runner import BeamRunner

TOUCH_PLAIN_TASK = "G1-Beam-Touch-Curved-Plain"
BLIND_TASK = "G1-Beam-Curved-Blind"
TOUCH_MLP_PLAIN_TASK = "G1-Beam-Touch-Curved-MLP-Plain"
BLIND_MLP_PLAIN_TASK = "G1-Beam-Curved-Blind-MLP-Plain"
TOUCH_LSTM_TASK = "G1-Beam-Touch-Curved-LSTM"
BLIND_LSTM_TASK = "G1-Beam-Curved-Blind-LSTM"
LSTM_ROLLOUT_STEPS = 64
LSTM_LEARNING_RATE = 4.0e-4


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


_LSTM = {"num_steps_per_env": LSTM_ROLLOUT_STEPS, "learning_rate": LSTM_LEARNING_RATE}
_TASKS = (
  # task, tactile, actor, experiment, extra PPO settings
  (TOUCH_PLAIN_TASK, "touch", "gru", "beam_touch_curved_plain", {}),
  (BLIND_TASK, "none", "gru", "beam_curved_blind", {}),
  (TOUCH_MLP_PLAIN_TASK, "touch", "mlp", "beam_touch_curved_mlp_plain", {}),
  (BLIND_MLP_PLAIN_TASK, "none", "mlp", "beam_curved_blind_mlp_plain", {}),
  (TOUCH_LSTM_TASK, "touch", "lstm", "beam_touch_curved_lstm", _LSTM),
  (BLIND_LSTM_TASK, "none", "lstm", "beam_curved_blind_lstm", _LSTM),
)
for _task, _tactile, _actor, _experiment, _ppo in _TASKS:
  _kw = {"curves": True, "tactile": _tactile}
  for _suffix, _evaluation in (("", False), ("-Eval", True)):
    _env = make_beam_env_cfg(evaluation=_evaluation, **_kw)
    assert_actor_is_deployable(_env.observations)
    register_mjlab_task(
      task_id=_task + _suffix,
      env_cfg=_env,
      play_env_cfg=make_beam_env_cfg(play=True, evaluation=_evaluation, **_kw),
      rl_cfg=beam_ppo_cfg(experiment_name=_experiment, actor=_actor, **_ppo),
      runner_cls=BeamRunner,
    )
