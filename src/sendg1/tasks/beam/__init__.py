"""Beam walking, plain PPO, on straight and bent beams.

The main pair, ``G1-Beam-Touch-Curved-LSTM`` and ``G1-Beam-Curved-Blind-LSTM``:
the standard actor and PPO every task shares (``sendg1.common.ppo``: LSTM(256),
24-step rollouts, LR 4e-4 adaptive) on the robot's own signals, plus for touch
the tactile slipper's touch features (``tactile_touch``). The LSTM tasks used
64-step rollouts by default until 2026-11-25; the v3/v4 runs overrode them to 24.

Actor ablations, identical but for the actor's model (``rl_cfg.ActorKind``):
a GRU(256), ``G1-Beam-Touch-Curved-Plain`` / ``G1-Beam-Curved-Blind``, and a
memoryless MLP, ``G1-Beam-Touch-Curved-MLP-Plain`` /
``G1-Beam-Curved-Blind-MLP-Plain``. Until 2026-11-25 these trained 32-step
rollouts at LR 1e-3.

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
  # task, tactile, actor, experiment
  (TOUCH_LSTM_TASK, "touch", "lstm", "beam_touch_curved_lstm"),
  (BLIND_LSTM_TASK, "none", "lstm", "beam_curved_blind_lstm"),
  (TOUCH_PLAIN_TASK, "touch", "gru", "beam_touch_curved_plain"),
  (BLIND_TASK, "none", "gru", "beam_curved_blind"),
  (TOUCH_MLP_PLAIN_TASK, "touch", "mlp", "beam_touch_curved_mlp_plain"),
  (BLIND_MLP_PLAIN_TASK, "none", "mlp", "beam_curved_blind_mlp_plain"),
)
for _task, _tactile, _actor, _experiment in _TASKS:
  _kw = {"curves": True, "tactile": _tactile}
  for _suffix, _evaluation in (("", False), ("-Eval", True)):
    _env = make_beam_env_cfg(evaluation=_evaluation, **_kw)
    assert_actor_is_deployable(_env.observations)
    register_mjlab_task(
      task_id=_task + _suffix,
      env_cfg=_env,
      play_env_cfg=make_beam_env_cfg(play=True, evaluation=_evaluation, **_kw),
      rl_cfg=beam_ppo_cfg(experiment_name=_experiment, actor=_actor),
      runner_cls=BeamRunner,
    )
