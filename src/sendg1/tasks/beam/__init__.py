"""Beam walking, plain PPO: ``G1-Beam-E2E`` (straight beams) and
``G1-Beam-E2E-Curved`` (straight and bent beams).

Touch-feature experiments, both on curved beams: ``G1-Beam-Touch-Curved``
(symmetric GRU) and ``G1-Beam-Touch-Curved-Plain`` (plain GRU), with the
``touch`` tactile input (``env_cfg.Tactile``). Same rewards, curriculum and
PPO settings as ``G1-Beam-E2E-Curved``. Their blind baseline,
``G1-Beam-Curved-Blind``, is ``G1-Beam-Touch-Curved-Plain`` without the
tactile actor input; ``G1-Beam-Curved-Blind-Sym`` is the same with the
symmetric GRU (blind ``G1-Beam-Touch-Curved``).

Memoryless actors, an MLP on the current observation only (``rl_cfg.ActorKind``),
symmetric or plain, touch or blind: ``G1-Beam-Touch-Curved-MLP-{Sym,Plain}``,
``G1-Beam-Curved-Blind-MLP-{Sym,Plain}``.

Reward study on the touch plain MLP: ``G1-Beam-Touch-Curved-MLP-Plain-R-<Variant>``,
one per ``env_cfg.REWARD_VARIANTS`` entry (``R-Base`` is the unchanged reward),
everything else as ``G1-Beam-Touch-Curved-MLP-Plain``.

Memory over a longer window, touch input, unchanged reward:
``G1-Beam-Touch-Curved-Transformer``, a small causal transformer over the last
``TRANSFORMER_HISTORY`` frames (``sendg1.rl.transformer``), and
``G1-Beam-Touch-Curved-Plain-R64``, ``G1-Beam-Touch-Curved-Plain`` with
64-step rollouts (the GRU learns over 1.28 s, about one full gait cycle,
instead of 0.64 s).

Round two's rules with round three's 35 s episodes, plain GRU, touch input:
``G1-Beam-Touch-Curved-Plain-R2-35s`` (32-step rollouts) and ``...-R64``
(``make_beam_env_cfg(round2=True)``). Both round-three runs climbed to ~0.25 m
and fell back; these tell whether the long episodes alone do that.
``G1-Beam-Touch-Curved-LSTM-R2-35s`` (and ``-R64``) are the same with an
LSTM(256) in place of the GRU.

Every task has an evaluation-only twin, ``<task>-Eval``: held-out beams
(``make_beam_env_cfg(evaluation=True)``), same observations and episode
length, so its training checkpoints load. Not for training. Every task trains
with ``runner.BeamRunner``, whose checkpoints carry the curriculum.

A GRU actor on the robot's own signals and the tactile slipper, a privileged
critic. Trained with mjlab's ``train``; evaluated with
``scripts/beam_eval.py``.
"""

from __future__ import annotations

from mjlab.tasks.registry import register_mjlab_task

from sendg1.common.beam.terms import DEPLOYABLE, TERMS
from sendg1.tasks.beam.env_cfg import REWARD_VARIANTS, make_beam_env_cfg
from sendg1.tasks.beam.rl_cfg import beam_ppo_cfg
from sendg1.tasks.beam.runner import BeamRunner

TASK = "G1-Beam-E2E"
CURVED_TASK = "G1-Beam-E2E-Curved"
TOUCH_TASK = "G1-Beam-Touch-Curved"
TOUCH_PLAIN_TASK = "G1-Beam-Touch-Curved-Plain"
BLIND_TASK = "G1-Beam-Curved-Blind"
BLIND_SYM_TASK = "G1-Beam-Curved-Blind-Sym"
TOUCH_MLP_SYM_TASK = "G1-Beam-Touch-Curved-MLP-Sym"
TOUCH_MLP_PLAIN_TASK = "G1-Beam-Touch-Curved-MLP-Plain"
BLIND_MLP_SYM_TASK = "G1-Beam-Curved-Blind-MLP-Sym"
BLIND_MLP_PLAIN_TASK = "G1-Beam-Curved-Blind-MLP-Plain"
TOUCH_TRANSFORMER_TASK = "G1-Beam-Touch-Curved-Transformer"
TOUCH_PLAIN_R64_TASK = "G1-Beam-Touch-Curved-Plain-R64"
TOUCH_PLAIN_R2_TASK = "G1-Beam-Touch-Curved-Plain-R2-35s"
TOUCH_PLAIN_R2_R64_TASK = "G1-Beam-Touch-Curved-Plain-R2-35s-R64"
TOUCH_LSTM_R2_TASK = "G1-Beam-Touch-Curved-LSTM-R2-35s"
TOUCH_LSTM_R2_R64_TASK = "G1-Beam-Touch-Curved-LSTM-R2-35s-R64"
TRANSFORMER_HISTORY = 64
"""Frames the transformer sees: 1.28 s at 50 Hz, about one gait cycle of the
slow beam gait (~1.4 steps/s at 0.2 m/s)."""


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
  # task, curves, tactile, actor, experiment
  (TASK, False, "map", "sym_gru", "beam_e2e"),
  (CURVED_TASK, True, "map", "sym_gru", "beam_e2e_curved"),
  (TOUCH_TASK, True, "touch", "sym_gru", "beam_touch_curved"),
  (TOUCH_PLAIN_TASK, True, "touch", "gru", "beam_touch_curved_plain"),
  (BLIND_TASK, True, "none", "gru", "beam_curved_blind"),
  (BLIND_SYM_TASK, True, "none", "sym_gru", "beam_curved_blind_sym"),
  (TOUCH_MLP_SYM_TASK, True, "touch", "sym_mlp", "beam_touch_curved_mlp_sym"),
  (TOUCH_MLP_PLAIN_TASK, True, "touch", "mlp", "beam_touch_curved_mlp_plain"),
  (BLIND_MLP_SYM_TASK, True, "none", "sym_mlp", "beam_curved_blind_mlp_sym"),
  (BLIND_MLP_PLAIN_TASK, True, "none", "mlp", "beam_curved_blind_mlp_plain"),
)

def reward_task(variant: str) -> str:
  return f"{TOUCH_MLP_PLAIN_TASK}-R-" + "-".join(w.title() for w in variant.split("_"))


_REWARD_TASKS = tuple(
  (reward_task(v), True, "touch", "mlp", f"beam_touch_mlp_plain_r_{v}", v) for v in REWARD_VARIANTS
)
# task, curves, tactile, actor, experiment, reward, actor history, rollout steps, round2
_ALL_TASKS = (
  *((*t, "base", 0, 32, False) for t in _TASKS),
  *((*t, 0, 32, False) for t in _REWARD_TASKS),
  (TOUCH_TRANSFORMER_TASK, True, "touch", "transformer", "beam_touch_curved_transformer", "base",
   TRANSFORMER_HISTORY, 32, False),
  (TOUCH_PLAIN_R64_TASK, True, "touch", "gru", "beam_touch_curved_plain_r64", "base", 0, 64, False),
  (TOUCH_PLAIN_R2_TASK, True, "touch", "gru", "beam_touch_curved_plain_r2_35s", "base", 0, 32, True),
  (TOUCH_PLAIN_R2_R64_TASK, True, "touch", "gru", "beam_touch_curved_plain_r2_35s_r64", "base",
   0, 64, True),
  (TOUCH_LSTM_R2_TASK, True, "touch", "lstm", "beam_touch_curved_lstm_r2_35s", "base", 0, 32, True),
  (TOUCH_LSTM_R2_R64_TASK, True, "touch", "lstm", "beam_touch_curved_lstm_r2_35s_r64", "base",
   0, 64, True),
)
for _task, _curves, _tactile, _actor, _experiment, _reward, _hist, _steps, _r2 in _ALL_TASKS:
  _kw = dict(curves=_curves, tactile=_tactile, reward=_reward, actor_history=_hist, round2=_r2)
  for _suffix, _evaluation in (("", False), ("-Eval", True)):
    _env = make_beam_env_cfg(evaluation=_evaluation, **_kw)
    assert_actor_is_deployable(_env.observations)
    register_mjlab_task(
      task_id=_task + _suffix,
      env_cfg=_env,
      play_env_cfg=make_beam_env_cfg(play=True, evaluation=_evaluation, **_kw),
      rl_cfg=beam_ppo_cfg(experiment_name=_experiment, actor=_actor, num_steps_per_env=_steps),
      runner_cls=BeamRunner,
    )
