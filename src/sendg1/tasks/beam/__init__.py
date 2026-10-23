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

Every task has an evaluation-only twin, ``<task>-Eval``: held-out beams and
35 s episodes (``make_beam_env_cfg(evaluation=True)``), same observations, so
its training checkpoints load. Not for training.

A GRU actor on the robot's own signals and the tactile slipper, a privileged
critic. Trained with mjlab's ``train``; evaluated with
``scripts/beam_eval.py``.
"""

from __future__ import annotations

from mjlab.tasks.registry import register_mjlab_task

from sendg1.common.beam.terms import DEPLOYABLE, TERMS
from sendg1.tasks.beam.env_cfg import make_beam_env_cfg
from sendg1.tasks.beam.rl_cfg import beam_ppo_cfg

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
for _task, _curves, _tactile, _actor, _experiment in _TASKS:
  _env = make_beam_env_cfg(curves=_curves, tactile=_tactile)
  assert_actor_is_deployable(_env.observations)
  register_mjlab_task(
    task_id=_task,
    env_cfg=_env,
    play_env_cfg=make_beam_env_cfg(play=True, curves=_curves, tactile=_tactile),
    rl_cfg=beam_ppo_cfg(experiment_name=_experiment, actor=_actor),
  )
  _eval = make_beam_env_cfg(curves=_curves, tactile=_tactile, evaluation=True)
  assert_actor_is_deployable(_eval.observations)
  register_mjlab_task(
    task_id=f"{_task}-Eval",
    env_cfg=_eval,
    play_env_cfg=make_beam_env_cfg(play=True, curves=_curves, tactile=_tactile, evaluation=True),
    rl_cfg=beam_ppo_cfg(experiment_name=_experiment, actor=_actor),
  )
