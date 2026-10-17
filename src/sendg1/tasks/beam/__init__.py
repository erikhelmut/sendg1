"""Beam walking, plain PPO: ``G1-Beam-E2E`` (straight beams) and
``G1-Beam-E2E-Curved`` (straight and bent beams).

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


for _task, _curves, _experiment in ((TASK, False, "beam_e2e"), (CURVED_TASK, True, "beam_e2e_curved")):
  _env = make_beam_env_cfg(curves=_curves)
  assert_actor_is_deployable(_env.observations)
  register_mjlab_task(
    task_id=_task,
    env_cfg=_env,
    play_env_cfg=make_beam_env_cfg(play=True, curves=_curves),
    rl_cfg=beam_ppo_cfg(experiment_name=_experiment),
  )
