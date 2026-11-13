"""Pedalo riding, plain PPO: ``G1-Pedalo``, ``G1-Pedalo-Sport`` and
``G1-Pedalo-Sport-Blind``.

The G1 rides a Pedalo Classic (``G1-Pedalo``) or a Pedalo Sport
(``G1-Pedalo-Sport``) forward at a joystick speed, or holds it still. Same
actor inputs and PPO settings as the beam task's ``G1-Beam-Touch-Curved-Plain``
(plain GRU on the robot's own signals and the tactile touch features), a
privileged critic. ``G1-Pedalo-Sport-Blind`` is the Sport task with the touch
features taken out of the actor and everything else identical, as the
tactile-vs-blind baseline.

``G1-Pedalo-Sym`` and ``G1-Pedalo-Sport-Sym`` keep the symmetric GRU actor
(``sendg1.rl.symmetric_rnn``) the first pedalo runs were trained with, so
their checkpoints still load for eval and video. Not for new runs: the
symmetric actor needs the mirror maps at run time and has no export yet.
"""

from __future__ import annotations

from mjlab.tasks.registry import register_mjlab_task

from sendg1.common.pedalo.terms import DEPLOYABLE, TERMS
from sendg1.common.pedalo.variants import CLASSIC, SPORT
from sendg1.tasks.beam.rl_cfg import beam_ppo_cfg
from sendg1.tasks.pedalo.env_cfg import make_pedalo_env_cfg

TASK = "G1-Pedalo"
SPORT_TASK = "G1-Pedalo-Sport"
SPORT_BLIND_TASK = "G1-Pedalo-Sport-Blind"
SYM_TASK = "G1-Pedalo-Sym"
SPORT_SYM_TASK = "G1-Pedalo-Sport-Sym"


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
  # task, variant, blind, symmetric actor, experiment
  (TASK, CLASSIC, False, False, "pedalo_plain"),
  (SPORT_TASK, SPORT, False, False, "pedalo_sport_plain"),
  (SPORT_BLIND_TASK, SPORT, True, False, "pedalo_sport_blind"),
  (SYM_TASK, CLASSIC, False, True, "pedalo"),
  (SPORT_SYM_TASK, SPORT, False, True, "pedalo_sport"),
)
for _task, _variant, _blind, _symmetric, _experiment in _TASKS:
  _env = make_pedalo_env_cfg(variant=_variant, blind=_blind)
  assert_actor_is_deployable(_env.observations)
  register_mjlab_task(
    task_id=_task,
    env_cfg=_env,
    play_env_cfg=make_pedalo_env_cfg(play=True, variant=_variant, blind=_blind),
    rl_cfg=beam_ppo_cfg(experiment_name=_experiment, symmetric=_symmetric),
  )
