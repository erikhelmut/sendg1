"""Pedalo riding: ``G1-Pedalo``, ``G1-Pedalo-Sport`` and
``G1-Pedalo-Sport-Blind``.

The G1 rides a Pedalo Classic (``G1-Pedalo``) or a Pedalo Sport
(``G1-Pedalo-Sport``) forward at a joystick speed, or holds it still. Same
actor inputs as the beam task (the robot's own signals and the tactile touch
features), the standard LSTM actor and PPO settings (``sendg1.common.ppo``), a
privileged critic. ``G1-Pedalo-Sport-Blind`` is the Sport task with the
touch features taken out of the actor and everything else identical, as the
tactile-vs-blind baseline.
"""

from __future__ import annotations

from sendg1.common.pedalo.terms import DEPLOYABLE, TERMS
from sendg1.common.pedalo.variants import CLASSIC, SPORT
from sendg1.tasks._register import register_task
from sendg1.tasks.beam.rl_cfg import beam_ppo_cfg
from sendg1.tasks.pedalo.env_cfg import make_pedalo_env_cfg

TASK = "G1-Pedalo"
SPORT_TASK = "G1-Pedalo-Sport"
SPORT_BLIND_TASK = "G1-Pedalo-Sport-Blind"


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
  # task, variant, blind, experiment, family
  (TASK, CLASSIC, False, "pedalo_plain", "pedalo_classic"),
  (SPORT_TASK, SPORT, False, "pedalo_sport_plain", "pedalo_sport"),
  (SPORT_BLIND_TASK, SPORT, True, "pedalo_sport_blind", "pedalo_sport"),
)
for _task, _variant, _blind, _experiment, _family in _TASKS:
  _env = make_pedalo_env_cfg(variant=_variant, blind=_blind)
  assert_actor_is_deployable(_env.observations)
  register_task(
    _task,
    family=_family,
    condition="blind" if _blind else "tactile",
    env_cfg=_env,
    play_env_cfg=make_pedalo_env_cfg(play=True, variant=_variant, blind=_blind),
    rl_cfg=beam_ppo_cfg(experiment_name=_experiment),
  )
