"""Slackline: ``G1-Slackline-Touch`` and ``G1-Slackline-Blind``.

The G1 stands in a tandem stance on a tensioned slackline and, when
commanded, walks along it. A paired comparison: the two tasks are the same
env, critic, reward and PPO settings, and differ only in whether the actor
gets the tactile slipper's touch features. Actor and PPO:
the settings every task shares (``sendg1.common.ppo``).
"""

from __future__ import annotations

from mjlab.tasks.registry import register_mjlab_task

from sendg1.common.slackline.terms import DEPLOYABLE, TERMS
from sendg1.tasks.beam.rl_cfg import beam_ppo_cfg
from sendg1.tasks.slackline.env_cfg import make_slackline_env_cfg

TOUCH_TASK = "G1-Slackline-Touch"
BLIND_TASK = "G1-Slackline-Blind"
SURFACE_TASKS = {"box": "Slackline", "ribbon": "SlacklineFlex", "cable": "SlacklineCable"}
"""Line model (``sm.SURFACES``) -> task stem: ``G1-<stem>-{Touch,Blind}``. The
flex variants (a seamless ribbon, a round rope) share everything else."""
MAX_ITERATIONS = 20_000
"""The beam's: balance and walking on one line, no curriculum yet."""


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


BALANCE_TASKS = {True: "G1-SlacklineBalance-Touch", False: "G1-SlacklineBalance-Blind"}
"""Balance only: always 'stand' on the 5 cm box line with stiff contact,
levels from stiff to free, arms free (``make_slackline_env_cfg(balance_only)``)."""


def task_id(surface: str, tactile: bool) -> str:
  return f"G1-{SURFACE_TASKS[surface]}-{'Touch' if tactile else 'Blind'}"


for _surface, _tactile in ((s, t) for s in SURFACE_TASKS for t in (True, False)):
  _task = task_id(_surface, _tactile)
  _experiment = (
    f"{'slackline' if _surface == 'box' else f'slackline_{_surface}'}"
    f"_{'touch' if _tactile else 'blind'}"
  )
  _env = make_slackline_env_cfg(tactile=_tactile, surface=_surface)
  assert_actor_is_deployable(_env.observations)
  register_mjlab_task(
    task_id=_task,
    env_cfg=_env,
    play_env_cfg=make_slackline_env_cfg(play=True, tactile=_tactile, surface=_surface),
    rl_cfg=beam_ppo_cfg(
      experiment_name=_experiment, max_iterations=MAX_ITERATIONS
    ),
  )

for _tactile, _task in BALANCE_TASKS.items():
  _kw = {"tactile": _tactile, "contact": "stiff", "balance_only": True}
  _env = make_slackline_env_cfg(**_kw)
  assert_actor_is_deployable(_env.observations)
  register_mjlab_task(
    task_id=_task,
    env_cfg=_env,
    play_env_cfg=make_slackline_env_cfg(play=True, **_kw),
    rl_cfg=beam_ppo_cfg(
      experiment_name=f"slackline_balance_{'touch' if _tactile else 'blind'}",
      max_iterations=MAX_ITERATIONS,
    ),
  )
