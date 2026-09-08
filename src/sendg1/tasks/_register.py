"""Fan a task definition across every observation set and register it.

Adding task N+1 is one directory plus one ``register_variants`` call. Adding an
observation set is one entry in ``OBS_SETS``; it fans out across every existing
task automatically.
"""

from __future__ import annotations

from typing import Callable

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.rl import RslRlBaseRunnerCfg
from mjlab.tasks.registry import register_mjlab_task

from sendg1.common.obs.obs_sets import OBS_SETS, ObsSet
from sendg1.common.obs.terms import TERMS

EnvCfgFactory = Callable[..., ManagerBasedRlEnvCfg]
RlCfgFactory = Callable[[str, dict], RslRlBaseRunnerCfg]

_VARIANTS: dict[str, tuple[str, str]] = {}
"""task_id -> (base_id, obs_set). Lets scripts and tests enumerate the sweep."""


def _slug(obs_set: str) -> str:
  return "".join(part.capitalize() for part in obs_set.split("_"))


def assert_actor_is_legal(
  task_id: str,
  obs_set_name: str,
  obs_set: ObsSet,
  observations: dict,
) -> None:
  """Fail at import if a policy group carries a term it must not.

  This is the same rule tests/test_actor_allowlist.py enforces, applied at
  registration time so that ``train`` refuses to start rather than failing only
  under pytest.
  """
  for group_name in obs_set.actor:
    group = observations.get(group_name)
    if group is None:
      raise KeyError(
        f"{task_id}: obs_set {obs_set_name!r} maps the actor to group "
        f"{group_name!r}, which the env config does not define."
      )
    for term_name in group.terms:
      spec = TERMS.get(term_name)
      if spec is None:
        raise KeyError(
          f"{task_id}: actor group {group_name!r} contains term {term_name!r}, "
          "which is not declared in sendg1.common.obs.terms.TERMS. Every actor "
          "term must declare its hardware availability."
        )
      if spec.availability == "privileged":
        raise ValueError(
          f"{task_id}: actor group {group_name!r} contains PRIVILEGED term "
          f"{term_name!r}. The actor may only observe quantities available on "
          f"the real blind G1. Rationale on record: {spec.rationale}"
        )
      if spec.availability == "oracle" and not obs_set.diagnostic:
        raise ValueError(
          f"{task_id}: actor group {group_name!r} contains ORACLE term "
          f"{term_name!r}, but obs_set {obs_set_name!r} is not marked "
          "diagnostic. Oracle terms are idealised and do not describe a "
          "deployable policy."
        )


def register_variants(
  base_id: str,
  env_cfg_factory: EnvCfgFactory,
  rl_cfg_factory: RlCfgFactory,
  *,
  experiment_prefix: str,
) -> None:
  """Register one task id per observation set."""
  for obs_set_name, obs_set in OBS_SETS.items():
    task_id = f"{base_id}-{_slug(obs_set_name)}"
    env_cfg = env_cfg_factory(obs_set=obs_set_name, play=False)
    play_cfg = env_cfg_factory(obs_set=obs_set_name, play=True)

    assert_actor_is_legal(task_id, obs_set_name, obs_set, env_cfg.observations)

    rl_cfg = rl_cfg_factory(
      f"{experiment_prefix}__{obs_set_name}",
      obs_set.obs_groups(),
    )
    register_mjlab_task(
      task_id=task_id,
      env_cfg=env_cfg,
      play_env_cfg=play_cfg,
      rl_cfg=rl_cfg,
    )
    _VARIANTS[task_id] = (base_id, obs_set_name)


def variants() -> dict[str, tuple[str, str]]:
  """All registered task ids -> (base_id, obs_set)."""
  return dict(_VARIANTS)
