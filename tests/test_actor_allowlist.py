"""The actor-observation allowlist.

Constraint: the actor may only observe quantities available on the real G1
(IMU angular velocity, projected gravity, joint positions/velocities, previous
action, user command). The critic may use privileged information.

This test fails if a non-deployable term ends up in a policy observation group.
It checks three distinct failure modes:

  1. a privileged term reaching an actor group;
  2. an oracle term reaching an actor group of a non-diagnostic obs set;
  3. a term appearing anywhere without being declared in TERMS -- which closes
     the bypass where someone adds an observation and simply forgets to
     register its availability.
"""

from __future__ import annotations

import pytest
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

import sendg1.tasks  # noqa: F401  (registers tasks)
from sendg1.common.obs.obs_sets import OBS_SETS
from sendg1.common.obs.terms import TERMS
from sendg1.tasks._register import variants

ALL_VARIANTS = sorted(variants())

# The exact set the research question permits an actor to see.
EXPECTED_DEPLOYABLE = {
  "base_ang_vel",
  "projected_gravity",
  "joint_pos",
  "joint_vel",
  "actions",
  "command",
}


def _actor_terms(task_id: str) -> dict[str, list[str]]:
  """group name -> term names, for the groups feeding the actor."""
  env_cfg = load_env_cfg(task_id)
  rl_cfg = load_rl_cfg(task_id)
  return {
    group: list(env_cfg.observations[group].terms)
    for group in rl_cfg.obs_groups["actor"]
  }


@pytest.mark.parametrize("task_id", ALL_VARIANTS)
def test_actor_contains_no_privileged_terms(task_id: str) -> None:
  obs_set = OBS_SETS[variants()[task_id][1]]
  for group, terms in _actor_terms(task_id).items():
    for term in terms:
      spec = TERMS[term]
      assert spec.availability != "privileged", (
        f"{task_id}: actor group {group!r} contains privileged term {term!r}. "
        f"Rationale on record: {spec.rationale}"
      )
      if spec.availability == "oracle":
        assert obs_set.diagnostic, (
          f"{task_id}: actor group {group!r} contains oracle term {term!r} but "
          "the obs set is not marked diagnostic."
        )


@pytest.mark.parametrize("task_id", ALL_VARIANTS)
def test_every_actor_term_is_declared(task_id: str) -> None:
  for group, terms in _actor_terms(task_id).items():
    for term in terms:
      assert term in TERMS, (
        f"{task_id}: actor group {group!r} contains undeclared term {term!r}. "
        "Add it to sendg1.common.obs.terms.TERMS with an availability tag."
      )


@pytest.mark.parametrize("task_id", ALL_VARIANTS)
def test_every_observation_term_anywhere_is_declared(task_id: str) -> None:
  """Critic terms must be declared too -- otherwise availability tags rot."""
  env_cfg = load_env_cfg(task_id)
  for group_name, group in env_cfg.observations.items():
    for term in group.terms:
      assert term in TERMS, (
        f"{task_id}: group {group_name!r} contains undeclared term {term!r}."
      )


def test_blind_actor_is_exactly_the_deployable_set() -> None:
  """Pin the baseline actor's contents, not just their availability class."""
  for task_id, (_, obs_set) in variants().items():
    if obs_set != "blind":
      continue
    terms = {t for group in _actor_terms(task_id).values() for t in group}
    assert terms == EXPECTED_DEPLOYABLE, (
      f"{task_id}: blind actor observes {sorted(terms)}, expected "
      f"{sorted(EXPECTED_DEPLOYABLE)}"
    )


def test_no_actor_sees_height_scan_or_base_lin_vel() -> None:
  """Explicit regression guard on the two terms most likely to creep back in.

  Both are in mjlab's default G1 actor group; both are excluded here by
  deliberate decision (blind policies, no state estimator).
  """
  for task_id in ALL_VARIANTS:
    terms = {t for group in _actor_terms(task_id).values() for t in group}
    assert "height_scan" not in terms, f"{task_id}: actor regained height_scan"
    assert "base_lin_vel" not in terms, f"{task_id}: actor regained base_lin_vel"


def test_critic_is_identical_across_obs_sets() -> None:
  """The critic is a control variable: only the actor may differ.

  If the critic changed alongside the actor, a measured difference could not be
  attributed to the actor's observation space.
  """
  by_base: dict[str, dict[str, tuple[str, ...]]] = {}
  for task_id, (base_id, obs_set) in variants().items():
    rl_cfg = load_rl_cfg(task_id)
    by_base.setdefault(base_id, {})[obs_set] = rl_cfg.obs_groups["critic"]

  for base_id, per_set in by_base.items():
    distinct = set(per_set.values())
    assert len(distinct) == 1, (
      f"{base_id}: critic observation groups differ across obs sets: {per_set}"
    )


def test_diagnostic_sets_are_flagged_and_not_default() -> None:
  from sendg1.common.obs.obs_sets import DEFAULT_OBS_SET, research_obs_sets

  assert not OBS_SETS[DEFAULT_OBS_SET].diagnostic
  assert "contact_oracle" in OBS_SETS
  assert OBS_SETS["contact_oracle"].diagnostic
  assert "contact_oracle" not in research_obs_sets()


# ---------------------------------------------------------------------------
# Negative tests: prove the guard actually fires.
#
# An allowlist that has never rejected anything is not evidence of anything.
# These construct deliberately illegal configs and assert they are refused.
# ---------------------------------------------------------------------------


def _fake_group(*term_names: str):
  from mjlab.managers.observation_manager import ObservationGroupCfg

  return ObservationGroupCfg(
    terms={n: TERMS[n].build() for n in term_names},
    concatenate_terms=True,
  )


def test_guard_rejects_privileged_term_in_actor() -> None:
  from sendg1.tasks._register import assert_actor_is_legal

  obs_set = OBS_SETS["blind"]
  bad = {"proprio": _fake_group("base_ang_vel", "height_scan")}
  with pytest.raises(ValueError, match="PRIVILEGED"):
    assert_actor_is_legal("Fake-Task", "blind", obs_set, bad)


def test_guard_rejects_base_lin_vel_in_actor() -> None:
  from sendg1.tasks._register import assert_actor_is_legal

  obs_set = OBS_SETS["blind"]
  bad = {"proprio": _fake_group("base_ang_vel", "base_lin_vel")}
  with pytest.raises(ValueError, match="PRIVILEGED"):
    assert_actor_is_legal("Fake-Task", "blind", obs_set, bad)


def test_guard_rejects_oracle_term_in_non_diagnostic_set() -> None:
  from sendg1.tasks._register import assert_actor_is_legal

  obs_set = OBS_SETS["blind"]
  bad = {"proprio": _fake_group("base_ang_vel", "foot_contact_ideal")}
  with pytest.raises(ValueError, match="ORACLE"):
    assert_actor_is_legal("Fake-Task", "blind", obs_set, bad)


def test_guard_accepts_oracle_term_in_diagnostic_set() -> None:
  from sendg1.tasks._register import assert_actor_is_legal

  obs_set = OBS_SETS["contact_oracle"]
  ok = {
    "proprio": _fake_group("base_ang_vel"),
    "contact_ideal": _fake_group("foot_contact_ideal"),
  }
  assert_actor_is_legal("Fake-Task", "contact_oracle", obs_set, ok)


def test_guard_rejects_undeclared_term() -> None:
  from mjlab.envs import mdp as envs_mdp
  from mjlab.managers.observation_manager import (
    ObservationGroupCfg,
    ObservationTermCfg,
  )

  from sendg1.tasks._register import assert_actor_is_legal

  obs_set = OBS_SETS["blind"]
  bad = {
    "proprio": ObservationGroupCfg(
      terms={
        "smuggled_in": ObservationTermCfg(
          func=envs_mdp.base_lin_vel  # never registered in TERMS
        )
      },
      concatenate_terms=True,
    )
  }
  with pytest.raises(KeyError, match="not declared"):
    assert_actor_is_legal("Fake-Task", "blind", obs_set, bad)


def test_guard_rejects_actor_mapped_to_missing_group() -> None:
  from sendg1.tasks._register import assert_actor_is_legal

  obs_set = OBS_SETS["blind"]
  with pytest.raises(KeyError, match="does not define"):
    assert_actor_is_legal("Fake-Task", "blind", obs_set, {"privileged": _fake_group()})
