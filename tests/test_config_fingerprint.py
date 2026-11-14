"""Upstream-drift tripwire.

A pinned mjlab version is necessary but not sufficient: it does not tell you
*what* changed when you deliberately upgrade, and it does not protect against a
local edit that silently retunes the baseline. These fingerprints do.

When one fails: run ``python scripts/show_fingerprints.py --task <id> --dump`` to
see the canonical config, diff it against the previous commit, and decide whether
the change was intended. If it was, update the expected hash here IN THE SAME
COMMIT as the change, and re-run every baseline that moved.
"""

from __future__ import annotations

import pytest
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

import sendg1.tasks  # noqa: F401
from sendg1.fingerprint import fingerprint
from sendg1.tasks._register import variants

# Regenerate with: python scripts/show_fingerprints.py
EXPECTED: dict[str, tuple[str, str]] = {
  # task_id: (env_fingerprint, rl_fingerprint)
  "Sendg1-BoxPush-G1-Blind": ("ce9384031658ef67", "c16b31a0a6530469"),
  "Sendg1-BoxPush-G1-ContactOracle": ("c9a59f010fd39429", "60a8f3e89832aa13"),
  "Sendg1-BoxPush-G1-Tactile": ("4b0d1b6c4555b884", "06be4eaf3e4d3ce8"),
  "Sendg1-BoxPushCurriculum-G1-Blind": ("09953bcccd9d544e", "e98b9cbb19a07132"),
  "Sendg1-BoxPushCurriculum-G1-ContactOracle": ("17b9012a9b98a547", "721027b642ad80c8"),
  "Sendg1-BoxPushCurriculum-G1-Tactile": ("35ef61a46cd76e09", "3286945506885d8f"),
  "Sendg1-BoxPushPatches-G1-Blind": ("bddd7fddae4db332", "e51d215245977ada"),
  "Sendg1-BoxPushPatches-G1-ContactOracle": ("90f84414891ff12b", "2d2fe677a07f7c74"),
  "Sendg1-BoxPushPatches-G1-Tactile": ("c835ca55909ea0d3", "5ed81f293074d7b2"),
  "Sendg1-Velocity-Flat-G1-Blind": ("912f562976564235", "4eca2b5efbd3cacf"),
  "Sendg1-Velocity-Flat-G1-ContactOracle": ("749e5d6497f1b988", "ef6cb7888e840c6e"),
  "Sendg1-Velocity-Flat-G1-Tactile": ("2cd95a2d14a7cbb7", "f37b879819fe3569"),
  "Sendg1-Velocity-Rough-G1-Blind": ("9e4a616371d70df4", "f9c4880fbb9131ba"),
  "Sendg1-Velocity-Rough-G1-ContactOracle": ("d56bf871a3aff876", "bbe10c74e54dce3c"),
  "Sendg1-Velocity-Rough-G1-Tactile": ("081e6e612cb7313d", "351907bde540e06b"),
  "Sendg1-VelocityPush-G1-Blind": ("712d67743087b88b", "c10fe67780511509"),
  "Sendg1-VelocityPush-G1-ContactOracle": ("440002d06098c736", "e9076274d270ebd5"),
  "Sendg1-VelocityPush-G1-Tactile": ("69ea67fcc3a0cf4f", "8423d0976865d29c"),
}


@pytest.mark.parametrize("task_id", sorted(variants()))
def test_fingerprint_is_stable_within_process(task_id: str) -> None:
  """Canonicalisation must not depend on object identity or dict ordering."""
  a = fingerprint(load_env_cfg(task_id))
  b = fingerprint(load_env_cfg(task_id))
  assert a == b, f"{task_id}: fingerprint is not deterministic"


@pytest.mark.parametrize("task_id", sorted(variants()))
def test_fingerprint_matches_recorded(task_id: str) -> None:
  if not EXPECTED:
    pytest.skip(
      "No fingerprints recorded yet. Run `python scripts/show_fingerprints.py "
      "--write` once the configs are settled."
    )
  assert task_id in EXPECTED, f"{task_id}: no recorded fingerprint"
  env_fp = fingerprint(load_env_cfg(task_id))
  rl_fp = fingerprint(load_rl_cfg(task_id))
  exp_env, exp_rl = EXPECTED[task_id]
  assert (env_fp, rl_fp) == (exp_env, exp_rl), (
    f"{task_id}: config changed.\n"
    f"  env: {exp_env} -> {env_fp}\n"
    f"  rl:  {exp_rl} -> {rl_fp}\n"
    "If intentional, update EXPECTED in this file and re-run affected baselines."
  )


def test_reward_set_is_identical_across_obs_sets() -> None:
  """Rewards are a control variable. Contact-dependent terms are kept
  deliberately (see sendg1/common/rewards.py) but must not differ by condition."""
  by_base: dict[str, dict[str, str]] = {}
  for task_id, (base_id, obs_set) in variants().items():
    by_base.setdefault(base_id, {})[obs_set] = fingerprint(load_env_cfg(task_id).rewards)
  for base_id, per_set in by_base.items():
    assert len(set(per_set.values())) == 1, (
      f"{base_id}: reward set differs across observation sets: {per_set}"
    )


def test_events_are_identical_across_obs_sets() -> None:
  by_base: dict[str, dict[str, str]] = {}
  for task_id, (base_id, obs_set) in variants().items():
    by_base.setdefault(base_id, {})[obs_set] = fingerprint(load_env_cfg(task_id).events)
  for base_id, per_set in by_base.items():
    assert len(set(per_set.values())) == 1, (
      f"{base_id}: events differ across observation sets: {per_set}"
    )


def test_network_architecture_is_identical_across_obs_sets() -> None:
  """Architecture is a control variable too -- see tasks/velocity/rl_cfg.py."""
  by_base: dict[str, dict[str, tuple]] = {}
  for task_id, (base_id, obs_set) in variants().items():
    rl = load_rl_cfg(task_id)
    by_base.setdefault(base_id, {})[obs_set] = (
      tuple(rl.actor.hidden_dims),
      rl.actor.activation,
      tuple(rl.critic.hidden_dims),
      rl.critic.activation,
    )
  for base_id, per_set in by_base.items():
    assert len(set(per_set.values())) == 1, (
      f"{base_id}: network architecture differs across obs sets: {per_set}"
    )


def test_fingerprint_is_stable_across_processes() -> None:
  """The recorded hashes are worthless if they move between interpreters.

  Within-process stability is not enough: a hash that varied per process would
  fail on every CI run, and a tripwire that always fires gets ignored. This
  catches the classic causes -- a ``repr`` carrying a memory address, an
  unordered set, or dict iteration order leaking into the digest.
  """
  import subprocess
  import sys

  code = (
    "import sendg1.tasks;"
    "from mjlab.tasks.registry import load_env_cfg, load_rl_cfg;"
    "from sendg1.fingerprint import fingerprint;"
    "from sendg1.tasks._register import variants;"
    "print(' '.join(f'{t}:{fingerprint(load_env_cfg(t))}:"
    "{fingerprint(load_rl_cfg(t))}' for t in sorted(variants())))"
  )
  runs = []
  for _ in range(2):
    proc = subprocess.run(
      [sys.executable, "-c", code], capture_output=True, text=True, timeout=600
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    runs.append(proc.stdout.strip().split()[-len(variants()) :])

  assert runs[0] == runs[1], (
    "fingerprints differ between processes:\n"
    f"  run 1: {runs[0]}\n  run 2: {runs[1]}"
  )
