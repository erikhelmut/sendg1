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
  "Sendg1-Beam-G1-Blind": ("75e0eb08d6580e25", "0c2b4d917e7cad8a"),
  "Sendg1-Beam-G1-ContactOracle": ("f45a6debc2e8e90b", "8b480392886bedce"),
  "Sendg1-Beam-G1-TactileArray": ("431abce03559626b", "6e81b2cd00f2f549"),
  "Sendg1-BoxPush-G1-Blind": ("7533a1505c85e786", "98bf70d84bc5f8f4"),
  "Sendg1-BoxPush-G1-ContactOracle": ("42594245c7c3f067", "d93f5da5e7ba6f3f"),
  "Sendg1-BoxPush-G1-TactileArray": ("7018b33ba0c4aedf", "6db095ff927a2deb"),
  "Sendg1-BoxPushCurriculum-G1-Blind": ("72b05ea69922d437", "2df9427cfb5bac97"),
  "Sendg1-BoxPushCurriculum-G1-ContactOracle": ("ca231c822fd6903f", "df8dff01001f971d"),
  "Sendg1-BoxPushCurriculum-G1-TactileArray": ("03b0796ddd0a7b2c", "173f446b583305d3"),
  "Sendg1-BoxPushPatches-G1-Blind": ("5dbfb5550ed27c5b", "8eee971493c901a8"),
  "Sendg1-BoxPushPatches-G1-ContactOracle": ("a183ed7449143d7a", "f6b7529b1cd4dc76"),
  "Sendg1-BoxPushPatches-G1-TactileArray": ("12ff0f8b87671550", "c1d0517dff465260"),
  "Sendg1-Velocity-Flat-G1-Blind": ("4b16b31d3100856e", "4eca2b5efbd3cacf"),
  "Sendg1-Velocity-Flat-G1-ContactOracle": ("793b9160cd3a52eb", "ef6cb7888e840c6e"),
  "Sendg1-Velocity-Flat-G1-TactileArray": ("1f0aaab06be8c661", "894b9801885aa05e"),
  "Sendg1-Velocity-Rough-G1-Blind": ("abb89a02f83b898c", "f9c4880fbb9131ba"),
  "Sendg1-Velocity-Rough-G1-ContactOracle": ("130c4a42148b67be", "bbe10c74e54dce3c"),
  "Sendg1-Velocity-Rough-G1-TactileArray": ("f2cecf46fe0ecdd8", "462f06834bdebc26"),
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
