"""Hard pin on the mjlab version this study is calibrated against.

Why this exists
---------------
Every result in this repository is a comparison between a baseline policy and a
tactile-augmented policy. Those comparisons are only meaningful if the environment
is bit-stable across the months the study runs. mjlab is young (12 releases at time
of writing) and ships breaking changes between minor versions, so a silent upgrade
that retunes, say, ``foot_clearance`` from -2.0 to -1.5 would invalidate every
cross-run comparison without producing a single error.

Pinning in pyproject.toml is necessary but not sufficient: it does not protect a
developer who installed mjlab by hand, and it does not tell you *what changed*.
So we also assert at import, and we fingerprint the fully-assembled configs
(see tests/test_config_fingerprint.py).

Upgrading mjlab is allowed. It is a deliberate act with a checklist:
  1. bump MJLAB_VERSION / MJLAB_COMMIT here and in pyproject.toml,
  2. run `pytest` and inspect the config-fingerprint diff,
  3. re-run every baseline whose fingerprint moved.
"""

import json
from importlib.metadata import distribution, version

MJLAB_VERSION = "1.6.0"
MJLAB_COMMIT = "1b5c91c923ee4bc4159f3dce9cc85b52f88b6bd5"
"""github.com/mujocolab/mjlab main of 2026-11-13, the first with MuJoCo 4
(mujoco-core / mujoco-warp 4.0.1, torch >= 2.14, rsl-rl 5.5.2). No release has
it yet and the package still reports 1.6.0, so the version alone no longer
identifies the code: the commit is checked too (the 1.6.0 release was 0fb8a68)."""


def installed_mjlab_commit() -> str | None:
  """The commit a git install of mjlab came from (PEP 610 direct_url.json), or
  None for an install from a wheel or sdist."""
  raw = distribution("mjlab").read_text("direct_url.json")
  if raw is None:
    return None
  return json.loads(raw).get("vcs_info", {}).get("commit_id")


def check_mjlab_version(strict: bool = True) -> str:
  """Verify the installed mjlab matches the pin (version and commit). Returns the
  installed version."""
  installed = version("mjlab")
  commit = installed_mjlab_commit()
  if (installed != MJLAB_VERSION or commit != MJLAB_COMMIT) and strict:
    raise RuntimeError(
      f"sendg1 is pinned to mjlab {MJLAB_VERSION} at commit {MJLAB_COMMIT[:7]} but "
      f"found {installed} at {commit[:7] if commit else 'no recorded commit'}.\n"
      "Results are not comparable across mjlab versions. Either reinstall the "
      "pinned version (`uv sync --frozen`) or, if the upgrade is intentional, "
      "follow the checklist in src/sendg1/_pin.py."
    )
  return installed
