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

from importlib.metadata import version

MJLAB_VERSION = "1.6.0"
MJLAB_COMMIT = "0fb8a681136be94ffc636a3dd423cabb97d91f10"
"""github.com/mujocolab/mjlab tag v1.6.0. The PyPI wheel embeds no SHA; this was
resolved from the GitHub tags API and is recorded for provenance only."""


def check_mjlab_version(strict: bool = True) -> str:
  """Verify the installed mjlab matches the pin. Returns the installed version."""
  installed = version("mjlab")
  if installed != MJLAB_VERSION and strict:
    raise RuntimeError(
      f"sendg1 is pinned to mjlab=={MJLAB_VERSION} but found {installed}.\n"
      "Results are not comparable across mjlab versions. Either reinstall the "
      f"pinned version (`pip install mjlab=={MJLAB_VERSION}`) or, if the upgrade "
      "is intentional, follow the checklist in src/sendg1/_pin.py."
    )
  return installed
