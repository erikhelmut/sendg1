"""Importing any sendg1 module first must work.

mjlab imports sendg1.tasks (via the mjlab.tasks entry point) while sendg1 may
itself be mid-import. That cycle is only safe because sendg1/__init__ pins where
it starts. This test runs each entry point in a *fresh interpreter*, because an
import cycle can only be observed on a cold module cache.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

ENTRY_POINTS = [
  "import sendg1",
  "import sendg1.tasks",
  "from sendg1.common.obs.obs_sets import OBS_SETS",
  "from sendg1.common.obs.groups import build_groups",
  "from sendg1.common.obs.terms import TERMS",
  "from sendg1.common.robots.g1 import get_g1_cfg",
  "from sendg1.common.rewards import build_rewards",
  "from sendg1.eval.protocol import EvalProtocol",
  "from sendg1.fingerprint import fingerprint",
  "from sendg1.tasks.velocity.env_cfg import make_g1_velocity_env_cfg",
]


@pytest.mark.parametrize("stmt", ENTRY_POINTS)
def test_cold_import_succeeds(stmt: str) -> None:
  proc = subprocess.run(
    [sys.executable, "-c", f"{stmt}\nprint('OK')"],
    capture_output=True,
    text=True,
    timeout=300,
  )
  assert proc.returncode == 0, f"{stmt!r} failed:\n{proc.stderr[-2000:]}"
  assert "OK" in proc.stdout
  # mjlab's entry point loader swallows failures into a [WARN] + traceback
  # rather than raising, so a broken cycle would otherwise pass silently.
  assert "Failed to load task package" not in proc.stderr, (
    f"{stmt!r} left mjlab unable to load sendg1.tasks:\n{proc.stderr[-2000:]}"
  )
