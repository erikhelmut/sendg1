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
  # A deep start: the term registry pulls in the beam and box-push mdp modules
  # (and through them robots, sensors, tactile) before sendg1.tasks exists.
  "from sendg1.common.obs.terms import TERMS",
]
"""Each costs a fresh interpreter (~4 s); 12 entry points were trimmed to these
three on 2026-11-15, which between them start the import at the package root,
at the task registry, and deep inside ``common``."""


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
