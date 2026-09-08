"""Stable fingerprints for assembled configs.

Purpose: catch the failure mode that version pinning alone cannot. If an mjlab
upgrade (or an accidental local edit) changes any number that defines the
baseline -- a reward weight, a DR range, a noise bound, a curriculum stage --
the fingerprint moves and CI says so, naming the task. Upgrading then becomes a
deliberate act with a visible diff and a "re-run these baselines" list.

Callables are canonicalised by ``module.qualname`` rather than ``repr``, because
a function's repr contains its memory address and would make every hash unstable
across processes.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from enum import Enum
from typing import Any


def canonicalize(obj: Any) -> Any:
  """Convert a config object into a deterministically JSON-serialisable form."""
  if obj is None or isinstance(obj, (bool, int, str)):
    return obj
  if isinstance(obj, float):
    # Guard against platform float repr drift.
    return f"{obj:.12g}"
  if isinstance(obj, Enum):
    return f"{type(obj).__name__}.{obj.name}"
  if callable(obj):
    module = getattr(obj, "__module__", "?")
    qualname = getattr(obj, "__qualname__", getattr(obj, "__name__", repr(obj)))
    return f"<callable {module}.{qualname}>"
  if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
    return {
      "__type__": type(obj).__name__,
      **{f.name: canonicalize(getattr(obj, f.name)) for f in dataclasses.fields(obj)},
    }
  if isinstance(obj, dict):
    return {str(k): canonicalize(obj[k]) for k in sorted(obj, key=str)}
  if isinstance(obj, (list, tuple)):
    return [canonicalize(v) for v in obj]
  if isinstance(obj, (set, frozenset)):
    return sorted((canonicalize(v) for v in obj), key=str)
  # numpy scalars/arrays and anything else exotic.
  tolist = getattr(obj, "tolist", None)
  if callable(tolist):
    return canonicalize(tolist())
  return f"<{type(obj).__name__}>"


def fingerprint(obj: Any) -> str:
  """SHA-256 over the canonical form. Stable across processes and machines."""
  blob = json.dumps(canonicalize(obj), sort_keys=True, separators=(",", ":"))
  return hashlib.sha256(blob.encode()).hexdigest()[:16]


def config_report(obj: Any) -> str:
  """Human-readable canonical form, for diffing when a fingerprint moves."""
  return json.dumps(canonicalize(obj), sort_keys=True, indent=2)
