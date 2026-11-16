"""Move the slipper CAD export into the G1 ankle-roll-link frame.

Usage: ``uv run python scripts/make_slipper_asset.py <G1_slipper.stl>``

The CAD export sits in an assembly frame (mm, toe toward -x, tilted ~1 deg).
This fits the plane of its flat bottom, rotates that plane level, turns the toe
to +x, centres the outline on the G1 foot mesh and puts the inside of the 3 mm
floor on the bare G1 sole (``G1_SOLE_Z``). The result (metres) is written to
``src/g1_tactile/assets/g1_slipper.stl``; the robot model reads it
from there (``g1_tactile.slipper``).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from g1_tactile.layout import G1_SOLE_Z
from g1_tactile.slipper import (
  ASSET,
  G1_FOOT_CENTER_XY,
  SLIPPER_FLOOR,
  read_stl,
  write_stl,
)


def _rotation_to(a: np.ndarray, b: np.ndarray) -> np.ndarray:
  """Smallest rotation taking unit vector a to unit vector b."""
  v, c = np.cross(a, b), float(a @ b)
  k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
  return np.eye(3) + k + k @ k / (1 + c)


def main(src: str) -> None:
  tris = read_stl(Path(src)) / 1000.0  # [N, 3, 3] m
  e1, e2 = tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]
  n = np.cross(e1, e2)
  n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
  # The flat bottom: the largest set of downward faces, all near the lowest point.
  down = tris[n[:, 2] < -0.99]
  pts = down.reshape(-1, 3)
  # Least-squares plane z = ax + by + c through the downward faces' vertices,
  # keeping only those within 1 mm of the plane (drops pockets and fillets).
  keep = np.ones(len(pts), bool)
  for _ in range(5):
    A = np.column_stack([pts[keep, 0], pts[keep, 1], np.ones(keep.sum())])
    a, b, c = np.linalg.lstsq(A, pts[keep, 2], rcond=None)[0]
    res = pts[:, 2] - (a * pts[:, 0] + b * pts[:, 1] + c)
    keep = np.abs(res) < 0.001
  normal = np.array([-a, -b, 1.0])
  normal /= np.linalg.norm(normal)
  level = _rotation_to(normal, np.array([0.0, 0.0, 1.0]))
  yaw180 = np.diag([-1.0, -1.0, 1.0])  # the export's toe points along -x
  rot = yaw180 @ level
  out = tris @ rot.T
  flat = out.reshape(-1, 3)
  bottom = flat[:, 2].min()
  near = flat[flat[:, 2] < bottom + 0.0003]
  lo, hi = near[:, :2].min(0), near[:, :2].max(0)
  shift = np.array(
    [*(G1_FOOT_CENTER_XY - (lo + hi) / 2), G1_SOLE_Z - SLIPPER_FLOOR - bottom]
  )
  out = out + shift
  print(f"tilt removed: {np.degrees(np.arccos(normal[2])):.2f} deg")
  print(f"plane fit kept {keep.sum()}/{len(keep)} bottom vertices")
  flat = out.reshape(-1, 3)
  print(
    "bbox min (mm)",
    np.round(flat.min(0) * 1000, 1),
    "max",
    np.round(flat.max(0) * 1000, 1),
  )
  write_stl(ASSET, out.astype(np.float32))
  print("wrote", ASSET)


if __name__ == "__main__":
  if len(sys.argv) != 2:
    sys.exit(__doc__)
  main(sys.argv[1])
