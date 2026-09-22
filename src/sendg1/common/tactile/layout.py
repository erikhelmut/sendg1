"""Where the taxels sit on the sole.

Everything is expressed in the **sole frame**, which is the ``*_ankle_roll_link``
body frame: x forward, y left, z up. The sole plane is the bottom of the 7 foot
collision capsules (and, to 0.4 mm, of the foot's visual mesh), at
``SOLE_Z = -0.035`` m.

The model uses ``slipper_v1``: the real sensor on the slipper, 26 square
patches of 12 mm, read from ``layouts/slipper_v1.csv`` (millimetres, exported
from the slipper CAD). ``placeholder_8x3`` is the earlier stand-in grid, kept
because the model's unit tests are written against a regular grid. Everything
downstream reads ``TaxelLayout`` and never assumes a grid.

Right foot: ``slipper_v1`` is the SAME part on both feet (1:1, not a mirrored
version), so taxel ``i`` sits at the same (x, y) on both soles -- channel ``i`` is
lateral on one foot and medial on the other. ``placeholder_8x3`` mirrors
(y -> -y) instead. ``TaxelLayout.mirror_right`` selects which.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

Side = Literal["left", "right"]
SIDES: tuple[Side, ...] = ("left", "right")

SOLE_Z = -0.035
"""Sole plane in the ankle-roll-link frame: capsule axis z=-0.025 minus radius
0.01. The visual mesh bottoms out at -0.0354."""

SOLE_EXTENT_X = (-0.066, 0.142)
SOLE_EXTENT_Y = (-0.038, 0.038)
"""Footprint of the G1 foot mesh in the sole frame (metres)."""


@dataclass(frozen=True)
class TaxelLayout:
  """Taxel patches on the LEFT sole, in the sole frame (metres).

  ``centers`` and ``half_sizes`` are tuples of (x, y) pairs so the layout is
  hashable and fingerprints cleanly.
  """

  name: str
  centers: tuple[tuple[float, float], ...]
  half_sizes: tuple[tuple[float, float], ...]
  """Half edge lengths of each (axis-aligned rectangular) patch."""
  mirror_right: bool = True
  """True: the right sole carries a mirrored sensor (y -> -y). False: the same
  sensor part is used on both feet, at identical (x, y)."""

  @property
  def num_taxels(self) -> int:
    return len(self.centers)

  def areas(self) -> tuple[float, ...]:
    return tuple(4.0 * hx * hy for hx, hy in self.half_sizes)

  def for_side(self, side: Side) -> tuple[tuple[float, float], ...]:
    """Taxel centres on ``side``'s sole; the right foot mirrors y only if
    ``mirror_right``."""
    if side == "left" or not self.mirror_right:
      return self.centers
    return tuple((x, -y) for x, y in self.centers)

  def sample_points(
    self, side: Side, per_axis: int
  ) -> tuple[tuple[float, float], ...]:
    """``per_axis`` x ``per_axis`` points spread over each patch, taxel-major.

    Used to estimate what fraction of a patch is over supporting ground (a
    patch half over a beam edge should read half-supported).
    """
    out: list[tuple[float, float]] = []
    offsets = [(2 * k + 1) / (2 * per_axis) * 2 - 1 for k in range(per_axis)]
    for (cx, cy), (hx, hy) in zip(self.for_side(side), self.half_sizes):
      for u in offsets:
        for v in offsets:
          out.append((cx + u * hx, cy + v * hy))
    return tuple(out)


def grid_layout(
  name: str,
  rows: int,
  cols: int,
  patch: float,
  extent_x: tuple[float, float] = SOLE_EXTENT_X,
  extent_y: tuple[float, float] = SOLE_EXTENT_Y,
) -> TaxelLayout:
  """``rows`` (along x, heel -> toe) x ``cols`` (along y) square patches,
  evenly spread over the footprint. Row-major, heel row first, right-to-left
  within a row (y ascending)."""
  (x0, x1), (y0, y1) = extent_x, extent_y
  px, py = (x1 - x0) / rows, (y1 - y0) / cols
  if patch > min(px, py):
    raise ValueError(f"patch {patch} m overlaps at pitch ({px:.4f}, {py:.4f}) m")
  centers = tuple(
    (round(x0 + (r + 0.5) * px, 6), round(y0 + (c + 0.5) * py, 6))
    for r in range(rows)
    for c in range(cols)
  )
  half = (patch / 2, patch / 2)
  return TaxelLayout(name=name, centers=centers, half_sizes=(half,) * len(centers))


LAYOUT_DIR = Path(__file__).parent / "layouts"


def csv_layout(
  name: str, path: Path | None = None, *, mirror_right: bool = False
) -> TaxelLayout:
  """Load ``layouts/<name>.csv``: columns ``id, cx, cy, width_x, length_y,
  rotation_deg`` in millimetres, left foot, sole frame. Rows are ordered by
  ``id``, which must run 0..N-1 -- taxel ``i`` in the model is sensor id ``i``.

  Rejects patches that are rotated (not modelled yet), overlap, or stick out
  past the foot: each of those is far more likely a typo than intent.
  """
  path = path or LAYOUT_DIR / f"{name}.csv"
  with open(path, newline="") as f:
    rows = sorted(csv.DictReader(f), key=lambda r: int(r["id"]))
  ids = [int(r["id"]) for r in rows]
  if ids != list(range(len(rows))):
    raise ValueError(f"{path.name}: ids must be 0..{len(rows) - 1}, got {ids}")

  (x0, x1), (y0, y1) = SOLE_EXTENT_X, SOLE_EXTENT_Y
  centers, half_sizes = [], []
  for r in rows:
    if float(r["rotation_deg"]) != 0.0:
      raise NotImplementedError(f"{path.name}: taxel {r['id']} is rotated")
    cx, cy = float(r["cx"]) / 1000, float(r["cy"]) / 1000
    hx, hy = float(r["width_x"]) / 2000, float(r["length_y"]) / 2000
    if not (x0 <= cx - hx and cx + hx <= x1 and y0 <= cy - hy and cy + hy <= y1):
      raise ValueError(
        f"{path.name}: taxel {r['id']} at ({r['cx']}, {r['cy']}) mm extends past "
        f"the foot (x {x0 * 1000:.0f}..{x1 * 1000:.0f}, "
        f"y {y0 * 1000:.0f}..{y1 * 1000:.0f} mm)"
      )
    centers.append((cx, cy))
    half_sizes.append((hx, hy))

  for i in range(len(rows)):
    for j in range(i):
      (ax, ay), (ahx, ahy) = centers[i], half_sizes[i]
      (bx, by), (bhx, bhy) = centers[j], half_sizes[j]
      if abs(ax - bx) < ahx + bhx and abs(ay - by) < ahy + bhy:
        raise ValueError(f"{path.name}: taxels {j} and {i} overlap")
  return TaxelLayout(
    name=name,
    centers=tuple(centers),
    half_sizes=tuple(half_sizes),
    mirror_right=mirror_right,
  )


LAYOUTS: dict[str, TaxelLayout] = {
  "placeholder_8x3": grid_layout("placeholder_8x3", rows=8, cols=3, patch=0.020),
  # One sensor part, used 1:1 on both feet.
  "slipper_v1": csv_layout("slipper_v1", mirror_right=False),
}
"""Named layouts. Observation terms reference a layout by name so configs stay
plain data and fingerprint stably."""

DEFAULT_LAYOUT = "slipper_v1"
