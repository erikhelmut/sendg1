"""Where the taxels sit on the sole.

Everything is expressed in the **sole frame**, which is the ``*_ankle_roll_link``
body frame: x forward, y left, z up. The sole plane is the bottom of the 7 foot
collision capsules (and, to 0.4 mm, of the foot's visual mesh), at
``SOLE_Z = -0.035`` m.

The layout is a placeholder until the real slipper geometry is exported from
CAD: a regular 8 x 3 grid of 20 mm square patches covering the sole footprint.
Only this module needs to change when the real layout arrives -- everything
downstream reads ``TaxelLayout`` and never assumes a grid.

Right-foot layouts are the mirror image (y -> -y) of the left, so taxel ``i`` is
the same anatomical patch on both feet.
"""

from __future__ import annotations

from dataclasses import dataclass
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

  @property
  def num_taxels(self) -> int:
    return len(self.centers)

  def areas(self) -> tuple[float, ...]:
    return tuple(4.0 * hx * hy for hx, hy in self.half_sizes)

  def for_side(self, side: Side) -> tuple[tuple[float, float], ...]:
    """Taxel centres on ``side``'s sole; the right foot mirrors y."""
    if side == "left":
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


LAYOUTS: dict[str, TaxelLayout] = {
  "placeholder_8x3": grid_layout("placeholder_8x3", rows=8, cols=3, patch=0.020),
}
"""Named layouts. Observation terms reference a layout by name so configs stay
plain data and fingerprint stably."""

DEFAULT_LAYOUT = "placeholder_8x3"
