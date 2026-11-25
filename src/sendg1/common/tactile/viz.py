"""Drawing the sole: foot mesh, collision capsules, taxels, and live readings.

Matplotlib only, imported lazily so the training path never pays for it. All
drawing is in the sole frame (metres), plan view from above: x forward (toe to
the right), y left (up on the page).
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from sendg1.common.tactile.layout import Side, TaxelLayout

# Reference palette (dataviz skill, light mode).
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MESH = "#d9d8d4"
CAPSULE = "#eb6834"  # categorical slot 2
FORCE_RAMP = (
  "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b",
)
UNLOADED = "#f0efec"


def force_cmap():
  from matplotlib.colors import LinearSegmentedColormap

  return LinearSegmentedColormap.from_list("sole_force", FORCE_RAMP)


def _g1_xml_dir() -> Path:
  import mjlab

  return Path(os.path.dirname(mjlab.__file__)) / "asset_zoo/robots/unitree_g1/xmls"


@dataclass(frozen=True)
class Capsule:
  name: str
  p0: tuple[float, float, float]
  p1: tuple[float, float, float]
  radius: float


@lru_cache(maxsize=2)
def foot_capsules(side: Side) -> tuple[Capsule, ...]:
  """The 7 collision capsules of one foot, straight from the G1 MJCF."""
  root = ET.parse(_g1_xml_dir() / "g1.xml").getroot()
  radius = None
  for d in root.iter("default"):
    if d.get("class") == "foot_capsule":
      radius = float(d.find("geom").get("size"))  # type: ignore[union-attr]
  assert radius is not None
  out = []
  for g in root.iter("geom"):
    name = g.get("name", "")
    if name.startswith(f"{side}_foot") and name.endswith("_collision"):
      v = [float(t) for t in g.get("fromto", "").split()]
      out.append(Capsule(name, tuple(v[:3]), tuple(v[3:]), radius))  # type: ignore[arg-type]
  return tuple(sorted(out, key=lambda c: c.name))


@lru_cache(maxsize=2)
def foot_mesh(side: Side):
  """The visual foot mesh as a trimesh, in the ankle-roll-link frame."""
  import trimesh

  path = next(_g1_xml_dir().rglob(f"{side}_ankle_roll_link.STL"))
  return trimesh.load(path)


def draw_mesh_plan(ax, side: Side, **kw) -> None:
  from matplotlib.collections import PolyCollection

  m = foot_mesh(side)
  tris = m.vertices[m.faces][:, :, :2]
  ax.add_collection(
    PolyCollection(tris, facecolor=kw.get("color", MESH), edgecolor="none", zorder=0)
  )


def taxel_patches(layout: TaxelLayout, side: Side):
  """One Rectangle per taxel, in layout order."""
  from matplotlib.patches import Rectangle

  return [
    Rectangle((cx - hx, cy - hy), 2 * hx, 2 * hy)
    for (cx, cy), (hx, hy) in zip(layout.for_side(side), layout.half_sizes)
  ]


def style_plan_axes(ax, title: str | None = None) -> None:
  ax.set_aspect("equal")
  ax.set_xlim(-0.085, 0.16)
  ax.set_ylim(-0.055, 0.055)
  ax.set_facecolor(SURFACE)
  ax.tick_params(colors=INK_2, labelsize=8)
  for s in ax.spines.values():
    s.set_color(MESH)
  ax.set_xlabel("x  (m, forward)", color=INK_2, fontsize=8)
  ax.set_ylabel("y  (m, left)", color=INK_2, fontsize=8)
  if title:
    ax.set_title(title, color=INK, fontsize=10, loc="left")


