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

import numpy as np

from sendg1.common.tactile.layout import SOLE_Z, Side, TaxelLayout

# Reference palette (dataviz skill, light mode).
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MESH = "#d9d8d4"
CAPSULE = "#eb6834"  # categorical slot 2
TAXEL_EDGE = "#2a78d6"  # categorical slot 1
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


def flat_ground_contact_points(side: Side) -> np.ndarray:
  """Where MuJoCo puts contacts when the foot is flat on a plane: under each
  end of every capsule. [N, 2] in the sole frame."""
  pts = []
  for c in foot_capsules(side):
    pts += [c.p0[:2], c.p1[:2]]
  return np.array(pts)


def draw_mesh_plan(ax, side: Side, **kw) -> None:
  from matplotlib.collections import PolyCollection

  m = foot_mesh(side)
  tris = m.vertices[m.faces][:, :, :2]
  ax.add_collection(
    PolyCollection(tris, facecolor=kw.get("color", MESH), edgecolor="none", zorder=0)
  )


def draw_capsules_plan(
  ax, side: Side, fill_alpha: float = 0.08, lw: float = 1.0, label: bool = False
) -> None:
  """Capsule outlines plus their axes. They overlap (1 cm radius, 0.8-1 cm
  apart), so outlines read better than fills."""
  from matplotlib.patches import FancyBboxPatch

  for i, c in enumerate(foot_capsules(side)):
    x0, x1 = sorted((c.p0[0], c.p1[0]))
    y = 0.5 * (c.p0[1] + c.p1[1])
    r = c.radius
    ax.add_patch(
      FancyBboxPatch(
        (x0, y), x1 - x0, 0.0,
        boxstyle=f"round,pad={r}",
        facecolor=(*_rgb(CAPSULE), fill_alpha), edgecolor=CAPSULE, lw=lw, zorder=2,
      )
    )
    ax.plot([c.p0[0], c.p1[0]], [c.p0[1], c.p1[1]], color=CAPSULE, lw=0.8, zorder=2)
    if label:
      ax.text(x1 + r + 0.003, y, str(i + 1), color=CAPSULE, fontsize=7,
              va="center", ha="left", zorder=6)


def _rgb(hex_color: str) -> tuple[float, float, float]:
  h = hex_color.lstrip("#")
  return tuple(int(h[i : i + 2], 16) / 255 for i in (0, 2, 4))  # type: ignore[return-value]


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


def draw_side_view(ax, side: Side, ray_lift: float) -> None:
  """x-z profile: mesh, capsules end-on-ish, sole plane, taxel ray origins."""
  from matplotlib.collections import PolyCollection
  from matplotlib.patches import FancyBboxPatch

  m = foot_mesh(side)
  tris = m.vertices[m.faces][:, :, [0, 2]]
  ax.add_collection(PolyCollection(tris, facecolor=MESH, edgecolor="none", zorder=0))
  for c in foot_capsules(side):
    x0, x1 = sorted((c.p0[0], c.p1[0]))
    ax.add_patch(
      FancyBboxPatch(
        (x0, c.p0[2]), x1 - x0, 0.0, boxstyle=f"round,pad={c.radius}",
        facecolor=CAPSULE, edgecolor=CAPSULE, alpha=0.25, lw=0.8, zorder=2,
      )
    )
  ax.axhline(SOLE_Z, color=TAXEL_EDGE, lw=1.5, zorder=3)
  ax.text(0.155, SOLE_Z - 0.004, f"sole plane  z = {SOLE_Z} m", color=INK_2,
          fontsize=8, ha="right", va="top")
  ax.axhline(SOLE_Z + ray_lift, color=INK_2, lw=0.8, ls=(0, (3, 3)), zorder=3)
  ax.text(0.155, SOLE_Z + ray_lift + 0.002, "taxel rays start here and point down",
          color=INK_2, fontsize=8, ha="right", va="bottom")
  ax.set_aspect("equal")
  ax.set_xlim(-0.085, 0.16)
  ax.set_ylim(-0.05, 0.03)
  ax.set_facecolor(SURFACE)
  ax.tick_params(colors=INK_2, labelsize=8)
  for s in ax.spines.values():
    s.set_color(MESH)
  ax.set_xlabel("x  (m, forward)", color=INK_2, fontsize=8)
  ax.set_ylabel("z  (m, up)", color=INK_2, fontsize=8)
