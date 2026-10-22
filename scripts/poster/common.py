"""Shared helpers for the poster figures: palette, soles, arrows."""

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon
from PIL import Image
from scipy.interpolate import splev, splprep
from scipy.spatial import ConvexHull

from sendg1.common.tactile import viz
from sendg1.common.tactile.layout import LAYOUTS

# Work dir for intermediate renders and outputs (media/ is gitignored).
D = os.environ.get("POSTER_DIR", str(Path(__file__).resolve().parents[2] / "media" / "poster")) + "/"
os.makedirs(D, exist_ok=True)
SOLE, EMPTY = "#15191f", "#2a313b"
INK, INK_2, YELLOW, KYBER, DASH = "#1d2430", "#6b7380", "#fac055", "#4cc9f0", "#9aa3ae"
cmap = LinearSegmentedColormap.from_list(
  "kyber", [EMPTY, "#264f78", "#2f8fc4", KYBER, "#bdeeff"])
norm = PowerNorm(0.5, vmin=0, vmax=60, clip=True)

w = np.load(D + "walk.npz")
STAND, MID = 60, 233
layout = LAYOUTS["slipper_v1"]


def trim(path):
  im = Image.open(path)
  return im.crop(im.getchannel("A").point(lambda v: 255 if v > 8 else 0).getbbox())


def up(xy):
  """Sole frame -> page with the toe up."""
  xy = np.asarray(xy)
  return np.stack([-xy[..., 1], xy[..., 0]], -1)


def g1_outline(side="right"):
  pts = viz.foot_mesh(side).vertices[:, :2]
  hull = pts[ConvexHull(pts).vertices]
  ang = np.linspace(0, 2 * np.pi, 48, endpoint=False)
  ring = (hull[:, None, :] + 0.009 * np.stack([np.cos(ang), np.sin(ang)], -1)).reshape(-1, 2)
  return ring[ConvexHull(ring).vertices]


def human_outline():
  """Schematic right footprint, metres, toe to +x, roughly centred like the G1 sole."""
  ctrl = np.array([
    [0.000, 0.000], [0.010, 0.026], [0.050, 0.032], [0.100, 0.030], [0.150, 0.040],
    [0.200, 0.048], [0.240, 0.040], [0.258, 0.018], [0.252, -0.008], [0.225, -0.028],
    [0.180, -0.036], [0.120, -0.022], [0.070, -0.020], [0.020, -0.026],
  ])
  ctrl[:, 0] -= 0.07
  ctrl[:, 1] -= 0.006
  tck, _ = splprep([*ctrl.T], s=0, per=True)
  return np.stack(splev(np.linspace(0, 1, 200), tck), -1)


def draw_g1_sole(ax, t, foot=1, side="right", clip=None, cop=True):
  outline = Polygon(up(g1_outline(side)), closed=True, facecolor=SOLE, edgecolor="none")
  ax.add_patch(outline)
  arts = [outline]
  normal = w["normal"][t, foot]
  for i, ((cx, cy), (hx, hy)) in enumerate(zip(layout.for_side(side), layout.half_sizes)):
    px, py = up([cx, cy])
    p = FancyBboxPatch((px - hy + 0.0012, py - hx + 0.0012), 2 * hy - 0.0024, 2 * hx - 0.0024,
                       boxstyle="round,pad=0.0012,rounding_size=0.0018",
                       facecolor=cmap(norm(normal[i])), edgecolor="none", zorder=2)
    ax.add_patch(p)
    arts.append(p)
  if cop and normal.sum() > 1:
    arts.append(ax.scatter(*up(w["cop"][t, foot]), s=110, color=YELLOW, edgecolor=SOLE,
                           lw=2, zorder=4))
  if clip is not None:
    for a in arts:
      a.set_clip_path(clip)


def draw_human_sole(ax, clip=None, q=True):
  p = Polygon(up(human_outline()), closed=True, facecolor="#f3f5f7", edgecolor=DASH,
              lw=2, ls=(0, (4, 3)))
  ax.add_patch(p)
  if clip is not None:
    p.set_clip_path(clip)
  if q:
    ax.text(*up([0.06, -0.006]), "?", ha="center", va="center", fontsize=34, color=DASH,
            fontweight="bold")


def sole_axes(ax):
  ax.set_aspect("equal")
  ax.set_xlim(-0.07, 0.07)
  ax.set_ylim(-0.09, 0.2)
  ax.axis("off")


def arrow(fig, a, b, text, color, rad=0.0, above=True, size=18):
  fig.add_artist(FancyArrowPatch(a, b, transform=fig.transFigure, arrowstyle="-|>",
                                 mutation_scale=26, lw=2.6, color=color,
                                 connectionstyle=f"arc3,rad={rad}"))
  mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
  fig.text(mx, my + (0.025 if above else -0.025), text, ha="center",
           va="bottom" if above else "top", fontsize=size, color=INK)
