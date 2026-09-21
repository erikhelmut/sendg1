#!/usr/bin/env python
"""Draw the G1 sole as simulated: foot mesh, the 7 collision capsules, and the
taxel layout on top.

    python scripts/show_sole.py                       # -> media/tactile_sole_layout.png
    python scripts/show_sole.py --layout placeholder_8x3 --show

Plan views look down through the foot from above (x forward, y left), so the
left and right feet are mirror images.
"""

from __future__ import annotations

import argparse
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--layout", default=None, help="key into tactile.layout.LAYOUTS")
  ap.add_argument("--rays-per-axis", type=int, default=None)
  ap.add_argument("--out", default=str(REPO / "media" / "tactile_sole_layout.png"))
  ap.add_argument("--show", action="store_true")
  args = ap.parse_args()

  import matplotlib

  if not args.show:
    matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  from matplotlib.collections import PatchCollection
  from matplotlib.lines import Line2D
  from matplotlib.patches import Patch

  from sendg1.common.sensors import TAXEL_RAY_LIFT
  from sendg1.common.tactile import viz
  from sendg1.common.tactile.layout import LAYOUTS
  from sendg1.common.tactile.sensing import TactileParams

  params = TactileParams()
  layout = LAYOUTS[args.layout or params.layout]
  per_axis = args.rays_per_axis or params.rays_per_axis

  fig = plt.figure(figsize=(13, 8.2), facecolor=viz.SURFACE)
  gs = fig.add_gridspec(2, 2, height_ratios=[1.35, 1], hspace=0.28, wspace=0.12)
  fig.suptitle(
    f"G1 sole in simulation  ·  taxel layout '{layout.name}' "
    f"({layout.num_taxels} taxels per foot)",
    color=viz.INK, fontsize=13, x=0.06, ha="left",
  )

  for col, side in enumerate(("left", "right")):
    ax = fig.add_subplot(gs[0, col])
    viz.draw_mesh_plan(ax, side)
    viz.draw_capsules_plan(ax, side, label=True)
    ax.add_collection(
      PatchCollection(
        viz.taxel_patches(layout, side), facecolor="none",
        edgecolor=viz.TAXEL_EDGE, lw=1.2, zorder=4,
      )
    )
    for i, ((x, y), (hx, hy)) in enumerate(zip(layout.for_side(side), layout.half_sizes)):
      ax.text(x - hx + 0.001, y + hy - 0.001, str(i), ha="left", va="top",
              fontsize=6.5, color=viz.TAXEL_EDGE, zorder=5)
    medial = -1 if side == "left" else 1
    ax.text(0.155, medial * 0.047, "medial (inner) edge", fontsize=7,
            color=viz.INK_2, ha="right", va="center")
    pts = viz.flat_ground_contact_points(side)
    ax.scatter(pts[:, 0], pts[:, 1], marker="x", s=22, color=viz.INK, lw=1.0, zorder=6)
    samples = layout.sample_points(side, per_axis)
    sx, sy = zip(*samples)
    ax.scatter(sx, sy, s=2, color=viz.TAXEL_EDGE, zorder=5)
    ax.plot(0, 0, marker="+", ms=12, mew=1.5, color=viz.INK_2, zorder=6)
    ax.annotate("ankle-roll axis\n(sole frame origin)", (0, 0), (-0.02, 0.047),
                fontsize=7, color=viz.INK_2, ha="center",
                arrowprops=dict(arrowstyle="-", color=viz.INK_2, lw=0.6))
    viz.style_plan_axes(ax, f"{side} foot  ·  plan view from above")

  ax = fig.add_subplot(gs[1, :])
  viz.draw_side_view(ax, "left", TAXEL_RAY_LIFT)
  ax.set_title("left foot  ·  side view", color=viz.INK, fontsize=10, loc="left")

  handles = [
    Patch(facecolor=viz.MESH, label="foot mesh (visual only)"),
    Patch(facecolor="none", edgecolor=viz.CAPSULE,
          label="7 collision capsules, r = 1 cm (what touches the ground)"),
    Patch(facecolor="none", edgecolor=viz.TAXEL_EDGE, label="taxel patch"),
    Line2D([], [], color=viz.TAXEL_EDGE, marker=".", ls="none",
           label=f"support ray origins ({per_axis}×{per_axis} per taxel)"),
    Line2D([], [], color=viz.INK, marker="x", ls="none",
           label="where MuJoCo puts contacts when the foot is flat"),
  ]
  fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
             fontsize=8.5, labelcolor=viz.INK_2)
  fig.subplots_adjust(left=0.06, right=0.98, top=0.9, bottom=0.12)

  out = Path(args.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(out, dpi=150, facecolor=viz.SURFACE)
  print(f"[sole] wrote {out}")
  if args.show:
    plt.show()


if __name__ == "__main__":
  main()
