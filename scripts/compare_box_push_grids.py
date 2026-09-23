#!/usr/bin/env python
"""Compare two box-push grid evaluations cell by cell.

    python scripts/compare_box_push_grids.py \
        results/it4000/box_push__tactile_array/box_push_grid_seed0.json \
        results/it4000/box_push__blind/box_push_grid_seed0.json \
        --labels tactile_array blind --out results/compare.png

Both runs must use the same protocol, so every cell holds the same pinned
episodes. Left/middle: the metric per condition, one blue ramp. Right: the
difference, a diverging ramp with a neutral zero -- the only place a
two-hue scale belongs.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

SEQ = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
DIV = ["#8c2f21", "#c0705f", "#f0efec", "#6da7ec", "#184f95"]
INK, INK_ON_DARK, MUTED = "#1f1f1f", "#ffffff", "#6b6b6b"


def grid(cells: list[dict], key: str, mus: list[float], masses: list[float]) -> np.ndarray:
  g = np.full((len(mus), len(masses)), np.nan)
  for c in cells:
    g[mus.index(c["floor_mu"]), masses.index(c["box_mass"])] = c[key]
  return g


def panel(ax, g, title, cmap, vmin, vmax, fmt, mus, masses, fig):
  im = ax.imshow(g, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower", aspect="auto")
  for i in range(len(mus)):
    for j in range(len(masses)):
      v = g[i, j]
      if not math.isfinite(v):
        continue
      shade = abs(v - (vmin + vmax) / 2) / max((vmax - vmin) / 2, 1e-9)
      ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=10,
              color=INK_ON_DARK if shade > 0.55 else INK)
  ax.set_xticks(range(len(masses)), [f"{m:.0f}" for m in masses])
  ax.set_yticks(range(len(mus)), [f"{m:.2f}" for m in mus])
  ax.set_xlabel("box mass (kg)", color=MUTED)
  ax.set_ylabel("floor friction μ", color=MUTED)
  ax.set_title(title, loc="left", fontsize=11, color=INK)
  ax.tick_params(length=0, colors=MUTED)
  for s in ax.spines.values():
    s.set_visible(False)
  cb = fig.colorbar(im, ax=ax, shrink=0.85)
  cb.outline.set_visible(False)
  cb.ax.tick_params(length=0, colors=MUTED)


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("grids", nargs=2, type=Path)
  ap.add_argument("--labels", nargs=2, default=["A", "B"])
  ap.add_argument("--key", default="success_rate")
  ap.add_argument("--fmt", default="{:.0%}")
  ap.add_argument("--out", type=Path, required=True)
  args = ap.parse_args()

  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  from matplotlib.colors import LinearSegmentedColormap

  seq = LinearSegmentedColormap.from_list("seq", SEQ)
  div = LinearSegmentedColormap.from_list("div", DIV)
  data = [json.loads(p.read_text()) for p in args.grids]
  mus = sorted({c["floor_mu"] for c in data[0]["cells"]})
  masses = sorted({c["box_mass"] for c in data[0]["cells"]})
  gs = [grid(d["cells"], args.key, mus, masses) for d in data]
  diff = gs[0] - gs[1]

  finite = np.isfinite(np.concatenate([g.ravel() for g in gs]))
  vmax = float(np.nanmax(np.concatenate([g.ravel() for g in gs])[finite]))
  vmin = 0.0 if args.key.endswith("rate") or "frac" in args.key else float(
    np.nanmin(np.concatenate([g.ravel() for g in gs])[finite])
  )
  dlim = float(np.nanmax(np.abs(diff))) or 1.0

  fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), constrained_layout=True)
  panel(axes[0], gs[0], args.labels[0], seq, vmin, vmax, args.fmt, mus, masses, fig)
  panel(axes[1], gs[1], args.labels[1], seq, vmin, vmax, args.fmt, mus, masses, fig)
  panel(axes[2], diff, f"{args.labels[0]} − {args.labels[1]}", div, -dlim, dlim,
        args.fmt if "%" in args.fmt else "{:+.1f}", mus, masses, fig)
  fig.suptitle(
    f"{args.key}, {data[0]['protocol']['episodes_per_cell']} pinned episodes per cell",
    x=0.01, ha="left", fontsize=11, color=INK,
  )
  args.out.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(args.out, dpi=150)
  print(f"wrote {args.out}")


if __name__ == "__main__":
  main()
