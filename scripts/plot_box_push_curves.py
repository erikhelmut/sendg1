#!/usr/bin/env python
"""Learning curves of box-pushing runs from their TensorBoard event files.

    python scripts/plot_box_push_curves.py runs/_checks/box_push__blind/learncheck_seed0 \
        --out results/box_push_learncheck.png

One small-multiple panel per quantity (no shared y axes); one line per run.
"""

from __future__ import annotations

import argparse
from pathlib import Path

PANELS: tuple[tuple[str, str], ...] = (
  ("Train/mean_reward", "Mean episode return"),
  ("Train/mean_episode_length", "Episode length (steps)"),
  ("Episode_Metrics/success", "Success rate"),
  ("Episode_Metrics/fell", "Fall rate"),
  ("Episode_Metrics/box_tipped", "Box tipped rate"),
  ("Episode_Metrics/box_to_target", "Final box-to-target distance (m)"),
  ("Episode_Metrics/slip_events", "Full-foot slip events / episode"),
  ("Episode_Metrics/mean_push_force", "Mean push force (N)"),
  ("Episode_Reward/box_progress", "Reward: box_progress"),
  ("Episode_Reward/hand_contact", "Reward: hand_contact"),
  ("Episode_Reward/foot_slip", "Reward: foot_slip"),
  ("Episode_Reward/action_rate_l2", "Reward: action_rate_l2"),
)
LINE_COLORS = ("#2a78d6", "#e8743b", "#1a9e77", "#8f5bd6")
INK, MUTED, GRID = "#1f1f1f", "#6b6b6b", "#e6e6e6"


def load(run: Path) -> dict[str, tuple[list[int], list[float]]]:
  from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

  acc = EventAccumulator(str(run), size_guidance={"scalars": 0})
  acc.Reload()
  out = {}
  for tag in acc.Tags()["scalars"]:
    ev = acc.Scalars(tag)
    out[tag] = ([e.step for e in ev], [e.value for e in ev])
  return out


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("runs", nargs="+", type=Path)
  ap.add_argument("--labels", nargs="+", default=None)
  ap.add_argument("--out", type=Path, required=True)
  ap.add_argument("--smooth", type=int, default=20, help="moving-average window (iterations)")
  args = ap.parse_args()

  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  import numpy as np

  data = [load(r) for r in args.runs]
  labels = args.labels or [r.name for r in args.runs]
  cols = 4
  rows = -(-len(PANELS) // cols)
  fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 2.8 * rows), constrained_layout=True)
  for ax, (tag, title) in zip(axes.flat, PANELS):
    for d, label, color in zip(data, labels, LINE_COLORS):
      if tag not in d:
        continue
      x, y = map(np.asarray, d[tag])
      if args.smooth > 1 and len(y) > args.smooth:
        k = np.ones(args.smooth) / args.smooth
        y = np.convolve(y, k, mode="valid")
        x = x[args.smooth - 1:]
      ax.plot(x, y, color=color, linewidth=2, label=label)
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.tick_params(colors=MUTED, length=0, labelsize=8)
    for s in ax.spines.values():
      s.set_visible(False)
  for ax in axes.flat[len(PANELS):]:
    ax.set_visible(False)
  if len(data) > 1:
    axes.flat[0].legend(frameon=False, fontsize=8)
  fig.supxlabel("PPO iteration", color=MUTED, fontsize=9)
  args.out.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(args.out, dpi=130)
  print(f"wrote {args.out}")


if __name__ == "__main__":
  main()
