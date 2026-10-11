#!/usr/bin/env python
"""Extract training curves of a beam teacher or student run to JSON.

    python scripts/beam_ts_curves.py runs/beam_ts_teacher/<run> --out results/beam_ts/teacher_curves.json

Keeps every Curriculum/*, Episode_Termination/*, Episode_Reward/*, Loss/* and
Train/* scalar, downsampled to at most ``--points`` points per tag.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

PREFIXES = ("Curriculum/", "Episode_Termination/", "Episode_Reward/", "Loss/", "Train/", "Policy/", "Perf/", "Metrics/")


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("run_dir")
  ap.add_argument("--out", required=True)
  ap.add_argument("--points", type=int, default=400)
  args = ap.parse_args()

  from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

  files = sorted(glob.glob(str(Path(args.run_dir) / "**" / "events.out.tfevents*"), recursive=True))
  if not files:
    raise SystemExit(f"no event files under {args.run_dir}")
  curves: dict[str, list[tuple[int, float]]] = {}
  for f in files:
    ea = EventAccumulator(f, size_guidance={"scalars": 0})
    ea.Reload()
    for tag in ea.Tags()["scalars"]:
      if not tag.startswith(PREFIXES):
        continue
      curves.setdefault(tag, []).extend((s.step, s.value) for s in ea.Scalars(tag))
  out = {}
  for tag, pts in curves.items():
    pts.sort()
    stride = max(1, len(pts) // args.points)
    out[tag] = pts[::stride] + ([pts[-1]] if (len(pts) - 1) % stride else [])
  Path(args.out).parent.mkdir(parents=True, exist_ok=True)
  json.dump({"run_dir": str(Path(args.run_dir).resolve()), "curves": out}, open(args.out, "w"))
  print(f"wrote {args.out}: {len(out)} tags")


if __name__ == "__main__":
  main()
