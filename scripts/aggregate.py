#!/usr/bin/env python
"""Collate results/ into a comparison table, paired by seed.

    python scripts/aggregate.py
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def main() -> None:
  from sendg1.eval.metrics import LOWER_IS_BETTER, PRIMARY_METRIC

  ap = argparse.ArgumentParser()
  ap.add_argument("--results-root", default=str(REPO / "results"))
  ap.add_argument("--metric", default=PRIMARY_METRIC)
  args = ap.parse_args()

  root = Path(args.results_root)
  if not root.exists():
    print(f"no results at {root}")
    return

  # experiment -> seed -> metric value
  table: dict[str, dict[int, float]] = defaultdict(dict)
  for path in sorted(root.rglob("seed*.json")):
    data = json.loads(path.read_text())
    table[path.parent.name][data["seed"]] = data["metrics"][args.metric]

  if not table:
    print("no result files found")
    return

  direction = "lower is better" if LOWER_IS_BETTER.get(args.metric, True) else "higher is better"
  print(f"\nmetric: {args.metric}  ({direction})\n")
  print(f"{'experiment':40s} {'n':>3s} {'mean':>10s} {'stdev':>10s}")
  print("-" * 66)
  for exp in sorted(table):
    vals = list(table[exp].values())
    sd = statistics.stdev(vals) if len(vals) > 1 else float("nan")
    print(f"{exp:40s} {len(vals):3d} {statistics.mean(vals):10.4f} {sd:10.4f}")

  # Paired comparison: same task, blind vs everything else, matched by seed.
  by_task: dict[str, dict[str, dict[int, float]]] = defaultdict(dict)
  for exp, per_seed in table.items():
    task, _, obs_set = exp.rpartition("__")
    by_task[task][obs_set] = per_seed

  print()
  for task, per_set in sorted(by_task.items()):
    base = per_set.get("blind")
    if base is None:
      continue
    for obs_set, vals in sorted(per_set.items()):
      if obs_set == "blind":
        continue
      shared = sorted(set(base) & set(vals))
      if not shared:
        continue
      deltas = [vals[s] - base[s] for s in shared]
      mean_delta = statistics.mean(deltas)
      print(
        f"{task}: {obs_set} vs blind  n={len(shared)}  "
        f"mean delta={mean_delta:+.4f}"
        + ("  (n=1: directional only)" if len(shared) == 1 else "")
      )


if __name__ == "__main__":
  main()
