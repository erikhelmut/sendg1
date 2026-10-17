#!/usr/bin/env python
"""Quasi-static feasibility of the box-pushing ranges.

    python scripts/box_push_feasibility.py
    python scripts/box_push_feasibility.py --alpha 0.3     # a curriculum level

Samples episodes from the task's own ranges (sendg1/common/box_push/params.py)
and reports:

  traction margin  1 - (mu_box m_box g) / (mu_foot m_robot g)
                   < 0: infeasible, the feet must slip to slide the box
                   0-20%: near the limit
  tipping          whether a horizontal push at a given height tips the box
                   before it slides (h > depth / (2 mu_box))

Both are idealised: horizontal push only, no vertical hand force, no dynamics,
the robot's full weight on its feet.
"""

from __future__ import annotations

import argparse

import numpy as np

from sendg1.common.box_push.params import (
  G1_MASS_KG,
  GRAVITY,
  BoxPushRanges,
  curriculum_ranges,
  mass_from_margin,
  tips_before_sliding,
  traction_margin,
)

NEAR_LIMIT = 0.20


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--samples", type=int, default=1_000_000)
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--alpha", type=float, default=1.0, help="curriculum level; 1 = full ranges")
  ap.add_argument("--push-heights", type=float, nargs="+", default=[0.35, 0.45, 0.55])
  args = ap.parse_args()

  r = BoxPushRanges()
  cur = curriculum_ranges(r, args.alpha)
  rng = np.random.default_rng(args.seed)
  n = args.samples
  floor_mu = rng.uniform(*cur["floor_mu"], n)
  box_mu = rng.uniform(*cur["box_mu"], n)
  wanted = rng.uniform(*cur["traction_margin"], n)
  box_mass = mass_from_margin(wanted, floor_mu, box_mu, r.box_mass)
  depth = rng.uniform(*r.box_depth, n)
  height = rng.uniform(*r.box_height, n)

  print(f"robot mass {G1_MASS_KG:.2f} kg, g {GRAVITY}, alpha {args.alpha}")
  print(f"margin drawn from {cur['traction_margin']}, floor mu {cur['floor_mu']}, "
        f"box mu {cur['box_mu']}, mass clipped to {r.box_mass} kg\n")

  m = traction_margin(floor_mu, box_mass, box_mu)
  clipped = np.mean(
    (box_mass <= r.box_mass[0] + 1e-9) | (box_mass >= r.box_mass[1] - 1e-9)
  )
  infeasible = m < 0
  near = (m >= 0) & (m < NEAR_LIMIT)
  print(f"margin = 1 - required/available, over {n:,} sampled episodes")
  print(f"  infeasible (margin < 0):       {infeasible.mean():6.1%}")
  print(f"  near the limit (0 <= m < 20%): {near.mean():6.1%}")
  print(f"  comfortable (m >= 20%):        {(m >= NEAR_LIMIT).mean():6.1%}")
  print(f"  mass hit a clip:               {clipped:6.1%}")
  q = np.percentile(m, [10, 25, 50, 75, 90])
  print("  margin percentiles 10/25/50/75/90: " + " ".join(f"{v:+.2f}" for v in q))
  q = np.percentile(box_mass, [5, 50, 95])
  print(f"  box mass p5/50/95: {q.round(1)} kg")
  print(f"  required push force, median: {np.median(box_mu * box_mass * GRAVITY):.0f} N"
        f"  ({np.median(box_mu * box_mass) / G1_MASS_KG:.0%} of the robot's weight)")

  print("\nInfeasible share by floor mu (rows) and box mass (cols):")
  mass_edges = np.linspace(*r.box_mass, 5)
  mu_edges = np.linspace(*r.floor_mu, 5)
  print("  floor mu \\ mass " + " ".join(
    f"{a:>4.0f}-{b:<4.0f}" for a, b in zip(mass_edges[:-1], mass_edges[1:])
  ))
  for lo, hi in zip(mu_edges[:-1], mu_edges[1:]):
    row = []
    for a, b in zip(mass_edges[:-1], mass_edges[1:]):
      sel = (floor_mu >= lo) & (floor_mu < hi) & (box_mass >= a) & (box_mass < b)
      row.append(f"{infeasible[sel].mean():>9.0%}" if sel.any() else "        -")
    print(f"  {lo:.1f}-{hi:.1f}         " + " ".join(row))

  print("\nTipping before sliding (horizontal push, no vertical hand force):")
  for h in args.push_heights:
    tips = tips_before_sliding(np.minimum(h, height), depth, box_mu)
    print(f"  push at {h:.2f} m: {tips.mean():6.1%} of boxes tip first")
  tips_top = tips_before_sliding(height - 0.05, depth, box_mu)
  print(f"  push 5 cm below the top: {tips_top.mean():6.1%}")
  either = infeasible | tips_before_sliding(np.minimum(0.5, height), depth, box_mu)
  print(f"  infeasible OR tips first at 0.5 m: {either.mean():6.1%}")


if __name__ == "__main__":
  main()
