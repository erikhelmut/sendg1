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
  tips_before_sliding,
  traction_margin,
)

NEAR_LIMIT = 0.20


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--samples", type=int, default=1_000_000)
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--alpha", type=float, default=1.0, help="curriculum level; 1 = full ranges")
  ap.add_argument("--push-heights", type=float, nargs="+", default=[0.3, 0.5, 0.7])
  args = ap.parse_args()

  r = BoxPushRanges()
  cur = curriculum_ranges(r, args.alpha)
  rng = np.random.default_rng(args.seed)
  n = args.samples
  floor_mu = rng.uniform(*cur["floor_mu"], n)
  box_mass = rng.uniform(*cur["box_mass"], n)
  box_mu = rng.uniform(*cur["box_mu"], n)
  depth = rng.uniform(*r.box_depth, n)
  height = rng.uniform(*r.box_height, n)

  print(f"robot mass {G1_MASS_KG:.2f} kg, g {GRAVITY}, alpha {args.alpha}")
  print(
    f"floor mu {cur['floor_mu']}, box mass {cur['box_mass']} kg, box mu {cur['box_mu']}\n"
  )

  print("Traction: mu_foot * m_robot * g  vs  mu_box * m_box * g  [N]")
  print(f"  available: {G1_MASS_KG * GRAVITY * cur['floor_mu'][0]:.0f}"
        f" .. {G1_MASS_KG * GRAVITY * cur['floor_mu'][1]:.0f}")
  print(f"  required:  {GRAVITY * cur['box_mu'][0] * cur['box_mass'][0]:.0f}"
        f" .. {GRAVITY * cur['box_mu'][1] * cur['box_mass'][1]:.0f}")

  m = traction_margin(floor_mu, box_mass, box_mu)
  infeasible = m < 0
  near = (m >= 0) & (m < NEAR_LIMIT)
  print(f"\nmargin = 1 - required/available, over {n:,} sampled episodes")
  print(f"  infeasible (margin < 0):      {infeasible.mean():6.1%}")
  print(f"  near the limit (0 <= m < 20%): {near.mean():6.1%}")
  print(f"  comfortable (m >= 20%):        {(m >= NEAR_LIMIT).mean():6.1%}")
  q = np.percentile(m, [10, 25, 50, 75, 90])
  print("  percentiles 10/25/50/75/90:   " + " ".join(f"{v:+.2f}" for v in q))

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
