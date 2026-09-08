#!/usr/bin/env python
"""Enumerate and run the (task x observation-set x seed) grid.

    # what would run
    python scripts/sweep.py --dry-run

    # the real thing
    python scripts/sweep.py --tasks flat --seeds 0 --max-iterations 3000

Diagnostic observation sets (contact_oracle) are NEVER included unless named
explicitly with --obs-sets, because their results are not claims about a
deployable policy.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

TASK_BASES = {
  "flat": "Sendg1-Velocity-Flat-G1",
  "rough": "Sendg1-Velocity-Rough-G1",
}


def main() -> None:
  from sendg1.common.obs.obs_sets import OBS_SETS, research_obs_sets
  from sendg1.tasks._register import _slug

  ap = argparse.ArgumentParser()
  ap.add_argument("--tasks", nargs="+", default=["flat"], choices=list(TASK_BASES))
  ap.add_argument(
    "--obs-sets",
    nargs="+",
    default=None,
    help=f"default: {list(research_obs_sets())}; diagnostic sets must be named",
  )
  ap.add_argument("--seeds", nargs="+", type=int, default=[0])
  ap.add_argument("--max-iterations", type=int, default=None)
  ap.add_argument("--num-envs", type=int, default=None)
  ap.add_argument("--log-root", default=str(REPO / "runs"))
  ap.add_argument("--logger", default=None, choices=["wandb", "tensorboard"],
                  help="override the configured logger (default: wandb)")
  ap.add_argument("--dry-run", action="store_true")
  args = ap.parse_args()

  obs_sets = args.obs_sets or list(research_obs_sets())
  for name in obs_sets:
    if name not in OBS_SETS:
      sys.exit(f"unknown obs set {name!r}; known: {sorted(OBS_SETS)}")
    if OBS_SETS[name].diagnostic:
      print(
        f"[sweep] NOTE: {name!r} is a DIAGNOSTIC set. Its actor sees idealised "
        "signals; results are an upper bound, not a deployable policy."
      )

  if len(args.seeds) == 1:
    print(
      "[sweep] NOTE: 1 seed. Enough to see whether the pipeline is sane and "
      "whether an effect is plausibly there; NOT enough to separate an effect "
      "from seed variance. Use 5 seeds for any reported claim."
    )

  jobs = [
    (f"{TASK_BASES[t]}-{_slug(o)}", o, s)
    for t in args.tasks
    for o in obs_sets
    for s in args.seeds
  ]

  print(f"[sweep] {len(jobs)} run(s):")
  for task_id, obs_set, seed in jobs:
    print(f"  {task_id}  seed={seed}")
  if args.dry_run:
    return

  for task_id, obs_set, seed in jobs:
    cmd = [
      "train", task_id,
      "--agent.seed", str(seed),
      "--agent.run-name", f"seed{seed}",
      "--log-root", args.log_root,
    ]
    if args.logger is not None:
      cmd += ["--agent.logger", args.logger]
    if args.max_iterations is not None:
      cmd += ["--agent.max-iterations", str(args.max_iterations)]
    if args.num_envs is not None:
      cmd += ["--env.scene.num-envs", str(args.num_envs)]
    print(f"\n[sweep] $ {' '.join(cmd)}\n")
    result = subprocess.run(cmd)
    if result.returncode != 0:
      sys.exit(f"[sweep] FAILED: {task_id} seed={seed}")


if __name__ == "__main__":
  main()
