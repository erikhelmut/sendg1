#!/usr/bin/env python
"""Compare the gait of tactile and blind policies (``scripts/gait_analysis.py``
output) and write an HTML report.

    python scripts/gait_report.py results/gait/velocity_flat_lstm

Reads ``<dir>/{tactile,blind}_s<seed>_<tag>_<condition>.json``. The unit of
comparison is the seed: each policy's stride means are one sample, so n is the
number of seeds per condition, not the number of strides. Scalars get the
difference, Hedges' g and a Welch t-test; curves get a permutation test with
the maximum t over the cycle (the nonparametric form of statistical parametric
mapping), which needs at least 4 seeds per group to reach p < 0.05.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import re
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
NAME = re.compile(r"^(tactile|blind)_s(\d+)_(.+)_(clean|perturbed)\.json$")
GROUPS = ("tactile", "blind")
CURVES = ("angle", "torque", "power", "pelvis", "grf_stance", "cop_stance")


def _hedges_g(a: np.ndarray, b: np.ndarray) -> float:
  na, nb = len(a), len(b)
  if na < 2 or nb < 2:
    return float("nan")
  sp = math.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
  if sp == 0:
    return float("nan")
  return (a.mean() - b.mean()) / sp * (1 - 3 / (4 * (na + nb) - 9))


def _welch_t(a: np.ndarray, b: np.ndarray) -> np.ndarray:
  va, vb = a.var(0, ddof=1) / len(a), b.var(0, ddof=1) / len(b)
  return (a.mean(0) - b.mean(0)) / np.sqrt(np.maximum(va + vb, 1e-12))


def max_t_permutation(a: np.ndarray, b: np.ndarray, alpha: float = 0.05) -> dict:
  """Two-sample curve test, family-wise over the whole cycle: the observed
  |t| curve against the permutation distribution of max |t|. Exact (all
  relabellings) for small groups."""
  both = np.concatenate([a, b])
  n, na = len(both), len(a)
  t_obs = _welch_t(a, b)
  maxes = []
  for idx in itertools.combinations(range(n), na):
    m = np.zeros(n, dtype=bool)
    m[list(idx)] = True
    maxes.append(np.abs(_welch_t(both[m], both[~m])).max())
  maxes = np.array(maxes)
  thresh = float(np.quantile(maxes, 1 - alpha))
  p = float((maxes >= np.abs(t_obs).max()).mean())
  return {
    "p": p,
    "min_p": 1.0 / len(maxes),
    "threshold": thresh,
    "significant": (np.abs(t_obs) > thresh).tolist()
    if 1.0 / len(maxes) <= alpha
    else None,
  }


def load(directory: Path) -> dict:
  runs: dict = {}
  for f in sorted(directory.glob("*.json")):
    m = NAME.match(f.name)
    if not m:
      continue
    group, seed, tag, cond = m.groups()
    runs.setdefault(cond, {}).setdefault(group, {})[int(seed)] = (
      tag,
      json.loads(f.read_text()),
    )
  return runs


def compare(runs: dict) -> dict:
  out: dict = {"conditions": {}}
  for cond, by_group in runs.items():
    seeds = {g: sorted(by_group.get(g, {})) for g in GROUPS}
    any_run = next(iter(next(iter(by_group.values())).values()))[1]
    speeds = list(any_run["speeds"])
    cdata: dict = {"seeds": seeds, "speeds": {}}
    out["meta"] = {
      "angles": any_run["angles"],
      "leg_joints": any_run["leg_joints"],
      "torque_limits_nm": any_run["torque_limits_nm"],
      "mass_kg": any_run["mass_kg"],
      "checkpoints": {
        g: {
          s: Path(t[1]["checkpoint"]).parent.name + "/" + Path(t[1]["checkpoint"]).name
          for s, t in by_group.get(g, {}).items()
        }
        for g in GROUPS
      },
      "protocol": any_run["protocol"],
    }
    for v in speeds:
      per = {g: [by_group[g][s][1]["speeds"][v] for s in seeds[g]] for g in GROUPS}
      sp: dict = {
        "strides": {g: [p["n_strides"] for p in per[g]] for g in GROUPS},
        "falls": {g: [p.get("falls", 0) for p in per[g]] for g in GROUPS},
        "scalars": {},
        "curves": {},
      }
      keys = list(per["tactile"][0]["scalars"]) + [
        f"cv_{k}" for k in per["tactile"][0]["variability_cv_pct"]
      ]
      for k in keys:

        def val(p, k=k):
          if k.startswith("cv_"):
            return p["variability_cv_pct"][k[3:]]
          return p["scalars"][k][0]

        a = np.array([val(p) for p in per["tactile"]], dtype=float)
        b = np.array([val(p) for p in per["blind"]], dtype=float)
        p_val = float("nan")
        if len(a) >= 2 and len(b) >= 2 and (a.std() > 0 or b.std() > 0):
          p_val = float(stats.ttest_ind(a, b, equal_var=False).pvalue)
        sp["scalars"][k] = {
          "tactile": a.round(5).tolist(),
          "blind": b.round(5).tolist(),
          "g": _hedges_g(a, b),
          "p": p_val,
        }
      for c in CURVES:
        a = np.array([p["curves"][c]["mean"] for p in per["tactile"]])  # [S, 101, K]
        b = np.array([p["curves"][c]["mean"] for p in per["blind"]])
        entry = {}
        for g, arr in (("tactile", a), ("blind", b)):
          entry[g] = {
            "mean": arr.mean(0).round(4).tolist(),
            "sd": (arr.std(0, ddof=1) if len(arr) > 1 else np.zeros_like(arr[0]))
            .round(4)
            .tolist(),
          }
        entry["test"] = [
          max_t_permutation(a[..., k], b[..., k]) if len(a) >= 2 and len(b) >= 2 else None
          for k in range(a.shape[-1])
        ]
        sp["curves"][c] = entry
      cdata["speeds"][v] = sp
    out["conditions"][cond] = cdata
  return out


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("directory")
  ap.add_argument("--out", default=None, help="default <directory>/report.html")
  ap.add_argument("--title", default="Flat walking gait: tactile vs blind")
  ap.add_argument("--video-note", default="", help="caption under the clips (policy, seed)")
  args = ap.parse_args()
  directory = Path(args.directory)
  data = compare(load(directory))
  data["title"] = args.title
  # Side-by-side clips (tactile left, blind right) from ``<dir>/videos``.
  views = (("side", "Sagittal plane, side view"), ("front", "Frontal plane, front view"),
           ("top", "Transverse plane, top view"))
  data["videos"] = [
    {"src": f"videos/{v}.mp4", "label": label}
    for v, label in views
    if (directory / "videos" / f"{v}.mp4").exists()
  ]
  data["video_note"] = args.video_note
  (directory / "summary.json").write_text(json.dumps(data))
  template = (HERE / "gait_report.html").read_text()
  html = template.replace("/*__DATA__*/null", json.dumps(data).replace("</", "<\\/"))
  out = Path(args.out) if args.out else directory / "report.html"
  out.write_text(html)
  print(f"[gait-report] wrote {out} and {directory / 'summary.json'}")


if __name__ == "__main__":
  main()
