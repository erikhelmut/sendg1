#!/usr/bin/env python
"""Evaluate a box-pushing checkpoint on the floor-mu x box-mass grid.

    python scripts/evaluate_box_push.py --task Sendg1-BoxPush-G1-Blind --seed 0
    python scripts/evaluate_box_push.py --task ... --checkpoint runs/.../model_2000.pt

Protocol: sendg1/eval/box_push.py (5 floor mu x 4 masses, 32 episodes per cell,
every episode's world pinned from the protocol seed, so every checkpoint and
every observation set is scored on the same 640 episodes).

Writes, under results/<experiment>/:
  box_push_grid_seed<S>.csv        one row per episode
  box_push_grid_seed<S>.json       per-cell summary + protocol + checkpoint
  box_push_heatmaps_seed<S>.png    success rate and slip events per cell
and videos (gitignored) under videos/<experiment>/seed<S>/: a few episodes of
every cell at the lowest and highest floor mu.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
os.environ.setdefault("MUJOCO_GL", "egl")

RAMP = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
"""Single-hue sequential ramp (blue, light -> dark): more is darker."""
INK, INK_ON_DARK, MUTED = "#1f1f1f", "#ffffff", "#6b6b6b"


def _nanmean(xs: list[float]) -> float:
  arr = np.asarray(xs, dtype=float)
  return float(np.nanmean(arr)) if np.isfinite(arr).any() else float("nan")


def summarize(rows: list[dict], protocol) -> list[dict]:
  cells = []
  for mu, mass in protocol.cells:
    sel = [r for r in rows if r["cell_floor_mu"] == mu and r["cell_box_mass"] == mass]
    n = len(sel)
    cells.append({
      "floor_mu": mu,
      "box_mass": mass,
      "episodes": n,
      "success_rate": _nanmean([r["success"] for r in sel]),
      "time_to_target_s": _nanmean([r["time_to_target"] for r in sel]),
      "slip_events": _nanmean([r["slip_events"] for r in sel]),
      "fall_rate": _nanmean([r["fell"] for r in sel]),
      "box_tipped_rate": _nanmean([r["box_tipped"] for r in sel]),
      "mean_push_force_n": _nanmean([r["mean_push_force"] for r in sel]),
      "double_support_impulse_frac": _nanmean([r["double_support_impulse_frac"] for r in sel]),
      "infeasible_frac": _nanmean([float(r["traction_margin"] < 0) for r in sel]),
    })
  return cells


def heatmaps(cells: list[dict], protocol, title: str, dest: Path) -> None:
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  from matplotlib.colors import LinearSegmentedColormap

  cmap = LinearSegmentedColormap.from_list("seq_blue", RAMP)
  cmap.set_bad("#e6e6e6")
  mus, masses = list(protocol.floor_mu), list(protocol.box_mass)

  def grid(key: str) -> np.ndarray:
    g = np.full((len(mus), len(masses)), np.nan)
    for c in cells:
      g[mus.index(c["floor_mu"]), masses.index(c["box_mass"])] = c[key]
    return g

  panels = [
    ("success_rate", "Success rate", (0.0, 1.0), "{:.0%}"),
    ("slip_events", "Full-foot slip events per episode", None, "{:.1f}"),
  ]
  fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
  for ax, (key, name, lim, fmt) in zip(axes, panels):
    g = grid(key)
    vmin, vmax = lim if lim else (0.0, max(1.0, float(np.nanmax(g)) if np.isfinite(g).any() else 1.0))
    im = ax.imshow(g, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower", aspect="auto")
    for i in range(len(mus)):
      for j in range(len(masses)):
        v = g[i, j]
        text = "–" if not math.isfinite(v) else fmt.format(v)
        dark = math.isfinite(v) and (v - vmin) / max(vmax - vmin, 1e-9) > 0.55
        ax.text(j, i, text, ha="center", va="center", fontsize=10,
                color=INK_ON_DARK if dark else INK)
    ax.set_xticks(range(len(masses)), [f"{m:.0f}" for m in masses])
    ax.set_yticks(range(len(mus)), [f"{m:.2f}" for m in mus])
    ax.set_xlabel("box mass (kg)", color=MUTED)
    ax.set_ylabel("floor friction μ", color=MUTED)
    ax.set_title(name, loc="left", fontsize=11, color=INK)
    ax.tick_params(length=0, colors=MUTED)
    for s in ax.spines.values():
      s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.85)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0, colors=MUTED)
  fig.suptitle(title, x=0.01, ha="left", fontsize=11, color=INK)
  fig.savefig(dest, dpi=150)
  plt.close(fig)


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--task", required=True)
  ap.add_argument("--seed", type=int, default=0, help="training seed of the checkpoint")
  ap.add_argument("--checkpoint", default=None, help="explicit .pt path")
  ap.add_argument("--log-root", default=str(REPO / "runs"))
  ap.add_argument("--results-root", default=str(REPO / "results"))
  ap.add_argument("--video-root", default=str(REPO / "videos"))
  ap.add_argument("--episodes-per-cell", type=int, default=None,
                  help="override the protocol's N (recorded in the output)")
  ap.add_argument("--no-video", action="store_true")
  ap.add_argument("--device", default=None)
  args = ap.parse_args()

  import torch
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_rl_cfg, load_runner_cls
  from mjlab.utils.os import get_checkpoint_path
  from mjlab.utils.torch import configure_torch_backends

  import sendg1.tasks  # noqa: F401
  from sendg1 import MJLAB_COMMIT, MJLAB_VERSION
  from sendg1.common.box_push.params import traction_margin
  from sendg1.eval.box_push import (
    BoxPushProtocol,
    build_eval_env_cfg,
    configure_video_view,
    draw_worlds,
    pin_worlds,
    record_row,
    roll_out_video,
  )
  from sendg1.tasks._register import variants

  configure_torch_backends()
  device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  protocol = BoxPushProtocol()
  if args.episodes_per_cell is not None:
    protocol = BoxPushProtocol(episodes_per_cell=args.episodes_per_cell)
  worlds = draw_worlds(protocol)
  total = protocol.num_episodes

  rl_cfg = load_rl_cfg(args.task)
  _, obs_set = variants()[args.task]
  ckpt = Path(args.checkpoint) if args.checkpoint else get_checkpoint_path(
    Path(args.log_root) / rl_cfg.experiment_name, f".*seed{args.seed}$", "model_.*.pt"
  )
  print(f"[eval] checkpoint: {ckpt}")

  def load_policy(wrapped):
    runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
    runner = runner_cls(wrapped, asdict(rl_cfg), device=device)
    runner.load(str(ckpt), load_cfg={"actor": True}, strict=True, map_location=device)
    return runner.get_inference_policy(device=device)

  # -- the grid: one episode per env ------------------------------------------
  env = ManagerBasedRlEnv(cfg=build_eval_env_cfg(args.task, protocol, total), device=device)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=rl_cfg.clip_actions)
  policy = load_policy(wrapped)
  pin_worlds(env, worlds, np.arange(total))
  obs, _ = wrapped.reset()  # the pins take effect here

  rows: list[dict | None] = [None] * total
  pending = torch.ones(total, dtype=torch.bool, device=device)
  with torch.inference_mode():
    for _ in range(env.max_episode_length + 2):
      obs, _, dones, _ = wrapped.step(policy(obs))
      done = dones.bool()
      if hasattr(policy, "reset"):
        policy.reset(done)
      for i in (done & pending).nonzero().flatten().tolist():
        rows[i] = record_row(env, i)
      pending &= ~done
      if not pending.any():
        break
  env.close()
  assert not pending.any(), "some episodes did not finish"

  out_rows = []
  for i, rec in enumerate(rows):
    assert rec is not None
    mu, mass = protocol.cells[worlds.cell[i]]
    w = {k: float(v[i]) for k, v in worlds.values.items()}
    out_rows.append({
      "task": args.task,
      "obs_set": obs_set,
      "train_seed": args.seed,
      "episode": i,
      "cell_floor_mu": mu,
      "cell_box_mass": mass,
      **{f"world_{k}": v for k, v in w.items()},
      "traction_margin": float(traction_margin(
        np.array(w["floor_mu"]), np.array(w["box_mass"]), np.array(w["box_mu"])
      )),
      **rec,
    })

  dest = Path(args.results_root) / rl_cfg.experiment_name
  dest.mkdir(parents=True, exist_ok=True)
  csv_path = dest / f"box_push_grid_seed{args.seed}.csv"
  with csv_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(out_rows[0]))
    writer.writeheader()
    writer.writerows(out_rows)

  cells = summarize(out_rows, protocol)
  (dest / f"box_push_grid_seed{args.seed}.json").write_text(json.dumps({
    "task": args.task,
    "seed": args.seed,
    "checkpoint": str(ckpt),
    "protocol": asdict(protocol),
    "mjlab_version": MJLAB_VERSION,
    "mjlab_commit": MJLAB_COMMIT,
    "cells": cells,
  }, indent=2))
  png = dest / f"box_push_heatmaps_seed{args.seed}.png"
  heatmaps(cells, protocol, f"{args.task}  seed {args.seed}  ({ckpt.name}, "
           f"{protocol.episodes_per_cell} episodes per cell)", png)

  print(f"[eval] {csv_path}\n[eval] {png}")
  print(f"{'mu':>4} {'kg':>4} {'success':>8} {'slips':>6} {'falls':>6} {'tipped':>7} {'infeas.':>8}")
  for c in cells:
    print(f"{c['floor_mu']:4.1f} {c['box_mass']:4.0f} {c['success_rate']:8.0%} "
          f"{c['slip_events']:6.1f} {c['fall_rate']:6.0%} {c['box_tipped_rate']:7.0%} "
          f"{c['infeasible_frac']:8.0%}")

  if args.no_video:
    return

  # -- videos: the first episodes of the lowest- and highest-mu cells ---------
  from sendg1.video import EvalVideo

  vcfg = build_eval_env_cfg(args.task, protocol, 1)
  configure_video_view(vcfg, 1280, 720)
  venv = ManagerBasedRlEnv(cfg=vcfg, device=device, render_mode="rgb_array")
  vwrapped = RslRlVecEnvWrapper(venv, clip_actions=rl_cfg.clip_actions)
  video = EvalVideo(venv, rl_cfg, title=args.task)
  vpolicy = load_policy(vwrapped)
  vdir = Path(args.video_root) / rl_cfg.experiment_name / f"seed{args.seed}"
  vdir.mkdir(parents=True, exist_ok=True)
  with torch.inference_mode():  # explicit resets write sensor buffers
    for c_idx, (mu, mass) in enumerate(protocol.cells):
      if mu not in protocol.video_floor_mu:
        continue
      for k in range(protocol.videos_per_cell):
        ep = c_idx * protocol.episodes_per_cell + k
        pin_worlds(venv, worlds, np.array([ep]))
        obs, _ = vwrapped.reset()
        if hasattr(vpolicy, "reset"):
          vpolicy.reset(torch.ones(1, dtype=torch.bool, device=device))
        frames, rec, _ = roll_out_video(
          vwrapped, vpolicy, obs, max_steps=venv.max_episode_length + 1,
          label=f"{obs_set} seed {args.seed}  mu {mu:.1f}  {mass:.0f} kg  episode {ep}",
          video=video,
        )
        path = vdir / f"mu{mu:.1f}_m{mass:.0f}_ep{ep}.mp4"
        video.save(path, frames)
        print(f"[eval] video {path}  success={rec.get('success')}")
  venv.close()


if __name__ == "__main__":
  main()
