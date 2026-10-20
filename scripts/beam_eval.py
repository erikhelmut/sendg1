#!/usr/bin/env python
"""Evaluate a beam policy per beam width.

    python scripts/beam_eval.py --checkpoint runs/beam_e2e/<run>/model_14999.pt --out results/beam/e2e_vN.json
    python scripts/beam_eval.py --task G1-Beam-E2E-Curved --checkpoint <pt> --out results/beam/<name>_curved.json
    python scripts/beam_eval.py --task G1-Beam-Touch-Curved-Plain-Eval --checkpoint <pt> --out results/beam/<name>_eval.json

Protocol (fixed)
---------------------------------------------------
- Every width level gets ``--envs-per-level`` envs, spread over the terrain
  columns (four beam heights; with ``G1-Beam-E2E-Curved`` also bend and turn
  columns, reported separately per path kind); each env runs ``--episodes``
  episodes. No curriculum.
- Every episode starts on the START PLATFORM, whatever the width, with a
  constant forward command ``--speed`` (default 0.3 m/s), for the task's
  episode length (20 s; 35 s on held-out beams with a ``-Eval`` task). Success =
  reaching the end platform with no early termination.
- Training DR stays on (friction, mass, CoM, motor gains/strength, latency,
  IMU noise/bias, tactile faults at the level's strength). Pushes off by
  default (``--pushes`` turns them on).
- Per episode: success, distance traversed along the beam (from the beam start,
  clipped to [0, beam length]), mean |v_cmd - v_along| while walking, and which
  termination fired.

Writes ``--out`` (default ``results/beam/eval[_<tag>].json``).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def build_env(args, task: str):
  import torch
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.curriculum_manager import CurriculumTermCfg
  from mjlab.tasks.registry import load_env_cfg

  from sendg1.common.beam.terrain import BEAM_WIDTHS

  cfg = load_env_cfg(task)
  n_levels = len(BEAM_WIDTHS)
  cfg.scene.num_envs = args.envs_per_level * n_levels
  cfg.seed = args.seed
  cfg.events["spawn"].params = {
    "platform_from_level": 0,
    "platform_lateral_range": (-args.start_lateral, args.start_lateral),
    "yaw_range": (-args.start_yaw, args.start_yaw),
  }
  if not args.pushes:
    cfg.events.pop("push")
  cfg.events.pop("spawn_from_bank", None)  # training-only mid-beam restarts
  twist = cfg.commands["twist"]
  twist.ranges.lin_vel_x = (args.speed, args.speed)
  twist.rel_standing_envs = 0.0
  twist.resampling_time_range = (1e6, 1e6)
  cfg.curriculum = {"record": CurriculumTermCfg(func=_record_episodes)}
  env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
  t = env.scene.terrain
  lv = torch.arange(env.num_envs, device=env.device) % n_levels
  t.terrain_levels[:] = lv
  t.env_origins[:] = t.terrain_origins[lv, t.terrain_types]
  env._beam_records = []  # type: ignore[attr-defined]
  subs = cfg.scene.terrain.terrain_generator.sub_terrains
  env._beam_column_kind = [  # type: ignore[attr-defined]
    "straight" if c.path_kind == "straight" else f"turn<={c.max_turn_deg:.0f}deg" for c in subs.values()
  ]
  return env


def _record_episodes(env, env_ids):
  """Curriculum-slot hook: runs at reset, before the spawn event clears the
  episode state, so it sees each finished episode once."""
  import torch

  from sendg1.common.beam.mdp import BeamEpisode

  if env.common_step_counter == 0 or env_ids is None or len(env_ids) == 0:
    return {}
  ep = BeamEpisode.get(env)
  st = ep.stats(env_ids)
  beam = ep.geo.beam_arc()[env_ids]
  arc = ep.spawn_arc[env_ids] + ep.max_progress[env_ids]
  traversed = (arc - beam[:, 0]).clamp(min=0.0)
  traversed = torch.minimum(traversed, beam[:, 1] - beam[:, 0])
  reached = st.reached_end
  traversed = torch.where(reached, beam[:, 1] - beam[:, 0], traversed)
  tm = env.termination_manager
  reasons = []
  for i in range(len(env_ids)):
    fired = [n for n in tm.active_terms if n != "time_out" and bool(tm.get_term(n)[env_ids[i]])]
    reasons.append(fired[0] if fired else ("success" if bool(reached[i]) else "time_out"))
  for i, e in enumerate(env_ids.tolist()):
    env._beam_records.append(
      {
        "env": e,
        "level": int(st.level[i]),
        "width": float(ep.geo.widths[st.level[i]]),
        "beam_height": float(ep.geo.top_z()[e]),
        "path": env._beam_column_kind[int(ep.geo.types[e])],
        "success": bool(reached[i] and not st.terminated[i]),
        "traversed_m": float(traversed[i]),
        "vel_err": float(st.vel_err[i]),
        "end": reasons[i],
        "steps": int(ep.steps[e]),
      }
    )
  return {}


def load_policy(args, env):
  """Actor only, so a checkpoint loads into either beam task (their critics
  differ by the curved task's ``beam_ahead``)."""
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_rl_cfg

  rl_cfg = load_rl_cfg(args.task)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=rl_cfg.clip_actions)
  runner = MjlabOnPolicyRunner(wrapped, asdict(rl_cfg), device="cuda:0")
  runner.load(args.checkpoint, load_cfg={"actor": True}, strict=True, map_location="cuda:0")
  return wrapped, runner.get_inference_policy(device="cuda:0")


def summarise(records: list[dict]) -> dict:
  by_w: dict[float, list[dict]] = defaultdict(list)
  for r in records:
    by_w[round(r["width"], 3)].append(r)
  return _per_width(by_w)


def summarise_by_path(records: list[dict]) -> dict:
  by_path: dict[str, list[dict]] = defaultdict(list)
  for r in records:
    by_path[r["path"]].append(r)
  return {p: summarise(rs) for p, rs in sorted(by_path.items())}


def _per_width(by_w: dict[float, list[dict]]) -> dict:
  out = {}
  for w in sorted(by_w, reverse=True):
    rs = by_w[w]
    n = len(rs)
    ends: dict[str, int] = defaultdict(int)
    for r in rs:
      ends[r["end"]] += 1
    out[f"{w:.2f}"] = {
      "n": n,
      "success_rate": sum(r["success"] for r in rs) / n,
      "traversed_m": sum(r["traversed_m"] for r in rs) / n,
      "traversed_frac": sum(r["traversed_m"] for r in rs) / n / 4.0,
      "vel_err": sum(r["vel_err"] for r in rs) / n,
      "ends": dict(ends),
    }
  return out


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--checkpoint", required=True)
  ap.add_argument("--envs-per-level", type=int, default=128)
  ap.add_argument("--episodes", type=int, default=2)
  ap.add_argument("--speed", type=float, default=0.3)
  ap.add_argument("--pushes", action="store_true")
  ap.add_argument("--task", default="G1-Beam-E2E")
  ap.add_argument("--start-lateral", type=float, default=0.08, help="platform start offset (+/- m)")
  ap.add_argument("--start-yaw", type=float, default=0.08, help="start yaw offset (+/- rad)")
  ap.add_argument("--seed", type=int, default=12345)
  ap.add_argument("--tag", default="")
  ap.add_argument("--out", default=None)
  args = ap.parse_args()

  import torch

  import sendg1.tasks  # noqa: F401
  env = build_env(args, args.task)
  wrapped, policy = load_policy(args, env)
  obs, _ = wrapped.reset()
  env._beam_records.clear()
  # Run until every env has finished ``episodes`` episodes (envs that fail fast
  # must not fill the quota for slow ones).
  done_per_env = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
  with torch.inference_mode():
    while bool((done_per_env < args.episodes).any()):
      obs, _, dones, _ = wrapped.step(policy(obs))
      policy.reset(dones.bool())
      done_per_env += dones.long()
  # Keep exactly the first ``episodes`` per env, so every width is weighted the same.
  kept, seen = [], defaultdict(int)
  for r in env._beam_records:
    if seen[r["env"]] < args.episodes:
      kept.append(r)
      seen[r["env"]] += 1
  summary = summarise(kept)
  by_path = summarise_by_path(kept)
  out = Path(args.out or REPO / "results" / "beam" / f"eval{('_' + args.tag) if args.tag else ''}.json")
  out.parent.mkdir(parents=True, exist_ok=True)
  json.dump(
    {
      "task": args.task,
      "checkpoint": str(Path(args.checkpoint).resolve()),
      "protocol": {
        "speed": args.speed,
        "envs_per_level": args.envs_per_level,
        "episodes": args.episodes,
        "pushes": args.pushes,
        "seed": args.seed,
        "spawn": "start platform at every width",
        "episode_s": env.cfg.episode_length_s,
        "terrain_seed": env.cfg.scene.terrain.terrain_generator.seed,
        "start_lateral": args.start_lateral,
        "start_yaw": args.start_yaw,
      },
      "per_width": summary,
      "per_path": by_path,
      "episodes": kept,
    },
    open(out, "w"),
    indent=1,
  )
  print(f"wrote {out}")
  for w, s in summary.items():
    print(
      f"  width {w}  n={s['n']:4d}  success {s['success_rate']:.3f}  "
      f"traversed {s['traversed_m']:.2f} m  vel_err {s['vel_err']:.3f}  {s['ends']}"
    )
  if len(by_path) > 1:
    for p, per_w in by_path.items():
      print(f"  {p:14s} " + " ".join(f"{w}:{s['success_rate']:.2f}" for w, s in per_w.items()))
  env.close()


if __name__ == "__main__":
  main()
