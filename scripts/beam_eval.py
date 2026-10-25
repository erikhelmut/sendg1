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
- Gait, per width (``gait``): swings (air phases of at least ``MIN_SWING_S``)
  per metre the pelvis travelled, their median / p90 apex (foot-site height
  gained between lift-off and touch-down), the share of lift-offs that are
  shorter contact chatter, stance slide (horizontal foot-site speed while in
  contact, integrated, per metre travelled) and the fraction of foot-time in
  the air. A shuffling gait shows up as few swings, a low apex and a high
  slide; success rates alone hide it, and a gait that relies on sliding is
  friction-dependent on hardware.

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


MIN_SWING_S = 0.1  # shorter air phases are contact chatter, not steps


class GaitMeter:
  """Accumulates the gait statistics of the running episode of every env.
  ``step`` runs after each env step; envs reset in that step are skipped (their
  state is already the next episode's spawn) and restarted."""

  def __init__(self, env):
    import torch

    from sendg1.common.beam.mdp import BeamEpisode

    self.env, self.robot = env, env.scene["robot"]
    self.sites = BeamEpisode.get(env)._foot_sites
    z = lambda: torch.zeros(env.num_envs, device=env.device)  # noqa: E731
    self.path, self.slide, self.air, self.steps, self.chatter = z(), z(), z(), z(), z()
    self.air_t = torch.zeros(env.num_envs, 2, device=env.device)
    self.apexes: list[list[float]] = [[] for _ in range(env.num_envs)]
    c, z_, xy = self._state()
    self.c, self.lift_z, self.apex, self.xy = c.clone(), z_.clone(), torch.zeros_like(z_), xy.clone()

  def _state(self):
    from sendg1.common.beam.mdp import _feet_in_contact

    d = self.robot.data
    return _feet_in_contact(self.env), d.site_pos_w[:, self.sites, 2], d.root_link_pos_w[:, :2]

  def step(self, dones):
    import torch

    live = ~dones
    c, z, xy = self._state()
    v = self.robot.data.site_lin_vel_w[:, self.sites, :2].norm(dim=-1)
    off = self.c & ~c & live[:, None]
    on = ~self.c & c & live[:, None]
    self.lift_z = torch.where(off, z, self.lift_z)
    self.apex = torch.where(off, torch.zeros_like(self.apex), torch.maximum(self.apex, z - self.lift_z))
    self.air_t = torch.where(c, self.air_t, self.air_t + self.env.step_dt)
    swing = on & (self.air_t >= MIN_SWING_S)
    for i, f in swing.nonzero().tolist():
      self.apexes[i].append(float(self.apex[i, f]))
    self.chatter += (on & ~swing).sum(1)
    self.air_t = torch.where(c, torch.zeros_like(self.air_t), self.air_t)
    m = live.float()
    self.air += (~c).float().sum(1) * m
    self.steps += m
    self.slide += (v * c.float()).sum(1) * self.env.step_dt * m
    self.path += (xy - self.xy).norm(dim=-1) * m
    # Reset envs restart from their spawn state.
    d = dones[:, None]
    self.c, self.xy = c, xy.clone()
    self.lift_z = torch.where(d, z, self.lift_z)
    self.apex = torch.where(d, torch.zeros_like(self.apex), self.apex)
    self.air_t = torch.where(d, torch.zeros_like(self.air_t), self.air_t)

  def pop(self, e: int) -> dict:
    """The finished episode of env ``e``; zeroes its counters."""
    out = {
      "path_m": float(self.path[e]),
      "slide_m": float(self.slide[e]),
      "chatter": int(self.chatter[e]),
      "air_frac": float(self.air[e] / (2 * self.steps[e].clamp(min=1))),
      "_apex": self.apexes[e],
    }
    for t in (self.path, self.slide, self.air, self.steps, self.chatter):
      t[e] = 0.0
    self.apexes[e] = []
    return out


def _gait(rs: list[dict]) -> dict:
  import numpy as np

  apex = np.array([a for r in rs for a in r["_apex"]]) * 100
  path = max(sum(r["path_m"] for r in rs), 1e-6)
  chatter = sum(r["chatter"] for r in rs)
  return {
    "swings_per_m": apex.size / path,
    "swing_apex_cm_median": float(np.median(apex)) if apex.size else None,
    "swing_apex_cm_p90": float(np.quantile(apex, 0.9)) if apex.size else None,
    "chatter_frac": chatter / max(chatter + apex.size, 1),
    "slide_cm_per_m": 100 * sum(r["slide_m"] for r in rs) / path,
    "air_frac": sum(r["air_frac"] for r in rs) / len(rs),
  }


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
    if "_apex" in rs[0]:
      out[f"{w:.2f}"]["gait"] = _gait(rs)
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
  gait = GaitMeter(env)
  with torch.inference_mode():
    while bool((done_per_env < args.episodes).any()):
      n0 = len(env._beam_records)
      obs, _, dones, _ = wrapped.step(policy(obs))
      policy.reset(dones.bool())
      done_per_env += dones.long()
      gait.step(dones.bool())
      for r in env._beam_records[n0:]:
        r.update(gait.pop(r["env"]))
  # Keep exactly the first ``episodes`` per env, so every width is weighted the same.
  kept, seen = [], defaultdict(int)
  for r in env._beam_records:
    if seen[r["env"]] < args.episodes:
      kept.append(r)
      seen[r["env"]] += 1
  summary = summarise(kept)
  by_path = summarise_by_path(kept)
  for r in kept:
    r.pop("_apex")  # pooled into the per-width gait; too long to keep per episode
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
    g = s["gait"]
    print(
      f"              swings {g['swings_per_m']:.1f}/m  apex {g['swing_apex_cm_median'] or 0:.2f}"
      f"/{g['swing_apex_cm_p90'] or 0:.2f} cm  chatter {g['chatter_frac']:.2f}"
      f"  slide {g['slide_cm_per_m']:.0f} cm/m"
      f"  air {g['air_frac']:.2f}"
    )
  if len(by_path) > 1:
    for p, per_w in by_path.items():
      print(f"  {p:14s} " + " ".join(f"{w}:{s['success_rate']:.2f}" for w, s in per_w.items()))
  env.close()


if __name__ == "__main__":
  main()
