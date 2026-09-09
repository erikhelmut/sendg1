#!/usr/bin/env python
"""Evaluate a trained checkpoint under the fixed protocol.

    python scripts/evaluate.py --task Sendg1-Velocity-Flat-G1-Blind --seed 0

Writes results/<experiment>/<seed>.json. Both conditions run through this same
code path with the same protocol, which is what makes the numbers comparable.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--task", required=True)
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--checkpoint", default=None, help="explicit .pt path")
  ap.add_argument("--log-root", default=str(REPO / "runs"))
  ap.add_argument("--results-root", default=str(REPO / "results"))
  ap.add_argument("--num-envs", type=int, default=None)
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
  from sendg1.common.terrains import TERRAIN_ROLES
  from sendg1.eval.metrics import EpisodeStats, TerrainStats, summarize
  from sendg1.eval.protocol import EvalProtocol, build_eval_env_cfg

  configure_torch_backends()
  device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  protocol = EvalProtocol()
  if args.num_envs is not None:
    protocol = EvalProtocol(num_envs=args.num_envs)

  rl_cfg = load_rl_cfg(args.task)
  rl_cfg.seed = args.seed

  if args.checkpoint:
    ckpt = Path(args.checkpoint)
  else:
    exp_dir = Path(args.log_root) / rl_cfg.experiment_name
    ckpt = get_checkpoint_path(exp_dir, f".*seed{args.seed}$", "model_.*.pt")
  print(f"[eval] checkpoint: {ckpt}")

  env_cfg = build_eval_env_cfg(args.task, protocol)
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=rl_cfg.clip_actions)

  runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
  runner = runner_cls(wrapped, asdict(rl_cfg), device=device)
  runner.load(str(ckpt), load_cfg={"actor": True}, strict=True, map_location=device)
  policy = runner.get_inference_policy(device=device)

  # Pin one command per env, drawn from the protocol grid by env index. Every
  # condition therefore sees the same command on the same env index, with the
  # same terrain patch and the same domain-randomisation draw.
  grid = torch.tensor(protocol.command_grid, device=device, dtype=torch.float32)
  assign = torch.arange(protocol.num_envs, device=device) % grid.shape[0]
  fixed_cmd = grid[assign]

  cmd_term = env.command_manager.get_term("twist")
  robot = env.scene["robot"]
  stats = EpisodeStats()

  # Per-terrain-type accumulators. In curriculum mode each sub-terrain owns one
  # column, so terrain_types is an index into the sub_terrains ordering.
  terrain = env.scene.terrain
  per_terrain: dict[int, TerrainStats] = {}
  origins_xy = None
  num_cols = 0
  strayed = 0
  total_att = 0
  if terrain is not None and terrain.cfg.terrain_generator is not None:
    names = list(terrain.cfg.terrain_generator.sub_terrains)
    for i, nm in enumerate(names):
      role = TERRAIN_ROLES[nm].role if nm in TERRAIN_ROLES else "unknown"
      per_terrain[i] = TerrainStats(name=nm, role=role)
    # Attribute by the patch the robot is STANDING ON, not the column it was
    # assigned. Nothing confines a robot to its patch: out_of_terrain_bounds
    # only fires at the outer edge of the whole grid, and a 20 s episode at
    # 1 m/s covers ~2.5 patch widths. Attributing by assignment would let
    # discriminator terrain leak into the control columns and vice versa --
    # exactly the signal this study is trying to separate.
    #
    # Patches are a uniform axis-aligned grid, so the nearest origin IS the
    # containing patch. Positions out in the flat border clamp to the nearest
    # edge patch, which is the sensible reading of "which terrain is it on".
    assert terrain.terrain_origins is not None
    num_rows, num_cols = terrain.terrain_origins.shape[:2]
    origins_xy = terrain.terrain_origins[..., :2].reshape(-1, 2).contiguous()

  obs, _ = wrapped.reset()
  steps = int(protocol.episode_length_s / env.step_dt) * protocol.num_episodes
  print(f"[eval] {steps} steps x {protocol.num_envs} envs")

  # Track episode length ourselves: mjlab zeroes env.episode_length_buf for done
  # envs inside step(), so reading it after step() always yields 0.
  ep_steps = torch.zeros(protocol.num_envs, device=device)

  with torch.inference_mode():
    for _ in range(steps):
      cmd_term.vel_command_b[:] = fixed_cmd
      actions = policy(obs)
      obs, _, dones, _ = wrapped.step(actions)
      cmd_term.vel_command_b[:] = fixed_cmd
      ep_steps += 1

      lin_err = torch.norm(
        fixed_cmd[:, :2] - robot.data.root_link_lin_vel_b[:, :2], dim=-1
      )
      ang_err = (fixed_cmd[:, 2] - robot.data.root_link_ang_vel_b[:, 2]).abs()
      stats.add_step(lin_err, ang_err)

      cur_col = cur_row = None
      if origins_xy is not None:
        cell = torch.cdist(robot.data.root_link_pos_w[:, :2], origins_xy).argmin(dim=1)
        cur_row, cur_col = cell // num_cols, cell % num_cols
        strayed += int((cur_col != terrain.terrain_types).sum())
        total_att += int(cur_col.numel())
        for idx, ts in per_terrain.items():
          m = cur_col == idx
          if not bool(m.any()):
            continue
          ts.lin_vel_error_sum += float(lin_err[m].sum())
          ts.ang_vel_error_sum += float(ang_err[m].sum())
          ts.step_count += int(m.sum())

      done_mask = dones.bool()
      if done_mask.any():
        # A fall is a real termination; a time-out is a successful episode.
        fell = env.termination_manager.terminated[done_mask]
        stats.add_terminations(fell, ep_steps[done_mask] * env.step_dt)
        ep_steps[done_mask] = 0.0

        if cur_col is not None and cur_row is not None:
          done_cols = cur_col[done_mask]
          done_rows = cur_row[done_mask].float()
          for idx, ts in per_terrain.items():
            m = done_cols == idx
            if not bool(m.any()):
              continue
            ts.episodes += int(m.sum())
            ts.falls += int(fell[m].sum())
            ts.level_sum += float(done_rows[m].sum())
            ts.level_n += int(m.sum())

  env.close()

  out = {
    "task": args.task,
    "seed": args.seed,
    "checkpoint": str(ckpt),
    "protocol": asdict(protocol),
    "mjlab_version": MJLAB_VERSION,
    "mjlab_commit": MJLAB_COMMIT,
    **summarize(stats),
  }
  if per_terrain:
    out["per_terrain"] = {
      ts.name: {"role": ts.role, **ts.summary()} for ts in per_terrain.values()
    }
    # How often a robot was standing on a patch other than the one it spawned
    # in. Quantifies the contamination that position-based attribution removes;
    # a large number also means episodes are long relative to patch size.
    out["metrics"]["strayed_frac"] = strayed / max(total_att, 1)
  dest = Path(args.results_root) / rl_cfg.experiment_name
  dest.mkdir(parents=True, exist_ok=True)
  path = dest / f"seed{args.seed}.json"
  path.write_text(json.dumps(out, indent=2))
  print(f"[eval] wrote {path}")
  for k, v in out["metrics"].items():
    print(f"  {k:20s} {v:.4f}")

  if "per_terrain" in out:
    print(f"\n  {'terrain':22s} {'role':14s} {'lin_err':>8s} {'fall':>7s} {'level':>7s}")
    for nm, m in out["per_terrain"].items():
      print(
        f"  {nm:22s} {m['role']:14s} {m['lin_vel_error']:8.3f} "
        f"{m['fall_rate']:7.2%} {m['mean_terrain_level']:7.2f}"
      )


if __name__ == "__main__":
  main()
