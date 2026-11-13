#!/usr/bin/env python
"""Render beam episodes of a policy, headless.

    python scripts/beam_video.py --checkpoint <pt> --width 0.10 --want success
    python scripts/beam_video.py --checkpoint <pt> --width 0.05 --want failure
    python scripts/beam_video.py --task G1-Beam-E2E-Curved --path bend --checkpoint <pt> --width 0.20

Runs one env per try (same protocol as ``beam_eval.py``: start platform,
constant forward command) until an episode with the wanted outcome turns up,
then writes that episode as an MP4 (``sendg1.video.EvalVideo``: studio theme,
caption, foot-tactile card). A failure clip ends on the terminating step and
its caption names the termination.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
os.environ.setdefault("MUJOCO_GL", "egl")


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--label", default="", help="caption title, e.g. the run name (default: the task)")
  ap.add_argument("--path", choices=["straight", "bend", "turn"], default="straight",
                  help="beam shape (bend / turn need G1-Beam-E2E-Curved)")
  ap.add_argument("--checkpoint", required=True)
  ap.add_argument("--task", default="G1-Beam-E2E")
  ap.add_argument("--start-lateral", type=float, default=0.08, help="platform start offset (+/- m)")
  ap.add_argument("--start-yaw", type=float, default=0.08, help="start yaw offset (+/- rad)")
  ap.add_argument("--width", type=float, required=True)
  ap.add_argument("--want", choices=["success", "failure", "any"], default="success")
  ap.add_argument("--speed", type=float, default=0.3)
  ap.add_argument("--tries", type=int, default=20)
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--out", default=None)
  ap.add_argument("--width-px", type=int, default=1280)
  ap.add_argument("--height-px", type=int, default=720)
  ap.add_argument("--theme", default="studio", help="studio | gray | mujoco")
  ap.add_argument("--qp", type=int, default=28, help="H.264 quantiser (lower = better, ~18 is near lossless)")
  ap.add_argument("--no-caption", action="store_true", help="drop the top-left caption card")
  ap.add_argument("--logo", action="append", default=[],
                  help="top-left logo, repeatable (left to right): an image path or 'ias' / 'flexoo'")
  ap.add_argument("--sensor-brand", default=None,
                  help="logo for a 'sensor by' row in the tactile card: a path or 'flexoo'")
  ap.add_argument("--distance", type=float, default=3.2)
  ap.add_argument("--elevation", type=float, default=-18.0)
  ap.add_argument("--azimuth", type=float, default=135.0,
                  help="camera azimuth [deg], fixed in the world (90 = side on to the beam)")
  ap.add_argument("--arrow-scale", type=float, default=None,
                  help="size of the command arrows above the robot (default: the task's)")
  ap.add_argument("--arrow-height", type=float, default=None,
                  help="how far above the robot the command arrows float [m]")
  args = ap.parse_args()

  import torch
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

  import sendg1.tasks  # noqa: F401
  from sendg1.common.beam.mdp import BeamEpisode
  from sendg1.video import LOGOS, EvalVideo, prepare_video_cfg

  task = args.task
  cfg = load_env_cfg(task)
  columns = list(cfg.scene.terrain.terrain_generator.sub_terrains)
  # The task's own widths: the round-two tasks keep the old seven.
  widths = cfg.scene.terrain.terrain_generator.sub_terrains[columns[0]].widths
  level = min(range(len(widths)), key=lambda i: abs(widths[i] - args.width))
  kind = lambda n: next((k for k in ("bend", "turn") if n.endswith(k)), "straight")  # noqa: E731
  same_kind = [i for i, n in enumerate(columns) if kind(n) == args.path]
  if not same_kind:
    raise SystemExit(f"{task} has no {args.path} beams (use G1-Beam-E2E-Curved)")
  cfg.scene.num_envs = 1
  cfg.seed = args.seed
  cfg.curriculum = {}
  cfg.events["spawn"].params = {
    "platform_from_level": 0,
    "platform_lateral_range": (-args.start_lateral, args.start_lateral),
    "yaw_range": (-args.start_yaw, args.start_yaw),
  }
  cfg.events.pop("push")
  cfg.events.pop("spawn_from_bank", None)  # training-only mid-beam restarts
  twist = cfg.commands["twist"]
  twist.ranges.lin_vel_x = (args.speed, args.speed)
  twist.rel_standing_envs = 0.0
  twist.resampling_time_range = (1e6, 1e6)
  prepare_video_cfg(cfg, args.width_px, args.height_px, args.theme)
  cfg.viewer.distance = args.distance
  cfg.viewer.elevation = args.elevation
  cfg.viewer.azimuth = args.azimuth
  if args.arrow_scale is not None:
    twist.viz.scale = args.arrow_scale
  if args.arrow_height is not None:
    twist.viz.z_offset = args.arrow_height
  env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0", render_mode="rgb_array")
  t = env.scene.terrain
  t.terrain_levels[:] = level
  t.terrain_types[:] = same_kind[args.seed % len(same_kind)]
  t.env_origins[:] = t.terrain_origins[level, t.terrain_types]

  rl_cfg = load_rl_cfg(task)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=rl_cfg.clip_actions)
  runner = MjlabOnPolicyRunner(wrapped, asdict(rl_cfg), device="cuda:0")
  runner.load(args.checkpoint, load_cfg={"actor": True}, strict=True, map_location="cuda:0")
  policy = runner.get_inference_policy(device="cuda:0")
  video = EvalVideo(env, rl_cfg, title=args.label or task, theme=args.theme,
                    caption=not args.no_caption,
                    logo=[LOGOS.get(name, name) for name in args.logo] or None,
                    sensor_brand=LOGOS.get(args.sensor_brand, args.sensor_brand))
  ep = BeamEpisode.get(env)
  tm = env.termination_manager

  chosen = fallback = None
  for attempt in range(args.tries):
    video.take()
    video.new_episode()
    outcome, reason = None, ""
    stand_after = None
    # Reset inside inference mode too: buffers first written during a step
    # are inference tensors and cannot be reset in-place outside it.
    with torch.inference_mode():
      obs, _ = wrapped.reset()
      policy.reset(torch.ones(1, dtype=torch.bool, device=env.device))
      for step in range(int(cfg.episode_length_s / env.step_dt)):
        act = policy(obs)
        obs, _, dones, _ = wrapped.step(act)
        if bool(dones[0]):
          # Time-out terms (``time_out``, ``finished`` after the end platform) are not falls.
          fired = [
            n for n in tm.active_terms
            if not tm.get_term_cfg(n).time_out and bool(tm.get_term(n)[0])
          ]
          outcome = "failure" if fired else ("success" if stand_after is not None else "timeout")
          reason = fired[0] if fired else ""
          break
        if bool(ep.reached_end[0]) and stand_after is None:
          stand_after = step
        progress = float(ep.max_progress[0])
        lines = [
          f"{args.path} beam {widths[level]:.2f} m",
          f"command {args.speed:.2f} m/s   progress {max(progress, 0):.2f} m",
        ]
        if stand_after is not None:
          lines.append("end platform reached - command 0")
        video.capture(lines)
        # Stop 2 s after reaching the end: the rest is standing still.
        if stand_after is not None and step - stand_after > 100:
          outcome = "success"
          break
    if outcome is None:
      outcome = "success" if stand_after is not None else "timeout"
    print(f"[video] try {attempt}: {outcome} {reason} ({len(video.frames)} frames)")
    if args.want == "any" or outcome == args.want:
      if outcome == "failure":
        video.hold(0.8, [f"{args.path} beam {widths[level]:.2f} m", f"FAILED: {reason}"])
      chosen = (video.take(), outcome, reason)
      break
    if outcome == "failure":
      # A policy that never succeeds still gets a clip: its last failure.
      video.hold(0.8, [f"{args.path} beam {widths[level]:.2f} m", f"FAILED: {reason}"])
    fallback = (video.take(), outcome, reason)

  env.close()
  if chosen is None:
    print(f"[video] no {args.want} episode in {args.tries} tries; keeping the last ({fallback[1]})")
    chosen = fallback
  frames, outcome, reason = chosen
  out = Path(
    args.out
    or REPO / "media" / "beam" / f"{args.path}_w{widths[level]:.2f}_{outcome}.mp4"
  )
  out.parent.mkdir(parents=True, exist_ok=True)
  video.save(out, frames, qp=args.qp)
  meta = out.with_suffix(".json")
  meta.write_text(json.dumps({"outcome": outcome, "reason": reason, "width": widths[level]}))
  print(f"[video] wrote {out} ({outcome} {reason})")


if __name__ == "__main__":
  main()
