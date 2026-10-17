#!/usr/bin/env python
"""Render beam episodes of a policy, headless.

    python scripts/beam_video.py --checkpoint <pt> --width 0.10 --want success
    python scripts/beam_video.py --checkpoint <pt> --width 0.05 --want failure
    python scripts/beam_video.py --task G1-Beam-E2E-Curved --path bend --checkpoint <pt> --width 0.20

Runs one env per try (same protocol as ``beam_eval.py``: start platform,
constant forward command) until an episode with the wanted outcome turns up,
then writes that episode as an MP4 with a small caption. A failure clip ends
on the terminating step and its caption names the termination.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def caption(frame, lines):
  import numpy as np
  from PIL import Image, ImageDraw, ImageFont

  img = Image.fromarray(frame)
  d = ImageDraw.Draw(img, "RGBA")
  try:
    import matplotlib

    ttf = matplotlib.get_data_path() + "/fonts/ttf/"
    big = ImageFont.truetype(ttf + "DejaVuSans-Bold.ttf", 18)
    small = ImageFont.truetype(ttf + "DejaVuSans.ttf", 14)
  except Exception:
    big = small = ImageFont.load_default()
  h = 14 + 22 * len(lines)
  d.rectangle([0, 0, 360, h], fill=(6, 12, 18, 190))
  for i, (txt, bold) in enumerate(lines):
    d.text((12, 8 + 22 * i), txt, font=big if bold else small, fill=(235, 244, 250, 255))
  return np.asarray(img)


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--label", default="", help="caption prefix, e.g. the run name")
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
  ap.add_argument("--width-px", type=int, default=960)
  ap.add_argument("--height-px", type=int, default=540)
  args = ap.parse_args()

  import mediapy
  import torch
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

  import sendg1.tasks  # noqa: F401
  from sendg1.common.beam.mdp import BeamEpisode
  from sendg1.common.beam.terrain import BEAM_WIDTHS

  level = min(range(len(BEAM_WIDTHS)), key=lambda i: abs(BEAM_WIDTHS[i] - args.width))
  task = args.task
  cfg = load_env_cfg(task)
  columns = list(cfg.scene.terrain.terrain_generator.sub_terrains)
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
  cfg.viewer.width, cfg.viewer.height = args.width_px, args.height_px
  cfg.viewer.distance = 3.2
  cfg.viewer.elevation = -18.0
  cfg.viewer.azimuth = 135.0
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
  ep = BeamEpisode.get(env)
  tm = env.termination_manager

  chosen = None
  for attempt in range(args.tries):
    frames, outcome, reason = [], None, ""
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
          fired = [n for n in tm.active_terms if n != "time_out" and bool(tm.get_term(n)[0])]
          outcome = "failure" if fired else ("success" if stand_after is not None else "timeout")
          reason = fired[0] if fired else ""
          break
        if bool(ep.reached_end[0]) and stand_after is None:
          stand_after = step
        progress = float(ep.max_progress[0])
        lines = [
          (f"{args.label + '   ' if args.label else ''}{args.path} beam {BEAM_WIDTHS[level]:.2f} m", True),
          (f"command {args.speed:.2f} m/s   progress {max(progress, 0):.2f} m", False),
        ]
        if stand_after is not None:
          lines.append(("end platform reached - command 0", False))
        frame = env.render()
        if frame is not None:
          frames.append(caption(frame, lines))
        # Stop 2 s after reaching the end: the rest is standing still.
        if stand_after is not None and step - stand_after > 100:
          outcome = "success"
          break
    if outcome is None:
      outcome = "success" if stand_after is not None else "timeout"
    print(f"[video] try {attempt}: {outcome} {reason} ({len(frames)} frames)")
    if args.want == "any" or outcome == args.want:
      if outcome == "failure" and frames:
        frames += [caption(frames[-1], [(f"FAILED: {reason}", True)])] * 40
      chosen = (frames, outcome, reason)
      break

  env.close()
  if chosen is None:
    raise SystemExit(f"no {args.want} episode in {args.tries} tries")
  frames, outcome, reason = chosen
  out = Path(
    args.out
    or REPO / "media" / "beam" / f"{args.path}_w{BEAM_WIDTHS[level]:.2f}_{outcome}.mp4"
  )
  out.parent.mkdir(parents=True, exist_ok=True)
  mediapy.write_video(str(out), frames, fps=int(round(1 / env.step_dt)), qp=28)
  meta = out.with_suffix(".json")
  meta.write_text(json.dumps({"outcome": outcome, "reason": reason, "width": BEAM_WIDTHS[level]}))
  print(f"[video] wrote {out} ({outcome} {reason})")


if __name__ == "__main__":
  main()
