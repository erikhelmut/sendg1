#!/usr/bin/env python
"""Evaluate a pedalo policy at fixed forward commands, and film one episode.

    python scripts/pedalo_eval.py --checkpoint <pt> --out results/pedalo/<name>

Every env gets one constant command (``--commands``, ``--reps`` envs each)
for one full episode. Per command: share of episodes that end without a fall,
the pedalo's mean forward speed over the episode after the first second, and
how far the arms are out (shoulder roll beyond the default, mean of the
episode max). Env 0 is filmed (``--film-command``) with a caption. Writes
``<out>.json`` and ``<out>.mp4``.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--checkpoint", required=True)
  ap.add_argument("--task", default="G1-Pedalo")
  ap.add_argument("--commands", type=float, nargs="+", default=[0.0, 0.15, 0.3, 0.45, 0.6])
  ap.add_argument("--reps", type=int, default=4)
  ap.add_argument("--film-command", type=float, default=0.45)
  ap.add_argument("--film-seconds", type=float, default=12.0)
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--label", default="")
  ap.add_argument("--out", required=True, help="output path without extension")
  args = ap.parse_args()

  import mediapy
  import torch
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

  import sendg1.tasks  # noqa: F401
  from sendg1.common.pedalo.mdp import PedaloState

  cmds = [args.film_command] + [c for c in args.commands for _ in range(args.reps)]
  cfg = load_env_cfg(args.task)
  cfg.scene.num_envs = len(cmds)
  cfg.seed = args.seed
  twist = cfg.commands["twist"]
  twist.rel_standing_envs = 0.0
  twist.resampling_time_range = (1e6, 1e6)
  cfg.viewer.width, cfg.viewer.height = 960, 540
  cfg.viewer.distance = 2.6
  cfg.viewer.elevation = -12.0
  cfg.viewer.azimuth = 90.0
  env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0", render_mode="rgb_array")
  rl_cfg = load_rl_cfg(args.task)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=rl_cfg.clip_actions)
  runner = MjlabOnPolicyRunner(wrapped, asdict(rl_cfg), device="cuda:0")
  runner.load(args.checkpoint, load_cfg={"actor": True}, strict=True, map_location="cuda:0")
  policy = runner.get_inference_policy(device="cuda:0")

  robot = env.scene["robot"]
  roll_ids, _ = robot.find_joints(("left_shoulder_roll_joint", "right_shoulder_roll_joint"), preserve_order=True)
  out_sign = torch.tensor([1.0, -1.0], device=env.device)  # + = arm out sideways
  command = torch.tensor(cmds, device=env.device)
  n = len(cmds)
  steps = int(cfg.episode_length_s / env.step_dt)
  settle = int(1.0 / env.step_dt)
  tm = env.termination_manager

  alive = torch.ones(n, dtype=torch.bool, device=env.device)
  fell = [""] * n
  v_sum = torch.zeros(n, device=env.device)
  v_cnt = torch.zeros(n, device=env.device)
  dist = torch.zeros(n, device=env.device)
  arm_max = torch.zeros(n, device=env.device)
  arm_sum = torch.zeros(n, device=env.device)
  frames = []
  with torch.inference_mode():
    obs, _ = wrapped.reset()
    policy.reset(torch.ones(n, dtype=torch.bool, device=env.device))
    term = env.command_manager.get_term("twist")
    st = PedaloState.get(env)
    term.vel_command_b[:, 0] = command
    term.vel_command_b[:, 1:] = 0.0
    obs = wrapped.get_observations()  # with the fixed commands
    for step in range(steps):
      term.vel_command_b[:, 0] = command
      term.vel_command_b[:, 1:] = 0.0
      act = policy(obs)
      obs, _, dones, _ = wrapped.step(act)
      for i in torch.nonzero(dones.bool() & alive).flatten().tolist():
        fired = [k for k in tm.active_terms if k != "time_out" and bool(tm.get_term(k)[i])]
        fell[i] = fired[0] if fired else ""
      alive &= ~dones.bool()
      v, _ = st.velocity()
      arm = ((robot.data.joint_pos[:, roll_ids] - robot.data.default_joint_pos[:, roll_ids]) * out_sign).mean(-1)
      a = alive.float()
      if step >= settle:
        v_sum += v * a
        v_cnt += a
      dist += v * env.step_dt * a
      arm_max = torch.maximum(arm_max, arm * a)
      arm_sum += arm * a
      if step * env.step_dt < args.film_seconds and bool(alive[0]):
        frame = env.render()
        if frame is not None:
          frames.append(caption(frame, [
            (f"{args.label + '   ' if args.label else ''}{'Pedalo Sport' if 'Sport' in args.task else 'Pedalo Classic'}", True),
            (f"command {cmds[0]:.2f} m/s   pedalo {float(v[0]):+.2f} m/s   {float(dist[0]):.2f} m", False),
            (f"arms out {float(arm[0]):+.2f} rad (shoulder roll beyond default)", False),
          ]))
  env.close()

  rows = []
  for c in args.commands:
    idx = [i for i in range(1, n) if cmds[i] == c]
    t = torch.tensor(idx, device=command.device)
    surv = sum(not fell[i] for i in idx) / len(idx)
    rows.append({
      "command": c,
      "no_fall": surv,
      "falls": [fell[i] for i in idx if fell[i]],
      "mean_speed": float((v_sum[t] / v_cnt[t].clamp(min=1)).mean()),
      "distance_m": float(dist[t].mean()),
      "arm_out_max_rad": float(arm_max[t].mean()),
      "arm_out_mean_rad": float((arm_sum[t] / steps).mean()),
    })
  for r in rows:
    print(f"cmd {r['command']:.2f}: no-fall {r['no_fall']:.0%}  speed {r['mean_speed']:+.3f}  "
          f"dist {r['distance_m']:.2f} m  arms out max {r['arm_out_max_rad']:.2f} mean {r['arm_out_mean_rad']:.2f}  "
          f"falls {r['falls']}")
  out = Path(args.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  out.with_suffix(".json").write_text(json.dumps({"checkpoint": args.checkpoint, "rows": rows}, indent=2))
  if frames:
    mediapy.write_video(str(out.with_suffix(".mp4")), frames, fps=int(round(1 / env.step_dt)), qp=28)
  print(f"wrote {out}.json / .mp4 ({len(frames)} frames, film env fell: {fell[0] or 'no'})")


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
  d.rectangle([0, 0, 470, h], fill=(6, 12, 18, 190))
  for i, (txt, bold) in enumerate(lines):
    d.text((12, 8 + 22 * i), txt, font=big if bold else small, fill=(235, 244, 250, 255))
  return np.asarray(img)


if __name__ == "__main__":
  main()
