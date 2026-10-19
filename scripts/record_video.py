#!/usr/bin/env python
"""Render a headless video of a trained policy.

    python scripts/record_video.py --task Sendg1-Velocity-Flat-G1-Blind --seed 0

mjlab's `play --video` also records, but it always constructs an interactive
viewer, which needs a display and hangs on a headless box. This drives the env
directly with render_mode="rgb_array".

Frames go through ``sendg1.video.EvalVideo``: the dark studio theme, a caption
with the current command, and the foot-tactile card (zero for blind policies).

Commands follow a scripted schedule so the clip shows the behaviours worth
seeing (stand, walk, turn, strafe) instead of whatever the sampler happened to
draw.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
os.environ.setdefault("MUJOCO_GL", "egl")

# (seconds, vx, vy, wz, label)
SCHEDULE = [
  (2.0, 0.0, 0.0, 0.0, "stand"),
  (4.0, 0.8, 0.0, 0.0, "forward"),
  (3.0, 0.8, 0.0, 0.6, "forward + turn"),
  (3.0, 0.0, 0.5, 0.0, "strafe left"),
  (3.0, -0.5, 0.0, 0.0, "reverse"),
  (3.0, 1.0, 0.0, -0.6, "fast + turn right"),
]


SCHEDULE_SHORT = [
  (3.0, 0.8, 0.0, 0.0, "forward"),
  (2.5, 0.8, 0.0, 0.5, "forward + turn"),
  (2.5, 0.6, 0.0, 0.0, "forward"),
]


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--task", required=True)
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--checkpoint", default=None)
  ap.add_argument("--log-root", default=str(REPO / "runs"))
  ap.add_argument("--out", default=None)
  ap.add_argument("--width", type=int, default=1280)
  ap.add_argument("--height", type=int, default=720)
  ap.add_argument("--fps", type=int, default=50)
  ap.add_argument("--distance", type=float, default=4.2)
  ap.add_argument("--terrain", default=None,
                  help="pin every env to this sub-terrain column (e.g. tilted_grid)")
  ap.add_argument("--level", type=int, default=9,
                  help="difficulty row to pin to when --terrain is given")
  ap.add_argument("--qp", type=int, default=None,
                  help="H.264 quantiser; higher = smaller file (try 30)")
  ap.add_argument("--subsample", type=int, default=1,
                  help="keep every Nth frame and divide fps to match")
  ap.add_argument("--short", action="store_true",
                  help="8 s clip instead of the full 18 s schedule")
  ap.add_argument("--no-overlay", action="store_true", help="no caption lines")
  ap.add_argument("--theme", default="studio", help="studio | gray | mujoco")
  ap.add_argument("--no-tactile-card", action="store_true")
  args = ap.parse_args()

  import torch
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
  from mjlab.utils.os import get_checkpoint_path
  from mjlab.utils.torch import configure_torch_backends

  import sendg1.tasks  # noqa: F401
  from sendg1.video import EvalVideo, prepare_video_cfg

  configure_torch_backends()
  device = "cuda:0" if torch.cuda.is_available() else "cpu"

  rl_cfg = load_rl_cfg(args.task)
  ckpt = (
    Path(args.checkpoint)
    if args.checkpoint
    else get_checkpoint_path(
      Path(args.log_root) / rl_cfg.experiment_name, f".*seed{args.seed}$", "model_.*.pt"
    )
  )
  print(f"[video] checkpoint: {ckpt}")

  pin = args.terrain is not None
  env_cfg = load_env_cfg(args.task, play=not pin)
  env_cfg.scene.num_envs = 1

  if pin:
    # Play mode rebuilds the generator as a 5x5 random grid, which destroys the
    # one-column-per-terrain layout we need to pin to. So start from the
    # TRAINING scene and apply the play-like bits by hand.
    env_cfg.episode_length_s = 1e9
    env_cfg.curriculum = {}
    env_cfg.events.pop("push_robot", None)
    for _g in env_cfg.observations.values():
      _g.enable_corruption = False

  prepare_video_cfg(env_cfg, args.width, args.height, args.theme)
  # Default distance 3.0 with the torso as origin puts the head and the command
  # arrow drawn above it right at the top edge. Pull back and raise the eye line.
  env_cfg.viewer.distance = args.distance
  env_cfg.viewer.elevation = -12.0

  # Enlarge the commanded-velocity arrow drawn above the robot. At the default
  # scale it is a few pixels and reads as a speck against the sky.
  from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

  twist_cfg = env_cfg.commands.get("twist")
  if twist_cfg is not None:
    assert isinstance(twist_cfg, UniformVelocityCommandCfg)
    twist_cfg.viz.scale = 1.4
    twist_cfg.viz.z_offset = 0.55
  env_cfg.observations["proprio"].enable_corruption = False

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode="rgb_array")

  if pin:
    terrain = env.scene.terrain
    assert terrain is not None and terrain.terrain_origins is not None
    names = list(env_cfg.scene.terrain.terrain_generator.sub_terrains)
    if args.terrain not in names:
      raise SystemExit(f"unknown terrain {args.terrain!r}; known: {names}")
    col = names.index(args.terrain)
    row = min(args.level, terrain.terrain_origins.shape[0] - 1)
    terrain.terrain_types[:] = col
    terrain.terrain_levels[:] = row
    # env_origins must be set BEFORE reset: reset_base places the robot
    # relative to them, and a fall mid-clip respawns on the same patch.
    terrain.env_origins[:] = terrain.terrain_origins[row, col]
    print(f"[video] pinned to {args.terrain} (col {col}) at level {row}")
  wrapped = RslRlVecEnvWrapper(env, clip_actions=rl_cfg.clip_actions)

  video = EvalVideo(env, rl_cfg, title=args.task, theme=args.theme,
                    show_tactile=not args.no_tactile_card)

  runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
  runner = runner_cls(wrapped, asdict(rl_cfg), device=device)
  runner.load(str(ckpt), load_cfg={"actor": True}, strict=True, map_location=device)
  policy = runner.get_inference_policy(device=device)

  # Tasks without a velocity command just run for the schedule's length.
  cmd_term = env.command_manager.get_term("twist") if twist_cfg is not None else None
  schedule = SCHEDULE_SHORT if args.short else SCHEDULE
  if cmd_term is None:
    schedule = [(sum(s[0] for s in schedule), 0.0, 0.0, 0.0, "")]
  obs, _ = wrapped.reset()

  with torch.inference_mode():
    for secs, vx, vy, wz, label in schedule:
      cmd = torch.tensor([[vx, vy, wz]], device=device, dtype=torch.float32)
      for _ in range(int(secs / env.step_dt)):
        if cmd_term is not None:
          cmd_term.vel_command_b[:] = cmd
        obs, _, dones, _ = wrapped.step(policy(obs))
        # A recurrent actor must forget an episode that just ended.
        policy.reset(dones.bool())
        if cmd_term is not None:
          cmd_term.vel_command_b[:] = cmd
        if bool(dones[0]):
          video.new_episode()
        lines = []
        if cmd_term is not None and not args.no_overlay:
          lines = [f"command  vx {vx:+.1f}  vy {vy:+.1f}  wz {wz:+.1f}   {label}"]
        video.capture(lines)

  env.close()

  out = Path(args.out or REPO / "media" / f"{rl_cfg.experiment_name}_seed{args.seed}.mp4")
  video.save(out, subsample=args.subsample, fps=args.fps, qp=args.qp)
  size_mb = out.stat().st_size / 1e6
  print(f"[video] wrote {out}  ({len(video.frames)} frames, {size_mb:.1f} MB)")


if __name__ == "__main__":
  main()
