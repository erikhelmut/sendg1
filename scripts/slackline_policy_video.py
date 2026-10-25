"""Video of a trained slackline policy, through ``EvalVideo``.

One clip, a few episodes back to back on one curriculum level: first a
'stand' command, then 'walk' at a fixed speed, alternating. Each episode
runs until it times out, falls (held for a moment with the reason), or the
clip's per-episode limit.

  uv run python scripts/slackline_policy_video.py \\
    --checkpoint logs/rsl_rl/slackline_touch/<run>/model_19999.pt --level 0
"""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
os.environ.setdefault("MUJOCO_GL", "egl")


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--checkpoint", required=True)
  ap.add_argument("--task", default=None, help="default: from the checkpoint's experiment")
  ap.add_argument("--level", type=int, default=0, help="curriculum level (0 plank .. 6 free)")
  ap.add_argument("--speed", type=float, default=0.25, help="walking command (m/s)")
  ap.add_argument("--episodes", type=int, default=4)
  ap.add_argument("--seconds", type=float, default=8.0, help="cap per episode")
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--label", default="")
  ap.add_argument("--out", default=None)
  ap.add_argument("--width-px", type=int, default=1280)
  ap.add_argument("--height-px", type=int, default=720)
  ap.add_argument("--theme", default="studio", help="studio | gray | mujoco")
  args = ap.parse_args()

  import torch
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

  import sendg1.tasks  # noqa: F401
  from sendg1.common.slackline import mdp
  from sendg1.tasks.slackline import BLIND_TASK, TOUCH_TASK
  from sendg1.video import EvalVideo, prepare_video_cfg

  task = args.task or (TOUCH_TASK if "slackline_touch" in args.checkpoint else BLIND_TASK)
  level = mdp.LEVELS[args.level]
  cfg = load_env_cfg(task, play=True)
  cfg.scene.num_envs = 1
  cfg.seed = args.seed
  cfg.events["line_level"].params = {"fixed_level": args.level}
  twist = cfg.commands["twist"]
  twist.rel_standing_envs = 0.0
  twist.resampling_time_range = (1e6, 1e6)
  prepare_video_cfg(cfg, args.width_px, args.height_px, args.theme)
  cfg.viewer.distance, cfg.viewer.azimuth, cfg.viewer.elevation = 3.4, 140.0, -12.0
  env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0", render_mode="rgb_array")

  rl_cfg = load_rl_cfg(task)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=rl_cfg.clip_actions)
  runner = MjlabOnPolicyRunner(wrapped, asdict(rl_cfg), device="cuda:0")
  runner.load(args.checkpoint, load_cfg={"actor": True}, strict=True, map_location="cuda:0")
  policy = runner.get_inference_policy(device="cuda:0")
  it = Path(args.checkpoint).stem.split("_")[-1]
  video = EvalVideo(env, rl_cfg, title=args.label or f"{task}, iteration {it}", theme=args.theme)
  st = mdp.LineState.get(env)
  tm = env.termination_manager
  cmd = env.command_manager.get_term("twist")
  where = f"level {args.level}: {100 * level.width:.0f} cm webbing, helper {level.stiffness:.0f} N/m"

  for ep in range(args.episodes):
    speed = 0.0 if ep % 2 == 0 else args.speed
    video.new_episode()
    with torch.inference_mode():
      obs, _ = wrapped.reset()
      policy.reset(torch.ones(1, dtype=torch.bool, device=env.device))
      outcome = "time limit"
      for k in range(round(args.seconds / env.step_dt)):
        cmd.vel_command_b[:] = 0.0
        if float(st.base()[0, 0]) < 2.0 - mdp.END_MARGIN:
          cmd.vel_command_b[:, 0] = speed
        obs, _, dones, _ = wrapped.step(policy(obs))
        if bool(dones[0]):
          fired = [n for n in tm.active_terms if n != "time_out" and bool(tm.get_term(n)[0])]
          outcome = f"FELL: {fired[0]}" if fired else "survived the episode"
          break
        line = st.line_under_base()
        sag = float(line.z[0, 0]) - float(st.origin()[0, 2]) - 1.0
        task_line = "stand still" if speed == 0 else f"walk {speed:.2f} m/s along the line"
        video.capture(
          [
            f"episode {ep + 1}/{args.episodes}: {task_line}   t = {k * env.step_dt:4.1f} s",
            f"{where}   progress {max(float(st.max_progress[0]), 0):.2f} m",
            f"line under the robot: sag {-100 * sag:4.1f} cm, roll "
            f"{float(line.roll[0, 0]) * 57.3:+5.1f} deg",
          ]
        )
    print(f"[video] episode {ep}: {outcome} after {k * env.step_dt:.1f} s")
    video.hold(0.8, [f"episode {ep + 1}/{args.episodes}: {outcome} after {k * env.step_dt:.1f} s", where])

  env.close()
  kind = "touch" if task == TOUCH_TASK else "blind"
  out = Path(args.out or REPO / "results" / "slackline" / f"policy_{kind}_L{args.level}.mp4")
  out.parent.mkdir(parents=True, exist_ok=True)
  video.save(out, video.take(), qp=28)
  print(f"[video] wrote {out}")


if __name__ == "__main__":
  main()
