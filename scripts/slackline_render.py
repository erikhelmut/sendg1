"""Stills of the G1 on the slackline in the task's spawn stance: the real env
(``G1-Slackline-Touch``, one world), reset, then held for a moment with the
solved stance as joint targets (no policy), rendered through ``EvalVideo``
so the look and the tactile card match the policy clips.

  uv run python scripts/slackline_render.py --out results/slackline
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import torch
from PIL import Image

VIEWS = {
  # name: (tracked body, distance, azimuth, elevation); "{front}" is the front foot
  "wide": ("torso_link", 3.0, 120.0, -10.0),
  "feet": ("{front}_ankle_roll_link", 1.1, 140.0, -32.0),
}


def render(view: str, out: Path, settle_steps: int, front: str, width: int, height: int) -> Path:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

  import sendg1.tasks  # noqa: F401
  from sendg1.common.slackline import mdp
  from sendg1.tasks.slackline import TOUCH_TASK
  from sendg1.video import EvalVideo, prepare_video_cfg

  cfg = load_env_cfg(TOUCH_TASK, play=True)
  cfg.scene.num_envs = 1
  cfg.events["reset_on_line"].params = {"joint_noise": 0.0, "heading_noise": 0.0, "along_range": (-0.3, -0.3)}
  prepare_video_cfg(cfg, width, height)
  body, cfg.viewer.distance, cfg.viewer.azimuth, cfg.viewer.elevation = VIEWS[view]
  cfg.viewer.body_name = body.format(front=front)
  env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0", render_mode="rgb_array")
  try:
    env.reset()
    st = mdp.LineState.get(env)
    stances = mdp._stance_tensors(env)
    robot = env.scene["robot"]
    term = env.action_manager.get_term("joint_pos")
    scale = term._scale if torch.is_tensor(term._scale) else torch.tensor(term._scale)
    # The reset draws direction and front foot at random: redo it until it
    # shows the asked-for front foot, facing the hi anchor.
    for _ in range(50):
      if bool(st.front_left[0]) == (front == "left") and float(st.direction[0]) > 0:
        break
      env.reset()
    q = stances[front]
    action = ((q - robot.data.default_joint_pos[0]) / scale).reshape(1, -1)
    video = EvalVideo(env, load_rl_cfg(TOUCH_TASK), title="G1-Slackline-Touch")
    for _ in range(settle_steps):
      env.step(action)
    sag = float(st.line_under_base().z[0, 0]) - 1.0
    frame = video.capture(
      [
        f"spawn stance, {front} foot in front, back foot turned out 45 deg",
        f"4 m line, {float(mdp.line_tension(env)[0]):.0f} N; sags {-100 * sag:.0f} cm under the robot",
      ]
    )
    assert frame is not None
    path = out / f"g1_on_slackline_{view}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frame).save(path)
    return path
  finally:
    env.close()


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--out", type=Path, default=Path("results/slackline"))
  ap.add_argument("--views", nargs="+", default=list(VIEWS), choices=list(VIEWS))
  ap.add_argument("--front", choices=("left", "right"), default="left")
  ap.add_argument("--settle-steps", type=int, default=2, help="policy steps (0.02 s) before the still: the stance loaded, before the line rolls")
  ap.add_argument("--width", type=int, default=1280)
  ap.add_argument("--height", type=int, default=720)
  args = ap.parse_args()
  for view in args.views:
    print(f"wrote {render(view, args.out, args.settle_steps, args.front, args.width, args.height)}")


if __name__ == "__main__":
  main()
