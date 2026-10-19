"""The one way evaluation videos are rendered.

Every script that writes a policy video goes through :class:`EvalVideo`:

    prepare_video_cfg(env_cfg, width, height)        # before building the env
    env = ManagerBasedRlEnv(cfg=env_cfg, render_mode="rgb_array", ...)
    video = EvalVideo(env, task)                     # restyles the renderer
    ... step ...; video.capture(["caption", ...])
    video.save("media/clip.mp4")

Each frame is the robot, rendered in a dark studio theme, plus a HUD with both
soles' taxels. The HUD shows what the tactile sole measures when the policy's
actor reads tactile input, and stays at zero for blind policies (and contact
oracles), so paired clips line up panel for panel.
"""

from sendg1.video.recorder import (
  EvalVideo,
  policy_sees_tactile,
  prepare_video_cfg,
)
from sendg1.video.theme import THEMES, apply_theme

__all__ = [
  "THEMES",
  "EvalVideo",
  "apply_theme",
  "policy_sees_tactile",
  "prepare_video_cfg",
]
