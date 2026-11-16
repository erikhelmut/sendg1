"""Videos of the G1 with the tactile soles: a themed render plus a live HUD.

    prepare_video_cfg(env_cfg, width, height)        # before building the env
    env = ManagerBasedRlEnv(cfg=env_cfg, render_mode="rgb_array", ...)
    video = TactileVideo(env, title="...")           # restyles the renderer
    ... step ...; video.capture(["caption", ...])
    video.save("media/clip.mp4")

Each frame is env 0's robot in a dark studio theme, a caption card with the
IAS logo top left, and a card with both soles' taxels, per-foot load and a
force trace bottom right, with a "sensor by Flexoo" row.
"""

from g1_tactile.video.recorder import (
  FLEXOO_LOGO,
  IAS_LOGO,
  LOGOS,
  NO_LOGO,
  TactileVideo,
  prepare_video_cfg,
)
from g1_tactile.video.theme import THEMES, apply_theme

__all__ = [
  "FLEXOO_LOGO",
  "IAS_LOGO",
  "LOGOS",
  "NO_LOGO",
  "THEMES",
  "TactileVideo",
  "apply_theme",
  "prepare_video_cfg",
]
