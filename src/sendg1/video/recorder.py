"""``EvalVideo``: render, overlay, collect and write the frames of a rollout."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Iterable

import numpy as np

from sendg1.video import hud
from sendg1.video.theme import apply_theme

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg


def prepare_video_cfg(
  cfg: ManagerBasedRlEnvCfg, width: int = 1280, height: int = 720, theme: str = "studio"
) -> None:
  """Video settings that must be in place before the env is built: frame size,
  and no sensor debug drawing (height-scan rays are critic-only; drawing them
  would suggest the actor sees them), and a sky the theme can colour."""
  from sendg1.video.theme import SKY_TEXTURE, sky_texture_cfg

  for sensor in cfg.scene.sensors or ():
    if hasattr(sensor, "debug_vis"):
      sensor.debug_vis = False
  cfg.viewer.width = width
  cfg.viewer.height = height
  terrain = cfg.scene.terrain
  if theme != "mujoco" and terrain is not None and not any(t.name == SKY_TEXTURE for t in terrain.textures):
    terrain.textures = (*terrain.textures, sky_texture_cfg())


def _actor_term_funcs(env_cfg, rl_cfg) -> list[str]:
  groups = (getattr(rl_cfg, "obs_groups", None) or {}).get("actor", ("actor",))
  names = []
  for g in groups:
    group = env_cfg.observations.get(g)
    if group is not None:
      names += [getattr(t.func, "__name__", type(t.func).__name__) for t in group.terms.values()]
  return names


def policy_sees_tactile(env_cfg, rl_cfg) -> bool:
  """True when any actor observation term reads the tactile sole."""
  return any("tactile" in n for n in _actor_term_funcs(env_cfg, rl_cfg))


def _badge(env_cfg, rl_cfg) -> str:
  funcs = _actor_term_funcs(env_cfg, rl_cfg)
  if any("tactile" in n for n in funcs):
    return "TACTILE"
  if "foot_contact" in funcs:
    return "CONTACT ORACLE"
  return "BLIND"


def _robot_weight(env) -> float | None:
  """Weight of env 0's robot in N (its masses as randomised), the reference
  line on the force trace: two feet together should carry about this."""
  try:
    ids = env.scene["robot"].indexing.body_ids
    return float(env.sim.model.body_mass[0, ids].sum()) * 9.81
  except Exception:
    return None


class EvalVideo:
  """Frames of env 0 of a rendering env, styled and annotated the same way for
  every task.

  ``tactile`` defaults to whether the actor reads the tactile sole. When it is
  False the sole card is drawn but every taxel reads zero: what the policy
  gets is nothing, and the clip says so.
  """

  def __init__(
    self,
    env: ManagerBasedRlEnv,
    rl_cfg=None,
    *,
    title: str = "",
    tactile: bool | None = None,
    theme: str = "studio",
    show_tactile: bool = True,
    vmax: float = 30.0,
    history_s: float = 3.0,
  ):
    from sendg1.common.tactile.layout import LAYOUTS
    from sendg1.common.tactile.sensing import TactileParams

    self.env = env
    self.badge = _badge(env.cfg, rl_cfg) if rl_cfg is not None else "POLICY"
    self.tactile = (self.badge == "TACTILE") if tactile is None else tactile
    self.title = title
    self.params = TactileParams()
    self.vignette = theme != "mujoco"
    apply_theme(env, theme)
    h, w = env.cfg.viewer.height, env.cfg.viewer.width
    self.card = (
      hud.TactileCard(LAYOUTS[self.params.layout], (h, w), live=self.tactile, vmax=vmax,
                      history=max(1, int(history_s / env.step_dt)), history_s=history_s,
                      weight=_robot_weight(env))
      if show_tactile
      else None
    )
    self.frames: list[np.ndarray] = []
    self._raw: np.ndarray | None = None
    self._normal = np.zeros((2, LAYOUTS[self.params.layout].num_taxels))

  def new_episode(self) -> None:
    """Clear the force trace (call after a reset)."""
    if self.card is not None:
      self.card.reset()

  def _normals(self) -> np.ndarray:
    if not self.tactile:
      return np.zeros_like(self._normal)
    from sendg1.common.tactile.sensing import cached_reading

    return cached_reading(self.env, self.params).normal[0].clamp_min(0).cpu().numpy()

  def _compose(self, raw: np.ndarray, normal: np.ndarray, lines, advance: bool) -> np.ndarray:
    frame = raw
    if self.vignette:
      frame = (frame * hud.vignette(*frame.shape[:2])).astype(np.uint8)
    if self.card is not None:
      if advance:
        frame = self.card.draw(frame, normal)
      else:  # frozen frame: redraw without advancing the trace
        trace = self.card.trace.copy()
        frame = self.card.draw(frame, normal)
        self.card.trace = trace
    return hud.caption(frame, self.title, self.badge, self.tactile, lines)

  def capture(self, lines: Iterable[str] = ()) -> np.ndarray | None:
    """Render env 0 now and append the annotated frame."""
    raw = self.env.render()
    if raw is None:
      return None
    self._raw = raw
    self._normal = self._normals()
    frame = self._compose(raw, self._normal, list(lines), advance=True)
    self.frames.append(frame)
    return frame

  def hold(self, seconds: float, lines: Iterable[str] = ()) -> None:
    """Freeze the last frame for ``seconds`` with new caption lines, e.g. to
    name the termination at the end of a failure clip."""
    if self._raw is None:
      return
    frame = self._compose(self._raw, self._normal, list(lines), advance=False)
    self.frames += [frame] * int(round(seconds / self.env.step_dt))

  def take(self) -> list[np.ndarray]:
    frames, self.frames = self.frames, []
    return frames

  def save(
    self,
    path: str | Path,
    frames: list[np.ndarray] | None = None,
    *,
    subsample: int = 1,
    fps: float | None = None,
    qp: int | None = 24,
  ) -> Path:
    """Write H.264 at ``fps`` (default: the env's control rate), divided by
    ``subsample`` so the clip stays real time."""
    import mediapy

    frames = self.frames if frames is None else frames
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fps = round((fps or 1 / self.env.step_dt) / subsample)
    kw = {"fps": fps} if qp is None else {"fps": fps, "qp": qp}
    mediapy.write_video(str(path), frames[::subsample], **kw)
    return path
