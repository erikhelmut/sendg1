"""Render-only restyling of the scene: dark grid floor, dark sky, fog.

Everything here edits the offscreen renderer's PRIVATE copy of the MjModel
(mjlab's ``OffscreenRenderer`` copies the model so render tweaks never reach
the simulation). Physics, sensors and the tactile rays see the unchanged
model; only the pixels differ.

Textures and fog are cached in the OpenGL context when it is made, so they are
re-uploaded into it after the edit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import mujoco
import numpy as np

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


@dataclass(frozen=True)
class Theme:
  floor: tuple[float, float, float]
  """Base floor colour (linear 0-1)."""
  minor_line: tuple[float, float, float]
  major_line: tuple[float, float, float]
  """Grid lines every 0.25 m (minor) and 1 m (major). A grid, not a plain
  floor: with the camera tracking the robot, the lines are the only cue that
  it is moving."""
  reflectance: float
  fog: tuple[float, float, float]
  """Fog colour; also the sky at the horizon, so the floor fades into it."""
  sky_top: tuple[float, float, float]
  fog_start: float
  fog_end: float
  """In units of the render extent (max(4, 1.5 * camera distance) m)."""
  headlight_ambient: float
  headlight_diffuse: float
  terrain_lo: tuple[float, float, float]
  terrain_hi: tuple[float, float, float]
  """Procedural terrain (beams, platforms, rough patches) loses mjlab's colours:
  its luminance is remapped onto this ramp, which keeps the shading that tells
  a beam from the ground and a step from the next."""


THEMES: dict[str, Theme | None] = {
  "studio": Theme(
    floor=(0.085, 0.09, 0.10),
    minor_line=(0.13, 0.14, 0.15),
    major_line=(0.20, 0.21, 0.23),
    reflectance=0.06,
    fog=(0.0, 0.0, 0.0),
    sky_top=(0.10, 0.11, 0.13),
    fog_start=1.2,
    fog_end=3.5,
    headlight_ambient=0.25,
    headlight_diffuse=0.6,
    terrain_lo=(0.07, 0.075, 0.085),
    terrain_hi=(0.46, 0.48, 0.52),
  ),
  "gray": Theme(
    floor=(0.30, 0.31, 0.32),
    minor_line=(0.34, 0.35, 0.36),
    major_line=(0.42, 0.43, 0.44),
    reflectance=0.05,
    fog=(0.16, 0.17, 0.18),
    sky_top=(0.36, 0.37, 0.39),
    fog_start=1.5,
    fog_end=4.0,
    headlight_ambient=0.3,
    headlight_diffuse=0.6,
    terrain_lo=(0.22, 0.23, 0.24),
    terrain_hi=(0.75, 0.76, 0.78),
  ),
  "mujoco": None,  # mjlab's stock blue checker, untouched
}


SKY_TEXTURE = "video_sky"
"""A white-to-black gradient skybox ``prepare_video_cfg`` adds to the scene.
Its pixel values are the gradient weight, so a theme recolours it in place."""


def sky_texture_cfg():
  from mjlab.utils.spec_config import TextureCfg

  return TextureCfg(name=SKY_TEXTURE, type="skybox", builtin="gradient",
                    rgb1=(1.0, 1.0, 1.0), rgb2=(0.0, 0.0, 0.0), width=256, height=1536)


def _grid_texture(theme: Theme, size: int, nchannel: int) -> np.ndarray:
  """One 1 m tile: a major line on the tile edge, minor lines at quarters."""
  img = np.empty((size, size, 3), np.float32)
  img[:] = theme.floor
  coord = np.arange(size)
  for k in (1, 2, 3):
    on = np.abs(coord - k * size / 4) < 0.6
    img[on, :] = theme.minor_line
    img[:, on] = theme.minor_line
  edge = (coord < 2) | (coord >= size - 1)
  img[edge, :] = theme.major_line
  img[:, edge] = theme.major_line
  out = (img * 255).round().astype(np.uint8)
  if nchannel == 4:
    out = np.concatenate([out, np.full((size, size, 1), 255, np.uint8)], -1)
  return out


def _ramp(lum: np.ndarray, theme: Theme) -> np.ndarray:
  lo, hi = np.array(theme.terrain_lo), np.array(theme.terrain_hi)
  return lo + np.clip(lum, 0.0, 1.0)[..., None] ** 1.4 * (hi - lo)


def _luminance(rgb: np.ndarray) -> np.ndarray:
  return rgb @ np.array([0.2126, 0.7152, 0.0722])


def _restyle_terrain(m: mujoco.MjModel, theme: Theme) -> list[int]:
  """Grey out the terrain body's geoms other than the ground plane. Returns
  the textures it rewrote (they need re-uploading)."""
  body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "terrain")
  plane = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MATERIAL, "groundplane")
  if body < 0:
    return []
  geoms = [g for g in np.flatnonzero(m.geom_bodyid == body)
           if plane < 0 or m.geom_matid[g] != plane]
  textures: set[int] = set()
  for g in geoms:
    mat = int(m.geom_matid[g])
    if mat < 0:
      m.geom_rgba[g, :3] = _ramp(_luminance(m.geom_rgba[g, :3]), theme)
      continue
    tex = int(m.mat_texid[mat, mujoco.mjtTextureRole.mjTEXROLE_RGB])
    if tex >= 0:
      textures.add(tex)
      m.mat_rgba[mat, :3] = 1.0
    else:
      m.mat_rgba[mat, :3] = _ramp(_luminance(m.mat_rgba[mat, :3]), theme)
  for tex in textures:
    c, adr = int(m.tex_nchannel[tex]), int(m.tex_adr[tex])
    n = int(m.tex_width[tex] * m.tex_height[tex])
    data = m.tex_data[adr : adr + n * c].reshape(n, c)
    rgb = _ramp(_luminance(data[:, :3] / 255.0), theme)
    data[:, :3] = (rgb * 255).round().astype(np.uint8)
  return sorted(textures)


def _restyle_model(m: mujoco.MjModel, theme: Theme) -> list[int]:
  """Edit ``m`` in place; returns the ids of the textures that changed."""
  tex = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_TEXTURE, "groundplane")
  if tex >= 0 and m.tex_width[tex] == m.tex_height[tex]:
    n, c, adr = int(m.tex_width[tex]), int(m.tex_nchannel[tex]), int(m.tex_adr[tex])
    m.tex_data[adr : adr + n * n * c] = _grid_texture(theme, n, c).ravel()
  mat = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MATERIAL, "groundplane")
  if mat >= 0:
    m.mat_texrepeat[mat] = (1.0, 1.0)  # texuniform: one tile per metre
    m.mat_rgba[mat] = (1.0, 1.0, 1.0, 1.0)
    m.mat_reflectance[mat] = theme.reflectance
  sky = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_TEXTURE, SKY_TEXTURE)
  if sky >= 0:
    n = int(m.tex_width[sky] * m.tex_height[sky] * m.tex_nchannel[sky])
    adr = int(m.tex_adr[sky])
    data = m.tex_data[adr : adr + n].reshape(-1, int(m.tex_nchannel[sky]))
    w = data[:, :1].astype(np.float32) / 255
    rgb = w * np.array(theme.sky_top) + (1 - w) * np.array(theme.fog)
    data[:, :3] = (rgb * 255).round().astype(np.uint8)
  m.vis.rgba.fog[:] = (*theme.fog, 1.0)
  m.vis.rgba.haze[:] = (*theme.fog, 1.0)
  m.vis.map.fogstart = theme.fog_start
  m.vis.map.fogend = theme.fog_end
  m.vis.headlight.ambient[:] = theme.headlight_ambient
  m.vis.headlight.diffuse[:] = theme.headlight_diffuse
  changed = _restyle_terrain(m, theme)
  for name in ("groundplane", SKY_TEXTURE):
    tex = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_TEXTURE, name)
    if tex >= 0:
      changed.append(tex)
  return changed


def apply_theme(env: ManagerBasedRlEnv, name: str = "studio") -> None:
  """Restyle ``env``'s offscreen renderer in place. No-op for ``"mujoco"``."""
  if name not in THEMES:
    raise ValueError(f"unknown theme {name!r}; known: {list(THEMES)}")
  theme = THEMES[name]
  renderer = getattr(env, "_offline_renderer", None)
  if theme is None or renderer is None:
    return
  # Edit the live context rather than re-creating it: a second GL context in
  # the process makes EGL complain at interpreter exit.
  model = renderer._model
  changed = _restyle_model(model, theme)
  r = renderer.renderer
  if r._gl_context is not None:
    r._gl_context.make_current()
  ctx = r._mjr_context
  for tex in changed:
    mujoco.mjr_uploadTexture(model, ctx, tex)
  # mjr_makeContext caches these from the model.
  ctx.fogStart = model.vis.map.fogstart * model.stat.extent
  ctx.fogEnd = model.vis.map.fogend * model.stat.extent
  ctx.fogRGBA[:] = model.vis.rgba.fog
  r.scene.flags[mujoco.mjtRndFlag.mjRND_FOG] = 1
