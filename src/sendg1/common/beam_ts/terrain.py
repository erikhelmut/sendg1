"""The beam tile: start platform, beam, end platform, ground below.

Layout of one tile (local frame, x along the beam)
--------------------------------------------------

    ground (z = 0) everywhere, a box whose top is the tile floor
    |-- margin --|== start platform ==|------ beam ------|== end platform ==|-- margin --|
                                      ^ origin (beam start, centreline, top height)

Beam and platforms are boxes standing on the ground, so their sides are real
vertical faces and stepping off puts a foot on the ground ``beam_height`` below.

The beam follows a centreline PATH, a polyline in the tile's xy plane. Only
straight paths are generated today (``path_kind="straight"``), but every
consumer (rewards, terminations, observations) projects onto the polyline and
never assumes the beam is along x, so curved or bent beams are a new path
generator plus geometry, not a rewrite of the MDP.

Curriculum axes
---------------
Row = width level, ``BEAM_WIDTHS[row]``. Column = beam height (a small domain
randomisation that is fixed per column). Both platforms are always built:
whether an episode starts on the start platform or directly on the beam is
the curriculum's decision (``mdp.PLATFORM_FROM_LEVEL``), not the geometry's.
That keeps one terrain for every level and lets the curriculum move an env
between modes without regenerating anything. ``start_platform`` and
``end_platform`` can still be switched off per tile config.

Handing the geometry to the MDP
-------------------------------
Through the terrain generator's ``flat_patches`` side channel, which the
generator shifts into world coordinates per (row, column):

- ``BEAM_PATH_PATCH``: ``NUM_PATH_POINTS`` points, evenly spaced by arc length,
  from the back edge of the start platform along the beam to the far edge of
  the end platform, at the top surface height. Arc 0 is the start platform's
  back edge, so its centre is at ``platform_length / 2``.
- ``BEAM_ENDS_PATCH``: 2 points, where the beam proper starts and ends.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import mujoco
import numpy as np
from mjlab.terrains.terrain_generator import (
  FlatPatchSamplingCfg,
  SubTerrainCfg,
  TerrainGeneratorCfg,
  TerrainGeometry,
  TerrainOutput,
)

BEAM_PATH_PATCH = "beam_ts_path"
BEAM_ENDS_PATCH = "beam_ts_ends"
NUM_PATH_POINTS = 48

BEAM_WIDTHS: tuple[float, ...] = (0.40, 0.30, 0.20, 0.15, 0.10, 0.07, 0.05)
"""Width per curriculum level (m). The G1's default stance is 0.31 m from outer
sole edge to outer sole edge (feet 0.238 m apart, soles 0.076 m wide): levels 0-1
admit it, level 2 needs a narrowed stance, 0.15 m and below need near-tandem
feet, and 0.076 m is the sole's own width."""

BEAM_HEIGHTS: tuple[float, ...] = (0.15, 0.20, 0.25, 0.30)
"""Beam top above the ground, one per terrain column (m)."""

TERRAIN_SEED = 20261011

_GROUND_RGBA = (0.18, 0.20, 0.22, 1.0)
_PLATFORM_RGBA = (0.85, 0.55, 0.20, 1.0)
_BEAM_RGBA = (0.20, 0.45, 0.95, 1.0)


def width_for_difficulty(difficulty: float, widths: tuple[float, ...] = BEAM_WIDTHS) -> float:
  """Curriculum rows get difficulty ``row / (rows - 1)``; map back to a level."""
  idx = int(round(difficulty * (len(widths) - 1)))
  return widths[min(max(idx, 0), len(widths) - 1)]


def resample_polyline(points: np.ndarray, n: int) -> np.ndarray:
  """``n`` points evenly spaced by arc length along a 2D polyline."""
  seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
  cum = np.concatenate([[0.0], np.cumsum(seg)])
  s = np.linspace(0.0, cum[-1], n)
  return np.stack([np.interp(s, cum, points[:, 0]), np.interp(s, cum, points[:, 1])], 1)


@dataclass(kw_only=True)
class BeamTileCfg(SubTerrainCfg):
  """One beam tile. Distances in metres, local tile frame."""

  size: tuple[float, float] = (8.0, 2.0)
  widths: tuple[float, ...] = BEAM_WIDTHS
  beam_height: float = 0.25
  beam_length: float = 4.0
  path_kind: Literal["straight"] = "straight"
  """Centreline generator. Only straight beams exist yet; see module docstring."""
  start_platform: bool = True
  end_platform: bool = True
  platform_length: float = 1.0
  """Along the path (m)."""
  platform_width: float = 1.0
  end_margin: float = 0.75
  """Bare ground before the start platform and after the end platform."""
  ground_depth: float = 0.2
  """Thickness of the ground box below z = 0."""
  flat_patch_sampling: dict[str, FlatPatchSamplingCfg] | None = field(
    default_factory=lambda: {
      BEAM_PATH_PATCH: FlatPatchSamplingCfg(num_patches=NUM_PATH_POINTS, patch_radius=0.01),
      BEAM_ENDS_PATCH: FlatPatchSamplingCfg(num_patches=2, patch_radius=0.01),
    }
  )

  def __post_init__(self) -> None:
    need = 2 * self.end_margin + 2 * self.platform_length + self.beam_length
    if need > self.size[0] + 1e-9:
      raise ValueError(f"tile length {self.size[0]} m < {need} m needed")

  # -- path ------------------------------------------------------------------

  def beam_polyline(self) -> np.ndarray:
    """Beam centreline joints [K, 2] in tile coordinates, start to end."""
    x0 = self.end_margin + self.platform_length
    yc = self.size[1] / 2
    if self.path_kind == "straight":
      return np.array([[x0, yc], [x0 + self.beam_length, yc]])
    raise NotImplementedError(self.path_kind)

  # -- geometry --------------------------------------------------------------

  def function(
    self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
  ) -> TerrainOutput:
    del rng
    body = spec.body("terrain")
    sx, sy = self.size
    h = self.beam_height
    width = width_for_difficulty(difficulty, self.widths)
    geoms: list[TerrainGeometry] = []

    ground = body.add_geom(
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=(sx / 2, sy / 2, self.ground_depth / 2),
      pos=(sx / 2, sy / 2, -self.ground_depth / 2),
    )
    geoms.append(TerrainGeometry(geom=ground, color=_GROUND_RGBA))

    joints = self.beam_polyline()
    for a, b in zip(joints[:-1], joints[1:], strict=True):
      d = b - a
      yaw = math.atan2(d[1], d[0])
      mid = (a + b) / 2
      seg = body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(float(np.linalg.norm(d)) / 2, width / 2, h / 2),
        pos=(mid[0], mid[1], h / 2),
      )
      seg.quat = np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])
      geoms.append(TerrainGeometry(geom=seg, color=_BEAM_RGBA))

    def platform(center: np.ndarray, yaw: float) -> None:
      g = body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(self.platform_length / 2, self.platform_width / 2, h / 2),
        pos=(center[0], center[1], h / 2),
      )
      g.quat = np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])
      geoms.append(TerrainGeometry(geom=g, color=_PLATFORM_RGBA))

    d0 = joints[1] - joints[0]
    d0 = d0 / np.linalg.norm(d0)
    d1 = joints[-1] - joints[-2]
    d1 = d1 / np.linalg.norm(d1)
    start_c = joints[0] - d0 * self.platform_length / 2
    end_c = joints[-1] + d1 * self.platform_length / 2
    if self.start_platform:
      platform(start_c, math.atan2(d0[1], d0[0]))
    if self.end_platform:
      platform(end_c, math.atan2(d1[1], d1[0]))

    path_start = start_c - d0 * self.platform_length / 2
    path_end = end_c + d1 * self.platform_length / 2
    path_xy = resample_polyline(np.vstack([path_start, joints, path_end]), NUM_PATH_POINTS)
    path = np.column_stack([path_xy, np.full(NUM_PATH_POINTS, h)])
    ends = np.array([[*joints[0], h], [*joints[-1], h]])
    origin = np.array([joints[0][0], joints[0][1], h])
    return TerrainOutput(
      origin=origin,
      geometries=geoms,
      flat_patches={BEAM_PATH_PATCH: path, BEAM_ENDS_PATCH: ends},
    )


def make_beam_ts_terrain_cfg(
  widths: tuple[float, ...] = BEAM_WIDTHS,
  heights: tuple[float, ...] = BEAM_HEIGHTS,
  seed: int | None = TERRAIN_SEED,
  **tile_kwargs,
) -> TerrainGeneratorCfg:
  """Rows = width levels, columns = beam heights."""
  sub_terrains: dict[str, SubTerrainCfg] = {
    f"beam_h{round(h * 100):02d}": BeamTileCfg(widths=widths, beam_height=h, **tile_kwargs)
    for h in heights
  }
  size = next(iter(sub_terrains.values())).size
  return TerrainGeneratorCfg(
    seed=seed,
    size=size,
    border_width=2.0,
    border_height=0.2,
    num_rows=len(widths),
    num_cols=len(sub_terrains),
    curriculum=True,
    add_lights=True,
    sub_terrains=sub_terrains,
  )
