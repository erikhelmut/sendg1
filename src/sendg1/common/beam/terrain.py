"""The beam tile: start platform, beam, end platform, ground below.

Layout of one tile (local frame, x along the beam)
--------------------------------------------------

    ground (z = 0) everywhere, a box whose top is the tile floor
    |-- margin --|== start platform ==|------ beam ------|== end platform ==|-- margin --|
                                      ^ origin (beam start, centreline, top height)

Beam and platforms are boxes standing on the ground, so their sides are real
vertical faces and stepping off puts a foot on the ground ``beam_height`` below.

The beam follows a centreline PATH, a polyline in the tile's xy plane, and
every consumer (rewards, terminations, observations) projects onto it, never
assuming the beam runs along x. ``path_kind="straight"`` is one segment.
``path_kind="bent"`` (from the retired first beam task) is straight segments of
random length, 0.6-1.4 m, joined at random heading changes of up to
``max_turn_deg``; a disc at every joint fills the wedge a bend would otherwise
open on its outside edge. The total heading stays within ``max_heading_deg``
(< 90 deg), so the beam never doubles back. The end platform lines up with the
last segment.

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

- ``BEAM_PATH_PATCH``: ``NUM_PATH_POINTS`` points, every polyline joint plus
  points evenly spaced by arc length in between, from the back edge of the start platform along the beam to the far edge of
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

BEAM_PATH_PATCH = "beam_path"
BEAM_ENDS_PATCH = "beam_ends"
NUM_PATH_POINTS = 64

BEAM_WIDTHS: tuple[float, ...] = (0.40, 0.30, 0.20, 0.15, 0.10, 0.07, 0.05)
"""Width per curriculum level (m). The G1's default stance is 0.31 m from outer
sole edge to outer sole edge (feet 0.238 m apart, soles 0.076 m wide): levels 0-1
admit it, level 2 needs a narrowed stance, 0.15 m and below need near-tandem
feet, and 0.076 m is the sole's own width."""

BEAM_HEIGHTS: tuple[float, ...] = (0.15, 0.20, 0.25, 0.30)
"""Beam top above the ground, one per terrain column (m)."""

BEND_MAX_TURN_DEG: dict[str, float] = {"bend": 25.0, "turn": 60.0}
"""Largest heading change at one joint per bent column kind (the first beam
task's values). At wide levels ignoring a bend costs little; at narrow ones a
missed bend puts a foot over the edge."""

CURVED_TILE_SIZE = (8.0, 6.0)
"""Tile size when bent beams are generated: room for the beam to wander
sideways (straight tiles are 8 x 2 m)."""

TERRAIN_SEED = 20261011

_GROUND_RGBA = (0.18, 0.20, 0.22, 1.0)
_PLATFORM_RGBA = (0.85, 0.55, 0.20, 1.0)
_BEAM_RGBA = (0.20, 0.45, 0.95, 1.0)


def width_for_difficulty(difficulty: float, widths: tuple[float, ...] = BEAM_WIDTHS) -> float:
  """Curriculum rows get difficulty ``row / (rows - 1)``; map back to a level."""
  idx = int(round(difficulty * (len(widths) - 1)))
  return widths[min(max(idx, 0), len(widths) - 1)]


def resample_polyline(points: np.ndarray, n: int) -> np.ndarray:
  """``n`` points along a 2D polyline: every vertex, plus points evenly spaced
  by arc length in between. Keeping the vertices keeps bends sharp (an evenly
  spaced resample cuts their corners by up to a few cm, a lot on a 5 cm beam)."""
  seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
  cum = np.concatenate([[0.0], np.cumsum(seg)])
  if n < len(points) + 2:
    raise ValueError(f"{n} path points cannot hold a polyline with {len(points)} vertices")
  even = np.linspace(0.0, cum[-1], n - len(points) + 2)[1:-1]
  s = np.sort(np.concatenate([cum, even]))
  return np.stack([np.interp(s, cum, points[:, 0]), np.interp(s, cum, points[:, 1])], 1)


@dataclass(kw_only=True)
class BeamTileCfg(SubTerrainCfg):
  """One beam tile. Distances in metres, local tile frame."""

  size: tuple[float, float] = (8.0, 2.0)
  widths: tuple[float, ...] = BEAM_WIDTHS
  beam_height: float = 0.25
  beam_length: float = 4.0
  path_kind: Literal["straight", "bent"] = "straight"
  """Centreline generator; see module docstring."""
  max_turn_deg: float = 25.0
  """``bent``: largest heading change at one joint."""
  max_heading_deg: float = 75.0
  """``bent``: largest total heading away from +x."""
  segment_length_range: tuple[float, float] = (0.6, 1.4)
  """``bent``: length of one straight piece (m)."""
  side_margin: float = 0.6
  """``bent``: least distance from the beam or a platform corner to the tile's
  long sides (m)."""
  max_tries: int = 100
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

  def beam_polyline(self, rng: np.random.Generator | None = None) -> np.ndarray:
    """Beam centreline joints [K, 2] in tile coordinates, start to end."""
    x0 = self.end_margin + self.platform_length
    yc = self.size[1] / 2
    straight = np.array([[x0, yc], [x0 + self.beam_length, yc]])
    if self.path_kind == "straight":
      return straight
    if self.path_kind != "bent":
      raise NotImplementedError(self.path_kind)
    if rng is None:
      raise ValueError("a bent beam needs an rng")
    max_turn = math.radians(self.max_turn_deg)
    max_heading = math.radians(self.max_heading_deg)
    seg_lo, seg_hi = self.segment_length_range
    for _ in range(self.max_tries):
      joints = [straight[0]]
      heading = 0.0
      remaining = self.beam_length
      while remaining > 1e-9:
        seg = min(rng.uniform(seg_lo, seg_hi), remaining)
        if remaining - seg < 0.5 * seg_lo:
          seg = remaining  # no stub segment at the end
        remaining -= seg
        joints.append(joints[-1] + seg * np.array([math.cos(heading), math.sin(heading)]))
        heading = float(np.clip(heading + rng.uniform(-max_turn, max_turn), -max_heading, max_heading))
      pts = np.stack(joints)
      if self._fits(pts):
        return pts
    return straight  # a straight beam always fits

  def _fits(self, joints: np.ndarray) -> bool:
    """The beam and the end platform's corners stay inside the tile."""
    d = joints[-1] - joints[-2]
    d = d / np.linalg.norm(d)
    n = np.array([-d[1], d[0]])
    far = joints[-1] + d * self.platform_length
    hw = self.platform_width / 2
    corners = np.stack([joints[-1] + n * hw, joints[-1] - n * hw, far + n * hw, far - n * hw])
    pts = np.vstack([joints, corners])
    sx, sy = self.size
    m = self.side_margin
    return bool(
      (pts[:, 1] >= m).all() and (pts[:, 1] <= sy - m).all() and (pts[:, 0] <= sx - self.end_margin).all()
    )

  # -- geometry --------------------------------------------------------------

  def function(
    self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
  ) -> TerrainOutput:
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

    joints = self.beam_polyline(rng)
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
    for j in joints[1:-1]:
      disc = body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=(width / 2, h / 2, 0.0),
        pos=(j[0], j[1], h / 2),
      )
      geoms.append(TerrainGeometry(geom=disc, color=_BEAM_RGBA))

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


def make_beam_terrain_cfg(
  widths: tuple[float, ...] = BEAM_WIDTHS,
  heights: tuple[float, ...] = BEAM_HEIGHTS,
  seed: int | None = TERRAIN_SEED,
  curves: bool = False,
  **tile_kwargs,
) -> TerrainGeneratorCfg:
  """Rows = width levels, columns = beam heights; with ``curves``, every
  height also gets a ``bend`` and a ``turn`` column (``BEND_MAX_TURN_DEG``),
  so a third of the envs walk straight beams."""
  if curves:
    tile_kwargs.setdefault("size", CURVED_TILE_SIZE)
  sub_terrains: dict[str, SubTerrainCfg] = {}
  for h in heights:
    name = f"beam_h{round(h * 100):02d}"
    sub_terrains[name] = BeamTileCfg(widths=widths, beam_height=h, **tile_kwargs)
    if curves:
      for kind, turn in BEND_MAX_TURN_DEG.items():
        sub_terrains[f"{name}_{kind}"] = BeamTileCfg(
          widths=widths, beam_height=h, path_kind="bent", max_turn_deg=turn, **tile_kwargs
        )
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
