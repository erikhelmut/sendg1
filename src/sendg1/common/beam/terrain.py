"""Beam-walking terrain: a small start platform and a narrow, bending beam over a pit.

Layout of one 8x8 m patch
-------------------------
A 0.6 m square platform near one edge, a beam leaving its front edge along +x,
and a pit 2 m deep everywhere else. Platform and beam are columns rising from
the pit floor, so every edge is a real drop.

The beam is a polyline of total arc length ``beam_length``: straight segments of
random length, joined at random heading changes of up to ``max_turn_deg``. A
cylinder at every joint fills the wedge that a bend would otherwise open on its
outside edge. The total heading is clamped to ``max_heading_deg`` (< 90 deg), so
the beam always advances in +x and never doubles back on itself.

Curriculum axes
---------------
Row = width. Widths are spaced logarithmically from 1.20 m to 0.15 m, because
difficulty scales roughly with 1/width:

  1.20 0.95 0.76 0.60 0.48 0.38 | 0.30 0.24 | 0.19 0.15

The G1's default stance is 0.238 m between foot centres, about 0.31 m outside
edge to outside edge with 0.07 m soles. Rows left of the first bar allow the
normal stance, the middle rows force a narrower one, and the last two need
near-tandem steps.

Column = bend type, which is not a curriculum axis. ``beam_straight`` is the
control: following it needs no information about where the beam goes. The bent
types get several columns each so a recurrent policy cannot memorise one
path per width; evaluation uses a held-out terrain seed on top of that.

Handing the path to the MDP
---------------------------
The centreline (platform centre, then the beam) is resampled to
``NUM_PATH_POINTS`` points evenly spaced by arc length and returned as a "flat
patch" named ``beam_path``. The generator shifts those points into world
coordinates and stores them per (row, column), and ``TerrainEntity`` exposes them
as a tensor -- so rewards and terminations can look up each env's beam without
any side channel.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import mujoco
import numpy as np
from mjlab.terrains.terrain_generator import (
  FlatPatchSamplingCfg,
  SubTerrainCfg,
  TerrainGeneratorCfg,
  TerrainGeometry,
  TerrainOutput,
)

from sendg1.common.terrains import TRAIN_TERRAIN_SEED, TerrainSpec

BEAM_PATH_PATCH = "beam_path"
NUM_PATH_POINTS = 64

WIDTH_RANGE: tuple[float, float] = (1.20, 0.15)
"""Beam width at difficulty 0 and difficulty 1 (m)."""


def beam_width(
  difficulty: float, widest: float = WIDTH_RANGE[0], narrowest: float = WIDTH_RANGE[1]
) -> float:
  """Log-interpolated width: each row narrows by the same factor."""
  return widest * (narrowest / widest) ** difficulty


BEAM_WIDTHS: tuple[float, ...] = tuple(beam_width(r / 9) for r in range(10))
"""Width per row for the default 10-row grid (m)."""

BEAM_ROLES: dict[str, TerrainSpec] = {
  "beam_straight": TerrainSpec(
    "control",
    "Straight beam. Staying on it needs balance and a narrow gait but no "
    "information about where the beam goes.",
  ),
  "beam_bend": TerrainSpec(
    "discriminator",
    "Joints turn by up to 25 deg. At wide rows the error from ignoring a bend "
    "is small; at narrow rows a missed bend puts a foot over the pit.",
  ),
  "beam_turn": TerrainSpec(
    "discriminator",
    "Joints turn by up to 60 deg. A blind policy must detect the turn from the "
    "support under its feet before committing weight.",
  ),
}


def bend_type(column_name: str) -> str:
  """``beam_bend_2`` -> ``beam_bend``; names without an instance suffix unchanged."""
  base, _, suffix = column_name.rpartition("_")
  return base if suffix.isdigit() else column_name


_PLATFORM_RGBA = (0.85, 0.55, 0.20, 1.0)
_BEAM_RGBA = (0.20, 0.45, 0.95, 1.0)
_FLOOR_RGBA = (0.10, 0.10, 0.10, 1.0)


def resample_polyline(points: np.ndarray, n: int) -> np.ndarray:
  """``n`` points evenly spaced by arc length along a 2D polyline."""
  seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
  cum = np.concatenate([[0.0], np.cumsum(seg)])
  s = np.linspace(0.0, cum[-1], n)
  return np.stack([np.interp(s, cum, points[:, 0]), np.interp(s, cum, points[:, 1])], 1)


@dataclass(kw_only=True)
class BeamPathTerrainCfg(SubTerrainCfg):
  beam_length: float = 5.0
  """Arc length of the beam, excluding the platform (m)."""
  width_range: tuple[float, float] = WIDTH_RANGE
  max_turn_deg: float = 0.0
  """Largest heading change at a single joint."""
  max_heading_deg: float = 75.0
  """Largest total heading away from +x. Below 90 so the beam never doubles back."""
  segment_length_range: tuple[float, float] = (0.6, 1.4)
  platform_size: float = 0.6
  platform_x: float = 1.0
  """Platform centre along x, from the patch edge (m). Centred in y."""
  beam_height: float = 0.2
  floor_depth: float = 2.0
  edge_margin: float = 0.8
  """Minimum distance from any beam joint to the patch edge (m)."""
  max_tries: int = 100
  flat_patch_sampling: dict[str, FlatPatchSamplingCfg] | None = field(
    default_factory=lambda: {
      BEAM_PATH_PATCH: FlatPatchSamplingCfg(
        num_patches=NUM_PATH_POINTS, patch_radius=0.02
      )
    }
  )

  def sample_joints(self, rng: np.random.Generator) -> np.ndarray:
    """Beam centreline joints [K, 2] in patch coordinates, from the platform edge."""
    size_x, size_y = self.size
    start = np.array([self.platform_x + self.platform_size / 2, size_y / 2])
    max_turn = math.radians(self.max_turn_deg)
    max_heading = math.radians(self.max_heading_deg)
    seg_lo, seg_hi = self.segment_length_range

    for _ in range(self.max_tries):
      joints = [start]
      heading = 0.0
      remaining = self.beam_length
      while remaining > 1e-9:
        seg = min(rng.uniform(seg_lo, seg_hi), remaining)
        if remaining - seg < 0.5 * seg_lo:
          seg = remaining  # no stub segment at the end
        remaining -= seg
        joints.append(joints[-1] + seg * np.array([math.cos(heading), math.sin(heading)]))
        heading = float(
          np.clip(heading + rng.uniform(-max_turn, max_turn), -max_heading, max_heading)
        )
      pts = np.stack(joints)
      m = self.edge_margin
      inside = (
        (pts[:, 0] >= m).all()
        and (pts[:, 0] <= size_x - m).all()
        and (pts[:, 1] >= m).all()
        and (pts[:, 1] <= size_y - m).all()
      )
      if inside:
        return pts
    # A straight beam always fits a patch this size.
    return np.stack([start, start + np.array([self.beam_length, 0.0])])

  def function(
    self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
  ) -> TerrainOutput:
    body = spec.body("terrain")
    size_x, size_y = self.size
    width = beam_width(difficulty, *self.width_range)
    z_center = (self.beam_height - self.floor_depth) / 2
    half_height = (self.beam_height + self.floor_depth) / 2
    geometries: list[TerrainGeometry] = []

    floor_h = 0.1
    floor = body.add_geom(
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=(size_x / 2, size_y / 2, floor_h / 2),
      pos=(size_x / 2, size_y / 2, -self.floor_depth - floor_h / 2),
    )
    geometries.append(TerrainGeometry(geom=floor, color=_FLOOR_RGBA))

    platform_xy = np.array([self.platform_x, size_y / 2])
    half_p = self.platform_size / 2
    platform = body.add_geom(
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=(half_p, half_p, half_height),
      pos=(platform_xy[0], platform_xy[1], z_center),
    )
    geometries.append(TerrainGeometry(geom=platform, color=_PLATFORM_RGBA))

    joints = self.sample_joints(rng)
    for a, b in zip(joints[:-1], joints[1:], strict=True):
      d = b - a
      yaw = math.atan2(d[1], d[0])
      mid = (a + b) / 2
      seg = body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(float(np.linalg.norm(d)) / 2, width / 2, half_height),
        pos=(mid[0], mid[1], z_center),
      )
      seg.quat = np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])
      geometries.append(TerrainGeometry(geom=seg, color=_BEAM_RGBA))
    for j in joints[1:-1]:
      disc = body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=(width / 2, half_height, 0.0),
        pos=(j[0], j[1], z_center),
      )
      geometries.append(TerrainGeometry(geom=disc, color=_BEAM_RGBA))

    centreline = resample_polyline(np.vstack([platform_xy, joints]), NUM_PATH_POINTS)
    path = np.column_stack([centreline, np.full(NUM_PATH_POINTS, self.beam_height)])
    origin = np.array([platform_xy[0], platform_xy[1], self.beam_height])
    return TerrainOutput(
      origin=origin, geometries=geometries, flat_patches={BEAM_PATH_PATCH: path}
    )


def make_beam_terrain_cfg(
  num_rows: int = 10,
  seed: int | None = TRAIN_TERRAIN_SEED,
  instances: int = 3,
) -> TerrainGeneratorCfg:
  """One straight column plus ``instances`` columns per bent type."""
  sub_terrains: dict[str, SubTerrainCfg] = {
    "beam_straight": BeamPathTerrainCfg(max_turn_deg=0.0),
  }
  for i in range(instances):
    sub_terrains[f"beam_bend_{i}"] = BeamPathTerrainCfg(max_turn_deg=25.0)
  for i in range(instances):
    sub_terrains[f"beam_turn_{i}"] = BeamPathTerrainCfg(max_turn_deg=60.0)
  return TerrainGeneratorCfg(
    seed=seed,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=num_rows,
    num_cols=len(sub_terrains),  # ignored in curriculum mode; kept honest.
    curriculum=True,
    add_lights=True,
    sub_terrains=sub_terrains,
  )
