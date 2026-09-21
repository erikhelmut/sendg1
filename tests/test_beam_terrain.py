"""Beam terrain geometry and path projection. CPU only, no simulation."""

from __future__ import annotations

import math

import mujoco
import numpy as np
import pytest
import torch

from sendg1.common.beam.mdp import project_onto_path
from sendg1.common.beam.terrain import (
  BEAM_PATH_PATCH,
  BEAM_WIDTHS,
  NUM_PATH_POINTS,
  BeamPathTerrainCfg,
  bend_type,
  beam_width,
  make_beam_terrain_cfg,
)


def _generate(cfg: BeamPathTerrainCfg, difficulty: float, seed: int):
  spec = mujoco.MjSpec()
  spec.worldbody.add_body(name="terrain")
  cfg.size = (8.0, 8.0)
  return cfg.function(difficulty, spec, np.random.default_rng(seed))


def _is_floor(geom) -> bool:
  return geom.pos[2] < -1.0


def _top(geom) -> float:
  half = geom.size[2] if geom.type == mujoco.mjtGeom.mjGEOM_BOX else geom.size[1]
  return float(geom.pos[2] + half)


def _supported(xy: np.ndarray, geometries, tol: float = 1e-6) -> bool:
  for tg in geometries:
    g = tg.geom
    if _is_floor(g):
      continue
    rel = xy - np.asarray(g.pos[:2])
    if g.type == mujoco.mjtGeom.mjGEOM_CYLINDER:
      if np.linalg.norm(rel) <= g.size[0] + tol:
        return True
      continue
    q = np.asarray(g.quat)
    yaw = 2.0 * math.atan2(q[3], q[0])
    c, s = math.cos(-yaw), math.sin(-yaw)
    local = np.array([c * rel[0] - s * rel[1], s * rel[0] + c * rel[1]])
    if abs(local[0]) <= g.size[0] + tol and abs(local[1]) <= g.size[1] + tol:
      return True
  return False


def test_width_schedule_is_log_spaced() -> None:
  assert BEAM_WIDTHS[0] == pytest.approx(1.20)
  assert BEAM_WIDTHS[-1] == pytest.approx(0.15)
  ratios = [b / a for a, b in zip(BEAM_WIDTHS[:-1], BEAM_WIDTHS[1:])]
  assert all(r == pytest.approx(ratios[0]) for r in ratios)


@pytest.mark.parametrize("difficulty", [0.0, 0.5, 1.0])
def test_path_export_matches_spawn_and_beam_top(difficulty: float) -> None:
  cfg = BeamPathTerrainCfg(max_turn_deg=60.0)
  out = _generate(cfg, difficulty, seed=1)
  path = out.flat_patches[BEAM_PATH_PATCH]
  assert path.shape == (NUM_PATH_POINTS, 3)
  np.testing.assert_allclose(path[:, 2], cfg.beam_height)
  np.testing.assert_allclose(path[0], out.origin)


def test_platform_and_beam_tops_are_flush() -> None:
  cfg = BeamPathTerrainCfg(max_turn_deg=60.0)
  out = _generate(cfg, 0.7, seed=2)
  tops = [_top(tg.geom) for tg in out.geometries if not _is_floor(tg.geom)]
  np.testing.assert_allclose(tops, cfg.beam_height, atol=1e-9)


@pytest.mark.parametrize("seed", range(25))
def test_path_stays_inside_patch(seed: int) -> None:
  # The generator overwrites size with its own; sampled directly, set it by hand.
  cfg = BeamPathTerrainCfg(max_turn_deg=60.0, size=(8.0, 8.0))
  joints = cfg.sample_joints(np.random.default_rng(seed))
  m = cfg.edge_margin - 1e-9
  assert (joints >= m).all() and (joints <= 8.0 - m).all()
  seg = np.linalg.norm(np.diff(joints, axis=0), axis=1)
  assert seg.sum() == pytest.approx(cfg.beam_length)


@pytest.mark.parametrize("seed", range(10))
def test_centreline_is_supported_at_narrowest_width(seed: int) -> None:
  """Every path point, and the chord between neighbours, lies over the beam.

  Chord midpoints cut the corners at sharp joints; this is what proves the joint
  cylinders actually close the gap on a bend.
  """
  cfg = BeamPathTerrainCfg(max_turn_deg=60.0)
  out = _generate(cfg, 1.0, seed)
  xy = out.flat_patches[BEAM_PATH_PATCH][:, :2]
  mids = 0.5 * (xy[:-1] + xy[1:])
  for p in np.vstack([xy, mids]):
    assert _supported(p, out.geometries), f"point {p} is over the pit"


def test_path_points_are_evenly_spaced() -> None:
  cfg = BeamPathTerrainCfg(max_turn_deg=60.0)
  xy = _generate(cfg, 0.0, seed=3).flat_patches[BEAM_PATH_PATCH][:, :2]
  spacing = np.linalg.norm(np.diff(xy, axis=0), axis=1)
  # Resampled along the true polyline, so chords only shrink where they cut a
  # corner: at most by cos(30 deg) for a 60 deg joint.
  nominal = (cfg.platform_size / 2 + cfg.beam_length) / (NUM_PATH_POINTS - 1)
  assert spacing.max() == pytest.approx(nominal, rel=1e-6)
  assert spacing.min() >= 0.85 * nominal


def test_straight_column_is_straight() -> None:
  out = _generate(BeamPathTerrainCfg(max_turn_deg=0.0), 0.5, seed=4)
  np.testing.assert_allclose(out.flat_patches[BEAM_PATH_PATCH][:, 1], 4.0)


def test_bend_type_strips_instance_suffix() -> None:
  assert bend_type("beam_straight") == "beam_straight"
  assert bend_type("beam_bend_0") == "beam_bend"
  assert bend_type("beam_turn_12") == "beam_turn"


def test_generator_exports_one_path_per_patch() -> None:
  from mjlab.terrains.terrain_entity import TerrainEntity, TerrainEntityCfg

  gen = make_beam_terrain_cfg()
  entity = TerrainEntity(
    TerrainEntityCfg(terrain_type="generator", terrain_generator=gen), device="cpu"
  )
  paths = entity.flat_patches[BEAM_PATH_PATCH]
  assert tuple(paths.shape) == (10, len(gen.sub_terrains), NUM_PATH_POINTS, 3)
  origins = torch.as_tensor(entity.terrain_origins, dtype=paths.dtype)
  torch.testing.assert_close(paths[:, :, 0, :], origins)
  names = list(gen.sub_terrains)
  a, b = names.index("beam_turn_0"), names.index("beam_turn_1")
  rel_a = paths[5, a, :, :2] - paths[5, a, :1, :2]
  rel_b = paths[5, b, :, :2] - paths[5, b, :1, :2]
  assert not torch.allclose(rel_a, rel_b), "instances of one bend type share a path"


def test_row_widths_follow_generator_difficulty() -> None:
  gen = make_beam_terrain_cfg()
  for row, width in enumerate(BEAM_WIDTHS):
    assert beam_width(row / (gen.num_rows - 1)) == pytest.approx(width)


def test_projection_on_l_shaped_path() -> None:
  corner = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
  points = corner[None].repeat(2, 1, 1)
  xy = torch.tensor([[0.5, 0.1], [1.1, 0.5]])
  proj = project_onto_path(points, xy)
  torch.testing.assert_close(proj.arc, torch.tensor([0.5, 1.5]))
  # Left of travel is positive: +y on the first leg, -x on the second.
  torch.testing.assert_close(proj.lateral, torch.tensor([0.1, -0.1]))
  torch.testing.assert_close(proj.tangent_yaw, torch.tensor([0.0, math.pi / 2]))
  torch.testing.assert_close(proj.length, torch.tensor([2.0, 2.0]))
