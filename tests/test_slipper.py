"""The tactile slipper on the G1: geometry, mass and contacts."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from sendg1.common.robots.g1 import G1_FOOT_GEOMS, STANDING_BASE_Z, get_g1_spec
from sendg1.common.robots.slipper import SLIPPER_MASS, sole_box, sole_outline
from sendg1.common.tactile.layout import (
  LAYOUTS,
  SOLE_EXTENT_X,
  SOLE_EXTENT_Y,
  SOLE_Z,
)


@pytest.fixture(scope="module")
def models():
  from mjlab.asset_zoo.robots.unitree_g1.g1_constants import get_spec

  return get_spec().compile(), get_g1_spec().compile()


def test_sole_extent_constants_match_the_cad() -> None:
  o = sole_outline()
  np.testing.assert_allclose(o[:, 0].min(), SOLE_EXTENT_X[0], atol=5e-4)
  np.testing.assert_allclose(o[:, 0].max(), SOLE_EXTENT_X[1], atol=5e-4)
  np.testing.assert_allclose(o[:, 1].min(), SOLE_EXTENT_Y[0], atol=5e-4)
  np.testing.assert_allclose(o[:, 1].max(), SOLE_EXTENT_Y[1], atol=5e-4)


def test_every_taxel_sits_inside_the_rubber_outline() -> None:
  from matplotlib.path import Path

  outline = Path(sole_outline())
  layout = LAYOUTS["slipper_v1"]  # the placeholder grid ignores the rounded toe
  for side in ("left", "right"):
    assert outline.contains_points(np.array(layout.for_side(side))).all()


def test_sole_box_bottom_is_the_sole_plane(models) -> None:
  _, m = models
  for name in G1_FOOT_GEOMS:
    g = m.geom(name)
    assert g.type[0] == mujoco.mjtGeom.mjGEOM_BOX
    assert g.pos[2] - g.size[2] == pytest.approx(SOLE_Z, abs=1e-9)
  center, half = sole_box()
  assert 2 * half[0] == pytest.approx(0.2176, abs=2e-3)  # 218 mm long
  assert 2 * half[1] == pytest.approx(0.0811, abs=2e-3)  # 81 mm wide


def test_stock_foot_capsules_are_gone(models) -> None:
  _, m = models
  names = [m.geom(i).name for i in range(m.ngeom)]
  assert not [n for n in names if "_foot" in n and n.endswith("_collision")]


def test_slipper_adds_its_mass_to_each_foot(models) -> None:
  stock, m = models
  for side in ("left", "right"):
    b = f"{side}_ankle_roll_link"
    assert m.body(b).mass[0] - stock.body(b).mass[0] == pytest.approx(SLIPPER_MASS)
  assert m.body_mass.sum() - stock.body_mass.sum() == pytest.approx(2 * SLIPPER_MASS)


def test_robot_stands_on_the_sole_box() -> None:
  """Spawned at STANDING_BASE_Z, the robot ends up on its sole boxes with
  only solver-sized penetration."""
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.registry import load_env_cfg

  import sendg1.tasks  # noqa: F401

  cfg = load_env_cfg("Sendg1-Velocity-Flat-G1-Blind")
  cfg.scene.num_envs = 2
  assert cfg.scene.entities["robot"].init_state.pos[2] == pytest.approx(STANDING_BASE_Z)
  cfg.events = {k: v for k, v in cfg.events.items() if k in ("reset_robot_joints",)}
  import torch

  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  try:
    env.reset()
    robot = env.scene["robot"]
    ids, _ = robot.find_geoms(G1_FOOT_GEOMS)
    for _ in range(10):
      env.step(torch.zeros(2, env.action_manager.total_action_dim, device=device))
    d = env.sim.data
    n = int(d.nacon[0])
    gids = {int(robot.indexing.geom_ids[i]) for i in ids}
    geom = d.contact.geom[:n].cpu().numpy()
    dist = d.contact.dist[:n].cpu().numpy()
    on_sole = np.isin(geom, list(gids)).any(-1)
    assert on_sole.any(), "the soles must touch the floor"
    assert dist[on_sole].min() > -0.003
  finally:
    env.close()
