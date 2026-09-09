#!/usr/bin/env python
"""Render the terrain curriculum: whole grid plus one tile per sub-terrain.

    python scripts/render_terrain.py

Writes media/terrain_grid.png and media/terrain_<name>.png. Offscreen via EGL,
so it works headless.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# Must precede any mujoco import: mujoco's gl_context reads MUJOCO_GL once at
# load time, and this script imports mujoco before mjlab gets a chance to set
# it. Without this, offscreen rendering falls back to GLFW and dies on a
# headless machine with "an OpenGL platform library has not been loaded".
os.environ.setdefault("MUJOCO_GL", "egl")

REPO = Path(__file__).resolve().parent.parent


def _render(model, data, cam, w, h):
  import mujoco

  with mujoco.Renderer(model, height=h, width=w) as r:
    mujoco.mj_forward(model, data)
    r.update_scene(data, camera=cam)
    return r.render()


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--out", default=str(REPO / "media"))
  ap.add_argument("--rows", type=int, default=10)
  ap.add_argument("--width", type=int, default=1280)
  ap.add_argument("--height", type=int, default=900)
  args = ap.parse_args()

  import imageio.v3 as iio
  import mujoco
  import numpy as np
  import torch

  from mjlab.terrains.terrain_entity import TerrainEntity, TerrainEntityCfg

  from sendg1.common.terrains import TERRAIN_ROLES, make_rough_terrain_cfg

  out = Path(args.out)
  out.mkdir(parents=True, exist_ok=True)
  device = "cuda:0" if torch.cuda.is_available() else "cpu"

  gen = make_rough_terrain_cfg(num_rows=args.rows)
  names = list(gen.sub_terrains)

  entity = TerrainEntity(
    TerrainEntityCfg(terrain_type="generator", terrain_generator=gen), device=device
  )
  # The offscreen framebuffer size is baked in at compile time and defaults to
  # 640x480, so it has to be raised on the spec before compiling.
  entity.spec.visual.global_.offwidth = max(args.width, 640)
  entity.spec.visual.global_.offheight = max(args.height, 480)
  model = entity.spec.compile()
  data = mujoco.MjData(model)
  print(f"[terrain] {model.ngeom:,} geoms  {model.nmesh:,} meshes  "
        f"{len(names)} columns x {args.rows} rows")

  size_x, size_y = gen.size
  span_x, span_y = args.rows * size_x, len(names) * size_y

  # Whole grid, looking down the difficulty axis.
  cam = mujoco.MjvCamera()
  cam.lookat[:] = [0.0, 0.0, 0.0]
  cam.distance = 1.15 * max(span_x, span_y)
  cam.azimuth = 90.0
  cam.elevation = -62.0
  iio.imwrite(out / "terrain_grid.png", _render(model, data, cam, args.width, args.height))
  print(f"[terrain] wrote {out / 'terrain_grid.png'}")

  # One close-up per column, at ~70% difficulty, framed on that patch.
  origins = entity.terrain_origins
  assert origins is not None
  org = origins.cpu().numpy() if hasattr(origins, "cpu") else np.asarray(origins)
  row = int(0.7 * (args.rows - 1))
  for col, nm in enumerate(names):
    c = mujoco.MjvCamera()
    c.lookat[:] = org[row, col]
    # Tight enough to frame a single 8x8 m patch; a wider shot pulls in the
    # neighbouring columns and it stops being obvious which terrain you are
    # looking at.
    c.distance = 0.85 * size_x
    c.azimuth = 90.0
    c.elevation = -46.0
    iio.imwrite(out / f"terrain_{nm}.png", _render(model, data, c, 640, 440))
  print(f"[terrain] wrote {len(names)} close-ups (row {row}/{args.rows - 1})")

  for nm in names:
    print(f"  {nm:22s} {TERRAIN_ROLES[nm].role}")


if __name__ == "__main__":
  main()
