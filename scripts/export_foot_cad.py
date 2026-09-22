#!/usr/bin/env python
"""Export the G1 foot in the tactile model's sole frame, for CAD.

    pip install ezdxf                       # or: pip install -e ".[cad]"
    python scripts/export_foot_cad.py       # -> media/cad/

Writes, per foot, in MILLIMETRES and in the ``*_ankle_roll_link`` frame (the
frame the taxel layout is defined in):

  g1_<side>_foot_sole_frame.dxf   layered 3D DXF: foot mesh, sole footprint,
                                  plan silhouette, collision capsules, frame
                                  axes, sole plane, current taxel layout
  g1_<side>_foot_sole_frame.stl   the foot mesh alone, for fitting a slipper
  g1_<side>_foot_sole_frame.png   preview of the DXF

The frame: origin on the ankle-roll joint axis; X forward along the foot, Y to
the robot's left, Z up when the foot is flat. The sole (bottom of the collision
capsules) is the plane Z = -35 mm. The right foot's frame is the same, with
the foot mirrored; the slipper sensor is the same part on both feet, so a
layout drawn on the left foot keeps its (X, Y) on the right.

DXF rather than DWG: DWG is a closed format with no reliable open-source
writer. Every major CAD package imports DXF, and can save it as DWG.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
MM = 1000.0

LAYERS = {
  # name: (AutoCAD colour index, description)
  "FOOT_MESH_3D": (8, "visual foot mesh (3D)"),
  "SOLE_FOOTPRINT": (1, "flat heel and forefoot pads: mesh section 0.5 mm above "
                        "its lowest point; the mid-foot band between them is "
                        "recessed ~1.5 mm"),
  "PLAN_SILHOUETTE": (9, "outline of the whole foot seen from above"),
  "COLLISION_CAPSULES": (30, "the 7 sim collision capsules, r = 10 mm, at Z = -35"),
  "FRAME": (7, "sole frame axes and origin"),
  "SOLE_PLANE": (5, "sole plane Z = -35 mm"),
  "TAXELS_CURRENT": (150, "taxel layout currently in the model"),
  "NOTES": (7, "labels"),
}


def _plan_silhouette(mesh, res_mm: float = 0.2) -> list[np.ndarray]:
  """Outline of the mesh projected onto XY, by rasterising the projected
  triangles and contouring. Returns closed polylines in mm."""
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  from matplotlib.path import Path as MplPath

  tris = mesh.vertices[mesh.faces][:, :, :2] * MM
  lo, hi = tris.reshape(-1, 2).min(0) - 2, tris.reshape(-1, 2).max(0) + 2
  xs = np.arange(lo[0], hi[0], res_mm)
  ys = np.arange(lo[1], hi[1], res_mm)
  gx, gy = np.meshgrid(xs, ys)
  pts = np.column_stack([gx.ravel(), gy.ravel()])
  inside = np.zeros(len(pts), dtype=bool)
  for tri in tris:
    tmin, tmax = tri.min(0), tri.max(0)
    cand = np.where(
      ~inside
      & (pts[:, 0] >= tmin[0]) & (pts[:, 0] <= tmax[0])
      & (pts[:, 1] >= tmin[1]) & (pts[:, 1] <= tmax[1])
    )[0]
    if len(cand):
      inside[cand[MplPath(tri).contains_points(pts[cand], radius=1e-9)]] = True
  fig, ax = plt.subplots()
  cs = ax.contour(gx, gy, inside.reshape(gx.shape).astype(float), levels=[0.5])
  loops = [p for p in cs.get_paths()[0].to_polygons() if len(p) > 20]
  plt.close(fig)
  return loops


def _section_loops(mesh, z_m: float) -> list[np.ndarray]:
  """Closed polylines (mm) where the plane Z = z_m cuts the mesh."""
  from trimesh.intersections import mesh_plane

  segs = mesh_plane(mesh, plane_normal=[0, 0, 1], plane_origin=[0, 0, z_m]) * MM
  segs = [s[:, :2] for s in segs]
  # Chain segments into loops by matching endpoints.
  key = lambda p: (round(p[0], 3), round(p[1], 3))  # noqa: E731
  adj: dict[tuple, list[int]] = {}
  for i, (a, b) in enumerate(segs):
    adj.setdefault(key(a), []).append(i)
    adj.setdefault(key(b), []).append(i)
  used = np.zeros(len(segs), dtype=bool)
  loops = []
  for start in range(len(segs)):
    if used[start]:
      continue
    used[start] = True
    a, b = segs[start]
    loop = [a, b]
    cur = b
    while True:
      nxt = [j for j in adj.get(key(cur), []) if not used[j]]
      if not nxt:
        break
      j = nxt[0]
      used[j] = True
      p, q = segs[j]
      cur = q if key(p) == key(cur) else p
      loop.append(cur)
    if len(loop) > 10:
      loops.append(np.array(loop))
  return loops


def _stadium(p0, p1, r, n=24) -> np.ndarray:
  """Plan outline (mm) of a capsule lying along X."""
  x0, x1 = sorted((p0[0], p1[0]))
  y = 0.5 * (p0[1] + p1[1])
  a = np.linspace(-np.pi / 2, np.pi / 2, n)
  right = np.column_stack([x1 + r * np.cos(a), y + r * np.sin(a)])
  left = np.column_stack([x0 - r * np.cos(a), y - r * np.sin(a)])
  return np.vstack([right, left])


def export(side: str, out_dir: Path) -> list[Path]:
  import ezdxf
  from ezdxf.enums import TextEntityAlignment
  from ezdxf.render import MeshBuilder

  from sendg1.common.tactile import viz
  from sendg1.common.tactile.layout import LAYOUTS, SOLE_Z
  from sendg1.common.tactile.sensing import TactileParams

  mesh = viz.foot_mesh(side)
  sole_mm = SOLE_Z * MM
  stem = out_dir / f"g1_{side}_foot_sole_frame"

  # -- STL -------------------------------------------------------------------
  mesh_mm = mesh.copy()
  mesh_mm.apply_scale(MM)
  mesh_mm.export(stem.with_suffix(".stl"))

  # -- DXF -------------------------------------------------------------------
  doc = ezdxf.new("R2010", setup=True)
  doc.units = ezdxf.units.MM
  doc.header["$INSUNITS"] = 4  # millimetres
  for name, (color, desc) in LAYERS.items():
    layer = doc.layers.add(name, color=color)
    layer.description = desc
  msp = doc.modelspace()

  mb = MeshBuilder()
  mb.vertices = [tuple(v) for v in mesh_mm.vertices]
  mb.faces = [tuple(f) for f in mesh_mm.faces]
  mb.render_mesh(msp, dxfattribs={"layer": "FOOT_MESH_3D"})

  lowest = float(mesh.vertices[:, 2].min())
  for loop in _section_loops(mesh, lowest + 0.0005):
    msp.add_polyline3d([(x, y, (lowest + 0.0005) * MM) for x, y in loop],
                       close=True, dxfattribs={"layer": "SOLE_FOOTPRINT"})
  for loop in _plan_silhouette(mesh):
    msp.add_polyline3d([(x, y, sole_mm) for x, y in loop], close=True,
                       dxfattribs={"layer": "PLAN_SILHOUETTE"})

  for c in viz.foot_capsules(side):
    p0 = np.array(c.p0) * MM
    p1 = np.array(c.p1) * MM
    outline = _stadium(p0, p1, c.radius * MM)
    msp.add_polyline3d([(x, y, sole_mm) for x, y in outline], close=True,
                       dxfattribs={"layer": "COLLISION_CAPSULES"})
    msp.add_line(tuple(p0), tuple(p1), dxfattribs={"layer": "COLLISION_CAPSULES"})

  # Frame: axes from the origin, and the same XY axes dropped to the sole plane
  # so they can be snapped to when sketching on the sole.
  axes = {"X": ((60, 0, 0), 1), "Y": ((0, 40, 0), 3), "Z": ((0, 0, 40), 5)}
  for label, (tip, color) in axes.items():
    msp.add_line((0, 0, 0), tip, dxfattribs={"layer": "FRAME", "color": color})
    t = msp.add_text(label, height=4, dxfattribs={"layer": "FRAME", "color": color})
    t.set_placement(tuple(np.array(tip) * 1.08), align=TextEntityAlignment.MIDDLE_CENTER)
  for label, (tip, color) in list(axes.items())[:2]:
    msp.add_line((0, 0, sole_mm), (tip[0], tip[1], sole_mm),
                 dxfattribs={"layer": "FRAME", "color": color, "linetype": "DASHED"})
  msp.add_circle((0, 0, 0), radius=2, dxfattribs={"layer": "FRAME"})
  msp.add_circle((0, 0, sole_mm), radius=2, dxfattribs={"layer": "FRAME"})

  x0, x1, y0, y1 = -80, 160, -50, 50
  msp.add_polyline3d([(x0, y0, sole_mm), (x1, y0, sole_mm), (x1, y1, sole_mm),
                      (x0, y1, sole_mm)], close=True, dxfattribs={"layer": "SOLE_PLANE"})

  layout = LAYOUTS[TactileParams().layout]
  for i, ((cx, cy), (hx, hy)) in enumerate(zip(layout.for_side(side), layout.half_sizes)):
    cx, cy, hx, hy = cx * MM, cy * MM, hx * MM, hy * MM
    msp.add_polyline3d([(cx - hx, cy - hy, sole_mm), (cx + hx, cy - hy, sole_mm),
                        (cx + hx, cy + hy, sole_mm), (cx - hx, cy + hy, sole_mm)],
                       close=True, dxfattribs={"layer": "TAXELS_CURRENT"})
    t = msp.add_text(str(i), height=2.5, dxfattribs={"layer": "TAXELS_CURRENT"})
    t.set_placement((cx, cy, sole_mm), align=TextEntityAlignment.MIDDLE_CENTER)

  notes = [
    f"G1 {side} foot - tactile sole frame ({side}_ankle_roll_link), units: mm",
    "Origin: ankle-roll joint axis.  X forward, Y left, Z up (foot flat).",
    f"Sole plane (bottom of collision capsules): Z = {sole_mm:.0f} mm",
    "Draw the sensor layout on the LEFT foot; the same part (same X, Y) goes on the right.",
    "Red: flat heel/forefoot pads. Mid-foot band (X ~ -10..43) is ~1.5 mm higher.",
  ]
  for k, line in enumerate(notes):
    t = msp.add_text(line, height=3, dxfattribs={"layer": "NOTES"})
    t.set_placement((x0, y1 + 30 - 5 * k, sole_mm), align=TextEntityAlignment.LEFT)

  dxf_path = stem.with_suffix(".dxf")
  doc.saveas(dxf_path)
  png_path = _preview(dxf_path, stem.with_suffix(".png"))
  return [dxf_path, stem.with_suffix(".stl"), png_path]


def _preview(dxf_path: Path, png_path: Path) -> Path:
  """Top-view render of everything but the 3D mesh, via ezdxf's drawing addon."""
  import ezdxf
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  from ezdxf.addons.drawing import Frontend, RenderContext
  from ezdxf.addons.drawing.config import Configuration
  from ezdxf.addons.drawing.matplotlib import MatplotlibBackend

  doc = ezdxf.readfile(dxf_path)
  doc.layers.get("FOOT_MESH_3D").off()
  fig = plt.figure(figsize=(12, 6.5))
  ax = fig.add_axes([0.02, 0.02, 0.96, 0.96])
  ctx = RenderContext(doc)
  Frontend(ctx, MatplotlibBackend(ax), config=Configuration()).draw_layout(
    doc.modelspace(), finalize=True
  )
  fig.savefig(png_path, dpi=150)
  plt.close(fig)
  return png_path


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--out-dir", default=str(REPO / "media" / "cad"))
  ap.add_argument("--side", choices=("left", "right", "both"), default="both")
  args = ap.parse_args()
  try:
    import ezdxf  # noqa: F401
  except ImportError as e:
    raise SystemExit("needs ezdxf: pip install ezdxf  (or pip install -e '.[cad]')") from e

  out = Path(args.out_dir)
  out.mkdir(parents=True, exist_ok=True)
  sides = ("left", "right") if args.side == "both" else (args.side,)
  for side in sides:
    for p in export(side, out):
      print(f"[cad] wrote {p}")


if __name__ == "__main__":
  main()
