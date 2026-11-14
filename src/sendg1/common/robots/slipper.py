"""The tactile slipper on the G1's feet: geometry, mass and how it goes into the model.

The real build, from the foot down:

  bare G1 sole          ``G1_SOLE_Z`` (-35 mm in the ankle-roll-link frame)
  TPU 95A slipper       3 mm floor (``SLIPPER_FLOOR``), walls up the side of the foot
  tactile foil          ~1 mm (``FOIL``), glued under the slipper
  Vibram 75A rubber     4 mm (``RUBBER``), protects the foil
                        -> new sole plane ``SOLE_Z`` = -43 mm

The slipper shape comes from its CAD export, moved into the ankle frame by
``scripts/make_slipper_asset.py`` and stored as ``assets/g1_slipper.stl``. The
foil and rubber are not in the CAD: they are prisms over the convex hull of the
slipper's flat bottom.

Collision: the 7 stock capsules per foot are replaced by one box spanning the
rubber sole's outline (``*_sole_collision``). A box is flat like the real sole
and makes 4 contacts on flat ground (the capsules made 14), and mujoco_warp
collides boxes with everything the tasks use (planes, heightfields, boxes,
capsules, cylinders, flex). Its corners overhang the rounded toe and heel
outline by up to ~1 cm; accepted.

Mass: slipper ~120 g (weighed) plus foil, glue and rubber, rounded to 200 g per
foot (``SLIPPER_MASS``), added to the ankle-roll link as a uniform slab.
"""

from __future__ import annotations

import struct
from functools import cache
from pathlib import Path

import mujoco
import numpy as np

from sendg1.common.tactile.layout import (
  FOIL,
  G1_SOLE_Z,
  SIDES,
  SLIPPER_FLOOR,
  SOLE_Z,
)

ASSET = Path(__file__).parent / "assets" / "g1_slipper.stl"

G1_FOOT_CENTER_XY = np.array([0.03826, 0.0])
"""Centre of the bare G1 foot mesh's plan-view bounding box (ankle frame, m);
the slipper outline is centred on it."""

SLIPPER_MASS = 0.200
"""kg per foot: slipper (120 g), rubber (~65 g for 4 mm), foil and glue."""

SOLE_GEOM = "{side}_sole_collision"
G1_SOLE_GEOMS: tuple[str, ...] = tuple(SOLE_GEOM.format(side=s) for s in SIDES)

# kyber palette: slipper in the accent blue, foil white, rubber black.
SLIPPER_RGBA = (0x4C / 255, 0xC9 / 255, 0xF0 / 255, 1.0)
FOIL_RGBA = (0.95, 0.95, 0.95, 1.0)
RUBBER_RGBA = (0.04, 0.04, 0.05, 1.0)

FOOT_SITE_Z = -0.037 - (G1_SOLE_Z - SOLE_Z)
"""The ``*_foot`` sites (clearance rewards, foot height scan) sat 2 mm below the
bare sole; they move down with it."""


def read_stl(path: Path) -> np.ndarray:
  """Binary STL -> triangles [N, 3, 3] (file units)."""
  raw = path.read_bytes()
  n = struct.unpack("<I", raw[80:84])[0]
  rec = np.dtype([("n", "<3f4"), ("v", "<9f4"), ("a", "<u2")])
  return (
    np.frombuffer(raw[84 : 84 + 50 * n], dtype=rec)["v"].reshape(-1, 3, 3).astype(float)
  )


def write_stl(path: Path, tris: np.ndarray) -> None:
  e1, e2 = tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]
  n = np.cross(e1, e2)
  n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
  rec = np.zeros(len(tris), dtype=[("n", "<3f4"), ("v", "<9f4"), ("a", "<u2")])
  rec["n"], rec["v"] = n, tris.reshape(-1, 9)
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_bytes(
    b"sendg1 G1 slipper, ankle-roll-link frame, m".ljust(80)
    + struct.pack("<I", len(tris))
    + rec.tobytes()
  )


@cache
def slipper_mesh() -> tuple[np.ndarray, np.ndarray]:
  """(vertices [V, 3], faces [F, 3]) of the slipper, ankle frame, metres."""
  tris = read_stl(ASSET)
  verts, faces = np.unique(tris.reshape(-1, 3).round(7), axis=0, return_inverse=True)
  return verts, faces.reshape(-1, 3)


@cache
def sole_outline() -> np.ndarray:
  """Convex hull [K, 2] of the slipper's flat bottom (ankle frame, m), counter-clockwise."""
  from scipy.spatial import ConvexHull

  v, _ = slipper_mesh()
  bottom = v[v[:, 2] < v[:, 2].min() + 0.0003, :2]
  return bottom[ConvexHull(bottom).vertices]


def _prism(
  outline: np.ndarray, z_top: float, z_bottom: float
) -> tuple[np.ndarray, np.ndarray]:
  k = len(outline)
  verts = np.concatenate(
    [
      np.column_stack([outline, np.full(k, z_top)]),
      np.column_stack([outline, np.full(k, z_bottom)]),
    ]
  )
  faces = []
  for i in range(1, k - 1):  # convex outline: fans
    faces += [(0, i, i + 1), (k, k + i + 1, k + i)]
  for i in range(k):
    j = (i + 1) % k
    faces += [(i, k + i, k + j), (i, k + j, j)]
  return verts, np.array(faces)


def sole_box() -> tuple[np.ndarray, np.ndarray]:
  """(centre [3], half sizes [3]) of the collision box under the rubber outline."""
  o = sole_outline()
  lo, hi = o.min(0), o.max(0)
  z_top = G1_SOLE_Z - SLIPPER_FLOOR
  return (
    np.array([*(lo + hi) / 2, (z_top + SOLE_Z) / 2]),
    np.array([*(hi - lo) / 2, (z_top - SOLE_Z) / 2]),
  )


def _add_mesh(
  spec: mujoco.MjSpec, name: str, verts: np.ndarray, faces: np.ndarray
) -> None:
  mesh = spec.add_mesh()
  mesh.name = name
  mesh.uservert = verts.astype(np.float32).ravel().tolist()
  mesh.userface = faces.astype(np.int32).ravel().tolist()
  # Visual only (density 0); shell inertia so the open CAD shell compiles.
  mesh.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL


def _add_mass(
  body: mujoco.MjsBody, mass: float, center: np.ndarray, half: np.ndarray
) -> None:
  """Combine the body's inertia with a uniform box (parallel-axis theorem)."""

  def full(m, diag, quat, pos):
    r = np.zeros(9)
    mujoco.mju_quat2Mat(r, np.asarray(quat, float))
    r = r.reshape(3, 3)
    i = r @ np.diag(diag) @ r.T
    return i, m, np.asarray(pos, float)

  i1, m1, p1 = full(body.mass, body.inertia, body.iquat, body.ipos)
  s = 2 * half
  i2 = (
    mass
    / 12
    * np.diag([s[1] ** 2 + s[2] ** 2, s[0] ** 2 + s[2] ** 2, s[0] ** 2 + s[1] ** 2])
  )
  m = m1 + mass
  com = (m1 * p1 + mass * center) / m

  def shift(i, mm, p):
    d = p - com
    return i + mm * (d @ d * np.eye(3) - np.outer(d, d))

  total = shift(i1, m1, p1) + shift(i2, mass, center)
  w, vecs = np.linalg.eigh(total)
  if np.linalg.det(vecs) < 0:
    vecs[:, 0] *= -1
  q = np.zeros(4)
  mujoco.mju_mat2Quat(q, vecs.ravel())
  body.mass, body.ipos, body.iquat, body.inertia = m, com, q, w


def add_slipper(spec: mujoco.MjSpec) -> mujoco.MjSpec:
  """Put the slipper on both feet of a G1 spec, in place: visuals, sole box,
  mass, and the foot sites moved to the new sole."""
  sv, sf = slipper_mesh()
  outline = sole_outline()
  z_floor = G1_SOLE_Z - SLIPPER_FLOOR
  _add_mesh(spec, "slipper", sv, sf)
  _add_mesh(spec, "slipper_foil", *_prism(outline, z_floor, z_floor - FOIL))
  _add_mesh(spec, "slipper_rubber", *_prism(outline, z_floor - FOIL, SOLE_Z))
  center, half = sole_box()
  for side in SIDES:
    body = spec.body(f"{side}_ankle_roll_link")
    for g in list(body.geoms):
      if g.name.startswith(f"{side}_foot") and g.name.endswith("_collision"):
        spec.delete(g)
    for mesh, rgba in (
      ("slipper", SLIPPER_RGBA),
      ("slipper_foil", FOIL_RGBA),
      ("slipper_rubber", RUBBER_RGBA),
    ):
      g = body.add_geom()
      g.name = f"{side}_{mesh}_visual"
      g.type = mujoco.mjtGeom.mjGEOM_MESH
      g.meshname = mesh
      g.rgba = rgba
      g.group = 2
      g.contype = g.conaffinity = 0
      g.density = 0.0
    col = body.add_geom()
    col.name = SOLE_GEOM.format(side=side)
    col.type = mujoco.mjtGeom.mjGEOM_BOX
    col.size = half
    col.pos = center
    col.group = 3
    col.density = 0.0
    # The stack's mass sits around the slipper floor (rubber below, walls above).
    _add_mass(body, SLIPPER_MASS, np.array([*center[:2], z_floor]), half)
    site = spec.site(f"{side}_foot")
    site.pos = [site.pos[0], site.pos[1], FOOT_SITE_Z]
  return spec
