"""The Pedalo Classic (Holz-Hoerz art. 552001) as an MjSpec.

Sources: the product page (pedalo.de/p/Pedalo-Pedalo-Classic-562/) and the
2025 system catalogue (pedalo-original-pedalo-system-de-en-25-2.pdf), which
give:

- overall 46 x 37.5 x 22 cm (product page; the catalogue rounds to 44x37x22),
  6 kg, rated to 200 kg;
- 6 wheels of 22 cm with PVC tyres, birch plywood, 2 outer + 1 inner per
  wheelset ("Rad außen" 511001, "Rad innen" 511002);
- step boards 30 x 14.5 cm (511030), each resting on two lower bearing blocks
  6 x 14.5 cm ("Trittschale unten", 511028);
- eccentric flange axles, 157.5 mm long, 21 mm diameter (512020).

Not published, estimated: the eccentric offset (crank radius). Measured on the
catalogue's photo of the Reha-S outer wheel, whose two flange screw holes sit
on a line through the wheel centre: their midpoint is 0.45-0.5 R off-centre,
i.e. 5-5.5 cm. ``ECCENTRIC`` takes 5.5 cm, an 11 cm board stroke. Also
estimated: wheel thickness (from the 37.5 cm width less two 14.5 cm boards),
board height above the axle, the mass split.

Kinematics
----------
Two wheelsets, front and rear, each rigid: outer-left wheel, left eccentric
axle, middle wheel, right eccentric axle, outer-right wheel. The left axle
sits ``ECCENTRIC`` from the wheel centre at crank angle phi, the right one at
phi + pi. Each board hangs on the left (right) axle of both wheelsets: a
parallelogram, so the boards stay level and both wheelsets turn together. A
board at the top of its circle is the one to step on; when one board is
straight above the axle and the other straight below (phi = +-90 deg), a
vertical push turns nothing: the dead centre a rider carries through with
momentum.

MuJoCo tree (one freejoint, three hinges, one loop closure)::

  front_wheels (free) --hinge--> left_board --hinge--> rear_wheels
                      --hinge--> right_board ==connect== rear_wheels

All hinges turn about the pedalo's lateral axis (y). Frames at qpos0: x
forward, z up, phi = 0 (left axle ahead of the wheel centre, right axle
behind, boards level).
"""

from __future__ import annotations

import math

import mujoco

# -- dimensions (m) ---------------------------------------------------------

WHEEL_RADIUS = 0.11
"""22 cm wheels, tyre included (catalogue: "Rad ... 22 cm", "PVC-Reifen 22 cm")."""
WHEEL_THICKNESS = 0.025
WIDTH = 0.375
BOARD_LENGTH = 0.30
BOARD_WIDTH = 0.145
BOARD_THICKNESS = 0.018
"""Birch multiplex with a grip layer."""
BLOCK_LENGTH = 0.06
"""Lower bearing block under each board end ("Trittschale unten 6x14,5 cm");
the eccentric axle runs through it."""
BLOCK_THICKNESS = 0.033
AXLE_RADIUS = 0.0105
"""21 mm axle (512020)."""
WHEELBASE = BOARD_LENGTH - BLOCK_LENGTH
"""Axle-to-axle distance along a board: the blocks are centred on the axles,
so 24 cm, and 24 + 22 cm wheels = the 46 cm overall length."""
ECCENTRIC = 0.055
"""Crank radius: wheel centre to eccentric axle. Estimated (module docstring)."""
BOARD_TOP_ABOVE_AXLE = 0.035
"""Board top above the axle centre: half the axle plus the block above it plus
the board. Board tops run from 9 cm to 20 cm above the ground."""

BOARD_Y = WHEEL_THICKNESS / 2 + 0.0025 + BOARD_WIDTH / 2
"""Lateral board centre: half the middle wheel, a 2.5 mm gap, half a board."""
OUTER_WHEEL_Y = WIDTH / 2 - WHEEL_THICKNESS / 2

# -- masses (kg): 6 kg in all -----------------------------------------------

WHEEL_MASS = 0.6
"""Plywood disc ~0.45 kg + tyre."""
AXLE_MASS = 0.3
BOARD_MASS = 0.6
"""Board + two bearing blocks."""
TOTAL_MASS = 6 * WHEEL_MASS + 4 * AXLE_MASS + 2 * BOARD_MASS

# -- contact ------------------------------------------------------------------

TYRE_FRICTION = 0.9
BOARD_FRICTION = 1.0
JOINT_DAMPING = 0.02
"""N m s / rad at each axle bearing: wood on a steel axle, no ball bearing."""
EQ_SOLREF = (0.01, 1.0)
EQ_SOLIMP = (0.95, 0.99, 0.001, 0.5, 2.0)
"""Loop closure: as stiff as a 5 ms step allows (time constant >= 2 dt).
MuJoCo's default (0.02 s, 0.9-0.95) let the joint open by 12 mm under a
240 N rider push."""
CONTYPE = 2
CONAFFINITY = 1
"""The pedalo collides with the floor and the robot (contype 1, conaffinity
1) but not with itself: the boards run 2.5 mm from the wheels."""

# -- names --------------------------------------------------------------------

ROOT_BODY = "front_wheels"
BOARD_BODIES = ("left_board", "right_board")
WHEEL_BODIES = ("front_wheels", "rear_wheels")
BOARD_GEOMS = ("left_board_top", "right_board_top")
CRANK_JOINT = "left_board_hinge"
"""The left board's hinge on the front wheelset turns by -phi when the
wheelset turns by phi (the board stays level)."""
HINGE_JOINTS = ("left_board_hinge", "right_board_hinge", "rear_wheels_hinge")

TYRE_RGBA = (0.08, 0.08, 0.08, 1.0)
WHEEL_RGBA = (0.98, 0.76, 0.05, 1.0)
LOGO_RGBA = (0.93, 0.33, 0.12, 1.0)
WOOD_RGBA = (0.86, 0.72, 0.52, 1.0)
GRIP_RGBA = (0.12, 0.12, 0.13, 1.0)
AXLE_RGBA = (0.75, 0.76, 0.78, 1.0)

_Y_AXIS_CYL = (math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0)
"""Turns a cylinder's axis (z) onto y."""


def _collide(geom, friction: float) -> None:
  geom.contype = CONTYPE
  geom.conaffinity = CONAFFINITY
  geom.friction = (friction, 0.005, 0.0001)
  geom.group = 0


def _visual(geom) -> None:
  geom.contype = 0
  geom.conaffinity = 0
  geom.mass = 0.0
  geom.group = 1


def _wheelset(body, prefix: str) -> None:
  """Three wheels and two eccentric axles, in a frame at the wheel centre."""
  for name, y in (("left", OUTER_WHEEL_Y), ("mid", 0.0), ("right", -OUTER_WHEEL_Y)):
    tyre = body.add_geom(
      name=f"{prefix}_{name}_wheel",
      type=mujoco.mjtGeom.mjGEOM_CYLINDER,
      size=(WHEEL_RADIUS, WHEEL_THICKNESS / 2, 0),
      pos=(0, y, 0),
      quat=_Y_AXIS_CYL,
      mass=WHEEL_MASS,
      rgba=TYRE_RGBA,
    )
    _collide(tyre, TYRE_FRICTION)
    face = body.add_geom(
      name=f"{prefix}_{name}_wheel_face",
      type=mujoco.mjtGeom.mjGEOM_CYLINDER,
      size=(WHEEL_RADIUS - 0.012, WHEEL_THICKNESS / 2 + 0.0005, 0),
      pos=(0, y, 0),
      quat=_Y_AXIS_CYL,
      rgba=WHEEL_RGBA,
    )
    _visual(face)
    # A spoke-like stripe on each wheel face, so turning is visible.
    stripe = body.add_geom(
      name=f"{prefix}_{name}_wheel_mark",
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=(0.6 * WHEEL_RADIUS, WHEEL_THICKNESS / 2 + 0.001, 0.008),
      pos=(0.25 * WHEEL_RADIUS, y, 0.03),
      rgba=LOGO_RGBA,
    )
    _visual(stripe)
  for name, sign in (("left", 1.0), ("right", -1.0)):
    axle = body.add_geom(
      name=f"{prefix}_{name}_axle",
      type=mujoco.mjtGeom.mjGEOM_CYLINDER,
      size=(AXLE_RADIUS, BOARD_Y - WHEEL_THICKNESS / 2, 0),
      pos=(sign * ECCENTRIC, sign * BOARD_Y, 0),
      quat=_Y_AXIS_CYL,
      mass=AXLE_MASS,
      rgba=AXLE_RGBA,
    )
    _visual(axle)
    axle.mass = AXLE_MASS


def _board(body, prefix: str) -> None:
  """A board in a frame at its front axle: blocks around both axles, the board
  on top, reaching ``WHEELBASE`` back."""
  top = body.add_geom(
    name=f"{prefix}_top",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=(BOARD_LENGTH / 2, BOARD_WIDTH / 2, BOARD_THICKNESS / 2),
    pos=(-WHEELBASE / 2, 0, BOARD_TOP_ABOVE_AXLE - BOARD_THICKNESS / 2),
    mass=0.6 * BOARD_MASS,
    rgba=GRIP_RGBA,
  )
  _collide(top, BOARD_FRICTION)
  block_top = BOARD_TOP_ABOVE_AXLE - BOARD_THICKNESS
  for name, x in (("front", 0.0), ("rear", -WHEELBASE)):
    block = body.add_geom(
      name=f"{prefix}_{name}_block",
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=(BLOCK_LENGTH / 2, BOARD_WIDTH / 2, BLOCK_THICKNESS / 2),
      pos=(x, 0, block_top - BLOCK_THICKNESS / 2),
      mass=0.2 * BOARD_MASS,
      rgba=WOOD_RGBA,
    )
    _collide(block, BOARD_FRICTION)


def _hinge(body, name: str, pos=(0.0, 0.0, 0.0)) -> None:
  body.add_joint(
    name=name,
    type=mujoco.mjtJoint.mjJNT_HINGE,
    axis=(0, 1, 0),
    pos=pos,
    damping=JOINT_DAMPING,
    armature=1e-4,
  )


def pedalo_spec() -> mujoco.MjSpec:
  """The Pedalo Classic, front wheel centre at the origin, at rest on the
  ground when the origin is ``WHEEL_RADIUS`` up."""
  spec = mujoco.MjSpec()
  front = spec.worldbody.add_body(name=ROOT_BODY)
  front.add_freejoint(name="pedalo_free")
  _wheelset(front, "front")

  left = front.add_body(name="left_board", pos=(ECCENTRIC, BOARD_Y, 0))
  _hinge(left, "left_board_hinge")
  _board(left, "left_board")

  rear = left.add_body(name="rear_wheels", pos=(-WHEELBASE - ECCENTRIC, -BOARD_Y, 0))
  _hinge(rear, "rear_wheels_hinge", pos=(ECCENTRIC, BOARD_Y, 0))
  _wheelset(rear, "rear")

  right = front.add_body(name="right_board", pos=(-ECCENTRIC, -BOARD_Y, 0))
  _hinge(right, "right_board_hinge")
  _board(right, "right_board")

  # Loop closure: the right board's rear block rides the rear right axle. The
  # second anchor is computed from qpos0, where the two points coincide.
  spec.add_equality(
    name="right_board_rear_axle",
    type=mujoco.mjtEq.mjEQ_CONNECT,
    objtype=mujoco.mjtObj.mjOBJ_BODY,
    name1="right_board",
    name2="rear_wheels",
    data=[-WHEELBASE, 0, 0] + [0.0] * 8,
    solref=EQ_SOLREF,
    solimp=EQ_SOLIMP,
  )
  return spec


def crank_qpos(phi: float) -> dict[str, float]:
  """Hinge angles that put both wheelsets at crank angle ``phi`` (the root
  pitched by ``phi``) with the boards level."""
  return {"left_board_hinge": -phi, "right_board_hinge": -phi, "rear_wheels_hinge": phi}


def board_top_height(phi: float) -> tuple[float, float]:
  """Ground-to-top height of the (left, right) board at crank angle ``phi``.
  phi > 0 lowers the left board: a positive turn about +y takes +x towards
  -z, and the left axle starts at +x. Forward rolling is phi increasing."""
  # Rotation about +y by phi maps (e, 0) to (e cos phi, -e sin phi) in (x, z).
  dz = ECCENTRIC * math.sin(phi)
  base = WHEEL_RADIUS + BOARD_TOP_ABOVE_AXLE
  return base - dz, base + dz
