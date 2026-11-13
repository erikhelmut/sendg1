"""The Pedalo Sport (Holz-Hoerz art. 551001) as an MjSpec.

Source: the 2025 system catalogue (pedalo-original-pedalo-system-de-en-25-2.pdf,
p. 9), which gives: turnable steps ("drehbare Trittflächen") of 6 x 14 cm
each, rated to 120 kg, 37 x 22 x 22 cm, 2.6 kg. The Classic's catalogue entry
says it converts into two Pedalo Sport ("Umbau zu zwei Pedalo Sport
möglich"), and the conversion picture shows how: each Classic wheelset,
unchanged, becomes one Sport, and the Classic's four bearing blocks
(6 x 14.5 cm, 511028) become the four steps. So wheels, axles, crank radius
and block size are the Classic's (``model``); only the boards are gone.

Kinematics
----------
One wheelset (outer-left wheel, left eccentric axle, middle wheel, right
eccentric axle, outer-right wheel) and two steps, each a block that turns
freely on its axle. Nothing keeps a step level but the foot on it, and
nothing keeps the pedalo upright fore-aft but the rider: the wheels touch the
ground along one line, so rider and pedalo balance like an inverted pendulum
on a wheel, and each foot stands across a 6 cm block that tips as soon as its
centre of pressure leaves the axle. That is why the catalogue calls the
Classic (1970) the easier one.

MuJoCo tree (one freejoint, two hinges, no loop)::

  wheels (free) --hinge--> left_step
                --hinge--> right_step

Frames at qpos0 as in ``model``: x forward, z up, phi = 0 (left axle ahead
of the wheel centre, right axle behind, steps level).
"""

from __future__ import annotations

import math

import mujoco

from sendg1.common.pedalo import model as pm

# -- dimensions (m) ---------------------------------------------------------

STEP_LENGTH = pm.BLOCK_LENGTH
"""6 cm along the direction of travel (catalogue: "je 6x14 cm")."""
STEP_WIDTH = pm.BOARD_WIDTH
"""14.5 cm across: the Classic's block (511028); the catalogue rounds to 14."""
STEP_THICKNESS = pm.BLOCK_THICKNESS
STEP_TOP_ABOVE_AXLE = pm.BOARD_TOP_ABOVE_AXLE - pm.BOARD_THICKNESS
"""The axle runs through the block's middle, as under a Classic board: step
tops run from 7 cm to 18 cm above the ground."""
LENGTH = 2 * pm.WHEEL_RADIUS
"""22 cm: the wheels are the whole length (catalogue: 37 x 22 x 22 cm)."""

# -- masses (kg): 2.6 kg in all ---------------------------------------------

STEP_MASS = 0.1
"""Bare birch block; with the Classic's wheels (0.6) and axles (0.3) this
makes the catalogue's 2.6 kg."""
TOTAL_MASS = 3 * pm.WHEEL_MASS + 2 * pm.AXLE_MASS + 2 * STEP_MASS

# -- contact ------------------------------------------------------------------

STEP_FRICTION = 0.6
"""Bare plywood, no grip layer (estimated). Below the foot DR range, so the
foot friction randomisation sets the contact (MuJoCo takes the larger)."""

# -- names --------------------------------------------------------------------

ROOT_BODY = "wheels"
STEP_BODIES = ("left_step", "right_step")
STEP_GEOMS = ("left_step_block", "right_step_block")
HINGE_JOINTS = ("left_step_hinge", "right_step_hinge")


def _step(body, name: str) -> None:
  """A block turning on its axle, in a frame at the axle."""
  block = body.add_geom(
    name=f"{name}_block",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=(STEP_LENGTH / 2, STEP_WIDTH / 2, STEP_THICKNESS / 2),
    pos=(0, 0, STEP_TOP_ABOVE_AXLE - STEP_THICKNESS / 2),
    mass=STEP_MASS,
    rgba=pm.WOOD_RGBA,
  )
  pm._collide(block, STEP_FRICTION)
  # The block's end grain, so its turning is visible.
  for x in (-1.0, 1.0):
    edge = body.add_geom(
      name=f"{name}_edge_{'front' if x > 0 else 'rear'}",
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=(0.002, STEP_WIDTH / 2 + 0.0005, STEP_THICKNESS / 2 + 0.0005),
      pos=(x * (STEP_LENGTH / 2 - 0.0015), 0, STEP_TOP_ABOVE_AXLE - STEP_THICKNESS / 2),
      rgba=(0.70, 0.55, 0.36, 1.0),
    )
    pm._visual(edge)


def pedalo_spec() -> mujoco.MjSpec:
  """The Pedalo Sport, wheel centre at the origin, at rest on the ground when
  the origin is ``WHEEL_RADIUS`` up."""
  spec = mujoco.MjSpec()
  wheels = spec.worldbody.add_body(name=ROOT_BODY)
  wheels.add_freejoint(name="pedalo_free")
  pm._wheelset(wheels, "sport")
  for name, sign in (("left", 1.0), ("right", -1.0)):
    step = wheels.add_body(name=f"{name}_step", pos=(sign * pm.ECCENTRIC, sign * pm.BOARD_Y, 0))
    pm._hinge(step, f"{name}_step_hinge")
    _step(step, f"{name}_step")
  return spec


def crank_qpos(phi: float) -> dict[str, float]:
  """Hinge angles that level both steps with the wheelset at crank angle ``phi``."""
  return {"left_step_hinge": -phi, "right_step_hinge": -phi}


def step_top_height(phi: float) -> tuple[float, float]:
  """Ground-to-top height of the (left, right) step at crank angle ``phi``,
  steps level (``model.board_top_height``)."""
  dz = pm.ECCENTRIC * math.sin(phi)
  base = pm.WHEEL_RADIUS + STEP_TOP_ABOVE_AXLE
  return base - dz, base + dz
