"""The rolling cylinder as an MjSpec: a rigid roller lying free on the floor.

The robot stands across its top, both soles at right angles to its axis, and
it rolls fore-aft under them. Unlike a rola-bola there is no board: each sole
touches the cylinder along one line, so where that line falls under the sole
is the balance cue, and it is exactly what the tactile slipper resolves and
the joints only infer.

The numbers are chosen, not sourced: a 30 cm roller is the size of a large
balance roller or a section of plastic drain pipe filled to 8 kg, and 80 cm
leaves ~28 cm of cylinder beyond either sole in the G1's default stance.
Radius is the main difficulty knob: the support line runs ahead of the sole
by R * pitch, and the cylinder top moves at twice the axle's speed.

Frame: x forward (the direction it rolls), y along the axis, z up; origin on
the axis, at rest on the ground when the origin is ``RADIUS`` up.
"""

from __future__ import annotations

import math

import mujoco

RADIUS = 0.15
LENGTH = 0.80
MASS = 8.0
"""Solid-cylinder inertia (MuJoCo computes it from the geom): I = m R^2 / 2."""

FRICTION = 0.9
"""Grip-taped or rubber-coated roller, on the floor and under the soles alike."""
CONTYPE = 2
CONAFFINITY = 1
"""Collides with the floor and the robot (as the pedalo does)."""

ROOT_BODY = "cylinder"
GEOM = "cylinder_shell"

SHELL_RGBA = (0.16, 0.19, 0.24, 1.0)
STRIPE_RGBA = (0.298, 0.788, 0.941, 1.0)
"""Kyber accent (#4cc9f0), so rolling is visible."""
END_RGBA = (0.98, 0.75, 0.33, 1.0)
"""#fac055 end caps: the axis direction reads at a glance."""

_Y_AXIS_CYL = (math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0)
"""Turns a cylinder's axis (z) onto y."""


def _visual(geom) -> None:
  geom.contype = 0
  geom.conaffinity = 0
  geom.mass = 0.0
  geom.group = 1


def cylinder_spec(radius: float = RADIUS, length: float = LENGTH, mass: float = MASS) -> mujoco.MjSpec:
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name=ROOT_BODY)
  body.add_freejoint(name="cylinder_free")
  shell = body.add_geom(
    name=GEOM,
    type=mujoco.mjtGeom.mjGEOM_CYLINDER,
    size=(radius, length / 2, 0),
    quat=_Y_AXIS_CYL,
    mass=mass,
    rgba=SHELL_RGBA,
  )
  shell.contype = CONTYPE
  shell.conaffinity = CONAFFINITY
  shell.friction = (FRICTION, 0.005, 0.0001)
  shell.group = 0
  # Two stripes along the axis, half a turn apart, just proud of the shell.
  for name, angle in (("a", 0.0), ("b", math.pi)):
    stripe = body.add_geom(
      name=f"cylinder_stripe_{name}",
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=(0.012, length / 2 - 0.01, 0.002),
      pos=(radius * math.sin(angle), 0.0, radius * math.cos(angle)),
      quat=(math.cos(angle / 2), 0.0, math.sin(angle / 2), 0.0),
      rgba=STRIPE_RGBA,
    )
    _visual(stripe)
  for name, y in (("left", 1.0), ("right", -1.0)):
    cap = body.add_geom(
      name=f"cylinder_cap_{name}",
      type=mujoco.mjtGeom.mjGEOM_CYLINDER,
      size=(radius - 0.01, 0.003, 0),
      pos=(0.0, y * (length / 2 + 0.001), 0.0),
      quat=_Y_AXIS_CYL,
      rgba=END_RGBA,
    )
    _visual(cap)
  return spec
