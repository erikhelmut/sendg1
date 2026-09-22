"""Scene pieces for the box-pushing task: the box, the target marker, the floor
friction priorities, the optional low-friction patches and the contact sensors.

Geometry and mass here are placeholders: every episode overwrites the box's
size, mass, inertia and friction, and the patches' size, position and friction,
from :mod:`sendg1.common.box_push.params`.
"""

from __future__ import annotations

import mujoco
from mjlab.entity import EntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from sendg1.common.box_push.params import (
  BOX_PRIORITY,
  FLOOR_PRIORITY,
  MAX_PATCHES,
  BoxPushRanges,
)

BOX_ENTITY = "box"
BOX_BODY = "box"
BOX_GEOM = "box_geom"
TARGET_ENTITY = "target"
TERRAIN_GEOM = "terrain"
"""mjlab's plane terrain: one body and one geom, both named ``terrain``."""


def patch_geom_name(i: int) -> str:
  return f"box_push_patch_{i}"


# Hands and forearms: the bodies that carry the elbow, wrist and hand collision
# capsules (G1 29-DoF MJCF).
HAND_BODIES: tuple[str, ...] = tuple(
  f"{side}_{link}"
  for side in ("left", "right")
  for link in ("elbow_link", "wrist_pitch_link", "wrist_yaw_link")
)

PUSH_FORCE_SENSOR = "robot_box_contact"
HAND_BOX_SENSOR = "hand_box_contact"
FEET_BOX_SENSOR = "feet_box_contact"


def box_spec() -> mujoco.MjSpec:
  """A free-floating box. Size and mass are the default ranges' midpoints until
  the first reset overwrites them."""
  ranges = BoxPushRanges()
  half = (
    0.25 * sum(ranges.box_depth),
    0.25 * sum(ranges.box_width),
    0.25 * sum(ranges.box_height),
  )
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name=BOX_BODY)
  body.add_freejoint(name="box_joint")
  body.add_geom(
    name=BOX_GEOM,
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=half,
    mass=0.5 * sum(ranges.box_mass),
    rgba=(0.72, 0.52, 0.30, 1.0),
    priority=BOX_PRIORITY,
    friction=(0.5 * sum(ranges.box_mu), 0.005, 0.0001),
  )
  return spec


def target_spec() -> mujoco.MjSpec:
  """A visual-only disc marking the target region. Group 2 and no collision,
  so neither physics nor any raycast sensor (all use group 0) sees it. Drawn at
  the default target radius; success is judged from the event's ranges."""
  radius = BoxPushRanges().target_radius
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="target", mocap=True)
  body.add_geom(
    name="target_disc",
    type=mujoco.mjtGeom.mjGEOM_CYLINDER,
    size=(radius, 0.002, 0),
    pos=(0, 0, 0.003),
    rgba=(0.1, 0.8, 0.3, 0.35),
    contype=0,
    conaffinity=0,
    group=2,
  )
  return spec


def box_entities() -> dict[str, EntityCfg]:
  # Named, argument-free spec functions: config fingerprints canonicalise a
  # callable by its qualified name, so a partial or lambda would not be stable.
  return {
    BOX_ENTITY: EntityCfg(spec_fn=box_spec),
    TARGET_ENTITY: EntityCfg(spec_fn=target_spec),
  }


def edit_scene_spec(spec: mujoco.MjSpec) -> None:
  """``SceneCfg.spec_fn`` without patches."""
  _edit_scene_spec(spec, low_friction_patches=False)


def edit_scene_spec_with_patches(spec: mujoco.MjSpec) -> None:
  """``SceneCfg.spec_fn`` with the low-friction patch geoms."""
  _edit_scene_spec(spec, low_friction_patches=True)


def _edit_scene_spec(spec: mujoco.MjSpec, *, low_friction_patches: bool) -> None:
  """``SceneCfg.spec_fn``: floor priority, and the patch geoms if enabled.

  Patches are geoms of the terrain body itself, not an entity of their own, so
  every sensor that matches ``terrain`` (foot contact, the tactile sole) sees a
  foot on a patch exactly as it sees a foot on the floor.
  """
  floor = spec.geom(TERRAIN_GEOM)
  if floor is None:
    raise RuntimeError("box-push needs mjlab's plane terrain (geom 'terrain')")
  floor.priority = FLOOR_PRIORITY
  if not low_friction_patches:
    return
  body = spec.body(TERRAIN_GEOM)
  for i in range(MAX_PATCHES):
    body.add_geom(
      name=patch_geom_name(i),
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=(0.3, 0.3, 0.005),
      pos=(0.0, 0.0, -1.0),  # parked below the floor until a reset places it
      rgba=(0.35, 0.55, 0.95, 1.0),
      priority=FLOOR_PRIORITY,
      friction=(0.2, 0.005, 0.0001),
    )


# ---------------------------------------------------------------------------
# Contact sensors
# ---------------------------------------------------------------------------


def robot_box_contact_sensor() -> ContactSensorCfg:
  """Net force of the whole robot on the box (logging, critic)."""
  return ContactSensorCfg(
    name=PUSH_FORCE_SENSOR,
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="body", pattern=BOX_BODY, entity=BOX_ENTITY),
    fields=("found", "force"),
    reduce="netforce",  # net force, always in the world frame
    num_slots=1,
  )


def hand_box_contact_sensor() -> ContactSensorCfg:
  """Strongest contact of each hand/forearm body with the box, with its
  position, so the reward can tell which face was touched."""
  return ContactSensorCfg(
    name=HAND_BOX_SENSOR,
    primary=ContactMatch(mode="body", pattern=HAND_BODIES, entity="robot"),
    secondary=ContactMatch(mode="body", pattern=BOX_BODY, entity=BOX_ENTITY),
    fields=("found", "force", "pos"),  # pos is always in the world frame
    reduce="maxforce",
    num_slots=1,
  )


def feet_box_contact_sensor() -> ContactSensorCfg:
  """Either foot touching the box (the kick penalty)."""
  return ContactSensorCfg(
    name=FEET_BOX_SENSOR,
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern=BOX_BODY, entity=BOX_ENTITY),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
  )


def box_push_sensors() -> tuple[ContactSensorCfg, ...]:
  return (
    robot_box_contact_sensor(),
    hand_box_contact_sensor(),
    feet_box_contact_sensor(),
  )
