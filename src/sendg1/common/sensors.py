"""Scene sensors for the G1 locomotion tasks.

The height scans and ground-contact sensors serve the critic and the reward
terms; the policies in this study are blind. The tactile sensors at the bottom
feed only the ``tactile_array`` observation group, and are added to a scene only
when its observation set uses that group.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import torch
from mjlab.sensor import (
  ContactMatch,
  ContactSensorCfg,
  GridPatternCfg,
  ObjRef,
  RayCastSensorCfg,
  RingPatternCfg,
  TerrainHeightSensorCfg,
)

from sendg1.common.robots.g1 import G1_FOOT_SITES, G1_PELVIS_BODY
from sendg1.common.tactile.layout import SIDES, SOLE_Z, TaxelLayout

TERRAIN_SCAN_MAX_DISTANCE = 5.0


def terrain_scan_sensor() -> RayCastSensorCfg:
  """Pelvis-mounted terrain height grid. Rough terrain only, critic only."""
  return RayCastSensorCfg(
    name="terrain_scan",
    frame=ObjRef(type="body", name=G1_PELVIS_BODY, entity="robot"),
    ray_alignment="yaw",
    pattern=GridPatternCfg(size=(1.6, 1.0), resolution=0.1),
    max_distance=TERRAIN_SCAN_MAX_DISTANCE,
    exclude_parent_body=True,
    include_geom_groups=(0,),
    debug_vis=True,
  )


def foot_height_sensor() -> TerrainHeightSensorCfg:
  """Per-foot clearance above terrain. Feeds foot_clearance / foot_swing_height
  rewards and the critic's foot_height observation."""
  return TerrainHeightSensorCfg(
    name="foot_height_scan",
    frame=tuple(ObjRef(type="site", name=s, entity="robot") for s in G1_FOOT_SITES),
    ray_alignment="yaw",
    pattern=RingPatternCfg.single_ring(radius=0.03, num_samples=6),
    max_distance=1.0,
    exclude_parent_body=True,
    include_geom_groups=(0,),
    debug_vis=True,
    viz=TerrainHeightSensorCfg.VizCfg(
      show_rays=True,
      hit_color=(1.0, 0.0, 1.0, 0.8),
      hit_sphere_color=(1.0, 0.0, 1.0, 1.0),
    ),
  )


def feet_ground_contact_sensor() -> ContactSensorCfg:
  """Net contact wrench per foot against the terrain.

  ``reduce="netforce"`` collapses all simultaneous contacts on a foot into one
  3D net force, i.e. an idealised load cell -- there is no spatial distribution
  over the sole. That distinction is the whole point of the tactile array, so
  the array reads a SEPARATE sensor (``sole_contacts`` below), not a
  reconfiguration of this one. This sensor stays exactly as-is so the critic
  and the reward terms never move underneath the study.
  """
  return ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )


def self_collision_sensor() -> ContactSensorCfg:
  """Pelvis-subtree self contacts, for the self_collisions penalty."""
  return ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )


# ---------------------------------------------------------------------------
# Tactile array (see sendg1.common.tactile)
# ---------------------------------------------------------------------------

SOLE_CONTACT_SENSOR = "sole_contacts"
SOLE_CONTACT_SLOTS = 16
"""Contacts kept per foot. Flat ground makes at most 14 (2 per capsule); the
strongest are kept if an uneven surface makes more."""

TAXEL_RAY_LIFT = 0.02
"""Taxel rays start this far above the sole plane, so a sole that has sunk
slightly into the ground still sees it."""


def taxel_ray_sensor_name(side: str) -> str:
  return f"tactile_rays_{side}"


@dataclass
class TaxelRayPatternCfg:
  """Parallel rays straight down the sole normal, from explicit sole points."""

  points: tuple[tuple[float, float], ...]
  z: float

  def generate_rays(
    self, mj_model: mujoco.MjModel | None, device: str
  ) -> tuple[torch.Tensor, torch.Tensor]:
    del mj_model
    offsets = torch.tensor(
      [(x, y, self.z) for x, y in self.points], device=device, dtype=torch.float32
    )
    directions = torch.zeros_like(offsets)
    directions[:, 2] = -1.0
    return offsets, directions


def sole_contact_sensor() -> ContactSensorCfg:
  """Every individual contact between each foot and the terrain.

  Unlike ``feet_ground_contact`` this keeps contacts separate (position and
  force of each), which the tactile model needs to recover the centre of
  pressure. Forces are rotated to the world frame.
  """
  return ContactSensorCfg(
    name=SOLE_CONTACT_SENSOR,
    primary=ContactMatch(
      mode="body",
      pattern=tuple(f"{side}_ankle_roll_link" for side in SIDES),
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force", "pos", "normal", "tangent"),
    reduce="maxforce",
    num_slots=SOLE_CONTACT_SLOTS,
    global_frame=True,
  )


def taxel_ray_sensor(side: str, layout: TaxelLayout, per_axis: int) -> RayCastSensorCfg:
  """Rays down from ``per_axis``^2 points on every taxel of one sole, to find
  which patches have ground beneath them."""
  return RayCastSensorCfg(
    name=taxel_ray_sensor_name(side),
    frame=ObjRef(type="body", name=f"{side}_ankle_roll_link", entity="robot"),
    pattern=TaxelRayPatternCfg(
      points=layout.sample_points(side, per_axis),  # type: ignore[arg-type]
      z=SOLE_Z + TAXEL_RAY_LIFT,
    ),
    ray_alignment="base",
    max_distance=TAXEL_RAY_LIFT + 0.05,
    exclude_parent_body=True,
    include_geom_groups=(0,),
  )


def tactile_sensors(layout: TaxelLayout, per_axis: int) -> tuple:
  """All scene sensors the tactile model reads."""
  return (
    sole_contact_sensor(),
    *(taxel_ray_sensor(side, layout, per_axis) for side in SIDES),
  )
