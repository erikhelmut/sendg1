"""Scene sensors for the G1 locomotion tasks.

All of these are simulation sensors. None of them feed the actor: the height
scans and contact sensors exist to serve the critic and the reward terms. The
policies in this study are blind.
"""

from __future__ import annotations

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
  when the 26-taxel sensor is modelled it will be a SEPARATE sensor over the
  per-foot collision geoms (see G1_FOOT_GEOMS), not a reconfiguration of this
  one. This sensor stays exactly as-is so the critic and the reward terms never
  move underneath the study.
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
