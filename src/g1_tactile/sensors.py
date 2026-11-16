"""Scene sensors the tactile model reads (``g1_tactile.sensing``).

``sole_contacts`` keeps every individual foot contact (position and force),
from which the model recovers the net force and centre of pressure; the
``tactile_rays_{left,right}`` raycasts find which taxels have ground beneath
them.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import torch
from mjlab.sensor import (
  ContactMatch,
  ContactSensorCfg,
  ObjRef,
  RayCastSensor,
  RayCastSensorCfg,
)
from mujoco_warp._src.ray import rays

from g1_tactile.layout import SIDES, SOLE_Z, TaxelLayout

SOLE_CONTACT_SENSOR = "sole_contacts"
SOLE_CONTACT_SLOTS = 16
"""Contacts kept per foot. Flat ground makes 4 (the sole box's corners); the
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


TERRAIN_MATCH = ContactMatch(mode="body", pattern="terrain")


def sole_contact_sensor(secondary: ContactMatch | None = TERRAIN_MATCH) -> ContactSensorCfg:
  """Every individual contact between each foot and the terrain (or whatever
  ``secondary`` names the feet stand on, e.g. a moving platform).

  Contacts are kept separate (position and force of each), not reduced to a
  net force, because the tactile model needs them to recover the centre of
  pressure. Forces are rotated to the world frame.
  """
  return ContactSensorCfg(
    name=SOLE_CONTACT_SENSOR,
    primary=ContactMatch(
      mode="body",
      pattern=tuple(f"{side}_ankle_roll_link" for side in SIDES),
      entity="robot",
    ),
    secondary=secondary,
    fields=("found", "force", "pos", "normal", "tangent"),
    reduce="maxforce",
    num_slots=SOLE_CONTACT_SLOTS,
    global_frame=True,
  )


class _GeomOnlyRayCastSensor(RayCastSensor):
  """Casts without the render context's BVH. mujoco_warp 3.11's BVH ray kernel
  reads out of bounds when the model has a flex (its geom and flex leaves are
  laid out with different strides; not re-checked on 4.0); the plain kernel
  tests geoms only, which is all the taxel rays need."""

  def raycast_kernel(self, rc) -> None:
    rays(
      m=self._model.struct,  # type: ignore[attr-defined]
      d=self._data.struct,  # type: ignore[attr-defined]
      pnt=self._ray_pnt,  # type: ignore[invalid-argument-type]
      vec=self._ray_vec,  # type: ignore[invalid-argument-type]
      geomgroup=self._geomgroup,  # pyright: ignore[reportArgumentType]
      flg_static=True,
      bodyexclude=self._ray_bodyexclude,
      dist=self._ray_dist,  # type: ignore[invalid-argument-type]
      geomid=self._ray_geomid,  # type: ignore[invalid-argument-type]
      normal=self._ray_normal,  # type: ignore[invalid-argument-type]
    )


@dataclass(kw_only=True)
class GeomOnlyRayCastSensorCfg(RayCastSensorCfg):
  def build(self) -> RayCastSensor:
    return _GeomOnlyRayCastSensor(self)


def taxel_ray_sensor(
  side: str, layout: TaxelLayout, per_axis: int, geom_only: bool = False
) -> RayCastSensorCfg:
  """Rays down from ``per_axis``^2 points on every taxel of one sole, to find
  which patches have ground beneath them. ``geom_only``: skip the BVH (needed
  when the scene has a flex, see ``_GeomOnlyRayCastSensor``)."""
  cls = GeomOnlyRayCastSensorCfg if geom_only else RayCastSensorCfg
  return cls(
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


def tactile_sensors(
  layout: TaxelLayout,
  per_axis: int,
  secondary: ContactMatch | None = TERRAIN_MATCH,
  geom_only_rays: bool = False,
) -> tuple:
  """All scene sensors the tactile model reads. ``secondary``: what the soles
  touch (``None``: anything); the taxel rays see every group-0 geom anyway."""
  return (
    sole_contact_sensor(secondary),
    *(taxel_ray_sensor(side, layout, per_axis, geom_only_rays) for side in SIDES),
  )
