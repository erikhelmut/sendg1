"""Run the tactile model on a live env: scene sensors in, taxel forces out.

Reads the ``sole_contacts`` contact sensor and the two ``tactile_rays_*``
raycast sensors (see ``sendg1.common.sensors.tactile_sensors``), moves
everything into each foot's sole frame and hands it to
``sendg1.common.tactile.model``.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING

import torch
from mjlab.utils.lab_api.math import matrix_from_quat

from sendg1.common.sensors import (
  SOLE_CONTACT_SENSOR,
  SOLE_CONTACT_SLOTS,
  TAXEL_RAY_LIFT,
  tactile_sensors,
  taxel_ray_sensor_name,
)
from sendg1.common.tactile import model
from sendg1.common.tactile.layout import DEFAULT_LAYOUT, LAYOUTS, SIDES, TaxelLayout

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


@dataclass(frozen=True)
class TactileParams:
  """Sensor-model settings. Kept separate from the layout: the layout is the
  hardware, these are modelling choices."""

  layout: str = DEFAULT_LAYOUT
  rays_per_axis: int = 2
  """Support is sampled on a rays_per_axis^2 grid within each patch."""
  compliance_depth: float = 0.004
  """Ground this far below a patch still counts as (partly) supporting it:
  the give of the slipper and sensor. Linear fall-off to zero."""


def tactile_scene_sensors(params: TactileParams = TactileParams()) -> tuple:
  """The scene sensors ``read_tactile`` needs, configured for ``params``."""
  return tactile_sensors(LAYOUTS[params.layout], params.rays_per_axis)


@lru_cache(maxsize=8)
def _layout_tensors(layout: TaxelLayout, device: str) -> tuple[torch.Tensor, ...]:
  centers = torch.tensor(
    [layout.for_side(s) for s in SIDES], device=device, dtype=torch.float32
  )  # [2, T, 2]
  areas = torch.tensor(layout.areas(), device=device, dtype=torch.float32)
  return centers, areas


def read_tactile(env: ManagerBasedRlEnv, params: TactileParams) -> model.TactileReading:
  """Taxel forces for every env. Leading dims of every field are ``[B, 2]``,
  feet ordered (left, right)."""
  layout = LAYOUTS[params.layout]
  robot = env.scene["robot"]
  device = str(robot.data.body_link_pos_w.device)
  body_ids, _ = robot.find_bodies([f"{s}_ankle_roll_link" for s in SIDES], preserve_order=True)
  foot_pos = robot.data.body_link_pos_w[:, body_ids]  # [B, 2, 3]
  foot_rot = matrix_from_quat(robot.data.body_link_quat_w[:, body_ids])  # [B, 2, 3, 3]
  batch = foot_pos.shape[0]

  # -- contacts, into each sole frame ----------------------------------------
  contacts = env.scene[SOLE_CONTACT_SENSOR].data
  assert contacts.pos is not None and contacts.force is not None
  assert contacts.normal is not None
  shape = (batch, len(SIDES), SOLE_CONTACT_SLOTS, 3)
  pos_w = contacts.pos.view(shape)
  # MuJoCo reports the force the primary (foot) exerts on the secondary
  # (ground). The sole feels the reaction.
  force_w = -contacts.force.view(shape)
  valid = contacts.normal.view(shape).norm(dim=-1) > 1e-8

  rot_t = foot_rot.transpose(-1, -2).unsqueeze(2)  # [B, 2, 1, 3, 3]
  pos_s = (rot_t @ (pos_w - foot_pos.unsqueeze(2)).unsqueeze(-1)).squeeze(-1)
  force_s = (rot_t @ force_w.unsqueeze(-1)).squeeze(-1)
  net_force, cop = model.net_force_and_cop(pos_s, force_s, valid)

  # -- support under each patch ----------------------------------------------
  per_taxel = params.rays_per_axis**2
  gaps = []
  for side in SIDES:
    dist = env.scene[taxel_ray_sensor_name(side)].data.distances  # [B, T*S]
    gap = torch.where(dist >= 0, dist - TAXEL_RAY_LIFT, torch.full_like(dist, torch.inf))
    gaps.append(gap.view(batch, layout.num_taxels, per_taxel))
  sample_support = model.support_from_gaps(torch.stack(gaps, 1), params.compliance_depth)
  support = sample_support.mean(-1)  # [B, 2, T]

  # -- split -----------------------------------------------------------------
  centers, areas = _layout_tensors(layout, device)
  normal, shear = model.distribute(
    net_force,
    cop,
    support,
    centers.unsqueeze(0),
    areas,
    fallback_sigma=min(min(h) for h in layout.half_sizes) * 2,
  )
  return model.TactileReading(
    normal=normal, shear=shear, support=support, net_force=net_force, cop=cop
  )


def tactile_array_obs(
  env: ManagerBasedRlEnv,
  layout: str = TactileParams.layout,
  rays_per_axis: int = TactileParams.rays_per_axis,
  compliance_depth: float = TactileParams.compliance_depth,
) -> torch.Tensor:
  """Observation: per foot, [normal(T), shear_x(T), shear_y(T)], each
  ``sign(x) * log1p(|x|)`` of newtons -- the same compression mjlab applies
  to its net contact-force observation. Shape [B, 2 * 3 * T]."""
  reading = read_tactile(env, TactileParams(layout, rays_per_axis, compliance_depth))
  per_foot = torch.cat(
    [reading.normal, reading.shear[..., 0], reading.shear[..., 1]], dim=-1
  )  # [B, 2, 3T]
  flat = per_foot.flatten(start_dim=1)
  return torch.sign(flat) * torch.log1p(flat.abs())
