"""Per-foot tactile features for beam walking, on top of the existing taxel model.

The physics -> taxel step is ``sendg1.common.tactile.sensing.read_tactile``
unchanged: per-contact forces give net force and CoP, 4 rays per taxel give
which patches have support, and a Winkler plane splits the normal force over
the supported taxels. This module only decides what a policy gets to see of it.

Representation (per foot, 33 dims; both feet 66)
------------------------------------------------
``shape[26]``   per-taxel normal force divided by the foot's total normal force:
                WHERE the load is, independent of how much. Under a global gain
                error (an uncalibrated sensor) these do not move at all.
``load``        total normal force / body weight (343 N), clipped to [0, 1.5]:
                HOW MUCH, the one quantity a gain error does move.
``cop[2]``      centre of pressure from the taxel readings (not the simulator's
                exact CoP -- the real sensor can only compute it this way),
                x and y normalised to the sole's half extents.
``area``        fraction of taxels above 2 N: how much of the sole is on support.
``edge[2]``     share of the load on the lateral and on the medial edge taxels
                (|y| >= 15 mm). On a beam narrower than the sole these are the
                direct signature of a foot hanging over an edge, and of which one.
``contact``     total normal force above 15 N.

Normal force only. The simulated shear is the net tangential force split in
proportion to normal load, so it carries no spatial information the shape does
not, and it is the channel least likely to survive contact with real hardware.

Both feet in one frame. The slipper layout is mirror-symmetric in y, so the
right foot's taxels are re-indexed into the left foot's mirror image and its
y features flipped: for both feet +y is lateral. One layer of weights (or one
encoder, if the actor grows one) then serves both feet.

Why hand-designed features rather than a learned encoder: the derived terms are
physical quantities that are invariant or graceful under the sensor faults we
randomise (gain, dead taxels, noise), and they are what makes the feature
vector small enough for a GRU actor. The per-taxel shape is kept alongside,
so a network can still find patterns the derived terms miss.

Sensor model (actor only; the critic's copy is noise-free)
---------------------------------------------------------------------
Per episode: a per-foot gain, a per-taxel gain, dead taxels (read 0). Per step:
additive Gaussian noise on each taxel, and a frame drop that holds a foot's
previous reading. Everything scales with :func:`noise_strength` of the env's
curriculum level.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING

import torch

from sendg1.common.tactile.layout import LAYOUTS, SIDES, TaxelLayout
from sendg1.common.tactile.sensing import TactileParams, read_tactile

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

BODY_WEIGHT_N = 343.0
"""Nominal G1 weight (35 kg). A known constant on the real robot."""
CONTACT_N = 15.0
TAXEL_ON_N = 2.0
EDGE_Y = 0.0145
"""Taxels with |y| at or beyond this (15 mm rows and the 25-29 mm outriggers)
count as edge taxels."""

FEATURES_PER_FOOT_EXTRA = 7  # load, cop x/y, area, edge lat/med, contact
CONTACT_MAP_RAMP_N = (2.0, 6.0)
"""Soft contact map: a taxel reads 0 at or below 2 N and 1 at or above 6 N."""


@dataclass(frozen=True)
class TactileNoiseCfg:
  """Sensor-fault ranges at noise strength 1.0 (they scale linearly)."""

  foot_gain: float = 0.20
  """Per-foot gain drawn from U(1 - g, 1 + g) per episode."""
  taxel_gain: float = 0.25
  """Per-taxel gain drawn from U(1 - g, 1 + g) per episode."""
  dead_prob: float = 0.08
  """Probability that a taxel reads zero for the whole episode."""
  noise_std_n: float = 3.0
  """Additive Gaussian noise per taxel per step (N)."""
  drop_prob: float = 0.05
  """Per step and foot, probability of repeating the previous frame."""


STRENGTH_RANGE = (0.3, 1.0)
"""Noise strength at the lowest and at the highest curriculum level."""


def noise_strength(level: torch.Tensor, num_levels: int) -> torch.Tensor:
  lo, hi = STRENGTH_RANGE
  return lo + (hi - lo) * level.float() / max(num_levels - 1, 1)


@lru_cache(maxsize=4)
def mirror_permutation(layout: TaxelLayout) -> tuple[int, ...]:
  """``perm[j]`` = index of the taxel at the mirror image (x, -y) of taxel j."""
  perm = []
  for x, y in layout.centers:
    d = [abs(x - x2) + abs(-y - y2) for x2, y2 in layout.centers]
    j = min(range(len(d)), key=d.__getitem__)
    if d[j] > 1e-4:
      raise ValueError(f"layout {layout.name} is not mirror-symmetric in y")
    perm.append(j)
  return tuple(perm)


@lru_cache(maxsize=4)
def _layout_consts(layout: TaxelLayout, device: str) -> tuple[torch.Tensor, ...]:
  centers = torch.tensor(layout.centers, device=device, dtype=torch.float32)  # [T, 2]
  if layout.mirror_right:
    # The right foot already carries a mirrored part; indices line up.
    perm = torch.arange(layout.num_taxels, device=device)
  else:
    perm = torch.tensor(mirror_permutation(layout), device=device)
  xs, ys = centers[:, 0], centers[:, 1]
  x_mid = 0.5 * (xs.max() + xs.min())
  x_half = 0.5 * (xs.max() - xs.min())
  y_half = ys.abs().max()
  lateral = (ys >= EDGE_Y).float()
  medial = (ys <= -EDGE_Y).float()
  return centers, perm, x_mid, x_half, y_half, lateral, medial


def canonical_normals(env: ManagerBasedRlEnv, params: TactileParams) -> torch.Tensor:
  """[B, 2, T] taxel normal forces (N), right foot re-indexed to the mirror
  image of the left, so taxel j on both feet sits at (x_j, +/- y_j) with +y
  lateral."""
  layout = LAYOUTS[params.layout]
  reading = read_tactile(env, params)
  normal = reading.normal
  _, perm, *_ = _layout_consts(layout, str(normal.device))
  return torch.stack([normal[:, 0], normal[:, 1, perm]], dim=1)


def features_from_normals(
  normal: torch.Tensor, layout: TaxelLayout, contact_map: bool = False
) -> torch.Tensor:
  """[B, 2, T] canonical normals -> [B, 2 * (T + 7)] features.

  ``contact_map``: the T per-taxel entries are a soft 0-1 "this taxel touches"
  map (``CONTACT_MAP_RAMP_N``) instead of each taxel's share of the foot's
  load. Which taxels touch is the beam edge's footprint under the sole, and a
  thresholded map is barely moved by gain errors."""
  centers, _, x_mid, x_half, y_half, lateral, medial = _layout_consts(
    layout, str(normal.device)
  )
  total = normal.sum(-1, keepdim=True)  # [B, 2, 1]
  # Fades to zero below CONTACT_N rather than amplifying noise on an unloaded foot.
  denom = total.clamp(min=CONTACT_N)
  shape = normal / denom
  load = (total / BODY_WEIGHT_N).clamp(0.0, 1.5)
  w = normal / total.clamp(min=1e-6)
  cop_x = ((w * centers[:, 0]).sum(-1, keepdim=True) - x_mid) / x_half
  cop_y = (w * centers[:, 1]).sum(-1, keepdim=True) / y_half
  in_contact = (total > CONTACT_N).float()
  cop_x, cop_y = cop_x * in_contact, cop_y * in_contact
  area = (normal > TAXEL_ON_N).float().mean(-1, keepdim=True)
  edge_lat = (shape * lateral).sum(-1, keepdim=True)
  edge_med = (shape * medial).sum(-1, keepdim=True)
  if contact_map:
    lo, hi = CONTACT_MAP_RAMP_N
    shape = ((normal - lo) / (hi - lo)).clamp(0.0, 1.0)
  per_foot = torch.cat(
    [shape, load, cop_x, cop_y, area, edge_lat, edge_med, in_contact], dim=-1
  )
  return per_foot.flatten(start_dim=1)


def cop_offset_from_support(env: ManagerBasedRlEnv, params: TactileParams) -> torch.Tensor:
  """[B, 2] lateral distance (m) of the true CoP from the centroid of the
  supported part of each sole -- where the foot actually rests on the beam.
  Zero for a foot without support. For the CoP-centring reward."""
  layout = LAYOUTS[params.layout]
  reading = read_tactile(env, params)
  device = str(reading.normal.device)
  centers = torch.tensor([layout.for_side(s) for s in SIDES], device=device)  # [2, T, 2]
  s = reading.support  # [B, 2, T]
  s_tot = s.sum(-1)
  centroid_y = (s * centers[None, :, :, 1]).sum(-1) / s_tot.clamp(min=1e-6)
  off = reading.cop[..., 1] - centroid_y
  loaded = (reading.net_force[..., 2] > CONTACT_N) & (s_tot > 1e-3)
  return torch.where(loaded, off, torch.zeros_like(off))


# ---------------------------------------------------------------------------
# Sensor fault state
# ---------------------------------------------------------------------------


class TactileNoiseState:
  """Per-env sensor faults, resampled at reset by ``reset_tactile_noise``."""

  _ATTR = "_sendg1_beam_tactile_noise"

  def __init__(self, env: ManagerBasedRlEnv, layout: TaxelLayout) -> None:
    n, t, dev = env.num_envs, layout.num_taxels, env.device
    self.strength = torch.zeros(n, device=dev)
    self.foot_gain = torch.ones(n, 2, 1, device=dev)
    self.taxel_gain = torch.ones(n, 2, t, device=dev)
    self.alive = torch.ones(n, 2, t, device=dev)
    self.prev = torch.zeros(n, 2, t, device=dev)
    self.cfg = TactileNoiseCfg()

  @classmethod
  def get(cls, env: ManagerBasedRlEnv, layout: TaxelLayout) -> TactileNoiseState:
    st = getattr(env, cls._ATTR, None)
    if st is None:
      st = cls(env, layout)
      setattr(env, cls._ATTR, st)
    return st

  def resample(self, env_ids: torch.Tensor, strength: torch.Tensor) -> None:
    c = self.cfg
    k = len(env_ids)
    s = strength[:, None, None]
    dev = strength.device
    t = self.taxel_gain.shape[-1]
    self.strength[env_ids] = strength
    self.foot_gain[env_ids] = 1.0 + c.foot_gain * s * (2 * torch.rand(k, 2, 1, device=dev) - 1)
    self.taxel_gain[env_ids] = 1.0 + c.taxel_gain * s * (2 * torch.rand(k, 2, t, device=dev) - 1)
    self.alive[env_ids] = (torch.rand(k, 2, t, device=dev) >= c.dead_prob * s).float()
    self.prev[env_ids] = 0.0

  def apply(self, normal: torch.Tensor) -> torch.Tensor:
    c = self.cfg
    s = self.strength[:, None, None]
    noisy = normal * self.foot_gain * self.taxel_gain
    noisy = noisy + c.noise_std_n * s * torch.randn_like(noisy)
    noisy = noisy.clamp(min=0.0) * self.alive
    drop = torch.rand(noisy.shape[:2], device=noisy.device) < c.drop_prob * s[..., 0]
    noisy = torch.where(drop[..., None], self.prev, noisy)
    self.prev = noisy.detach().clone()
    return noisy
