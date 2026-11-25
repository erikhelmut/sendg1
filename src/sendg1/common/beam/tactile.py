"""What a policy sees of the tactile slipper: the touch features every tactile
actor gets (``touch_features``), and the sensor faults applied first.

The physics -> taxel step is ``sendg1.common.tactile.sensing.read_tactile``:
per-contact forces give net force and CoP, 4 rays per taxel give which patches
have support, and a Winkler plane splits the normal force over the supported
taxels. This module only decides what a policy gets to see of it.

Representation (per foot, 29 dims; both feet 58)
------------------------------------------------
``bits[26]``    1 where a taxel reads above ``TOUCH_BIT_N`` (3 N): which part of
                the sole touches. A thresholded map barely moves under gain errors.
``load``        total normal force / body weight (343 N), clipped to [0, 1.5]:
                HOW MUCH, the one quantity a gain error does move.
``cop[2]``      centre of pressure from the taxel readings (not the simulator's
                exact CoP -- the real sensor can only compute it this way),
                x and y normalised to the sole's half extents.

Normal force only. The simulated shear is the net tangential force split in
proportion to normal load, so it carries no spatial information the normals do
not, and it is the channel least likely to survive contact with real hardware.

Both feet in one frame. The slipper layout is mirror-symmetric in y, so the
right foot's taxels are re-indexed into the left foot's mirror image and its
y features flipped: for both feet +y is lateral. One layer of weights then
serves both feet.

A soft per-taxel contact map with load, CoP, area, edge loading and a contact
flag (``tactile_map``, 33 per foot) was the beam E2E tasks' input until
2026-11-15; contact area and the contact flag left the touch features then too.

Sensor model (actor only; the critic's copy is noise-free)
---------------------------------------------------------------------
Per episode: a per-foot gain, a per-taxel gain, dead taxels (read 0). Per step:
additive Gaussian noise on each taxel, and a frame drop that holds a foot's
previous reading. Everything scales with one strength, the same in every task
(``FAULT_STRENGTH``), and is off unless ``FAULTS_ENABLED``.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING

import torch

from sendg1.common.tactile.layout import LAYOUTS, TaxelLayout
from sendg1.common.tactile.sensing import TactileParams, read_tactile

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

BODY_WEIGHT_N = 343.0
"""Nominal G1 weight (35 kg). A known constant on the real robot."""
CONTACT_N = 15.0
"""Below this total normal force a foot counts as not in contact (CoP reads 0)."""
TOUCH_BIT_N = 3.0
"""A taxel counts as touching above 3 N (``touch_features``). Standing on the
widest beam a loaded taxel reads ~6 N (median; 10th percentile 4 N), so the
threshold sits halfway between nothing and a typical load. With faults at
strength 1 (3 N std) ~16% of unloaded taxels flicker on; at 0.5 (1.5 N std)
about 5 %."""
TOUCH_PER_FOOT = 26 + 3  # bits, load, cop x/y


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


FAULTS_ENABLED = False
"""Whether the actor's tactile readings carry sensor faults, in every task.
Off for the first runs (2026-11-25): the 0-1 step latency stays, everything
else is clean. Switch on per run with the ``tactile_faults`` event's
``enabled`` parameter, or here for every task."""
FAULT_STRENGTH = 0.5
"""Fault strength when enabled, the same in every task and on every
curriculum level: 0.5 scales ``TactileNoiseCfg``'s ranges by half (foot gain
+/-10 %, taxel gain +/-12.5 %, 4 % dead taxels, 1.5 N noise, 2.5 % frame
drops)."""


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
  return centers, perm, x_mid, x_half, y_half


def canonical_normals(env: ManagerBasedRlEnv, params: TactileParams) -> torch.Tensor:
  """[B, 2, T] taxel normal forces (N), right foot re-indexed to the mirror
  image of the left, so taxel j on both feet sits at (x_j, +/- y_j) with +y
  lateral."""
  layout = LAYOUTS[params.layout]
  reading = read_tactile(env, params)
  normal = reading.normal
  _, perm, *_ = _layout_consts(layout, str(normal.device))
  return torch.stack([normal[:, 0], normal[:, 1, perm]], dim=1)


def touch_features(normal: torch.Tensor, layout: TaxelLayout) -> torch.Tensor:
  """[B, 2, T] canonical normals -> [B, 2 * (T + 3)] features, per foot:

  ``bits[T]``  1 where a taxel reads above ``TOUCH_BIT_N``, else 0: which part of
               the sole touches.
  ``load``     total normal force / body weight, clipped to [0, 1.5].
  ``cop[2]``   centre of pressure from the taxels, x and y scaled to the sole's
               half extents (+-1), 0 while the total normal force is at or
               below ``CONTACT_N`` (foot not in contact).

  Canonical frame (``canonical_normals``): +y is lateral on both feet."""
  centers, _, x_mid, x_half, y_half, *_ = _layout_consts(layout, str(normal.device))
  total = normal.sum(-1, keepdim=True)
  bits = (normal > TOUCH_BIT_N).float()
  load = (total / BODY_WEIGHT_N).clamp(0.0, 1.5)
  w = normal / total.clamp(min=1e-6)
  in_contact = (total > CONTACT_N).float()
  cop_x = (((w * centers[:, 0]).sum(-1, keepdim=True) - x_mid) / x_half).clamp(-1, 1) * in_contact
  cop_y = ((w * centers[:, 1]).sum(-1, keepdim=True) / y_half).clamp(-1, 1) * in_contact
  return torch.cat([bits, load, cop_x, cop_y], dim=-1).flatten(start_dim=1)


def canonical_forces(env: ManagerBasedRlEnv, params: TactileParams) -> torch.Tensor:
  """[B, 2, T, 3] clean per-taxel (normal, shear x, shear y) in newtons, in the
  canonical frame (right foot re-indexed, its shear y negated so +y is lateral
  on both feet)."""
  layout = LAYOUTS[params.layout]
  reading = read_tactile(env, params)
  _, perm, *_ = _layout_consts(layout, str(reading.normal.device))
  f = torch.cat([reading.normal[..., None], reading.shear[..., :2]], dim=-1)  # [B, 2, T, 3]
  right = f[:, 1, perm] * torch.tensor([1.0, 1.0, -1.0], device=f.device)
  return torch.stack([f[:, 0], right], dim=1)


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


def reset_tactile_noise(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  enabled: bool = FAULTS_ENABLED,
  strength: float = FAULT_STRENGTH,
) -> None:
  """Reset event of every task: resample the sensor faults, at ``strength``
  when ``enabled``, else none (strength 0: unit gains, no dead taxels, no
  noise, no frame drops). In every observation set's events, blind ones
  included, so the sets differ in the actor's input alone."""
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  st = TactileNoiseState.get(env, LAYOUTS[TactileParams().layout])
  s = strength if enabled else 0.0
  st.resample(env_ids, torch.full((len(env_ids),), s, device=env.device))


def tactile_faults_event():
  """The ``tactile_faults`` reset event, as every task registers it."""
  from mjlab.managers.event_manager import EventTermCfg

  return EventTermCfg(
    func=reset_tactile_noise,
    mode="reset",
    params={"enabled": FAULTS_ENABLED, "strength": FAULT_STRENGTH},
  )
