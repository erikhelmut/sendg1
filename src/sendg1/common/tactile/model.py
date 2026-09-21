"""From simulated foot contacts to per-taxel normal and shear force.

Pure torch, no simulator, so it can be unit-tested on hand-built inputs.

Why not just bin the raw contact points onto the taxels
-------------------------------------------------------
The simulated sole is 7 rigid capsules. On flat ground MuJoCo puts contacts
only at the two rounded ends of each capsule: one line under the heel
(x ~ -0.05) and one under the toe (x ~ +0.13). Binning those would leave every
mid-foot taxel at zero while the foot is flat, which no real sensor does. Where a
rigid contact solver *does* give meaningful answers is the net wrench: total
force and centre of pressure (CoP).

The model
---------
1. **How much.** Sum the per-contact forces into the net force in the sole
   frame, and take the normal-force-weighted mean contact position as the CoP.
2. **Where.** Each taxel gets a support fraction in [0, 1]: how much of the
   patch has ground within ``compliance_depth`` beneath it (from raycasts,
   computed by the caller). This is what tells a patch over a beam from one
   over a pit.
3. **Split.** Distribute the net normal force over the supported taxels as a
   rigid sole pressing into a thin elastic layer (a Winkler foundation): the
   pressure is a plane, ``p(x, y) = a + b x + c y``, fitted so the total force
   and the CoP match step 1 exactly. Taxels the plane puts in tension are
   dropped and the fit repeated -- a sole cannot pull on the ground.
4. **Shear.** The net tangential force is shared in proportion to normal load
   (uniform friction utilisation). This ignores torsional shear about the CoP.

Output units are newtons per taxel. Forces are the ground reaction ON the sole.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class TactileReading:
  """Per-taxel output for a batch of feet. Leading dims ``[...]`` are
  typically ``[B, 2]`` (envs x feet)."""

  normal: torch.Tensor
  """[..., T] normal force per taxel (N), >= 0."""
  shear: torch.Tensor
  """[..., T, 2] tangential force per taxel in the sole frame (N)."""
  support: torch.Tensor
  """[..., T] fraction of each patch with ground beneath it."""
  net_force: torch.Tensor
  """[..., 3] net ground reaction on the foot, sole frame (N)."""
  cop: torch.Tensor
  """[..., 2] centre of pressure in the sole frame (m). Zero without contact."""


def net_force_and_cop(
  contact_pos: torch.Tensor,
  contact_force: torch.Tensor,
  valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Net force and CoP from per-contact data, all in the sole frame.

  Args:
    contact_pos: [..., K, 3] contact positions.
    contact_force: [..., K, 3] ground reaction on the sole at each contact.
    valid: [..., K] bool, which of the K slots hold a real contact.
  """
  f = contact_force * valid.unsqueeze(-1)
  net = f.sum(dim=-2)
  fz = f[..., 2].clamp(min=0.0)
  total = fz.sum(dim=-1, keepdim=True)
  cop = (fz.unsqueeze(-1) * contact_pos[..., :2]).sum(dim=-2) / total.clamp(min=1e-6)
  cop = torch.where(total > 1e-6, cop, torch.zeros_like(cop))
  return net, cop


def distribute(
  net_force: torch.Tensor,
  cop: torch.Tensor,
  support: torch.Tensor,
  centers: torch.Tensor,
  areas: torch.Tensor,
  *,
  fallback_sigma: float,
  active_set_iters: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Winkler-plane split of a net force over the taxels.

  Args:
    net_force: [..., 3] net force on the sole (sole frame).
    cop: [..., 2] centre of pressure.
    support: [..., T] support fraction per taxel.
    centers: [..., T, 2] or [T, 2] taxel centres.
    areas: [T] patch areas.
    fallback_sigma: when force is present but no taxel reports support (a
      contact outside the sensed area, or on a capsule's rounded flank), load
      is spread with a Gaussian of this width around the CoP instead.
    active_set_iters: re-fits after dropping taxels in tension.

  Returns:
    normal [..., T] and shear [..., T, 2].
  """
  fn = net_force[..., 2].clamp(min=0.0)
  centers = centers.expand(*support.shape, 2)

  # Fallback support where the rays found nothing but there is load.
  d2 = ((centers - cop.unsqueeze(-2)) ** 2).sum(-1)
  gauss = torch.exp(-0.5 * d2 / fallback_sigma**2)
  has_support = support.sum(-1, keepdim=True) > 1e-6
  s = torch.where(has_support, support, gauss) * areas

  # Centre coordinates on the layout for conditioning.
  mean = centers.mean(dim=-2, keepdim=True)
  rel = centers - mean
  phi = torch.cat([torch.ones_like(rel[..., :1]), rel], dim=-1)  # [..., T, 3]
  rel_cop = cop - mean.squeeze(-2)
  b = fn.unsqueeze(-1) * torch.cat([torch.ones_like(fn).unsqueeze(-1), rel_cop], -1)

  eye = torch.eye(3, device=s.device, dtype=s.dtype)
  w = torch.zeros_like(s)
  for _ in range(active_set_iters):
    m = torch.einsum("...t,...ti,...tj->...ij", s, phi, phi)
    # Ridge: when the supported taxels are collinear (one row on a beam edge)
    # the slope across that line is undetermined -- pick zero.
    scale = m.diagonal(dim1=-2, dim2=-1).sum(-1).clamp(min=1e-12)
    m = m + 1e-6 * scale[..., None, None] * eye
    theta = torch.linalg.solve(m, b.unsqueeze(-1)).squeeze(-1)
    w = s * (phi @ theta.unsqueeze(-1)).squeeze(-1)
    kept = s * (w > 0)
    s = torch.where(kept.sum(-1, keepdim=True) > 0, kept, s)

  w = w.clamp(min=0.0)
  total = w.sum(-1, keepdim=True)
  normal = torch.where(total > 1e-9, w * fn.unsqueeze(-1) / total.clamp(min=1e-9), 0.0)

  share = torch.where(fn.unsqueeze(-1) > 1e-9, normal / fn.unsqueeze(-1).clamp(min=1e-9), 0.0)
  shear = share.unsqueeze(-1) * net_force[..., :2].unsqueeze(-2)
  return normal, shear


def support_from_gaps(gaps: torch.Tensor, compliance_depth: float) -> torch.Tensor:
  """Per-sample support in [0, 1] from sole-to-ground gaps (m).

  1 at contact or penetration, falling linearly to 0 at ``compliance_depth``.
  ``gaps`` should be +inf where a ray found no ground.
  """
  return (1.0 - gaps / compliance_depth).clamp(0.0, 1.0)
