"""The tactile force model on hand-built inputs -- no simulator.

What must hold whatever the layout: force is conserved (taxels sum to the net
force), the centre of pressure is reproduced when it lies over supported
taxels, unsupported taxels read zero, and nothing goes NaN in degenerate cases.
"""

from __future__ import annotations

import pytest
import torch

from sendg1.common.tactile import model
from sendg1.common.tactile.layout import LAYOUTS, grid_layout

LAYOUT = LAYOUTS["placeholder_8x3"]
T = LAYOUT.num_taxels
CENTERS = torch.tensor(LAYOUT.centers)
AREAS = torch.tensor(LAYOUT.areas())
SIGMA = 0.02


def _split(net_force, cop, support):
  return model.distribute(
    torch.as_tensor(net_force, dtype=torch.float32),
    torch.as_tensor(cop, dtype=torch.float32),
    torch.as_tensor(support, dtype=torch.float32),
    CENTERS,
    AREAS,
    fallback_sigma=SIGMA,
  )


def _cop_of(normal):
  return (normal.unsqueeze(-1) * CENTERS).sum(-2) / normal.sum(-1, keepdim=True)


def test_layout_is_24_taxels_mirrored_between_feet() -> None:
  assert T == 24
  for (lx, ly), (rx, ry) in zip(LAYOUT.for_side("left"), LAYOUT.for_side("right")):
    assert (lx, ly) == (rx, -ry)


def test_sample_points_stay_inside_their_patch() -> None:
  pts = LAYOUT.sample_points("left", per_axis=3)
  assert len(pts) == T * 9
  for i, ((cx, cy), (hx, hy)) in enumerate(zip(LAYOUT.centers, LAYOUT.half_sizes)):
    for x, y in pts[i * 9 : (i + 1) * 9]:
      assert abs(x - cx) < hx and abs(y - cy) < hy


def test_overlapping_patches_are_rejected() -> None:
  with pytest.raises(ValueError, match="overlaps"):
    grid_layout("bad", rows=8, cols=3, patch=0.03)


def test_centred_load_on_full_support_is_uniform_and_conserved() -> None:
  mean = CENTERS.mean(0)
  normal, shear = _split([10.0, -4.0, 300.0], mean, torch.ones(T))
  assert normal.sum().item() == pytest.approx(300.0, rel=1e-4)
  assert torch.allclose(normal, torch.full((T,), 300.0 / T), rtol=1e-3)
  assert torch.allclose(shear.sum(0), torch.tensor([10.0, -4.0]), atol=1e-3)


def test_cop_is_reproduced_and_load_shifts_toward_it() -> None:
  cop = torch.tensor([0.08, 0.01])
  normal, _ = _split([0.0, 0.0, 400.0], cop, torch.ones(T))
  assert normal.sum().item() == pytest.approx(400.0, rel=1e-4)
  assert torch.allclose(_cop_of(normal), cop, atol=1e-4)
  heel, toe = normal[:3].sum(), normal[-3:].sum()
  assert toe > heel


def test_cop_near_the_edge_drops_taxels_in_tension() -> None:
  """A CoP at the toe row cannot be made by a plane of pressure without going
  negative at the heel; those taxels must read zero, never negative."""
  cop = torch.tensor([CENTERS[-1, 0].item(), 0.0])
  normal, _ = _split([0.0, 0.0, 200.0], cop, torch.ones(T))
  assert (normal >= 0).all()
  assert normal[:3].sum() == 0
  assert normal.sum().item() == pytest.approx(200.0, rel=1e-4)


def test_unsupported_taxels_read_zero() -> None:
  support = torch.zeros(T)
  support[:9] = 1.0  # heel three rows only, e.g. heel strike
  cop = CENTERS[:9].mean(0)
  normal, shear = _split([5.0, 0.0, 150.0], cop, support)
  assert normal[9:].abs().max() == 0
  assert shear[9:].abs().max() == 0
  assert normal.sum().item() == pytest.approx(150.0, rel=1e-4)


def test_collinear_support_is_finite() -> None:
  """One column of taxels on a narrow beam: the cross-slope is undetermined."""
  support = torch.zeros(T)
  support[1::3] = 1.0
  normal, shear = _split([0.0, 3.0, 250.0], [0.03, 0.0], support)
  assert torch.isfinite(normal).all() and torch.isfinite(shear).all()
  assert normal.sum().item() == pytest.approx(250.0, rel=1e-4)
  assert normal[0::3].sum() == 0 and normal[2::3].sum() == 0


def test_load_without_support_falls_back_around_the_cop() -> None:
  cop = CENTERS[10]
  normal, _ = _split([0.0, 0.0, 100.0], cop, torch.zeros(T))
  assert normal.sum().item() == pytest.approx(100.0, rel=1e-4)
  assert normal.argmax().item() == 10


def test_no_contact_reads_zero() -> None:
  normal, shear = _split([0.0, 0.0, 0.0], [0.0, 0.0], torch.ones(T))
  assert normal.abs().max() == 0 and shear.abs().max() == 0


def test_pulling_force_reads_zero() -> None:
  """A net force away from the ground (impossible for a real contact, possible
  in solver noise) must not produce negative pressure."""
  normal, _ = _split([0.0, 0.0, -20.0], [0.0, 0.0], torch.ones(T))
  assert normal.abs().max() == 0


def test_batched_matches_unbatched() -> None:
  g = torch.Generator().manual_seed(0)
  net = torch.rand(4, 2, 3, generator=g) * torch.tensor([20.0, 20.0, 300.0])
  cop = CENTERS.mean(0) + 0.02 * torch.randn(4, 2, 2, generator=g)
  support = (torch.rand(4, 2, T, generator=g) > 0.3).float()
  batched, _ = model.distribute(net, cop, support, CENTERS, AREAS, fallback_sigma=SIGMA)
  for b in range(4):
    for f in range(2):
      one, _ = model.distribute(
        net[b, f], cop[b, f], support[b, f], CENTERS, AREAS, fallback_sigma=SIGMA
      )
      assert torch.allclose(batched[b, f], one, atol=1e-4)


def test_net_force_and_cop_from_contacts() -> None:
  pos = torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.02, 0.0], [9.0, 9.0, 9.0]])
  force = torch.tensor([[1.0, 0.0, 100.0], [0.0, 2.0, 300.0], [50.0, 50.0, 50.0]])
  valid = torch.tensor([True, True, False])
  net, cop = model.net_force_and_cop(pos, force, valid)
  assert torch.allclose(net, torch.tensor([1.0, 2.0, 400.0]))
  assert torch.allclose(cop, torch.tensor([0.075, 0.015]))


def test_support_from_gaps() -> None:
  gaps = torch.tensor([-0.001, 0.0, 0.002, 0.004, 0.01, torch.inf])
  s = model.support_from_gaps(gaps, 0.004)
  assert torch.allclose(s, torch.tensor([1.0, 1.0, 0.5, 0.0, 0.0, 0.0]))
