"""The tactile force model on hand-built inputs -- no simulator.

What must hold whatever the layout: force is conserved (taxels sum to the net
force), the centre of pressure is reproduced when it lies over supported
taxels, unsupported taxels read zero, and nothing goes NaN in degenerate cases.
"""

from __future__ import annotations

import pytest
import torch

from sendg1.common.tactile import model
from sendg1.common.tactile.layout import LAYOUTS

LAYOUT = LAYOUTS["slipper_v1"]
T = LAYOUT.num_taxels
CENTERS = torch.tensor(LAYOUT.centers)
AREAS = torch.tensor(LAYOUT.areas())
SIGMA = 0.02
HEEL = CENTERS[:, 0] < -0.02
"""The 7 heel taxels (x < -20 mm)."""
TOE = CENTERS[:, 0] > 0.10
"""The 5 toe taxels (x > 100 mm)."""
MIDLINE = CENTERS[:, 1].abs() < 1e-6
"""The taxels on the sole's centre line (y = 0), one straight column."""


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


def test_layout_is_the_same_part_on_both_feet() -> None:
  assert T == 26 and int(HEEL.sum()) == 7 and int(TOE.sum()) == 5
  assert LAYOUT.for_side("left") == LAYOUT.for_side("right")


def test_sample_points_stay_inside_their_patch() -> None:
  pts = LAYOUT.sample_points("left", per_axis=3)
  assert len(pts) == T * 9
  for i, ((cx, cy), (hx, hy)) in enumerate(zip(LAYOUT.centers, LAYOUT.half_sizes)):
    for x, y in pts[i * 9 : (i + 1) * 9]:
      assert abs(x - cx) < hx and abs(y - cy) < hy


def test_slipper_layout_is_the_default_and_matches_the_csv() -> None:
  from sendg1.common.tactile.layout import DEFAULT_LAYOUT

  assert DEFAULT_LAYOUT == "slipper_v1"
  slipper = LAYOUTS["slipper_v1"]
  assert slipper.num_taxels == 26
  assert slipper.centers[0] == pytest.approx((-0.05661, 0.0))
  assert slipper.centers[20] == pytest.approx((0.0931, -0.0257))
  assert all(h == pytest.approx((0.006, 0.006)) for h in slipper.half_sizes)


def test_slipper_sensor_is_the_same_part_on_both_feet() -> None:
  """Not mirrored: channel i sits at the same (x, y) on both soles."""
  slipper = LAYOUTS["slipper_v1"]
  assert slipper.for_side("right") == slipper.for_side("left")
  assert slipper.sample_points("right", 2) == slipper.sample_points("left", 2)


def _write_csv(tmp_path, rows):
  path = tmp_path / "layout.csv"
  lines = ["id,cx,cy,width_x,length_y,rotation_deg"]
  lines += [",".join(str(v) for v in r) for r in rows]
  path.write_text("\n".join(lines) + "\n")
  return path


@pytest.mark.parametrize(
  ("rows", "match"),
  [
    ([(0, -93.1, -25.7, 12, 12, 0)], "extends past the foot"),
    ([(0, 0, 0, 12, 12, 0), (1, 5, 0, 12, 12, 0)], "overlap"),
    ([(0, 0, 0, 12, 12, 0), (2, 20, 0, 12, 12, 0)], "ids must be"),
  ],
)
def test_csv_layout_rejects_bad_rows(tmp_path, rows, match) -> None:
  from sendg1.common.tactile.layout import csv_layout

  with pytest.raises(ValueError, match=match):
    csv_layout("bad", _write_csv(tmp_path, rows))


def test_csv_layout_rejects_rotation(tmp_path) -> None:
  from sendg1.common.tactile.layout import csv_layout

  with pytest.raises(NotImplementedError):
    csv_layout("bad", _write_csv(tmp_path, [(0, 0, 0, 12, 12, 15)]))


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
  heel, toe = normal[HEEL].sum(), normal[TOE].sum()
  assert toe > heel


def test_cop_near_the_edge_drops_taxels_in_tension() -> None:
  """A CoP at the toe cannot be made by a plane of pressure without going
  negative at the heel; those taxels must read zero, never negative."""
  cop = torch.tensor([CENTERS[:, 0].max().item(), 0.0])
  normal, _ = _split([0.0, 0.0, 200.0], cop, torch.ones(T))
  assert (normal >= 0).all()
  assert normal[HEEL].sum() == 0
  assert normal.sum().item() == pytest.approx(200.0, rel=1e-4)


def test_unsupported_taxels_read_zero() -> None:
  support = HEEL.float()  # heel only, e.g. heel strike
  cop = CENTERS[HEEL].mean(0)
  normal, shear = _split([5.0, 0.0, 150.0], cop, support)
  assert normal[~HEEL].abs().max() == 0
  assert shear[~HEEL].abs().max() == 0
  assert normal.sum().item() == pytest.approx(150.0, rel=1e-4)


def test_collinear_support_is_finite() -> None:
  """One column of taxels on a narrow beam: the cross-slope is undetermined."""
  support = MIDLINE.float()
  normal, shear = _split([0.0, 3.0, 250.0], [0.03, 0.0], support)
  assert torch.isfinite(normal).all() and torch.isfinite(shear).all()
  assert normal.sum().item() == pytest.approx(250.0, rel=1e-4)
  assert normal[~MIDLINE].sum() == 0


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
