"""The video HUD and the tactile/blind decision -- no renderer."""

from __future__ import annotations

import numpy as np
import pytest
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

import sendg1.tasks  # noqa: F401
from sendg1.common.tactile.layout import LAYOUTS
from sendg1.video import hud, policy_sees_tactile


@pytest.mark.parametrize(
  ("task", "tactile"),
  [
    ("Sendg1-Velocity-Flat-G1-Blind", False),
    ("Sendg1-Velocity-Flat-G1-ContactOracle", False),
    ("Sendg1-Velocity-Flat-G1-TactileArray", True),
    ("Sendg1-BoxPush-G1-TactileArray", True),
    ("G1-Beam-E2E", True),
    ("G1-Beam-Touch-Curved", True),
  ],
)
def test_policy_sees_tactile(task, tactile):
  assert policy_sees_tactile(load_env_cfg(task), load_rl_cfg(task)) is tactile


def test_card_draws_in_place_and_keeps_frame_size():
  layout = LAYOUTS["slipper_v1"]
  card = hud.TactileCard(layout, (360, 640), live=True, vmax=30.0, history=50,
                         history_s=1.0)
  frame = np.zeros((360, 640, 3), np.uint8)
  normal = np.zeros((2, layout.num_taxels))
  normal[0, 3] = 300.0
  out = card.draw(frame, normal)
  assert out.shape == frame.shape and out.dtype == np.uint8
  assert out.any() and not frame.any()
  assert card.trace[:, -1].tolist() == [300.0, 0.0]
  out = hud.caption(out, "task", "BLIND", False, ["line"])
  assert out.shape == frame.shape
