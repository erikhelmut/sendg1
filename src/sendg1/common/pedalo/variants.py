"""The two pedalos the task can put under the robot, as the few numbers and
names the MDP terms need (``mdp.PedaloState``, ``mdp.reset_on_pedalo``).

``step`` means what a foot stands on: a Classic board or a Sport block.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import mujoco
import torch

from sendg1.common.pedalo import model as pm
from sendg1.common.pedalo import sport


@dataclass(frozen=True)
class PedaloVariant:
  name: str
  spec_fn: Callable[[], mujoco.MjSpec]
  root_body: str
  """Holds the freejoint; a wheelset, pitched by the crank angle phi."""
  wheel_bodies: tuple[str, ...]
  """The wheelsets; their mean position is the pedalo centre."""
  step_bodies: tuple[str, str]
  """(left, right), each with its frame on its axle."""
  hinge_phi: tuple[tuple[str, float], ...]
  """Each hinge's angle per unit phi with the steps level (``crank_qpos``)."""
  root_ahead: float
  """Root wheel centre ahead of the pedalo centre."""
  step_top_local: tuple[float, float, float]
  """Centre of a step's top face in its body frame."""
  step_top_above_axle: float
  step_size: tuple[float, float]
  """(along, across) the direction of travel."""
  steps_turn: bool
  """Sport: the steps turn freely on their axles. Classic: the parallelogram
  keeps the boards level."""

  def step_top_height(self, phi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(left, right) step top above the ground with the steps level. phi > 0
    lowers the left step (``model.board_top_height``)."""
    base = pm.WHEEL_RADIUS + self.step_top_above_axle
    dz = pm.ECCENTRIC * torch.sin(phi)
    return base - dz, base + dz


CLASSIC = PedaloVariant(
  name="classic",
  spec_fn=pm.pedalo_spec,
  root_body=pm.ROOT_BODY,
  wheel_bodies=pm.WHEEL_BODIES,
  step_bodies=pm.BOARD_BODIES,
  hinge_phi=tuple(pm.crank_qpos(1.0).items()),
  root_ahead=pm.WHEELBASE / 2,
  step_top_local=(-pm.WHEELBASE / 2, 0.0, pm.BOARD_TOP_ABOVE_AXLE),
  step_top_above_axle=pm.BOARD_TOP_ABOVE_AXLE,
  step_size=(pm.BOARD_LENGTH, pm.BOARD_WIDTH),
  steps_turn=False,
)

SPORT = PedaloVariant(
  name="sport",
  spec_fn=sport.pedalo_spec,
  root_body=sport.ROOT_BODY,
  wheel_bodies=(sport.ROOT_BODY,),
  step_bodies=sport.STEP_BODIES,
  hinge_phi=tuple(sport.crank_qpos(1.0).items()),
  root_ahead=0.0,
  step_top_local=(0.0, 0.0, sport.STEP_TOP_ABOVE_AXLE),
  step_top_above_axle=sport.STEP_TOP_ABOVE_AXLE,
  step_size=(sport.STEP_LENGTH, sport.STEP_WIDTH),
  steps_turn=True,
)

VARIANTS: tuple[PedaloVariant, ...] = (CLASSIC, SPORT)
