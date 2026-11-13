"""Observation terms of the pedalo task, each with its availability.

Same contract as ``sendg1.common.beam.terms``: ``onboard`` and ``sensor``
terms may go to the actor, ``privileged`` ones only to the critic, and
``tasks/pedalo`` refuses to register an actor that breaks this.

The actor is the beam task's touch-feature actor unchanged: IMU, joints,
previous action, forward command, the tactile slipper's touch features. It
never sees the pedalo itself; the crank angle is in its joints (the two feet
sit at different heights) and its soles.
"""

from __future__ import annotations

from mjlab.managers.observation_manager import ObservationTermCfg

from sendg1.common.beam.terms import TERMS as BEAM_TERMS
from sendg1.common.beam.terms import TOUCH_ACTOR_TERMS
from sendg1.common.obs.terms import TermSpec
from sendg1.common.pedalo import mdp


def _plain(func):
  return lambda: ObservationTermCfg(func=func)


TERMS: dict[str, TermSpec] = {
  **{
    n: BEAM_TERMS[n]
    for n in (
      *TOUCH_ACTOR_TERMS,
      "tactile_touch_clean",
      "taxel_forces_clean",
      "base_lin_vel",
      "action_lag",
    )
  },
  "pedalo_motion": TermSpec(
    _plain(mdp.pedalo_motion),
    "privileged",
    "Pedalo forward/lateral speed, yaw rate, crank angle (sin, cos) and rate.",
  ),
  "feet_board_pose": TermSpec(
    _plain(mdp.feet_board_pose),
    "privileged",
    "Each sole relative to its own board's top centre, and its yaw error.",
  ),
  "base_pedalo_pose": TermSpec(
    _plain(mdp.base_pedalo_pose),
    "privileged",
    "Base and CoM offset from the pedalo centre, heading error.",
  ),
  "foot_slip_on_board": TermSpec(
    _plain(mdp.foot_slip_on_board), "privileged", "Planar foot velocity relative to its board."
  ),
  "foot_pedalo_forces": TermSpec(
    _plain(mdp.foot_pedalo_forces), "privileged", "Net force of each foot on the pedalo."
  ),
}

ACTOR_TERMS: tuple[str, ...] = TOUCH_ACTOR_TERMS
BLIND_ACTOR_TERMS: tuple[str, ...] = tuple(n for n in ACTOR_TERMS if n != "tactile_touch")
"""The blind baseline: the same actor without the soles' touch features."""
PRIVILEGED_TERMS: tuple[str, ...] = (
  "base_lin_vel",
  "pedalo_motion",
  "feet_board_pose",
  "base_pedalo_pose",
  "tactile_touch_clean",
  "taxel_forces_clean",
  "foot_slip_on_board",
  "foot_pedalo_forces",
  "action_lag",
)
"""The critic, on top of the actor's terms (the clean touch features replacing
the faulted ones)."""
DEPLOYABLE: frozenset[str] = frozenset(
  n for n, s in TERMS.items() if s.availability in ("onboard", "sensor")
)
