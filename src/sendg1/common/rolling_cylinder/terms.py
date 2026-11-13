"""Observation terms of the rolling-cylinder task, each with its availability.

Same contract as ``sendg1.common.beam.terms``: ``onboard`` and ``sensor``
terms may go to the actor, ``privileged`` ones only to the critic, and
``tasks/rolling_cylinder`` refuses to register an actor that breaks this.

Two actors, identical but for the slipper: ``BLIND_ACTOR_TERMS`` (IMU,
joints, previous action) and ``TOUCH_ACTOR_TERMS`` (the same plus the beam
task's ``tactile_touch`` features). Neither sees the cylinder; the blind one
has to infer the support line from its joints and IMU. There is no command
term: the task is balance only. The critic is the same for both.
"""

from __future__ import annotations

from mjlab.managers.observation_manager import ObservationTermCfg

from sendg1.common.beam.terms import TERMS as BEAM_TERMS
from sendg1.common.obs.terms import TermSpec
from sendg1.common.rolling_cylinder import mdp


def _plain(func):
  return lambda: ObservationTermCfg(func=func)


BLIND_ACTOR_TERMS: tuple[str, ...] = (
  "base_ang_vel",
  "projected_gravity",
  "joint_pos",
  "joint_vel",
  "actions",
)
TOUCH_ACTOR_TERMS: tuple[str, ...] = (*BLIND_ACTOR_TERMS, "tactile_touch")

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
  "cylinder_motion": TermSpec(
    _plain(mdp.cylinder_motion),
    "privileged",
    "Cylinder rolling and sliding speed, yaw rate, spin rate, drift from spawn.",
  ),
  "feet_cylinder_pose": TermSpec(
    _plain(mdp.feet_cylinder_pose),
    "privileged",
    "Each sole relative to the cylinder's top line, and its yaw error.",
  ),
  "base_cylinder_pose": TermSpec(
    _plain(mdp.base_cylinder_pose),
    "privileged",
    "Base and CoM offset from the cylinder, heading error.",
  ),
  "foot_slip_on_cylinder": TermSpec(
    _plain(mdp.foot_slip_on_cylinder),
    "privileged",
    "Planar foot velocity relative to the cylinder surface under it.",
  ),
  "foot_cylinder_forces": TermSpec(
    _plain(mdp.foot_cylinder_forces), "privileged", "Net force of each foot on the cylinder."
  ),
}

PRIVILEGED_TERMS: tuple[str, ...] = (
  "base_lin_vel",
  "cylinder_motion",
  "feet_cylinder_pose",
  "base_cylinder_pose",
  "tactile_touch_clean",
  "taxel_forces_clean",
  "foot_slip_on_cylinder",
  "foot_cylinder_forces",
  "action_lag",
)
"""The critic, on top of the blind actor's terms. It keeps the clean touch
features in both conditions: only the actor may differ."""
DEPLOYABLE: frozenset[str] = frozenset(
  n for n, s in TERMS.items() if s.availability in ("onboard", "sensor")
)
