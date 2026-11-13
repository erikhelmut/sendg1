"""Observation terms of the slackline task, each with its availability.

Same contract as ``sendg1.common.beam.terms``: ``onboard`` and ``sensor``
terms may go to the actor, ``privileged`` ones only to the critic, and
``tasks/slackline`` refuses to register an actor that breaks this.

Two actors, identical but for the slipper: ``BLIND_ACTOR_TERMS`` (IMU,
joints, previous action, forward command) and ``TOUCH_ACTOR_TERMS`` (the
same plus the beam task's ``tactile_touch`` features). Neither sees the
line: the blind one has to infer where the webbing is under its soles, and
how it is rolling, from its joints and IMU. The critic is the same for both.
"""

from __future__ import annotations

from mjlab.managers.observation_manager import ObservationTermCfg

from sendg1.common.beam.terms import TERMS as BEAM_TERMS
from sendg1.common.obs.terms import TermSpec
from sendg1.common.slackline import mdp


def _plain(func):
  return lambda: ObservationTermCfg(func=func)


BLIND_ACTOR_TERMS: tuple[str, ...] = (
  "base_ang_vel",
  "projected_gravity",
  "joint_pos",
  "joint_vel",
  "actions",
  "command_vx",
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
  "line_under_robot": TermSpec(
    _plain(mdp.line_under_robot),
    "privileged",
    "The webbing under the base and each sole (offset, sag, roll, velocity), "
    "distance to the anchor ahead, tension.",
  ),
  "feet_line_pose": TermSpec(
    _plain(mdp.feet_line_pose),
    "privileged",
    "Each sole relative to the webbing beneath it, and its yaw against the line.",
  ),
  "base_line_pose": TermSpec(
    _plain(mdp.base_line_pose),
    "privileged",
    "Base offset and height over the line, heading error, CoM offset.",
  ),
  "foot_slip_on_line": TermSpec(
    _plain(mdp.foot_slip_on_line),
    "privileged",
    "Each sole's velocity relative to the webbing under it.",
  ),
  "line_level": TermSpec(
    _plain(mdp.line_level_obs),
    "privileged",
    "Curriculum level: width and helper springs of this episode's line.",
  ),
  "foot_line_forces": TermSpec(
    _plain(mdp.foot_line_forces), "privileged", "Net force of each foot on the webbing."
  ),
}

PRIVILEGED_TERMS: tuple[str, ...] = (
  "base_lin_vel",
  "line_under_robot",
  "line_level",
  "feet_line_pose",
  "base_line_pose",
  "tactile_touch_clean",
  "taxel_forces_clean",
  "foot_slip_on_line",
  "foot_line_forces",
  "action_lag",
)
"""The critic, on top of the blind actor's terms. It keeps the clean touch
features in both conditions: only the actor may differ."""
DEPLOYABLE: frozenset[str] = frozenset(
  n for n, s in TERMS.items() if s.availability in ("onboard", "sensor")
)
