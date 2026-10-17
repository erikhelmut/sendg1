"""Assemble ``ObservationGroupCfg`` objects for a given observation set.

Groups are built on demand, so a set that does not use a group never pays to
compute it -- mjlab's ObservationManager evaluates every declared group on every
step regardless of whether a network consumes it.

Group vocabulary
----------------
``proprio``       Deployable proprioception. Actor and critic. 96-dim. On the
                  box-pushing task it also carries the task's ``perceived``
                  terms (box pose, target): 100-dim. The name is kept so every obs set maps the
                  actor exactly as on the other tasks.
``privileged``    Simulation state, critic only, permanently. Includes the
                  *idealised* foot contact terms: the critic's contact
                  information is deliberately the stable idealised version so
                  that improving tactile fidelity later never perturbs it.
``contact_ideal`` Oracle foot contact for the diagnostic set. Actor only.
``tactile_array`` The modelled taxel array on both soles. Actor only.
"""

from __future__ import annotations

from mjlab.managers.observation_manager import ObservationGroupCfg

from sendg1.common.obs.obs_sets import OBS_SETS
from sendg1.common.obs.terms import TERMS

PROPRIO_TERMS: tuple[str, ...] = (
  "base_ang_vel",
  "projected_gravity",
  "joint_pos",
  "joint_vel",
  "actions",
  "command",
)

PRIVILEGED_TERMS_FLAT: tuple[str, ...] = (
  "base_lin_vel",
  "joint_pos_true",
  "foot_height",
  "foot_air_time",
  "foot_contact_ideal",
  "foot_contact_forces_ideal",
)

CONTACT_IDEAL_TERMS: tuple[str, ...] = (
  "foot_contact_ideal",
  "foot_contact_forces_ideal",
)

TACTILE_ARRAY_TERMS: tuple[str, ...] = ("tactile_array",)

BOX_PUSH_PERCEIVED_TERMS: tuple[str, ...] = (
  "box_pos_b",
  "box_yaw_b",
  "target_rel_box",
)

BOX_PUSH_PRIVILEGED_TERMS: tuple[str, ...] = (
  "box_pos_b_true",
  "box_yaw_b_true",
  "target_rel_box_true",
  "floor_mu_under_feet",
  "box_mass",
  "box_mu",
  "box_size",
  "box_lin_vel",
  "box_ang_vel",
  "box_up",
  "foot_slip_vel",
  "push_force",
)


def _terms(names: tuple[str, ...]) -> dict:
  return {name: TERMS[name].build() for name in names}


def build_groups(
  obs_set: str,
  *,
  has_terrain_scan: bool,
  with_command: bool = True,
  box_push: bool = False,
) -> dict[str, ObservationGroupCfg]:
  """Build exactly the groups ``obs_set`` needs.

  Args:
    obs_set: key into :data:`OBS_SETS`.
    has_terrain_scan: whether the scene carries the ``terrain_scan`` raycast
      sensor. The rough task does, and the height scan is critic only --
      the actor is blind everywhere.
    with_command: include the operator command in ``proprio``. False for tasks
      that give the robot no velocity command (box pushing).
    box_push: add the box-pushing task's perceived terms to ``proprio`` and its
      privileged terms to the critic.
  """
  if obs_set not in OBS_SETS:
    raise KeyError(f"unknown obs_set {obs_set!r}; known: {sorted(OBS_SETS)}")
  wanted = set(OBS_SETS[obs_set].group_names)

  proprio = PROPRIO_TERMS
  if not with_command:
    proprio = tuple(t for t in proprio if t != "command")

  privileged = PRIVILEGED_TERMS_FLAT
  if has_terrain_scan:
    privileged = privileged + ("height_scan",)
  if box_push:
    proprio = proprio + BOX_PUSH_PERCEIVED_TERMS
    privileged = privileged + BOX_PUSH_PRIVILEGED_TERMS

  builders = {
    "proprio": lambda: ObservationGroupCfg(
      terms=_terms(proprio),
      concatenate_terms=True,
      enable_corruption=True,
    ),
    "privileged": lambda: ObservationGroupCfg(
      terms=_terms(privileged),
      concatenate_terms=True,
      enable_corruption=False,
    ),
    "contact_ideal": lambda: ObservationGroupCfg(
      terms=_terms(CONTACT_IDEAL_TERMS),
      concatenate_terms=True,
      enable_corruption=False,
    ),
    "tactile_array": lambda: ObservationGroupCfg(
      terms=_terms(TACTILE_ARRAY_TERMS),
      concatenate_terms=True,
      enable_corruption=False,
    ),
  }

  missing = wanted - set(builders)
  if missing:
    raise NotImplementedError(
      f"obs_set {obs_set!r} references group(s) {sorted(missing)} with no "
      "builder. Add one here (see the 'tactile' note in obs_sets.py)."
    )

  return {name: builders[name]() for name in sorted(wanted)}
