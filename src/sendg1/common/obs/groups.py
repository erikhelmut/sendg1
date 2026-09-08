"""Assemble ``ObservationGroupCfg`` objects for a given observation set.

Groups are built on demand, so a set that does not use a group never pays to
compute it -- mjlab's ObservationManager evaluates every declared group on every
step regardless of whether a network consumes it.

Group vocabulary
----------------
``proprio``       Deployable proprioception. Actor and critic. 96-dim.
``privileged``    Simulation state, critic only, permanently. Includes the
                  *idealised* foot contact terms: the critic's contact
                  information is deliberately the stable idealised version so
                  that improving tactile fidelity later never perturbs it.
``contact_ideal`` Oracle foot contact for the diagnostic set. Actor only.
``tactile``       (Phase 2) The modelled 26-taxel array. Actor only.
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


def _terms(names: tuple[str, ...]) -> dict:
  return {name: TERMS[name].build() for name in names}


def build_groups(
  obs_set: str,
  *,
  has_terrain_scan: bool,
) -> dict[str, ObservationGroupCfg]:
  """Build exactly the groups ``obs_set`` needs.

  Args:
    obs_set: key into :data:`OBS_SETS`.
    has_terrain_scan: whether the scene carries the ``terrain_scan`` raycast
      sensor. Only the rough-terrain task does, and the height scan is critic
      only -- the actor is blind on both terrains.
  """
  if obs_set not in OBS_SETS:
    raise KeyError(f"unknown obs_set {obs_set!r}; known: {sorted(OBS_SETS)}")
  wanted = set(OBS_SETS[obs_set].group_names)

  privileged = PRIVILEGED_TERMS_FLAT
  if has_terrain_scan:
    privileged = privileged + ("height_scan",)

  builders = {
    "proprio": lambda: ObservationGroupCfg(
      terms=_terms(PROPRIO_TERMS),
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
  }

  missing = wanted - set(builders)
  if missing:
    raise NotImplementedError(
      f"obs_set {obs_set!r} references group(s) {sorted(missing)} with no "
      "builder. Add one here (see the 'tactile' note in obs_sets.py)."
    )

  return {name: builders[name]() for name in sorted(wanted)}
