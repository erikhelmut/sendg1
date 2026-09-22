"""Every registered variant must actually build and step.

The GPU tests are marked ``slow`` so the fast structural checks (allowlist,
fingerprints) stay usable as a pre-commit gate:
    pytest -m "not slow"
"""

from __future__ import annotations

import pytest
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg

import sendg1.tasks  # noqa: F401
from sendg1.common.obs.obs_sets import OBS_SETS
from sendg1.tasks._register import variants

ALL_VARIANTS = sorted(variants())

BASE_ACTOR_DIM = {
  "Sendg1-Velocity-Flat-G1": 96,
  "Sendg1-Velocity-Rough-G1": 96,
  # No command: the beam task tells the actor nothing about where to go.
  "Sendg1-Beam-G1": 93,
  # The velocity actor exactly: proprioception + a 3-dim walking command,
  # here the scripted pilot's rather than the operator's.
  "Sendg1-BoxPush-G1": 96,
  "Sendg1-BoxPushCurriculum-G1": 96,
  "Sendg1-BoxPushPatches-G1": 96,
}
EXTRA_ACTOR_DIM = {
  "blind": 0,
  "contact_oracle": 8,
  # 2 feet x 26 taxels (slipper_v1) x (normal, shear_x, shear_y).
  "tactile_array": 156,
}


BOX_PUSH_TACTILE_EXTRA = 12
"""The box-push tactile group also carries the per-foot aggregate channels
(normal, shear x/y, CoP x/y, shear/normal ratio)."""


def _expected_actor_dim(task_id: str) -> int:
  base_id, obs_set = variants()[task_id]
  extra = EXTRA_ACTOR_DIM[obs_set]
  if base_id.startswith("Sendg1-BoxPush") and obs_set == "tactile_array":
    extra += BOX_PUSH_TACTILE_EXTRA
  return BASE_ACTOR_DIM[base_id] + extra


def test_expected_variants_registered() -> None:
  registered = {t for t in list_tasks() if t.startswith("Sendg1-")}
  expected = {
    f"{base_id}-{slug}"
    for base_id in BASE_ACTOR_DIM
    for slug in ("Blind", "ContactOracle", "TactileArray")
  }
  assert registered == expected


@pytest.mark.parametrize("task_id", ALL_VARIANTS)
def test_obs_groups_reference_existing_groups(task_id: str) -> None:
  env_cfg = load_env_cfg(task_id)
  rl_cfg = load_rl_cfg(task_id)
  for role, groups in rl_cfg.obs_groups.items():
    for group in groups:
      assert group in env_cfg.observations, (
        f"{task_id}: {role} maps to group {group!r}, not defined in the env cfg"
      )


@pytest.mark.parametrize("task_id", ALL_VARIANTS)
def test_no_unused_groups_are_computed(task_id: str) -> None:
  """Declared-but-unmapped groups are computed every step and thrown away."""
  env_cfg = load_env_cfg(task_id)
  rl_cfg = load_rl_cfg(task_id)
  mapped = {g for groups in rl_cfg.obs_groups.values() for g in groups}
  extra = set(env_cfg.observations) - mapped
  assert not extra, f"{task_id}: groups computed but unused: {sorted(extra)}"


@pytest.mark.parametrize("task_id", ALL_VARIANTS)
def test_action_space_is_full_29_dof(task_id: str) -> None:
  env_cfg = load_env_cfg(task_id)
  action = env_cfg.actions["joint_pos"]
  assert action.actuator_names == (".*",), (
    f"{task_id}: Phase 1 standardises on all 29 DoF actuated"
  )


@pytest.mark.slow
@pytest.mark.parametrize("task_id", ALL_VARIANTS)
def test_env_builds_and_steps(task_id: str) -> None:
  import torch
  from mjlab.envs import ManagerBasedRlEnv

  env_cfg = load_env_cfg(task_id)
  env_cfg.scene.num_envs = 4
  rl_cfg = load_rl_cfg(task_id)
  device = "cuda:0" if torch.cuda.is_available() else "cpu"

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  try:
    env.reset()
    for _ in range(3):
      env.step(torch.zeros(4, env.action_manager.total_action_dim, device=device))

    dims = env.observation_manager.group_obs_dim
    actor_dim = sum(dims[g][0] for g in rl_cfg.obs_groups["actor"])
    expected = _expected_actor_dim(task_id)
    assert actor_dim == expected, f"{task_id}: actor dim {actor_dim}, expected {expected}"
  finally:
    env.close()


@pytest.mark.slow
def test_blind_actor_dim_matches_across_terrains() -> None:
  """Dropping the height scan makes flat and rough share a policy input space,
  so one architecture serves both and flat->rough transfer stays meaningful."""
  import torch
  from mjlab.envs import ManagerBasedRlEnv

  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  dims = {}
  for task_id in ("Sendg1-Velocity-Flat-G1-Blind", "Sendg1-Velocity-Rough-G1-Blind"):
    cfg = load_env_cfg(task_id)
    cfg.scene.num_envs = 2
    env = ManagerBasedRlEnv(cfg=cfg, device=device)
    try:
      dims[task_id] = env.observation_manager.group_obs_dim["proprio"][0]
    finally:
      env.close()
  assert len(set(dims.values())) == 1, f"proprio dim differs across terrains: {dims}"


def test_obs_set_descriptions_are_informative() -> None:
  for name, obs_set in OBS_SETS.items():
    assert len(obs_set.description) > 40, f"{name}: describe what this set is for"
