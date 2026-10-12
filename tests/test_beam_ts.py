"""Beam teacher-student task: no privileged leak into the student, geometry,
tactile canonicalisation, and an env that builds and steps."""

from __future__ import annotations

import pytest
import torch
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

import sendg1.tasks  # noqa: F401
from sendg1.common.beam_ts import tactile as tac
from sendg1.common.beam_ts.geometry import project_points
from sendg1.common.beam_ts.terms import STUDENT_LEGAL, TERMS
from sendg1.common.tactile.layout import LAYOUTS
from sendg1.tasks.beam_ts import STUDENT_TASK, TEACHER_TASK, assert_student_is_deployable

STUDENT_EXPECTED = {
  "base_ang_vel",
  "projected_gravity",
  "joint_pos",
  "joint_vel",
  "actions",
  "command_vx",
  "tactile",
}


def _student_terms() -> set[str]:
  env_cfg = load_env_cfg(STUDENT_TASK)
  rl_cfg = load_rl_cfg(STUDENT_TASK)
  return {t for g in rl_cfg.obs_groups["student"] for t in env_cfg.observations[g].terms}


def test_student_sees_exactly_the_real_robot_signals() -> None:
  assert _student_terms() == STUDENT_EXPECTED


def test_student_terms_are_all_deployable() -> None:
  for term in _student_terms():
    assert TERMS[term].availability in ("onboard", "sensor"), term


def test_teacher_and_critic_see_privileged() -> None:
  rl = load_rl_cfg(TEACHER_TASK)
  assert "privileged" in rl.obs_groups["actor"]
  assert "privileged" in rl.obs_groups["critic"]


def test_leak_guard_fires() -> None:
  env_cfg = load_env_cfg(STUDENT_TASK)
  with pytest.raises(ValueError):
    assert_student_is_deployable(env_cfg.observations, ("real", "privileged"))
  env_cfg.observations["real"].terms["base_lin_vel"] = TERMS["base_lin_vel"].build()
  with pytest.raises(ValueError):
    assert_student_is_deployable(env_cfg.observations, ("real",))


def test_privileged_terms_never_legal() -> None:
  for name in ("base_lin_vel", "feet_beam_pose", "beam_params", "tactile_clean"):
    assert name not in STUDENT_LEGAL


def test_projection_on_straight_path() -> None:
  path = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]])
  pts = torch.tensor([[[0.5, 0.1], [1.5, -0.2], [-1.0, 0.0]]])
  p = project_points(path, pts)
  assert torch.allclose(p.arc, torch.tensor([[0.5, 1.5, 0.0]]))
  assert torch.allclose(p.lateral[:, :2], torch.tensor([[0.1, -0.2]]))
  assert torch.allclose(p.tangent_yaw, torch.zeros(1, 3))


def test_mirror_permutation_is_an_involution() -> None:
  layout = LAYOUTS["slipper_v1"]
  perm = tac.mirror_permutation(layout)
  assert sorted(perm) == list(range(layout.num_taxels))
  assert all(perm[perm[i]] == i for i in range(layout.num_taxels))


def test_features_are_mirror_consistent() -> None:
  """A load on the left foot's lateral edge and the same load on the right
  foot's lateral edge give identical canonical features."""
  layout = LAYOUTS["slipper_v1"]
  t = layout.num_taxels
  lateral_left = torch.tensor([float(y > 0.01) for _, y in layout.centers])
  normal = torch.zeros(1, 2, t)
  normal[0, 0] = 50.0 * lateral_left
  # Canonical right foot uses the mirrored indexing already, so feed the same.
  normal[0, 1] = 50.0 * lateral_left
  f = tac.features_from_normals(normal, layout).view(2, -1)
  assert torch.allclose(f[0], f[1])
  edge_lat = f[0, t + 4]
  assert edge_lat > 0.9


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_env_builds_steps_and_obs_are_finite() -> None:
  from mjlab.envs import ManagerBasedRlEnv

  cfg = load_env_cfg(TEACHER_TASK)
  cfg.scene.num_envs = 4
  env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
  try:
    obs, _ = env.reset()
    assert obs["real"].shape == (4, 160)
    for _ in range(10):
      obs, rew, *_ = env.step(torch.zeros(4, env.action_manager.total_action_dim, device=env.device))
      assert torch.isfinite(rew).all()
      assert all(torch.isfinite(v).all() for v in obs.values())
  finally:
    env.close()


def _swap_side(name: str) -> str:
  if name.startswith("left_"):
    return "right_" + name[len("left_") :]
  if name.startswith("right_"):
    return "left_" + name[len("right_") :]
  return name


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_mirror_maps_match_g1_kinematics_and_are_involutions() -> None:
  """Mirrored joint angles must give the mirror-image robot (every body at
  (x, -y, z) of its counterpart), the default pose must be its own mirror,
  and mirroring observations / actions twice must be the identity."""
  from mjlab.envs import ManagerBasedRlEnv

  from sendg1.common.beam_ts.symmetry import joint_mirror, mirror_obs_actions

  cfg = load_env_cfg(TEACHER_TASK)
  cfg.scene.num_envs = 2
  env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
  try:
    obs, _ = env.reset()
    robot = env.scene["robot"]
    names = list(robot.joint_names)
    src = torch.tensor([s for s, _ in joint_mirror(names)], device=env.device)
    sign = torch.tensor([g for _, g in joint_mirror(names)], device=env.device)

    default = robot.data.default_joint_pos[0]
    torch.testing.assert_close(default[src] * sign, default)

    gen = torch.Generator(device=env.device).manual_seed(0)
    q0 = default + (torch.rand(len(names), generator=gen, device=env.device) - 0.5) * 0.8
    q = torch.stack([q0, q0[src] * sign])
    robot.write_joint_state_to_sim(q, torch.zeros_like(q))
    pose = torch.zeros(2, 7, device=env.device)
    pose[:, :3] = env.scene.env_origins + torch.tensor([0.0, 0.0, 1.0], device=env.device)
    pose[:, 3] = 1.0
    robot.write_root_link_pose_to_sim(pose)
    env.sim.forward()
    rel = robot.data.body_link_pos_w - pose[:, None, :3]
    bodies = list(robot.body_names)
    mirrored = rel[0, [bodies.index(_swap_side(b)) for b in bodies]] * torch.tensor(
      [1.0, -1.0, 1.0], device=env.device
    )
    torch.testing.assert_close(rel[1], mirrored, atol=1e-4, rtol=0)

    class _Wrapped:
      unwrapped = env

    from tensordict import TensorDict

    w = _Wrapped()
    obs = TensorDict(dict(obs), batch_size=[2], device=env.device)
    actions = torch.randn(2, env.action_manager.total_action_dim, device=env.device)
    once, act_once = mirror_obs_actions(w, obs, actions)
    twice, act_twice = mirror_obs_actions(w, once[2:], act_once[2:])
    for k in obs.keys():
      torch.testing.assert_close(twice[2:][k], obs[k])
    torch.testing.assert_close(act_twice[2:], actions)
  finally:
    env.close()
