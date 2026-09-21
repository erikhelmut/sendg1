"""The tactile model inside a live env.

Checked against an independent reading of the same physics: MuJoCo's own
net-force contact sensor (``feet_ground_contact``), read straight from
``sensordata`` rather than through mjlab's per-step cache -- the cached value
is taken before mjlab's post-step forward pass and can differ by solver noise.
"""

from __future__ import annotations

import pytest
import torch

TASK = "Sendg1-Velocity-Flat-G1-TactileArray"
NUM_ENVS = 16


@pytest.fixture(scope="module")
def rollout():
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.registry import load_env_cfg

  import sendg1.tasks  # noqa: F401
  from sendg1.common.tactile.sensing import TactileParams, read_tactile

  cfg = load_env_cfg(TASK)
  cfg.scene.num_envs = NUM_ENVS
  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  mj_model = env.sim.mj_model

  def raw_net_force(side: str) -> torch.Tensor:
    sensor = mj_model.sensor(f"feet_ground_contact_{side}_ankle_roll_link_force")
    a = sensor.adr[0]
    return env.sim.data.sensordata[:, a : a + 3].clone()

  steps = []
  try:
    env.reset()
    g = torch.Generator(device=device).manual_seed(0)
    for _ in range(40):
      action = 0.3 * torch.randn(
        NUM_ENVS, env.action_manager.total_action_dim, device=device, generator=g
      )
      obs, *_ = env.step(action)
      reading = read_tactile(env, TactileParams())
      robot = env.scene["robot"]
      ids, _ = robot.find_bodies(
        ["left_ankle_roll_link", "right_ankle_roll_link"], preserve_order=True
      )
      steps.append(
        {
          "obs": obs["tactile_array"].clone(),
          "reading": reading,
          "quat": robot.data.body_link_quat_w[:, ids].clone(),
          "raw_net": torch.stack([raw_net_force("left"), raw_net_force("right")], 1),
        }
      )
  finally:
    env.close()
  return steps


@pytest.mark.slow
def test_observation_shape_and_finite(rollout) -> None:
  for step in rollout:
    assert step["obs"].shape == (NUM_ENVS, 144)
    assert torch.isfinite(step["obs"]).all()


@pytest.mark.slow
def test_net_force_matches_mujoco_net_sensor(rollout) -> None:
  """Frame, sign and slot capacity: the per-contact sum rotated back to the
  world must equal MuJoCo's net contact force, negated (MuJoCo reports the
  force the foot exerts on the ground)."""
  from mjlab.utils.lab_api.math import matrix_from_quat

  loaded = 0
  for step in rollout:
    rot = matrix_from_quat(step["quat"])
    net_w = (rot @ step["reading"].net_force.unsqueeze(-1)).squeeze(-1)
    expected = -step["raw_net"]
    mask = expected.norm(dim=-1) > 5.0
    loaded += int(mask.sum())
    err = (net_w - expected).norm(dim=-1)
    assert (err[mask] <= 1e-3 * expected.norm(dim=-1)[mask] + 1e-3).all(), err[mask].max()
  assert loaded > 50, "rollout too short to have exercised loaded feet"


@pytest.mark.slow
def test_taxels_conserve_normal_force_and_are_non_negative(rollout) -> None:
  for step in rollout:
    r = step["reading"]
    fn = r.net_force[..., 2].clamp(min=0)
    assert (r.normal >= 0).all()
    assert torch.allclose(r.normal.sum(-1), fn, rtol=1e-4, atol=1e-3)


@pytest.mark.slow
def test_loaded_foot_on_flat_ground_sees_support(rollout) -> None:
  for step in rollout:
    r = step["reading"]
    loaded = r.net_force[..., 2] > 50.0
    assert (r.support.amax(-1)[loaded] > 0).all()
