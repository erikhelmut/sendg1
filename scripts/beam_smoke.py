#!/usr/bin/env python
"""Smoke test of the beam env: builds, obs shapes, finite rewards,
terminations, curriculum movement.

    python scripts/beam_smoke.py --num-envs 16 --steps 300
    python scripts/beam_smoke.py --task G1-Beam-E2E-Curved
"""

from __future__ import annotations

import argparse


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--task", default="G1-Beam-E2E")
  ap.add_argument("--num-envs", type=int, default=16)
  ap.add_argument("--steps", type=int, default=300)
  ap.add_argument("--zero-actions", action="store_true")
  args = ap.parse_args()

  import torch
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.registry import load_env_cfg

  import sendg1.tasks  # noqa: F401
  from sendg1.common.beam.geometry import BeamGeometry
  from sendg1.common.beam.mdp import BeamEpisode

  cfg = load_env_cfg(args.task)
  cfg.scene.num_envs = args.num_envs
  env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
  obs, _ = env.reset()
  for k, v in obs.items():
    print(f"obs[{k}] {tuple(v.shape)}")
  om = env.observation_manager
  for g in ("actor", "critic"):
    print(g, dict(zip(om.active_terms[g], om.group_obs_term_dim[g])))

  geo = BeamGeometry.get(env)
  print("widths", geo.widths.tolist())
  print("top_z", geo.top_z()[:4].tolist(), "beam_arc", geo.beam_arc()[:2].tolist())
  ep = BeamEpisode.get(env)
  print("spawn arc", ep.spawn_arc[:4].tolist(), "platform", ep.platform_mode[:4].tolist())
  print("base proj", ep.base_proj().lateral[:4, 0].tolist())

  term_counts: dict[str, int] = {}
  rew_sum = torch.zeros(env.num_envs, device=env.device)
  nact = env.action_manager.total_action_dim
  for t in range(args.steps):
    a = torch.zeros(env.num_envs, nact, device=env.device)
    if not args.zero_actions:
      a += 0.3 * torch.randn_like(a)
    obs, rew, term, trunc, extras = env.step(a)
    assert torch.isfinite(rew).all(), "non-finite reward"
    for g, v in obs.items():
      assert torch.isfinite(v).all(), f"non-finite obs in {g}"
    rew_sum += rew
    for name in env.termination_manager.active_terms:
      n = int(env.termination_manager.get_term(name).sum())
      term_counts[name] = term_counts.get(name, 0) + n
    if t % 50 == 0:
      log = extras.get("log", {})
      lv = {k: round(float(v), 3) for k, v in log.items() if k.startswith("Curriculum")}
      print(t, "rew", round(float(rew.mean()), 3), lv)
  print("terminations", term_counts)
  print("reward per env", rew_sum.mean().item())
  print("levels", geo.levels.tolist())
  for name in env.reward_manager.active_terms:
    print(f"  {name:20s} {env.reward_manager._episode_sums[name].mean().item():+.4f}")
  env.close()


if __name__ == "__main__":
  main()
