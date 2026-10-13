#!/usr/bin/env python
"""Which inputs does a beam teacher's actor actually use?

    python scripts/beam_ts_sensitivity.py --task G1-Beam-Probe-Teacher \
        --checkpoint runs/.../model_16499.pt --out results/beam_ts/sens_probe_teacher.json

The teacher drives 512 envs spread over all seven widths. Every 5th step after
the first second, each input (a whole observation term, and each component of
the beam-pose terms) is replaced by its batch mean and the change of the
action is measured: mean over joints of (delta action)^2 weighted by the
joint's action scale squared, i.e. in joint-target space (the wrists' 0.075
scale would otherwise dominate). A large value means the actor depends on
that input; a student that cannot observe it has to infer it or fail.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

BEAM_POSE_LABELS = {
  "base": ["lat", "sin", "cos", "height", "com_lat"],
  "feet": [f"{f}_{c}" for f in ("L", "R") for c in ("lat", "margin", "height", "sin", "cos")],
}


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--task", required=True)
  ap.add_argument("--checkpoint", required=True)
  ap.add_argument("--num-envs", type=int, default=512)
  ap.add_argument("--steps", type=int, default=300)
  ap.add_argument("--out", required=True)
  args = ap.parse_args()

  import torch
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

  import sendg1.tasks  # noqa: F401
  from sendg1.common.beam_ts.mdp import set_levels

  cfg = load_env_cfg(args.task)
  cfg.scene.num_envs = args.num_envs
  env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
  wrapped = RslRlVecEnvWrapper(env)
  rl = load_rl_cfg(args.task)
  runner = MjlabOnPolicyRunner(wrapped, asdict(rl), device="cuda:0")
  runner.load(args.checkpoint, load_cfg={"actor": True}, strict=True, map_location="cuda:0")
  policy = runner.get_inference_policy(device="cuda:0")
  scale = env.action_manager.get_term("joint_pos").scale[0]
  weight = scale.square() / scale.square().mean()

  om = env.observation_manager
  probes = []  # (group, label, offset, width)
  for g in rl.obs_groups["actor"]:
    off = 0
    for name, dim in zip(om.active_terms[g], om.group_obs_term_dim[g]):
      k = int(torch.tensor(dim).prod())
      probes.append((g, name, off, k))
      kind = "base" if name.startswith("base_beam_pose") else "feet" if name.startswith("feet_beam_pose") else None
      if kind:
        probes += [(g, f"{name}.{lab}", off + j, 1) for j, lab in enumerate(BEAM_POSE_LABELS[kind])]
      off += k

  total = {p[1]: 0.0 for p in probes}
  count = 0
  with torch.inference_mode():
    set_levels(env, torch.arange(env.num_envs, device=env.device) % 7)
    obs, _ = wrapped.reset()
    for i in range(args.steps):
      act = policy(obs)
      if i >= 50 and i % 5 == 0:
        for g, label, off, k in probes:
          o2 = obs.clone()
          x = o2[g].clone()
          x[:, off : off + k] = x[:, off : off + k].mean(0, keepdim=True)
          o2[g] = x
          total[label] += (weight * (policy(o2) - act).square()).mean().item()
        count += 1
      obs, _, _, _ = wrapped.step(act)

  result = dict(sorted(((k, v / count) for k, v in total.items()), key=lambda kv: -kv[1]))
  for k, v in result.items():
    print(f"  {k:32s} {v:.4f}")
  Path(args.out).parent.mkdir(parents=True, exist_ok=True)
  Path(args.out).write_text(
    json.dumps({"task": args.task, "checkpoint": str(Path(args.checkpoint).resolve()), "action_change": result}, indent=1)
  )


if __name__ == "__main__":
  main()
