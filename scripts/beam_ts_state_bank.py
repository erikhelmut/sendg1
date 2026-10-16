#!/usr/bin/env python
"""Record mid-beam robot states of a beam teacher, as reset states for E2E PPO.

    python scripts/beam_ts_state_bank.py --task G1-Beam-Teacher \
        --checkpoint runs/beam_ts_teacher/.../model_10000.pt \
        --out results/beam_ts/v4_beam_states.pt

The privileged v4 teacher walks heel-to-toe on one line (feet ~1 cm apart
laterally, ~31 cm steps). The end-to-end policy shuffles with its feet side by
side and never reaches the narrow levels, where it could learn better; and
it cannot start on a narrow beam from the default stance, whose feet are 0.31 m
apart. These states, taken on the 0.05-0.10 m beams (valid on every wider one),
let a share of its episodes start mid-beam in the teacher's stance
(``mdp.spawn_from_bank``), i.e. reference-state initialisation.

Stored in the beam frame: joint state, base height above the beam top, lateral
offset, orientation and twist relative to the beam's yaw.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--task", default="G1-Beam-Teacher")
  ap.add_argument("--checkpoint", required=True)
  ap.add_argument("--envs-per-level", type=int, default=64)
  ap.add_argument("--min-level", type=int, default=4, help="widths from this level on (4 = 0.10 m)")
  ap.add_argument("--steps", type=int, default=900)
  ap.add_argument("--every", type=int, default=10)
  ap.add_argument("--out", required=True)
  args = ap.parse_args()

  import torch
  from mjlab.utils.lab_api.math import quat_apply_inverse, quat_from_euler_xyz, quat_mul

  import beam_ts_eval as ev
  import sendg1.tasks  # noqa: F401
  from sendg1.common.beam_ts.mdp import BeamEpisode, _feet_in_contact

  eargs = argparse.Namespace(
    envs_per_level=args.envs_per_level, seed=0, start_lateral=0.08, start_yaw=0.08,
    pushes=False, speed=0.3, task=args.task, checkpoint=args.checkpoint, policy="teacher",
  )
  env = ev.build_env(eargs, args.task)
  wrapped, policy = ev.load_policy(eargs, env)
  robot = env.scene["robot"]
  ep = BeamEpisode.get(env)
  geo = ep.geo
  keep: dict[str, list] = {k: [] for k in ("joint_pos", "joint_vel", "height", "lateral", "quat", "lin_vel", "ang_vel")}
  obs, _ = wrapped.reset()
  with torch.inference_mode():
    for i in range(args.steps):
      obs, _, dones, _ = wrapped.step(policy(obs))
      policy.reset(dones.bool())
      if i < 100 or i % args.every:
        continue
      feet = ep.feet_proj()
      beam = geo.beam_arc()
      on = ((feet.arc > beam[:, :1] + 0.3) & (feet.arc < beam[:, 1:] - 0.3)).all(-1)
      on &= feet.lateral.abs().max(-1).values < 0.5 * geo.width()
      on &= _feet_in_contact(env).any(-1) & ~dones.bool() & (geo.levels >= args.min_level)
      idx = on.nonzero()[:, 0]
      if len(idx) == 0:
        continue
      base = ep.base_proj()
      yaw = base.tangent_yaw[idx, 0]
      zero = torch.zeros_like(yaw)
      q_beam = quat_from_euler_xyz(zero, zero, yaw)
      d = robot.data
      keep["joint_pos"].append(d.joint_pos[idx])
      keep["joint_vel"].append(d.joint_vel[idx])
      keep["height"].append(d.root_link_pos_w[idx, 2] - geo.top_z()[idx])
      keep["lateral"].append(base.lateral[idx, 0])
      inv = q_beam * torch.tensor([1.0, -1.0, -1.0, -1.0], device=env.device)
      keep["quat"].append(quat_mul(inv, d.root_link_quat_w[idx]))
      keep["lin_vel"].append(quat_apply_inverse(q_beam, d.root_link_lin_vel_w[idx]))
      keep["ang_vel"].append(quat_apply_inverse(q_beam, d.root_link_ang_vel_w[idx]))
  bank = {k: torch.cat(v).cpu() for k, v in keep.items()}
  Path(args.out).parent.mkdir(parents=True, exist_ok=True)
  torch.save({**bank, "task": args.task, "checkpoint": str(Path(args.checkpoint).resolve())}, args.out)
  n = len(bank["height"])
  print(f"wrote {args.out}: {n} states, |lateral| {bank['lateral'].abs().mean() * 100:.1f} cm, "
        f"forward speed {bank['lin_vel'][:, 0].mean():.2f} m/s")


if __name__ == "__main__":
  main()
