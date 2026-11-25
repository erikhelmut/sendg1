#!/usr/bin/env python
"""Record a walking policy and measure its gait (``sendg1.eval.gait``).

    python scripts/gait_analysis.py --task Sendg1-Velocity-Flat-G1-Tactile \\
      --checkpoint <model.pt> --out results/gait/<name>.json

Straight walking at fixed forward speeds (``--speeds``), ``--envs-per-speed``
robots each, on the evaluation protocol's flat ground and fixed seed, so a
blind and a tactile policy face the same robots and the same randomisation.
``--condition clean`` (default) turns off observation noise, the tactile
sensor's faults and noise, and pushes: the gait itself. ``perturbed`` keeps
them all, as in training.

Every measure is simulator ground truth. The soles' contact sensors are added
to a blind task's scene for the measurement; they never reach its actor.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
REPO = Path(__file__).resolve().parent.parent


def _rpy(quat):
  """wxyz quaternions [..., 4] -> roll, pitch, yaw [..., 3]."""
  import torch

  w, x, y, z = quat.unbind(-1)
  roll = torch.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
  pitch = torch.asin((2 * (w * y - z * x)).clamp(-1, 1))
  yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
  return torch.stack([roll, pitch, yaw], -1)


def _leg_limits(joint_names: list[str]) -> list[float]:
  """Torque limits of ``joint_names`` from mjlab's G1 actuator table."""
  from mjlab.asset_zoo.robots.unitree_g1 import g1_constants as c

  table = [
    (expr, a.effort_limit)
    for n in dir(c)
    if n.startswith("G1_ACTUATOR_")
    for a in [getattr(c, n)]
    for expr in a.target_names_expr
  ]
  out = []
  for j in joint_names:
    hits = [lim for expr, lim in table if re.fullmatch(expr, j)]
    assert len(hits) == 1, (j, hits)
    out.append(float(hits[0]))
  return out


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--task", required=True)
  ap.add_argument("--checkpoint", required=True)
  ap.add_argument("--out", required=True)
  ap.add_argument("--speeds", type=float, nargs="+", default=[0.5, 1.0, 1.5])
  ap.add_argument("--envs-per-speed", type=int, default=32)
  ap.add_argument(
    "--warmup", type=float, default=4.0, help="seconds discarded per episode"
  )
  ap.add_argument("--duration", type=float, default=20.0, help="seconds measured")
  ap.add_argument("--condition", choices=("clean", "perturbed"), default="clean")
  args = ap.parse_args()

  import numpy as np
  import torch
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_rl_cfg, load_runner_cls
  from mjlab.utils.torch import configure_torch_backends

  import sendg1.tasks  # noqa: F401
  from sendg1.common.sensors import SOLE_CONTACT_SENSOR
  from sendg1.common.tactile.sensing import (
    TactileParams,
    read_tactile,
    tactile_scene_sensors,
  )
  from sendg1.eval import gait
  from sendg1.eval.protocol import EvalProtocol, build_eval_env_cfg

  configure_torch_backends()
  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  clean = args.condition == "clean"
  n_env = args.envs_per_speed * len(args.speeds)
  protocol = EvalProtocol(
    num_envs=n_env,
    episode_length_s=args.warmup + args.duration + 1.0,
    num_episodes=1,
    keep_corruption=not clean,
    keep_pushes=not clean,
  )
  env_cfg = build_eval_env_cfg(args.task, protocol)
  if clean:
    # The tactile sensor model (gain errors, dead taxels, noise, dropped
    # frames) lives inside the observation term, not in the observation
    # corruption switch, so it is swapped for the noise-free features here;
    # otherwise "clean" would be clean only for the blind policy.
    from sendg1.common.beam import mdp as beam_mdp

    for group in env_cfg.observations.values():
      for term in group.terms.values():
        if term.func is beam_mdp.tactile_touch_noisy:
          term.func = beam_mdp.tactile_touch_clean
  if not any(s.name == SOLE_CONTACT_SENSOR for s in env_cfg.scene.sensors):
    env_cfg.scene.sensors = (*env_cfg.scene.sensors, *tactile_scene_sensors())

  rl_cfg = load_rl_cfg(args.task)
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=rl_cfg.clip_actions)
  runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
  runner = runner_cls(wrapped, asdict(rl_cfg), device=device)
  runner.load(args.checkpoint, load_cfg={"actor": True}, strict=True, map_location=device)
  policy = runner.get_inference_policy(device=device)

  robot = env.scene["robot"]
  legs = [[f"{s}_{j}_joint" for j in gait.LEG_JOINTS] for s in gait.SIDES]
  leg_ids = [robot.find_joints(names, preserve_order=True)[0] for names in legs]
  limits = _leg_limits(legs[0])
  foot_ids, _ = robot.find_bodies(
    [f"{s}_ankle_roll_link" for s in gait.SIDES], preserve_order=True
  )
  mm = env.sim.mj_model
  pelvis = [i for i in range(mm.nbody) if mm.body(i).name.split("/")[-1] == "pelvis"]
  assert len(pelvis) == 1, pelvis
  mass = float(mm.body_subtreemass[pelvis[0]])
  params = TactileParams()

  speeds = torch.tensor(args.speeds, device=device)
  cmd = torch.zeros(n_env, 3, device=device)
  cmd[:, 0] = speeds.repeat_interleave(args.envs_per_speed)
  cmd_term = env.command_manager.get_term("twist")

  warm = round(args.warmup / env.step_dt)
  steps = warm + round(args.duration / env.step_dt)
  keys = (
    "leg_q",
    "leg_qd",
    "leg_qdd",
    "leg_tau",
    "all_tau",
    "all_qd",
    "grf",
    "cop",
    "foot_pos",
    "foot_vel",
    "base_pos",
    "base_rpy",
    "valid",
  )
  buf: dict[str, list] = {k: [] for k in keys}
  alive = torch.ones(n_env, dtype=torch.bool, device=device)
  obs, _ = wrapped.reset()
  print(f"[gait] {args.task} {args.condition}: {steps} steps x {n_env} envs")
  with torch.inference_mode():
    for t in range(steps):
      cmd_term.vel_command_b[:] = cmd
      obs, _, dones, _ = wrapped.step(policy(obs))
      cmd_term.vel_command_b[:] = cmd
      alive &= ~dones.bool()
      d = robot.data
      reading = read_tactile(env, params)
      cop = reading.cop.clone()
      cop[reading.net_force[..., 2] < gait.OFF_N] = float("nan")
      buf["leg_q"].append(torch.stack([d.joint_pos[:, i] for i in leg_ids], 1))
      buf["leg_qd"].append(torch.stack([d.joint_vel[:, i] for i in leg_ids], 1))
      buf["leg_qdd"].append(torch.stack([d.joint_acc[:, i] for i in leg_ids], 1))
      buf["leg_tau"].append(torch.stack([d.qfrc_actuator[:, i] for i in leg_ids], 1))
      buf["all_tau"].append(d.qfrc_actuator.clone())
      buf["all_qd"].append(d.joint_vel.clone())
      buf["grf"].append(reading.net_force.clone())
      buf["cop"].append(cop[..., :2])
      buf["foot_pos"].append(d.body_link_pos_w[:, foot_ids].clone())
      buf["foot_vel"].append(d.body_link_lin_vel_w[:, foot_ids].clone())
      buf["base_pos"].append(d.root_link_pos_w.clone())
      buf["base_rpy"].append(_rpy(d.root_link_quat_w))
      buf["valid"].append(alive & (t >= warm))
  env.close()
  arr = {k: torch.stack(v).cpu().numpy() for k, v in buf.items()}

  result = {
    "task": args.task,
    "checkpoint": args.checkpoint,
    "condition": args.condition,
    "protocol": {
      **asdict(protocol),
      "warmup_s": args.warmup,
      "duration_s": args.duration,
    },
    "mass_kg": mass,
    "leg_joints": list(gait.LEG_JOINTS),
    "torque_limits_nm": limits,
    "angles": [a[0] for a in gait.ANGLES],
    "falls": int((~alive).sum()),
    "speeds": {},
  }
  for i, v in enumerate(args.speeds):
    sl = slice(i * args.envs_per_speed, (i + 1) * args.envs_per_speed)
    rec = gait.Recording(
      dt=env.step_dt,
      mass=mass,
      leg_limit=np.array(limits),
      **{k: a[:, sl] for k, a in arr.items()},
    )
    summary = gait.summarize(gait.strides(rec))
    summary["falls"] = int((~alive[sl]).sum())
    result["speeds"][f"{v:.2f}"] = summary
    s = summary.get("scalars", {})
    if s:
      print(
        f"  {v:.2f} m/s: {summary['n_strides']} strides, speed {s['speed'][0]:.2f}, "
        f"cadence {s['cadence'][0]:.0f}/min, stride {s['stride_length'][0]:.2f} m, "
        f"ankle push-off {s['ankle_pitch_work_pos'][0]:.3f} J/kg, falls {summary['falls']}"
      )
  out = Path(args.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(json.dumps(result))
  print(f"[gait] wrote {out}")


if __name__ == "__main__":
  main()
