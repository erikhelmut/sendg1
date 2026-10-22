"""Roll out the flat walking policy; save qpos + tactile readings per step."""

import os
import sys
from pathlib import Path
from dataclasses import asdict

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

import sendg1.tasks  # noqa: F401
from sendg1.common.sensors import SOLE_CONTACT_SENSOR
from sendg1.common.tactile.sensing import TactileParams, read_tactile, tactile_scene_sensors

TASK = "Sendg1-Velocity-Flat-G1-Blind"
REPO = Path(__file__).resolve().parents[2]
CKPT = str(REPO / "runs/velocity_flat__blind/2026-09-08_18-22-37_seed0/model_29999.pt")
D = os.environ.get("POSTER_DIR", str(REPO / "media" / "poster")) + "/"
os.makedirs(D, exist_ok=True)
OUT = sys.argv[1] if len(sys.argv) > 1 else D + "walk.npz"

params = TactileParams()
cfg = load_env_cfg(TASK, play=True)
cfg.scene.num_envs = 1
sensors = tuple(cfg.scene.sensors or ())
if not any(s.name == SOLE_CONTACT_SENSOR for s in sensors):
  sensors += tactile_scene_sensors(params)
for s in sensors:
  if hasattr(s, "debug_vis"):
    s.debug_vis = False
cfg.scene.sensors = sensors
for g in cfg.observations.values():
  g.enable_corruption = False
env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
rl = load_rl_cfg(TASK)
w = RslRlVecEnvWrapper(env, clip_actions=rl.clip_actions)
runner = (load_runner_cls(TASK) or MjlabOnPolicyRunner)(w, asdict(rl), device="cuda:0")
runner.load(CKPT, load_cfg={"actor": True}, strict=True, map_location="cuda:0")
policy = runner.get_inference_policy(device="cuda:0")
cmd_term = env.command_manager.get_term("twist")

log = {k: [] for k in ("qpos", "normal", "cop", "fz")}
obs, _ = w.reset()
with torch.inference_mode():
  for i in range(300):
    vx = 0.0 if i < 100 else 0.6
    cmd_term.vel_command_b[:] = torch.tensor([[vx, 0.0, 0.0]], device="cuda:0")
    obs, _, dones, _ = w.step(policy(obs))
    policy.reset(dones.bool())
    r = read_tactile(env, params)
    log["qpos"].append(env.sim.data.qpos[0].cpu().numpy().copy())
    log["normal"].append(r.normal[0].cpu().numpy())
    log["cop"].append(r.cop[0].cpu().numpy())
    log["fz"].append(r.net_force[0, :, 2].cpu().numpy())
np.savez(OUT, dt=env.step_dt, **{k: np.stack(v) for k, v in log.items()})
# The model, for offline rendering with plain mujoco.
import mujoco  # noqa: E402

mujoco.mj_saveModel(env.sim.mj_model, OUT.replace(".npz", ".mjb"), None)
print("saved", OUT)
