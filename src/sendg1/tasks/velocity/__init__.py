"""Velocity tracking on flat and rough terrain, blind Unitree G1.

Phase 1 scope. Tasks 2-10 get their own sibling directories.
"""

from functools import partial

from sendg1.common.hand_forces import HandForceCfg
from sendg1.tasks._register import register_variants
from sendg1.tasks.velocity.env_cfg import make_g1_velocity_env_cfg
from sendg1.tasks.velocity.rl_cfg import LSTM_PPO, build_ppo_cfg

register_variants(
  "Sendg1-Velocity-Flat-G1",
  partial(make_g1_velocity_env_cfg, "flat"),
  build_ppo_cfg,
  experiment_prefix="velocity_flat",
)

# The beam LSTM's actor and PPO settings on flat ground, at the beam's 4096 envs:
# LSTM(256) -> MLP (256, 128), 64-step rollouts, LR 4e-4 at start (adaptive).
register_variants(
  "Sendg1-Velocity-Flat-G1-LSTM",
  partial(make_g1_velocity_env_cfg, "flat", num_envs=4096),
  partial(build_ppo_cfg, **LSTM_PPO),
  experiment_prefix="velocity_flat_lstm",
)

register_variants(
  "Sendg1-Velocity-Rough-G1",
  partial(make_g1_velocity_env_cfg, "rough"),
  build_ppo_cfg,
  experiment_prefix="velocity_rough",
)

# Flat ground plus VOFA's end-effector force curriculum, and nothing else. This
# is the LOW LEVEL for the box-push task: same 96-dim actor, same rewards, so a
# policy trained here drops straight into box pushing, where the scripted
# pilot's command takes the operator command's place.
register_variants(
  "Sendg1-VelocityPush-G1",
  partial(make_g1_velocity_env_cfg, "flat", hand_forces=HandForceCfg()),
  build_ppo_cfg,
  experiment_prefix="velocity_push",
)
