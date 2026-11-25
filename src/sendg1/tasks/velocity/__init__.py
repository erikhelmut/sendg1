"""Velocity tracking on flat and rough terrain, Unitree G1, blind or tactile.

Every task trains the standard actor and PPO (``sendg1.common.ppo``: LSTM,
24-step rollouts, LR 4e-4). ``Sendg1-Velocity-Flat-G1-LSTM`` was that same
setup under its own name until 2026-11-25; its checkpoints load in
``Sendg1-Velocity-Flat-G1``.
"""

from functools import partial

from sendg1.common.hand_forces import HandForceCfg
from sendg1.tasks._register import register_variants
from sendg1.tasks.velocity.env_cfg import make_g1_velocity_env_cfg
from sendg1.tasks.velocity.rl_cfg import build_ppo_cfg

register_variants(
  "Sendg1-Velocity-Flat-G1",
  partial(make_g1_velocity_env_cfg, "flat"),
  build_ppo_cfg,
  experiment_prefix="velocity_flat",
)

# The memoryless ablation of the flat task: an MLP actor on the current
# observation only, everything else as the standard (LSTM) task.
register_variants(
  "Sendg1-Velocity-Flat-G1-MLP",
  partial(make_g1_velocity_env_cfg, "flat"),
  partial(build_ppo_cfg, actor="mlp"),
  experiment_prefix="velocity_flat_mlp",
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
