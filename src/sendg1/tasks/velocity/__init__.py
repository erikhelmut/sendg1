"""Velocity tracking on flat and rough terrain, Unitree G1, blind or tactile.

Every task trains the standard actor and PPO (``sendg1.common.ppo``).
"""

from functools import partial

from sendg1.tasks._register import register_variants
from sendg1.tasks.velocity.env_cfg import make_g1_velocity_env_cfg
from sendg1.tasks.velocity.rl_cfg import build_ppo_cfg

register_variants(
  "Sendg1-Velocity-Flat-G1",
  partial(make_g1_velocity_env_cfg, "flat"),
  build_ppo_cfg,
  experiment_prefix="velocity_flat",
)

register_variants(
  "Sendg1-Velocity-Rough-G1",
  partial(make_g1_velocity_env_cfg, "rough"),
  build_ppo_cfg,
  experiment_prefix="velocity_rough",
)
