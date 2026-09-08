"""Velocity tracking on flat and rough terrain, blind Unitree G1.

Phase 1 scope. Tasks 2-10 get their own sibling directories.
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
