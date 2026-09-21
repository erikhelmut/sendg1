"""Beam walking over a pit: blind Unitree G1, no command."""

from functools import partial

from sendg1.tasks._register import register_variants
from sendg1.tasks.beam.env_cfg import STEPS_PER_ITERATION, make_g1_beam_env_cfg
from sendg1.tasks.velocity.rl_cfg import build_ppo_cfg

BEAM_ENTROPY_COEF = 0.005
"""Half the velocity tasks' 0.01. With 29 action dimensions and the beam's small
rewards, 0.01 let action std drift from 1.0 to 1.9 within 1,000 iterations in
two runs, and performance fell as it rose (posture reward 0.16 -> 0.004, episodes
~300 -> ~100 steps)."""

register_variants(
  "Sendg1-Beam-G1",
  make_g1_beam_env_cfg,
  partial(
    build_ppo_cfg,
    recurrent=True,
    num_steps_per_env=STEPS_PER_ITERATION,
    entropy_coef=BEAM_ENTROPY_COEF,
  ),
  experiment_prefix="beam",
)
