"""PPO config of the walking and box-push tasks.

Everything but the entropy coefficient and the observation groups comes from
``sendg1.common.ppo``, the settings every task shares: LSTM(256) -> MLP
(256, 128) actor, MLP (512, 256, 128) critic (privileged), 24-step rollouts,
LR 4e-4 (adaptive).
"""

from __future__ import annotations

from mjlab.rl import RslRlOnPolicyRunnerCfg

from sendg1.common.ppo import ppo_cfg
from sendg1.seeds import FIRST_SEED

ENTROPY_COEF = 0.01
"""mjlab's velocity default; box pushing halves it (``tasks/box_push``)."""


def build_ppo_cfg(
  experiment_name: str,
  obs_groups: dict[str, tuple[str, ...]],
  *,
  seed: int = FIRST_SEED,
  max_iterations: int = 30_000,
  entropy_coef: float = ENTROPY_COEF,
) -> RslRlOnPolicyRunnerCfg:
  return ppo_cfg(
    experiment_name,
    obs_groups,
    entropy_coef=entropy_coef,
    max_iterations=max_iterations,
    seed=seed,
  )
