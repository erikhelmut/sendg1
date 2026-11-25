"""PPO config of the beam-style tasks (beam, pedalo, rolling cylinder, slackline).

Everything but the entropy coefficient and the observation groups comes from
``sendg1.common.ppo``, the settings every task shares: LSTM(256) -> MLP
(256, 128) actor over the ``actor`` group, MLP (512, 256, 128) critic over the
``critic`` group (ground truth), 24-step rollouts, LR 4e-4 (adaptive).
``actor="gru"`` / ``"mlp"`` swap in an ablation actor.
"""

from __future__ import annotations

from mjlab.rl import RslRlOnPolicyRunnerCfg

from sendg1.common.ppo import STANDARD_ACTOR, ActorKind, ppo_cfg
from sendg1.seeds import FIRST_SEED

ENTROPY_COEF = 0.005
"""The first beam task's lesson: 0.01 let a 29-dim action std drift to ~2 and
collapse training."""

__all__ = ["ENTROPY_COEF", "ActorKind", "beam_ppo_cfg"]


def beam_ppo_cfg(
  seed: int = FIRST_SEED,
  max_iterations: int = 20_000,
  experiment_name: str = "beam_touch_curved_lstm",
  actor: ActorKind = STANDARD_ACTOR,
) -> RslRlOnPolicyRunnerCfg:
  return ppo_cfg(
    experiment_name,
    {"actor": ("actor",), "critic": ("critic",)},
    entropy_coef=ENTROPY_COEF,
    max_iterations=max_iterations,
    save_interval=500,
    seed=seed,
    actor=actor,
  )
