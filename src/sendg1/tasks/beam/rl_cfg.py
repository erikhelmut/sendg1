"""PPO config of the beam task.

Actor (``ActorKind``) over the ``actor`` group, all rsl_rl's own models:
``gru``, ``RNNModel`` with a GRU(256) -> MLP (256, 128); ``lstm``, the same
with an LSTM(256); ``mlp``, a memoryless ``MLPModel`` (512, 256, 128) on the
current observation only. Critic: MLP
(512, 256, 128) over the ``critic`` group (ground truth). Both normalise their
inputs. 32-step rollouts by default (``num_steps_per_env``): a GRU then learns
from gradients over 0.64 s, under one step of the slow beam gait.
"""

from __future__ import annotations

from typing import Literal

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

RNN_HIDDEN = 256
ACTOR_HEAD = (256, 128)
CRITIC_HIDDEN = (512, 256, 128)
MLP_ACTOR_HIDDEN = (512, 256, 128)
"""The memoryless actor's MLP: the critic's size, in place of GRU + head."""
ENTROPY_COEF = 0.005
"""The first beam task's lesson: 0.01 let a 29-dim action std drift to ~2 and
collapse training."""


ActorKind = Literal["gru", "lstm", "mlp"]
_DISTRIBUTION = {"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"}


def _actor_cfg(actor: ActorKind) -> RslRlModelCfg:
  if actor == "mlp":
    return RslRlModelCfg(
      class_name="MLPModel",
      hidden_dims=MLP_ACTOR_HIDDEN,
      activation="elu",
      obs_normalization=True,
      distribution_cfg=dict(_DISTRIBUTION),
    )
  return RslRlModelCfg(
    class_name="RNNModel",
    rnn_type="lstm" if actor == "lstm" else "gru",
    rnn_hidden_dim=RNN_HIDDEN,
    rnn_num_layers=1,
    hidden_dims=ACTOR_HEAD,
    activation="elu",
    obs_normalization=True,
    distribution_cfg=dict(_DISTRIBUTION),
  )


def beam_ppo_cfg(
  seed: int = 0,
  max_iterations: int = 20_000,
  experiment_name: str = "beam_touch_curved_plain",
  actor: ActorKind = "gru",
  num_steps_per_env: int = 32,
  learning_rate: float = 1.0e-3,
) -> RslRlOnPolicyRunnerCfg:
  """``learning_rate`` is the adaptive schedule's starting value (it then
  follows the KL target)."""
  return RslRlOnPolicyRunnerCfg(
    seed=seed,
    obs_groups={"actor": ("actor",), "critic": ("critic",)},
    actor=_actor_cfg(actor),
    critic=RslRlModelCfg(hidden_dims=CRITIC_HIDDEN, activation="elu", obs_normalization=True),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=ENTROPY_COEF,
      num_learning_epochs=5,
      # Minibatches of 4096 * 32 / 4 samples whatever the rollout length: a
      # 64-step rollout in 4 minibatches doubled the update's memory and ran
      # the simulator out of GPU memory on a 24 GB card.
      num_mini_batches=4 * num_steps_per_env // 32,
      learning_rate=learning_rate,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name=experiment_name,
    run_name=f"seed{seed}",
    save_interval=500,
    num_steps_per_env=num_steps_per_env,
    max_iterations=max_iterations,
    logger="wandb",
    wandb_project="sendg1",
    upload_model=False,
  )
