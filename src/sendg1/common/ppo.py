"""The one PPO configuration every task trains with.

ARCHITECTURE AND PPO SETTINGS ARE CONTROL VARIABLES, across observation sets
and across tasks: blind and tactile must train identically, and one task's
result should be comparable with another's. So the settings live here, once,
and task packages pick only what is genuinely theirs (the actor's and critic's
observation groups, the entropy coefficient, the iteration budget, the
experiment name).

Standard (2026-11-25): an LSTM(256) -> MLP (256, 128) actor, MLP (512, 256,
128) critic, both ELU with running obs normalisation; 24-step rollouts in 8
minibatches; adaptive learning rate starting at 4e-4. These are the settings
the beam v3/v4 and walking r24 runs used (there as CLI overrides of 64-step
defaults).

Changing anything here invalidates every run that came before it.
"""

from __future__ import annotations

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

from sendg1.seeds import FIRST_SEED

RNN_HIDDEN = 256
RNN_HEAD = (256, 128)
"""MLP head after the recurrent layer."""
CRITIC_HIDDEN = (512, 256, 128)
ACTIVATION = "elu"

NUM_ENVS = 4096
"""Default parallel envs of every task (2026-11-25). With ``ROLLOUT_STEPS`` it
sets the batch per update (98k samples), so it is a training setting, not a
speed knob. A task or GPU that cannot fit it is trained with fewer
(``--env.scene.num-envs``), the same number for blind and tactile, and the
run name says so (``_n2048``)."""
ROLLOUT_STEPS = 24
"""Steps per env per update, and the LSTM's truncated-BPTT window (0.48 s)."""
LEARNING_RATE = 4.0e-4
"""Start of the adaptive (KL-targeted) schedule."""
NUM_MINI_BATCHES = 8
NUM_LEARNING_EPOCHS = 5
DESIRED_KL = 0.01

_DISTRIBUTION = {"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"}


def actor_cfg() -> RslRlModelCfg:
  return RslRlModelCfg(
    class_name="RNNModel",
    rnn_type="lstm",
    rnn_hidden_dim=RNN_HIDDEN,
    rnn_num_layers=1,
    hidden_dims=RNN_HEAD,
    activation=ACTIVATION,
    obs_normalization=True,
    distribution_cfg=dict(_DISTRIBUTION),
  )


def ppo_cfg(
  experiment_name: str,
  obs_groups: dict[str, tuple[str, ...]],
  *,
  entropy_coef: float,
  max_iterations: int,
  save_interval: int,
  seed: int = FIRST_SEED,
) -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    seed=seed,
    obs_groups=obs_groups,
    actor=actor_cfg(),
    critic=RslRlModelCfg(
      hidden_dims=CRITIC_HIDDEN, activation=ACTIVATION, obs_normalization=True
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=entropy_coef,
      num_learning_epochs=NUM_LEARNING_EPOCHS,
      num_mini_batches=NUM_MINI_BATCHES,
      learning_rate=LEARNING_RATE,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=DESIRED_KL,
      max_grad_norm=1.0,
    ),
    experiment_name=experiment_name,
    run_name=f"seed{seed}",
    save_interval=save_interval,
    num_steps_per_env=ROLLOUT_STEPS,
    max_iterations=max_iterations,
    logger="wandb",
    wandb_project="sendg1",
    # Log metrics to W&B, but keep .pt/.onnx files local: checkpoints are
    # reproducible from a seed plus a pinned config.
    upload_model=False,
  )
