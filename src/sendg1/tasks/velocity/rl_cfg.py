"""PPO configuration.

ARCHITECTURE IS A CONTROL VARIABLE. It must be identical across observation
sets, and changing it invalidates every run that came before -- both conditions
must then be re-run. Keep it in this one function so the blast radius is
obvious.

Current: actor and critic are both MLPModel (512, 256, 128) with ELU and running
obs normalisation; the actor emits a scalar-std Gaussian. rsl_rl also supports
RNNModel (lstm/gru), CNNModel, and fully custom classes via a "module:Class"
string -- relevant later for a tactile encoder, and for the fact that blind
locomotion usually benefits from memory. A cheaper first step than an RNN is
per-term ``history_length`` on the proprio group (frame stacking), which is a
config change rather than a new model.
"""

from __future__ import annotations

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

HIDDEN_DIMS: tuple[int, ...] = (512, 256, 128)
ACTIVATION = "elu"

SAVE_INTERVAL = 2000
"""Iterations between checkpoints."""


def build_ppo_cfg(
  experiment_name: str,
  obs_groups: dict[str, tuple[str, ...]],
  *,
  seed: int = 0,
  max_iterations: int = 30_000,
) -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    seed=seed,
    obs_groups=obs_groups,
    actor=RslRlModelCfg(
      hidden_dims=HIDDEN_DIMS,
      activation=ACTIVATION,
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=HIDDEN_DIMS,
      activation=ACTIVATION,
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name=experiment_name,
    run_name=f"seed{seed}",
    # 30k iterations / 2000 = ~15 checkpoints per run. Enough to resume from and
    # to sample learning progress, without filling the disk: each G1 checkpoint
    # is a few MB and this study will accumulate many runs.
    save_interval=SAVE_INTERVAL,
    num_steps_per_env=24,
    max_iterations=max_iterations,
    logger="wandb",
    wandb_project="sendg1",
    # Log metrics to W&B, but keep .pt/.onnx files local. Checkpoints are
    # reproducible from a seed plus a pinned config; uploading every run's
    # weights burns storage for little benefit.
    upload_model=False,
  )
