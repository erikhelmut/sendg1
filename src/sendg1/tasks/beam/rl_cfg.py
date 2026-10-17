"""PPO config of the beam task.

Actor: GRU(256) -> MLP (256, 128) over the ``actor`` group, mirror-equivariant
by construction (``sendg1.rl.symmetric_rnn``; rsl_rl's symmetry augmentation
does not support recurrent actors). Critic: MLP (512, 256, 128) over the
``critic`` group (ground truth). Both normalise their inputs. 32-step
rollouts: BPTT over 0.64 s.
"""

from __future__ import annotations

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

RNN_HIDDEN = 256
ACTOR_HEAD = (256, 128)
CRITIC_HIDDEN = (512, 256, 128)
ENTROPY_COEF = 0.005
"""The first beam task's lesson: 0.01 let a 29-dim action std drift to ~2 and
collapse training."""


def beam_ppo_cfg(
  seed: int = 0, max_iterations: int = 20_000, experiment_name: str = "beam_e2e"
) -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    seed=seed,
    obs_groups={"actor": ("actor",), "critic": ("critic",)},
    actor=RslRlModelCfg(
      class_name="sendg1.rl.symmetric_rnn:SymmetricRNNModel",
      rnn_type="gru",
      rnn_hidden_dim=RNN_HIDDEN,
      rnn_num_layers=1,
      hidden_dims=ACTOR_HEAD,
      activation="elu",
      obs_normalization=True,
      distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
    ),
    critic=RslRlModelCfg(hidden_dims=CRITIC_HIDDEN, activation="elu", obs_normalization=True),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=ENTROPY_COEF,
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
    save_interval=500,
    num_steps_per_env=32,
    max_iterations=max_iterations,
    logger="wandb",
    wandb_project="sendg1",
    upload_model=False,
  )
