"""Teacher PPO and student distillation configs.

Teacher: actor and critic are both MLP (512, 256, 128), ELU, running obs
normalisation, and both read ``real`` + ``privileged``. No memory: everything a
memory would have to infer is handed to it.

Student: GRU(256) -> MLP (256, 128) over ``real`` only. The GRU is the history:
truncated BPTT over ``gradient_length`` = 25 steps (0.5 s), rollouts of 50
steps (1 s). Trained by DAgger (``sendg1.rl.dagger``): MSE onto the frozen
teacher's actions, teacher control faded out over the first 150 updates.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.rl.config import RslRlBaseRunnerCfg

TEACHER_OBS = ("real", "privileged")
STUDENT_OBS = ("real",)
HIDDEN = (512, 256, 128)
STUDENT_RNN_HIDDEN = 256
STUDENT_HEAD = (256, 128)
ENTROPY_COEF = 0.005
"""The old beam task's lesson: 0.01 let a 29-dim action std drift to ~2 and
collapse training."""


def _teacher_actor() -> RslRlModelCfg:
  return RslRlModelCfg(
    hidden_dims=HIDDEN,
    activation="elu",
    obs_normalization=True,
    distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
  )


@dataclass
class SymmetricPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
  """PPO with rsl_rl's symmetry extension: mirrored copies of every mini-batch
  plus a mirror loss on the actor (``sendg1.common.beam_ts.symmetry``)."""

  symmetry_cfg: dict | None = field(
    default_factory=lambda: {
      "data_augmentation_func": "sendg1.common.beam_ts.symmetry:mirror_obs_actions",
      "use_data_augmentation": True,
      "use_mirror_loss": True,
      "mirror_loss_coeff": 0.1,
    }
  )


def teacher_ppo_cfg(seed: int = 0, max_iterations: int = 15_000) -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    seed=seed,
    obs_groups={"actor": TEACHER_OBS, "critic": TEACHER_OBS},
    actor=_teacher_actor(),
    critic=RslRlModelCfg(hidden_dims=HIDDEN, activation="elu", obs_normalization=True),
    algorithm=SymmetricPpoAlgorithmCfg(
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
    experiment_name="beam_ts_teacher",
    run_name=f"seed{seed}",
    save_interval=500,
    num_steps_per_env=24,
    max_iterations=max_iterations,
    logger="wandb",
    wandb_project="sendg1",
    upload_model=False,
  )


@dataclass
class DAggerAlgorithmCfg:
  class_name: str = "sendg1.rl.dagger:DAggerDistillation"
  num_learning_epochs: int = 1
  gradient_length: int = 25
  learning_rate: float = 5e-4
  max_grad_norm: float = 1.0
  loss_type: str = "mse"
  optimizer: str = "adam"
  beta_start: float = 1.0
  fade_iterations: int = 150


@dataclass
class DistillationRunnerCfg(RslRlBaseRunnerCfg):
  class_name: str = "DistillationRunner"
  student: RslRlModelCfg = field(
    default_factory=lambda: RslRlModelCfg(
      class_name="RNNModel",
      rnn_type="gru",
      rnn_hidden_dim=STUDENT_RNN_HIDDEN,
      rnn_num_layers=1,
      hidden_dims=STUDENT_HEAD,
      activation="elu",
      obs_normalization=True,
      # Exploration noise only: the MSE loss never touches the std.
      distribution_cfg={"class_name": "GaussianDistribution", "init_std": 0.1, "std_type": "scalar"},
    )
  )
  teacher: RslRlModelCfg = field(default_factory=_teacher_actor)
  algorithm: DAggerAlgorithmCfg = field(default_factory=DAggerAlgorithmCfg)


def student_distill_cfg(seed: int = 0, max_iterations: int = 3_000) -> DistillationRunnerCfg:
  return DistillationRunnerCfg(
    seed=seed,
    obs_groups={"student": STUDENT_OBS, "teacher": TEACHER_OBS},
    experiment_name="beam_ts_student",
    run_name=f"seed{seed}",
    save_interval=250,
    num_steps_per_env=50,
    max_iterations=max_iterations,
    logger="wandb",
    wandb_project="sendg1",
    upload_model=False,
  )
