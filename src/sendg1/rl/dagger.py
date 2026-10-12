"""DAgger distillation of a privileged teacher into a deployable student.

rsl_rl's ``Distillation`` already does the core of DAgger: the student acts,
the frozen teacher labels every visited state, and the student regresses onto
the labels (MSE) with truncated BPTT through its recurrent state. Two things
are added here:

1. **Teacher-to-student control fade.** At the start the student's actions are
   garbage and it would only visit states the teacher never sees (falling off
   the beam within a second). For the first ``fade_iterations`` updates the
   teacher drives a share ``beta`` of the EPISODES, decaying linearly from
   ``beta_start`` to 0; after that the student drives alone. The draw is per
   env and per episode, not per step: mixing per step lets the untrained
   student's actions into every trajectory, and a robot on a 0.1 m beam falls
   off within a few such steps whoever drives the rest. Labels are always the
   teacher's, whoever drives.
2. **Loss in joint-target space.** The action is a joint position offset
   ``scale * a``, and the scales differ by 7x (wrist pitch/yaw 0.075, the rest
   0.35-0.55). Plain MSE on ``a`` let the wrists dominate: the teacher holds
   them at |a| ~ 4 (0.3 rad), so the first student's loss was mostly wrist
   error. Each dimension is weighted by ``scale^2`` (normalised to mean 1),
   i.e. MSE of the commanded joint targets.
3. **Bookkeeping** for the report: ``beta`` and the share of steps the teacher
   drove are logged with the losses.

The runner subclasses mjlab's runner so checkpoints keep the env's step counter
(curricula) and export to ONNX the same way as PPO runs.
"""

from __future__ import annotations

import torch
from rsl_rl.algorithms import Distillation
from tensordict import TensorDict

from mjlab.rl import MjlabOnPolicyRunner


class DAggerDistillation(Distillation):
  def __init__(
    self,
    *args,
    beta_start: float = 1.0,
    fade_iterations: int = 150,
    **kwargs,
  ) -> None:
    super().__init__(*args, **kwargs)
    self.beta_start = beta_start
    self.fade_iterations = fade_iterations
    self._teacher_steps = 0
    self._total_steps = 0
    self._teacher_drives: torch.Tensor | None = None

  def set_action_weights(self, weights: torch.Tensor) -> None:
    """Per action dimension loss weights (``DistillationRunner`` passes the
    normalised squared action scales)."""
    w = weights.to(self.device)

    def weighted_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
      return (w * (pred - target).square()).mean()

    self.loss_fn = weighted_mse

  @property
  def beta(self) -> float:
    if self.fade_iterations <= 0:
      return 0.0
    return max(0.0, self.beta_start * (1.0 - self.num_updates / self.fade_iterations))

  def act(self, obs: TensorDict) -> torch.Tensor:
    student = self.student(obs, stochastic_output=True).detach()
    teacher = self.teacher(obs).detach()
    self.transition.actions = student
    self.transition.privileged_actions = teacher
    self.transition.observations = obs
    if self._teacher_drives is None:
      self._teacher_drives = torch.rand(student.shape[0], device=student.device) < self.beta
    use_teacher = self._teacher_drives[:, None]
    self._teacher_steps += int(use_teacher.sum())
    self._total_steps += student.shape[0]
    return torch.where(use_teacher, teacher, student)

  def process_env_step(self, obs, rewards, dones, extras) -> None:
    super().process_env_step(obs, rewards, dones, extras)
    if self._teacher_drives is not None:
      done = dones.view(-1).bool()
      redraw = torch.rand(done.shape[0], device=done.device) < self.beta
      self._teacher_drives = torch.where(done, redraw, self._teacher_drives)

  def update(self) -> dict[str, float]:
    beta = self.beta
    share = self._teacher_steps / max(self._total_steps, 1)
    self._teacher_steps = self._total_steps = 0
    out = super().update()
    out["beta"] = beta
    out["teacher_drive_share"] = share
    return out


class DistillationRunner(MjlabOnPolicyRunner):
  """mjlab's runner (checkpoint env state, ONNX export) around a distillation
  algorithm. Refuses to train without a loaded teacher."""

  def __init__(self, env, train_cfg: dict, log_dir: str | None = None, device: str = "cpu"):
    # MjlabOnPolicyRunner strips None-valued options from actor/critic only.
    for key in ("student", "teacher"):
      if key in train_cfg:
        for opt in ("cnn_cfg", "distribution_cfg"):
          if train_cfg[key].get(opt) is None:
            train_cfg[key].pop(opt, None)
        if train_cfg[key].get("rnn_type") is None:
          for opt in ("rnn_type", "rnn_hidden_dim", "rnn_num_layers"):
            train_cfg[key].pop(opt, None)
    super().__init__(env, train_cfg, log_dir, device)
    scale = env.unwrapped.action_manager.get_term("joint_pos").scale
    scale = scale[0] if isinstance(scale, torch.Tensor) else torch.full((env.num_actions,), scale)
    w = scale.square()
    self.alg.set_action_weights(w / w.mean())

  def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
    if not self.alg.teacher_loaded:
      raise ValueError("load a teacher checkpoint before distilling")
    super().learn(num_learning_iterations, init_at_random_ep_len)

  def load_teacher(self, path: str, map_location: str | None = None) -> None:
    loaded = torch.load(path, map_location=map_location, weights_only=False)
    self.alg.load(loaded, {"teacher": True}, strict=True)
