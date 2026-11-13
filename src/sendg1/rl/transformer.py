"""A small causal transformer actor over a fixed window of past observations.

The actor group is built with ``history_length=H, flatten_history_dim=False``,
so every step hands the model the last H observation frames, ``[B, H, D]``,
oldest first. Every ``PATCH`` consecutive frames become one token (a linear
embedding of the stacked frames plus a learned position embedding; 64 frames
-> 16 tokens), a pre-norm transformer encoder runs over the tokens with a
causal mask (token t attends only to tokens <= t), and the newest token's
output feeds an MLP head and the action distribution.

Why patches: with one token per frame the PPO update took 2.9 s per iteration
at 1024 envs (the GRU's: 0.1 s), ~60 h for a 20k-iteration run at 4096 envs,
and its activations did not fit a 24 GB GPU. Four frames per token keeps the
whole 1.28 s window at a quarter of the tokens.

Why a window instead of a hidden state: the GRU learns from gradients that
span one rollout (32 steps, 0.64 s) and carries a state that can drift over a
long episode; here the context is exactly the window, in training as in
deployment, and PPO treats the model as feedforward (no hidden state, no
truncated BPTT). The causal mask means a frame's encoding never depends on
frames after it, so on the robot the per-frame encodings of earlier steps can
be cached and only the newest frame computed.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from rsl_rl.models.mlp_model import MLPModel
from tensordict import TensorDict

D_MODEL = 64
NUM_LAYERS = 2
NUM_HEADS = 4
FF_DIM = 128
"""Kept small: PPO is known to destabilise with large transformer actors."""
PATCH = 4
"""Frames per token (20 ms each: 80 ms per token)."""


class TransformerModel(MLPModel):
  """Use as ``class_name="sendg1.rl.transformer:TransformerModel"`` with an
  actor group of shape ``[B, H, D]`` (see module docstring)."""

  def __init__(self, obs: TensorDict, obs_groups, obs_set: str, output_dim: int, **kwargs) -> None:
    # MLPModel builds the normaliser over D and the head over D_MODEL inputs.
    super().__init__(obs, obs_groups, obs_set, output_dim, **kwargs)
    if self.history % PATCH:
      raise ValueError(f"history {self.history} is not a multiple of PATCH={PATCH}")
    h = self.history // PATCH
    self.embed = nn.Linear(PATCH * self.obs_dim, D_MODEL)
    self.pos = nn.Parameter(torch.randn(1, h, D_MODEL) * 0.02)
    self.layers = nn.ModuleList(
      nn.TransformerEncoderLayer(
        D_MODEL, NUM_HEADS, FF_DIM, dropout=0.0, activation="gelu", batch_first=True, norm_first=True
      )
      for _ in range(NUM_LAYERS)
    )
    self.out_norm = nn.LayerNorm(D_MODEL)
    self.register_buffer("causal_mask", nn.Transformer.generate_square_subsequent_mask(h), persistent=False)

  def _get_obs_dim(self, obs: TensorDict, obs_groups, obs_set: str) -> tuple[list[str], int]:
    groups = obs_groups[obs_set]
    shapes = [obs[g].shape for g in groups]
    if any(len(s) != 3 for s in shapes) or len({s[1] for s in shapes}) != 1:
      raise ValueError(f"TransformerModel needs [B, H, D] groups with one H, got {shapes}")
    self.history = int(shapes[0][1])
    return list(groups), sum(int(s[2]) for s in shapes)

  def _get_latent_dim(self) -> int:
    return D_MODEL

  def get_latent(self, obs: TensorDict, masks=None, hidden_state=None) -> torch.Tensor:
    x = self.obs_normalizer(torch.cat([obs[g] for g in self.obs_groups], dim=-1))
    b, h, d = x.shape
    # bf16 for the encoder only: on Ampere it runs the flash attention kernel
    # and halves the update time; the head and the log-probs stay in fp32.
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=x.is_cuda):
      z = self.embed(x.reshape(b, h // PATCH, PATCH * d)) + self.pos
      for layer in self.layers:
        z = layer(z, self.causal_mask, is_causal=True)
    return self.out_norm(z[:, -1].float())

  def update_normalization(self, obs: TensorDict) -> None:
    # Every frame of the window was the newest frame once: the newest alone
    # gives the same statistics at 1/H of the cost.
    if self.obs_normalization:
      self.obs_normalizer.update(torch.cat([obs[g][:, -1] for g in self.obs_groups], dim=-1))  # type: ignore[union-attr]

  def as_jit(self) -> nn.Module:
    raise NotImplementedError("export of the transformer actor is not implemented yet")

  def as_onnx(self, verbose: bool = False) -> nn.Module:
    raise NotImplementedError("export of the transformer actor is not implemented yet")
