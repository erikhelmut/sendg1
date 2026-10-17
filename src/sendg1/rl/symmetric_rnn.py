"""A recurrent actor that is left-right symmetric by construction.

rsl_rl refuses symmetry augmentation for recurrent policies, and without it
the early beam policies settled into lopsided gaits. Instead of augmenting data,
this actor runs one GRU on two streams, the observation and its mirror image,
each with its own hidden state, and averages:

    a = 0.5 * (f(o, h) + M_a f(M_o o, h'))

Mirroring the input sequence swaps the two streams, so the policy is exactly
equivariant: pi(M_o o) = M_a pi(o), at every step and for any weights.

The hidden state is stored as [layers, B, 2H] (the two streams side by side),
so rsl_rl's rollout storage and recurrent mini-batches handle it like a plain
GRU's. The observation normaliser is updated with both streams, which keeps
its statistics mirror-symmetric.

The mirror maps come from the env (``sendg1.common.beam.symmetry``) and are
registered by a startup event (``mdp.register_actor_mirror``), because rsl_rl
builds models from tensors only.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from rsl_rl.models.rnn_model import RNNModel
from rsl_rl.utils import unpad_trajectories
from tensordict import TensorDict

_MIRRORS: dict = {}


def set_mirrors(obs: dict[str, tuple[torch.Tensor, torch.Tensor]], actions: tuple[torch.Tensor, torch.Tensor]) -> None:
  """``obs[group]`` / ``actions``: (source index, sign) with
  ``mirrored = x[..., src] * sign``."""
  _MIRRORS["obs"] = obs
  _MIRRORS["actions"] = actions


def _apply(x: torch.Tensor, m: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
  src, sign = m
  return x[..., src.to(x.device)] * sign.to(x.device, x.dtype)


class _TwoStreamGRU(nn.Module):
  """rsl_rl ``RNN`` interface over two streams sharing one GRU."""

  def __init__(self, input_size: int, hidden_dim: int, num_layers: int) -> None:
    super().__init__()
    self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_dim, num_layers=num_layers)
    self.hidden_dim = hidden_dim
    self.hidden_state: torch.Tensor | None = None

  def _split(self, h: torch.Tensor | None, b: int, like: torch.Tensor) -> torch.Tensor:
    if h is None:
      return torch.zeros(self.rnn.num_layers, 2 * b, self.hidden_dim, device=like.device, dtype=like.dtype)
    return torch.cat([h[..., : self.hidden_dim], h[..., self.hidden_dim :]], dim=1)

  def _join(self, h: torch.Tensor, b: int) -> torch.Tensor:
    return torch.cat([h[:, :b], h[:, b:]], dim=-1)

  def forward(self, x: torch.Tensor, xm: torch.Tensor, masks=None, hidden_state=None) -> torch.Tensor:
    """Rollout: x [B, D] -> [B, 2H]. Update: x [T, B, D] padded -> unpadded [N, 2H]."""
    if masks is not None:
      if hidden_state is None:
        raise ValueError("Hidden states not passed to the RNN during the policy update")
      b = x.shape[1]
      out, _ = self.rnn(torch.cat([x, xm], dim=1), self._split(hidden_state, b, x))
      return torch.cat([unpad_trajectories(out[:, :b], masks), unpad_trajectories(out[:, b:], masks)], dim=-1)
    b = x.shape[0]
    out, h = self.rnn(torch.cat([x, xm], dim=0).unsqueeze(0), self._split(self.hidden_state, b, x))
    self.hidden_state = self._join(h, b)
    return torch.cat([out[0, :b], out[0, b:]], dim=-1)

  def reset(self, dones=None, hidden_state=None) -> None:
    if dones is None:
      self.hidden_state = hidden_state
    elif self.hidden_state is not None:
      if hidden_state is not None:
        raise NotImplementedError("resetting done envs to a custom hidden state")
      self.hidden_state[..., dones == 1, :] = 0.0

  def detach_hidden_state(self, dones=None) -> None:
    if self.hidden_state is None:
      return
    if dones is None:
      self.hidden_state = self.hidden_state.detach()
    else:
      self.hidden_state[..., dones == 1, :] = self.hidden_state[..., dones == 1, :].detach()


class SymmetricRNNModel(RNNModel):
  """GRU actor, mirror-equivariant by construction (see module docstring).
  Use as ``class_name="sendg1.rl.symmetric_rnn:SymmetricRNNModel"`` with
  ``rnn_type="gru"``."""

  def __init__(self, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    if not isinstance(self.rnn.rnn, nn.GRU):
      raise ValueError("SymmetricRNNModel supports rnn_type='gru' only")
    layers, hidden = self.rnn.rnn.num_layers, self.rnn.rnn.hidden_size
    self.rnn = _TwoStreamGRU(self.obs_dim, hidden, layers)
    self.hidden_size = hidden

  def _raw(self, obs: TensorDict) -> tuple[torch.Tensor, torch.Tensor]:
    if "obs" not in _MIRRORS:
      raise RuntimeError("mirror maps not registered (startup event mdp.register_actor_mirror)")
    x = torch.cat([obs[g] for g in self.obs_groups], dim=-1)
    xm = torch.cat([_apply(obs[g], _MIRRORS["obs"][g]) for g in self.obs_groups], dim=-1)
    return x, xm

  def get_latent(self, obs: TensorDict, masks=None, hidden_state=None) -> torch.Tensor:
    x, xm = self._raw(obs)
    return self.rnn(self.obs_normalizer(x), self.obs_normalizer(xm), masks, hidden_state)

  def update_normalization(self, obs: TensorDict) -> None:
    if self.obs_normalization:
      x, xm = self._raw(obs)
      self.obs_normalizer.update(torch.cat([x, xm], dim=0))  # type: ignore[union-attr]

  def forward(self, obs: TensorDict, masks=None, hidden_state=None, stochastic_output: bool = False) -> torch.Tensor:
    latent = self.get_latent(obs, masks, hidden_state)
    la, lb = latent[..., : self.hidden_size], latent[..., self.hidden_size :]
    out = 0.5 * (self.mlp(la) + _apply(self.mlp(lb), _MIRRORS["actions"]))
    if self.distribution is not None:
      if stochastic_output:
        self.distribution.update(out)
        return self.distribution.sample()
      return self.distribution.deterministic_output(out)
    return out

  def as_jit(self) -> nn.Module:
    raise NotImplementedError("export of the symmetric GRU actor is not implemented yet")

  def as_onnx(self, verbose: bool = False) -> nn.Module:
    raise NotImplementedError("export of the symmetric GRU actor is not implemented yet")
