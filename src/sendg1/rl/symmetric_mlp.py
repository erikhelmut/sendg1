"""A feedforward actor that is left-right symmetric by construction.

The memoryless counterpart of ``sendg1.rl.symmetric_rnn``: one MLP on the
current observation only (no history, no hidden state), run on the observation
and on its mirror image and averaged:

    a = 0.5 * (f(o) + M_a f(M_o o))

so pi(M_o o) = M_a pi(o) for any weights. The observation normaliser is updated
with both streams, which keeps its statistics mirror-symmetric. The mirror maps
are the ones the startup event ``mdp.register_actor_mirror`` registers for the
symmetric GRU.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from rsl_rl.models.mlp_model import MLPModel
from tensordict import TensorDict

from sendg1.rl.symmetric_rnn import _MIRRORS, _apply


class SymmetricMLPModel(MLPModel):
  """MLP actor, mirror-equivariant by construction (see module docstring).
  Use as ``class_name="sendg1.rl.symmetric_mlp:SymmetricMLPModel"``."""

  def _raw(self, obs: TensorDict) -> tuple[torch.Tensor, torch.Tensor]:
    if "obs" not in _MIRRORS:
      raise RuntimeError("mirror maps not registered (startup event mdp.register_actor_mirror)")
    x = torch.cat([obs[g] for g in self.obs_groups], dim=-1)
    xm = torch.cat([_apply(obs[g], _MIRRORS["obs"][g]) for g in self.obs_groups], dim=-1)
    return x, xm

  def update_normalization(self, obs: TensorDict) -> None:
    if self.obs_normalization:
      x, xm = self._raw(obs)
      self.obs_normalizer.update(torch.cat([x, xm], dim=0))  # type: ignore[union-attr]

  def forward(self, obs: TensorDict, masks=None, hidden_state=None, stochastic_output: bool = False) -> torch.Tensor:
    x, xm = self._raw(obs)
    out = 0.5 * (
      self.mlp(self.obs_normalizer(x)) + _apply(self.mlp(self.obs_normalizer(xm)), _MIRRORS["actions"])
    )
    if self.distribution is not None:
      if stochastic_output:
        self.distribution.update(out)
        return self.distribution.sample()
      return self.distribution.deterministic_output(out)
    return out

  def as_jit(self) -> nn.Module:
    raise NotImplementedError("export of the symmetric MLP actor is not implemented yet")

  def as_onnx(self, verbose: bool = False) -> nn.Module:
    raise NotImplementedError("export of the symmetric MLP actor is not implemented yet")
