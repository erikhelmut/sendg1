"""Left-right mirror of the pedalo task's observations and actions, for the
symmetric GRU actor (``sendg1.rl.symmetric_rnn``).

The robot's terms mirror as in the beam task (``beam.symmetry.term_mirrors``).
The pedalo is its own mirror image with the boards swapped: the right board
runs half a turn behind the left, so the mirrored state has crank angle
phi + pi (sin and cos flip sign, the crank rate does not).
"""

from __future__ import annotations

from functools import lru_cache

import torch

from sendg1.common.beam.symmetry import _signs, _swap_blocks, group_mirror, term_mirrors


def pedalo_term_mirrors(
  joint_names: list[str], action_names: list[str]
) -> dict[str, list[tuple[int, float]]]:
  return {
    **term_mirrors(joint_names, action_names),
    # v_fwd, v_lat, yaw rate, sin phi, cos phi, crank rate
    "pedalo_motion": _signs(1, -1, -1, -1, -1, 1),
    # per foot: along, across, height, sin yaw err, cos yaw err
    "feet_board_pose": _swap_blocks(5, (1, -1, 1, -1, 1)),
    # base fwd, lat, up, sin/cos heading err, CoM fwd, lat
    "base_pedalo_pose": _signs(1, -1, 1, -1, 1, 1, -1),
    # per foot (fwd, lat) relative to its board
    "foot_slip_on_board": _swap_blocks(2, (1, -1)),
    # per foot (fwd, lat, up) in the pedalo frame
    "foot_pedalo_forces": _swap_blocks(3, (1, -1, 1)),
  }


@lru_cache(maxsize=4)
def maps(env) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
  base = env.unwrapped
  om = base.observation_manager
  mirrors = pedalo_term_mirrors(
    list(base.scene["robot"].joint_names),
    list(base.action_manager.get_term("joint_pos")._target_names),
  )
  out = {}
  for group in om.active_terms:
    dims = [int(torch.tensor(d).prod()) for d in om.group_obs_term_dim[group]]
    s, g = group_mirror(om.active_terms[group], dims, mirrors)
    out[group] = (s.to(base.device), g.to(base.device))
  s, g = group_mirror(["actions"], [base.action_manager.total_action_dim], mirrors)
  out["__actions__"] = (s.to(base.device), g.to(base.device))
  return out
