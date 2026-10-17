"""Left-right mirror of the beam task's observations and actions.

The GRU actor is mirror-equivariant by construction
(``sendg1.rl.symmetric_rnn``): it runs on the observation and on its mirror
image and averages. This module supplies the maps. Without symmetry the early
beam policies settled into lopsided gaits (one foot stepping past the other on
99% of steps, with a fast lunge of one leg).

The mirror is the robot's sagittal plane. A straight beam is its own mirror
image, and a curved beam mirrors into one curving the other way, so the
symmetry holds on curved beams too (the path observations are in the beam
frame and only flip sign).

Per term (``term_mirrors``), a list of ``(source index, sign)`` pairs: output
element ``i`` = ``sign * input[source]``. Joint-space terms swap ``left_`` and
``right_`` joints and negate roll and yaw joints (their axes lie in the mirror
plane's normal direction); ``tests/test_beam.py`` checks this against the
G1's forward kinematics. Tactile features are already in a per-foot canonical
frame (``tactile.py``: +y lateral on both feet), so mirroring only swaps the
two feet's blocks.
"""

from __future__ import annotations

from functools import lru_cache

import torch

TACTILE_PER_FOOT = 33
BEAM_AHEAD_DIMS = 4
"""Must match ``mdp.BEAM_AHEAD_M``."""

_FLIP_JOINT = ("roll", "yaw")


def joint_mirror(names: list[str]) -> list[tuple[int, float]]:
  """(source index, sign) per joint for a list of G1 joint names."""
  index = {n: i for i, n in enumerate(names)}
  out = []
  for n in names:
    if n.startswith("left_"):
      src = "right_" + n[len("left_") :]
    elif n.startswith("right_"):
      src = "left_" + n[len("right_") :]
    else:
      src = n
    sign = -1.0 if any(k in n for k in _FLIP_JOINT) else 1.0
    out.append((index[src], sign))
  return out


def _signs(*s: float) -> list[tuple[int, float]]:
  return [(i, v) for i, v in enumerate(s)]


def _swap_blocks(block: int, signs: tuple[float, ...] | None = None) -> list[tuple[int, float]]:
  """Two per-foot blocks [left, right] of ``block`` elements each: swap them,
  applying ``signs`` within the block."""
  signs = signs or (1.0,) * block
  return [(((i // block) ^ 1) * block + i % block, signs[i % block]) for i in range(2 * block)]


def term_mirrors(
  joint_names: list[str], action_names: list[str] | None = None
) -> dict[str, list[tuple[int, float]]]:
  """``joint_names``: the robot's joint order (joint_pos/joint_vel obs);
  ``action_names``: the action term's order (actions obs and the actions)."""
  joints = joint_mirror(joint_names)
  acts = joint_mirror(action_names if action_names is not None else joint_names)
  tactile = _swap_blocks(TACTILE_PER_FOOT)
  return {
    # actor
    "base_ang_vel": _signs(-1, 1, -1),
    "projected_gravity": _signs(1, -1, 1),
    "joint_pos": joints,
    "joint_vel": joints,
    "actions": acts,
    "command_vx": _signs(1),
    "tactile_map": tactile,
    # privileged
    "tactile_map_clean": tactile,
    "base_lin_vel": _signs(1, -1, 1),
    # lateral, edge margin, height, sin(yaw err), cos(yaw err)
    "feet_beam_pose": _swap_blocks(5, (-1, 1, 1, -1, 1)),
    # lateral, sin, cos, height, CoM lateral
    "base_beam_pose": _signs(-1, -1, 1, 1, -1),
    "beam_progress": _signs(1, 1, 1),
    "beam_params": _signs(1, 1, 1),
    # turn angles ahead: a left turn mirrors into a right turn
    "beam_ahead": [(i, -1.0) for i in range(BEAM_AHEAD_DIMS)],
    # world-frame net force per foot (x, y, z); the beam runs along x
    "foot_contact_forces": _swap_blocks(3, (1, -1, 1)),
    "action_lag": _signs(1),
  }


def group_mirror(term_names, term_dims, mirrors) -> tuple[torch.Tensor, torch.Tensor]:
  """Concatenate per-term mirrors into (source index, sign) for a whole group."""
  src, sign, offset = [], [], 0
  for name, dim in zip(term_names, term_dims, strict=True):
    m = mirrors[name]
    if len(m) != dim:
      raise ValueError(f"mirror of {name!r} has {len(m)} entries, term has {dim}")
    src += [offset + s for s, _ in m]
    sign += [g for _, g in m]
    offset += dim
  return torch.tensor(src), torch.tensor(sign)


@lru_cache(maxsize=4)
def _maps(env) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
  base = env.unwrapped
  om = base.observation_manager
  mirrors = term_mirrors(
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
