"""A curriculum of sustained external forces at the hands.

VOFA (arXiv:2605.01518) does not train its low level on the pushing task. It
trains a locomotion controller (FALCON) separately, with "a curriculum that
progressively increases external forces at the end-effectors", then FREEZES it
and learns a high level that sends it walking commands. The low level therefore
learns one thing: walk as told while something heavy pushes back.

This module is that curriculum for our G1. It is deliberately the ONLY thing
added to the flat velocity task: the reward set, the command, the terrain and
the observation space are the velocity task's, untouched, so

  * the resulting policy is a drop-in for any task with the same 96-dim actor
    (the box-push low level takes the scripted pilot's command in exactly the
    place the operator's command sits here), and
  * whatever it gains is attributable to the force curriculum alone.

The force is a *sustained* load, not a bump: sampled once per episode and held,
which is what leaning into a box feels like. Its direction is mostly opposite
the robot's heading -- a box resists where you push it -- with lateral and
vertical components so the policy cannot assume a pure headwind.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

HAND_BODIES: tuple[str, ...] = ("left_wrist_yaw_link", "right_wrist_yaw_link")
"""Where the load is applied: the bodies that carry the hands."""


@dataclass(frozen=True)
class HandForceCfg:
  """The curriculum. Magnitudes are the TOTAL over both hands."""

  max_force: float = 200.0
  """N at full difficulty. The box-push task's median episode needs 125 N and
  its 95th percentile 198 N, so a policy comfortable here covers that task."""
  ramp_steps: int = 3000 * 24
  """Env steps over which the ceiling grows from zero (3,000 PPO iterations at
  24 steps per iteration). Ramped rather than fixed because a 200 N load from
  the first iteration is a policy that only ever learns to fall over."""
  backward_spread: float = math.radians(60.0)
  """The horizontal direction is within this of straight backward."""
  vertical_frac: float = 0.25
  """Vertical component, as a fraction of the horizontal magnitude."""
  min_frac: float = 0.0
  """Lower end of the per-episode magnitude draw, as a fraction of the ceiling.
  Zero keeps unloaded walking in the mix at every difficulty."""


def force_level(env: ManagerBasedRlEnv, cfg: HandForceCfg) -> float:
  """Fraction of ``max_force`` currently allowed, from the iteration count."""
  if cfg.ramp_steps <= 0:
    return 1.0
  return min(1.0, env.common_step_counter / cfg.ramp_steps)


class HandForceState:
  """The load each env is carrying this episode, in the robot's own frame.

  Kept in the robot frame and rotated into the world every step, for two
  reasons: a box resists where the robot pushes it, so the load should turn
  with the robot; and a reset event cannot read ``heading_w`` -- the base was
  repositioned earlier in the same reset and derived quantities do not refresh
  until after the events run, so a direction computed there points wherever the
  PREVIOUS episode happened to end.
  """

  _ATTR = "_sendg1_hand_force"

  def __init__(self, env: ManagerBasedRlEnv, cfg: HandForceCfg) -> None:
    self.cfg = cfg
    self._env = env
    self.robot = env.scene["robot"]
    self.body_ids, _ = self.robot.find_bodies(HAND_BODIES, preserve_order=True)
    n, dev = env.num_envs, env.device
    self.magnitude = torch.zeros(n, device=dev)
    self.bearing = torch.zeros(n, device=dev)
    """Horizontal direction relative to the robot's heading; pi is straight back."""
    self.vertical = torch.zeros(n, device=dev)

  @classmethod
  def get(cls, env: ManagerBasedRlEnv, cfg: HandForceCfg) -> "HandForceState":
    state = getattr(env, cls._ATTR, None)
    if state is None:
      state = cls(env, cfg)
      setattr(env, cls._ATTR, state)
    return state

  def resample(self, env_ids: torch.Tensor) -> None:
    cfg = self.cfg
    n, dev = len(env_ids), self._env.device
    ceiling = cfg.max_force * force_level(self._env, cfg)
    self.magnitude[env_ids] = (
      cfg.min_frac + (1.0 - cfg.min_frac) * torch.rand(n, device=dev)
    ) * ceiling
    self.bearing[env_ids] = math.pi + cfg.backward_spread * (
      2.0 * torch.rand(n, device=dev) - 1.0
    )
    self.vertical[env_ids] = (
      cfg.vertical_frac * self.magnitude[env_ids] * (2.0 * torch.rand(n, device=dev) - 1.0)
    )

  def apply(self) -> None:
    angle = self.robot.data.heading_w + self.bearing
    total = torch.stack(
      [self.magnitude * angle.cos(), self.magnitude * angle.sin(), self.vertical],
      dim=-1,
    )
    forces = (total / len(self.body_ids))[:, None, :].expand(-1, len(self.body_ids), -1)
    self.robot.write_external_wrench_to_sim(
      forces.contiguous(), torch.zeros_like(forces), body_ids=self.body_ids
    )


def reset_hand_forces(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  cfg: HandForceCfg = HandForceCfg(),
) -> None:
  """Event (mode="reset"): draw this episode's hand load."""
  state = HandForceState.get(env, cfg)
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  state.resample(env_ids)


def apply_hand_forces(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  cfg: HandForceCfg = HandForceCfg(),
) -> None:
  """Event (mode="step"): hold the load, rotated into the world frame."""
  del env_ids
  HandForceState.get(env, cfg).apply()


def hand_force_curriculum(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  cfg: HandForceCfg = HandForceCfg(),
) -> dict[str, float]:
  """Curriculum term: reports the ramp. The ramp itself is read per reset by
  :func:`apply_hand_forces`, so this only logs it."""
  del env_ids
  level = force_level(env, cfg)
  return {"level": level, "max_force_n": level * cfg.max_force}
