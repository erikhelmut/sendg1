"""The scripted high level: where to walk, as a velocity command.

VOFA (arXiv:2605.01518) splits box pushing in two: a high-level policy that
emits walking commands, and a frozen low-level controller that executes them
while absorbing the contact forces. We take the same split, but SCRIPT the high
level instead of learning it.

Why scripted: in simulation the box and target poses are known, so the
high-level geometry -- stand on the far side of the box, then push along the
box-to-target line -- is a closed-form calculation, not something worth
learning. Scripting it also makes it a control variable: byte-identical for the
blind, contact-oracle and tactile policies, so it cannot absorb any of the
difference between them. What remains to be learned is exactly what this study
is about: executing a walking command while pushing a heavy box on a floor of
unknown friction, without the feet slipping out.

The command is the same (vx, vy, wz) twist the velocity tasks use, in the
robot's yaw frame, so the low-level policy's observation space is the velocity
task's, unchanged.

Two phases, per env, recomputed every step (no state, so a slip that moves the
robot is corrected on the next step):

  APPROACH  the robot is not behind the box: walk to the push point along an
            arc around the box, so it never walks through it.
  PUSH      the robot is behind the box: walk along the box-to-target line,
            with a lateral term that keeps it on the line and a yaw term that
            keeps it facing the target.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from sendg1.common.box_push.mdp import BoxPushState


@dataclass(frozen=True)
class PilotCfg:
  """Gains and limits of the scripted high level."""

  standoff: float = 0.45
  """m behind the box's near face where the robot should stand to push."""
  approach_speed: float = 0.8
  """m/s commanded while walking to the push point."""
  push_speed: float = 0.5
  """m/s commanded while pushing. Below the approach speed: a heavy box moves
  slower than the robot can walk, and commanding more than it can deliver just
  asks the feet to slip."""
  arc_bearing: float = 0.6
  """rad. While approaching, aim at most this far around the box per step, so
  the robot circles it instead of cutting through it."""
  on_line_gain: float = 1.0
  """1/s. Lateral correction per metre off the box-to-target line."""
  yaw_gain: float = 2.0
  """1/s per rad of heading error."""
  max_lin: float = 1.0
  max_lat: float = 0.4
  max_yaw: float = 1.0
  behind_angle: float = 0.9
  """rad. The robot counts as behind the box when its bearing from the box is
  within this of the push direction's opposite."""


def _wrap(a: torch.Tensor) -> torch.Tensor:
  return (a + torch.pi) % (2 * torch.pi) - torch.pi


def pilot_command(st: BoxPushState, cfg: PilotCfg = PilotCfg()) -> torch.Tensor:
  """[B, 3] twist command (vx, vy, wz) in the robot's yaw frame."""
  box = st.box_pos[:, :2]
  target = st.target_xy
  robot = st.robot.data.root_link_pos_w[:, :2]
  yaw = st.robot.data.heading_w

  to_target = target - box
  dist_to_target = to_target.norm(dim=-1, keepdim=True).clamp_min(1e-6)
  u = to_target / dist_to_target  # push direction, box -> target

  # Push point: behind the box, on the line through the target.
  half_along = (st.box_half[:, :2] * u.abs()).sum(dim=-1, keepdim=True)
  push_point = box - u * (half_along + cfg.standoff)

  # Bearing of the robot around the box, relative to the push direction.
  from_box = robot - box
  bearing = _wrap(
    torch.atan2(from_box[:, 1], from_box[:, 0]) - torch.atan2(-u[:, 1], -u[:, 0])
  )
  behind = bearing.abs() <= cfg.behind_angle

  # -- approach: aim at a point on the circle around the box, at most
  # ``arc_bearing`` closer to the push point than the robot is now.
  radius = from_box.norm(dim=-1, keepdim=True).clamp_min(0.3)
  step = torch.sign(bearing) * torch.minimum(
    bearing.abs(), torch.full_like(bearing, cfg.arc_bearing)
  )
  aim_angle = torch.atan2(-u[:, 1], -u[:, 0]) + bearing - step
  arc_point = box + radius * torch.stack([aim_angle.cos(), aim_angle.sin()], dim=-1)
  # Once around, walk straight at the push point.
  goal_xy = torch.where(behind.unsqueeze(-1), push_point, arc_point)

  to_goal = goal_xy - robot
  approach = to_goal * (cfg.approach_speed / to_goal.norm(dim=-1, keepdim=True).clamp_min(0.2))

  # -- push: along the push direction, plus a pull back onto the line.
  off_line = (torch.stack([-u[:, 1], u[:, 0]], dim=-1) * from_box).sum(
    dim=-1, keepdim=True
  )
  lateral = torch.stack([-u[:, 1], u[:, 0]], dim=-1) * (-cfg.on_line_gain * off_line)
  push = u * cfg.push_speed + lateral

  vel_w = torch.where(behind.unsqueeze(-1), push, approach)

  # Heading: face along the push direction while pushing, face where you are
  # going while approaching.
  desired_yaw = torch.where(
    behind,
    torch.atan2(u[:, 1], u[:, 0]),
    torch.atan2(to_goal[:, 1], to_goal[:, 0]),
  )
  wz = (cfg.yaw_gain * _wrap(desired_yaw - yaw)).clamp(-cfg.max_yaw, cfg.max_yaw)

  # World -> robot yaw frame.
  c, s = torch.cos(yaw), torch.sin(yaw)
  vx = (c * vel_w[:, 0] + s * vel_w[:, 1]).clamp(-cfg.max_lin, cfg.max_lin)
  vy = (-s * vel_w[:, 0] + c * vel_w[:, 1]).clamp(-cfg.max_lat, cfg.max_lat)

  # Stop driving into a box that has arrived: the dwell then plays out.
  arrived = (dist_to_target.squeeze(-1) <= st.ranges.target_radius) & behind
  vx = torch.where(arrived, torch.zeros_like(vx), vx)
  vy = torch.where(arrived, torch.zeros_like(vy), vy)
  return torch.stack([vx, vy, wz], dim=-1)
