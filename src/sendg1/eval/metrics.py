"""Metrics computed under the evaluation protocol.

The PRIMARY metric must be chosen before results are collected, or the study
degenerates into a search for a metric where tactile wins. Declared here:

  primary   : linear velocity tracking error (m/s), lower is better.
  secondary : angular velocity tracking error, fall rate, mean episode length,
              foot slip, contact-force impulse at touchdown.

Reporting all of them is fine. Promoting a secondary to primary after seeing
results is not.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import torch

PRIMARY_METRIC = "lin_vel_error"
LOWER_IS_BETTER = {
  "lin_vel_error": True,
  "ang_vel_error": True,
  "fall_rate": True,
  "foot_slip": True,
  "touchdown_impulse": True,
  "episode_length_s": False,
  "mean_terrain_level": False,
}


@dataclass
class TerrainStats:
  """Per-terrain-type accumulators.

  Reporting one aggregate over a mixed curriculum would average the negative
  controls (flat, random_rough) into the discriminators and hide the effect the
  study is looking for. Terrain type index maps to a name by column order --
  curriculum mode gives each sub-terrain exactly one column.
  """

  name: str
  role: str
  lin_vel_error_sum: float = 0.0
  ang_vel_error_sum: float = 0.0
  step_count: int = 0
  falls: int = 0
  episodes: int = 0
  level_sum: float = 0.0
  level_n: int = 0

  def summary(self) -> dict[str, float]:
    steps = max(self.step_count, 1)
    eps = max(self.episodes, 1)
    return {
      "lin_vel_error": self.lin_vel_error_sum / steps,
      "ang_vel_error": self.ang_vel_error_sum / steps,
      "fall_rate": self.falls / eps,
      "mean_terrain_level": self.level_sum / max(self.level_n, 1),
      "n_episodes": float(self.episodes),
    }


@dataclass
class EpisodeStats:
  """Running accumulators over an evaluation run."""

  lin_vel_error_sum: float = 0.0
  ang_vel_error_sum: float = 0.0
  step_count: int = 0
  falls: int = 0
  episodes: int = 0
  episode_length_sum_s: float = 0.0
  per_command: dict[str, list[float]] = field(default_factory=dict)

  def add_step(self, lin_err: torch.Tensor, ang_err: torch.Tensor) -> None:
    self.lin_vel_error_sum += float(lin_err.sum())
    self.ang_vel_error_sum += float(ang_err.sum())
    self.step_count += int(lin_err.numel())

  def add_terminations(self, fell: torch.Tensor, lengths_s: torch.Tensor) -> None:
    self.falls += int(fell.sum())
    self.episodes += int(fell.numel())
    self.episode_length_sum_s += float(lengths_s.sum())

  def summary(self) -> dict[str, float]:
    steps = max(self.step_count, 1)
    eps = max(self.episodes, 1)
    return {
      "lin_vel_error": self.lin_vel_error_sum / steps,
      "ang_vel_error": self.ang_vel_error_sum / steps,
      "fall_rate": self.falls / eps,
      "episode_length_s": self.episode_length_sum_s / eps,
      "n_steps": float(self.step_count),
      "n_episodes": float(self.episodes),
    }


def summarize(stats: EpisodeStats) -> dict:
  return {"metrics": stats.summary(), "raw": asdict(stats)}
