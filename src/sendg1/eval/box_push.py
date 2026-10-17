"""Evaluation protocol for the box-pushing task.

A grid over floor friction x box mass, N episodes per cell. Every episode's
world -- floor mu, box mass, box mu, box size, spawn gap, target distance and
angle -- is drawn up front from a numpy RNG seeded by the protocol and PINNED
into the env (``BoxPushState.pins``), so every policy is scored on exactly the
same episodes, independent of how the simulator's own RNG is consumed. Only the
robot's +/-0.1 rad spawn yaw and the observation noise come from the env seed.

Each env runs exactly one episode. Its record is read from
``BoxPushState.last_episode`` on the step it ends.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from sendg1.common.box_push.mdp import PINNABLE, BoxPushState
from sendg1.common.box_push.params import BoxPushRanges

if TYPE_CHECKING:
  import torch
  from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg


@dataclass(frozen=True)
class BoxPushProtocol:
  version: str = "box_push_v1"
  floor_mu: tuple[float, ...] = (0.25, 0.35, 0.45, 0.55, 0.70)
  """Spans the training range. 0.2 and below would be mostly infeasible."""
  box_mass: tuple[float, ...] = (10.0, 20.0, 30.0, 40.0)
  episodes_per_cell: int = 32
  seed: int = 12345
  """Seeds both the world draws and the env."""
  keep_corruption: bool = True
  """Observation noise stays on, as in training."""
  video_floor_mu: tuple[float, ...] = (0.25, 0.70)
  """Cells recorded on video: lowest and highest floor mu, every mass."""
  videos_per_cell: int = 2

  @property
  def cells(self) -> list[tuple[float, float]]:
    return [(mu, m) for mu in self.floor_mu for m in self.box_mass]

  @property
  def num_episodes(self) -> int:
    return len(self.cells) * self.episodes_per_cell


EPISODE_FIELDS: tuple[str, ...] = (
  "success",
  "fell_before_contact",
  "fell_after_contact",
  "traction_margin",
  "time_to_target",
  "slip_events",
  "fell",
  "box_tipped",
  "mean_push_force",
  "double_support_impulse_frac",
  "floor_mu",
  "box_mass",
  "box_mu",
  "final_distance",
  "length_s",
)


@dataclass
class EpisodeWorlds:
  """One pinned world per episode, in cell-major order."""

  cell: np.ndarray
  values: dict[str, np.ndarray] = field(default_factory=dict)


def draw_worlds(
  protocol: BoxPushProtocol, ranges: BoxPushRanges = BoxPushRanges()
) -> EpisodeWorlds:
  """Every episode's world, from the protocol seed alone."""
  rng = np.random.default_rng(protocol.seed)
  cells = protocol.cells
  n = protocol.episodes_per_cell
  cell = np.repeat(np.arange(len(cells)), n)
  total = len(cell)
  u = lambda lo_hi: rng.uniform(*lo_hi, total)  # noqa: E731
  values = {
    "floor_mu": np.array([cells[c][0] for c in cell]),
    "box_mass": np.array([cells[c][1] for c in cell]),
    "box_mu": u(ranges.box_mu),
    "box_depth": u(ranges.box_depth),
    "box_width": u(ranges.box_width),
    "box_height": u(ranges.box_height),
    "robot_gap": u(ranges.robot_gap),
    "target_distance": u(ranges.target_distance),
    "target_angle": u(ranges.target_angle),
  }
  # A grid pins mass and floor mu directly; the traction margin is then
  # whatever they imply, so it is left unpinned.
  assert set(values) <= set(PINNABLE)
  return EpisodeWorlds(cell=cell, values=values)


def build_eval_env_cfg(
  task_id: str, protocol: BoxPushProtocol, num_envs: int
) -> ManagerBasedRlEnvCfg:
  from mjlab.tasks.registry import load_env_cfg

  cfg = load_env_cfg(task_id, play=False)
  cfg.scene.num_envs = num_envs
  cfg.seed = protocol.seed
  cfg.curriculum = {}  # the grid pins the ranges; alpha stays at 1.0
  if not protocol.keep_corruption:
    for group in cfg.observations.values():
      group.enable_corruption = False
  return cfg


def pin_worlds(env: ManagerBasedRlEnv, worlds: EpisodeWorlds, index: np.ndarray) -> None:
  """Pin world ``index[i]`` into env ``i``. Takes effect at the next reset."""
  import torch

  st = BoxPushState.get(env)
  st.alpha = 1.0
  for name in PINNABLE:
    if name not in worlds.values:
      # Not drawn by the protocol: the grid pins mass and floor mu, so the
      # traction margin is whatever they imply. Leave it free.
      st.pins[name][:] = float("nan")
      continue
    st.pins[name][:] = torch.as_tensor(
      worlds.values[name][index], device=env.device, dtype=torch.float32
    )


def record_row(env: ManagerBasedRlEnv, env_id: int) -> dict[str, float]:
  rec = BoxPushState.get(env).last_episode
  return {name: float(getattr(rec, name)[env_id]) for name in EPISODE_FIELDS}


def overlay(frame: np.ndarray, lines: list[str]) -> np.ndarray:
  from PIL import Image, ImageDraw, ImageFont

  img = Image.fromarray(frame)
  draw = ImageDraw.Draw(img)
  try:
    font = ImageFont.truetype("DejaVuSansMono.ttf", 18)
  except OSError:
    font = ImageFont.load_default()
  y = 10
  for line in lines:
    draw.text((12, y), line, fill=(255, 255, 255), font=font, stroke_width=2,
              stroke_fill=(0, 0, 0))
    y += 24
  return np.asarray(img)


def status_lines(env: ManagerBasedRlEnv, t: float, label: str = "") -> list[str]:
  st = BoxPushState.get(env)
  lines = [label] if label else []
  lines += [
    f"t {t:5.2f} s   floor mu {st.floor_mu[0]:.2f}   box {st.box_mass[0]:.0f} kg"
    f"   box mu {st.box_mu[0]:.2f}",
    f"box->target {st.dist[0]:.2f} m   slip events {int(st.slip_events[0])}"
    f"   push {st.horizontal_push_force()[0]:.0f} N",
  ]
  return lines


def to_dict(protocol: BoxPushProtocol) -> dict:
  return asdict(protocol)


def configure_video_view(cfg: ManagerBasedRlEnvCfg, width: int, height: int) -> None:
  for sensor in cfg.scene.sensors or ():
    if hasattr(sensor, "debug_vis"):
      sensor.debug_vis = False
  cfg.viewer.width = width
  cfg.viewer.height = height


def roll_out_video(
  wrapped,  # RslRlVecEnvWrapper around a 1-env ManagerBasedRlEnv
  act,  # obs -> actions
  obs,
  *,
  max_steps: int,
  label: str,
  reset_policy=None,
) -> tuple[list[np.ndarray], dict[str, float], object]:
  """Run env 0 for one episode, rendering every step. Returns the frames, the
  episode record and the observation after the auto-reset.

  The frame of the terminal step is dropped: auto-reset has already replaced
  the world by the time it could be rendered.
  """
  import torch

  env = wrapped.unwrapped
  frames = []
  record: dict[str, float] = {}
  with torch.inference_mode():
    for step in range(max_steps):
      obs, _, dones, _ = wrapped.step(act(obs))
      if bool(dones[0]):
        if reset_policy is not None:
          reset_policy(dones.bool())
        record = record_row(env, 0)
        break
      frame = env.render()
      if frame is not None:
        t = (step + 1) * env.step_dt
        frames.append(overlay(frame, status_lines(env, t, label)))
  return frames, record, obs
