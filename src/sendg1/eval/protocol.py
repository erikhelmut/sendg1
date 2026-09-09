"""The evaluation protocol. This is the product of the study, so it lives in
version control rather than in a command line.

Both conditions must see *identical* conditions: same command schedule, same
terrain, same domain-randomisation draws, same episode count and length. The
protocol is therefore code, versioned alongside the results it produced.

Deliberately NOT mjlab's play config: play mode is for looking at a policy
(infinite episodes, randomised terrain on reset, widened command ranges). It is
not a measurement instrument.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.tasks.registry import load_env_cfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg


@dataclass(frozen=True)
class EvalProtocol:
  """A fixed measurement setup. Change it and you must re-evaluate everything."""

  version: str = "v1"
  num_envs: int = 512
  episode_length_s: float = 20.0
  num_episodes: int = 4
  """Episodes per environment. Total episodes = num_envs * num_episodes."""
  seed: int = 12345
  """Environment seed. Fixed across conditions so DR draws and terrain match."""

  command_grid: tuple[tuple[float, float, float], ...] = field(
    default_factory=lambda: (
      # (vx, vy, wz). Spans the trained range; includes standing and pure turns.
      (0.0, 0.0, 0.0),
      (0.5, 0.0, 0.0),
      (1.0, 0.0, 0.0),
      (-0.5, 0.0, 0.0),
      (0.0, 0.5, 0.0),
      (0.0, 0.0, 0.5),
      (0.5, 0.0, 0.5),
      (1.0, 0.5, -0.5),
    )
  )

  keep_corruption: bool = True
  """Keep observation noise on. The question is whether tactile helps a policy
  operating under realistic sensing, so evaluating noise-free would flatter both
  conditions and could mask exactly the effect being measured."""

  keep_pushes: bool = True
  """Keep the interval push disturbances. Recovery from perturbation is where
  contact information is most plausibly useful."""

  terrain_seed: int = 7777
  """Terrain layout seed for evaluation. Deliberately different from
  TRAIN_TERRAIN_SEED so policies are scored on terrain instances they never saw,
  and pinned so every condition is scored on the *same* instances."""

  uniform_terrain_weights: bool = True
  """On generated terrain, give every sub-terrain the same number of
  environments regardless of its training spawn weight."""

  @property
  def total_episodes(self) -> int:
    return self.num_envs * self.num_episodes


def build_eval_env_cfg(
  task_id: str,
  protocol: EvalProtocol,
) -> ManagerBasedRlEnvCfg:
  """Training env cfg, pinned to the protocol. Not the play cfg."""
  cfg = load_env_cfg(task_id, play=False)

  cfg.scene.num_envs = protocol.num_envs
  cfg.episode_length_s = protocol.episode_length_s
  cfg.seed = protocol.seed

  # On generated terrain, evaluate across the FULL difficulty range rather than
  # the training spawn cap. max_init_terrain_level=None makes the terrain assign
  # levels uniformly over all rows, so the score covers easy through hardest
  # instead of whichever band training happened to reach.
  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_type == "generator":
    cfg.scene.terrain.max_init_terrain_level = None

    # Equal statistical power per terrain, not the training spawn weights.
    # Environments are split across columns in proportion to each sub-terrain's
    # ``proportion``, which is right for training (spend capacity where the
    # interesting terrain is) and wrong for measurement: at 512 envs the
    # 0.05-weighted ``flat`` control would get ~26 envs spread over 10 difficulty
    # rows, so the controls -- where a trustworthy null matters most -- would
    # carry the weakest statistics. Flattening to uniform gives every column
    # ~1/N of the envs while keeping each env pinned to one terrain type, so
    # per-terrain attribution stays clean.
    gen = cfg.scene.terrain.terrain_generator
    if gen is not None:
      gen.seed = protocol.terrain_seed
      if protocol.uniform_terrain_weights:
        for sub in gen.sub_terrains.values():
          sub.proportion = 1.0

  # Curricula are training machinery and would make the measurement depend on
  # how long the policy trained. Freeze them out.
  cfg.curriculum = {}

  if not protocol.keep_corruption:
    for group in cfg.observations.values():
      group.enable_corruption = False
  if not protocol.keep_pushes:
    cfg.events.pop("push_robot", None)

  # Pin the command distribution: no heading control, no standing/forward
  # subsampling. Commands are assigned from the grid by env index in
  # scripts/evaluate.py, so every condition sees the same command on the same
  # env index with the same terrain and the same DR draw.
  twist = cfg.commands["twist"]
  assert isinstance(twist, UniformVelocityCommandCfg)
  twist.heading_command = False
  twist.rel_standing_envs = 0.0
  twist.rel_heading_envs = 0.0
  twist.rel_forward_envs = 0.0
  twist.resampling_time_range = (protocol.episode_length_s, protocol.episode_length_s)
  twist.ranges = UniformVelocityCommandCfg.Ranges(
    lin_vel_x=(-1.0, 1.0),
    lin_vel_y=(-1.0, 1.0),
    ang_vel_z=(-0.5, 0.5),
    # Must be None: mjlab rejects a heading range with heading_command=False.
    # Commands are assigned from the protocol grid anyway, so the sampled
    # ranges only matter as a well-formedness constraint.
    heading=None,
  )

  return cfg
