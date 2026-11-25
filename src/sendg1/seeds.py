"""The seeds of the study, fixed for reproducibility.

Training: seeds 1-5, nothing else. A new experiment starts with seed 1 only
(directional); seeds 2-5 follow when a result is worth a claim, so every
reported number rests on the same five seeds. Seed 0 is never a training seed:
it is the evaluation seed, the env seed of every evaluation protocol, so all
checkpoints are scored on the same randomisation draws.
"""

from __future__ import annotations

TRAIN_SEEDS: tuple[int, ...] = (1, 2, 3, 4, 5)
FIRST_SEED = 1
"""The one seed a new experiment starts with."""
MORE_SEEDS: tuple[int, ...] = (2, 3, 4, 5)
"""The seeds to add once seed 1 looks worth a claim."""
EVAL_SEED = 0
"""Env seed of every evaluation; never a training seed."""


def check_train_seed(seed: int) -> int:
  if seed not in TRAIN_SEEDS:
    raise ValueError(
      f"training seed {seed} is not one of the study's seeds {TRAIN_SEEDS} "
      f"(seed {EVAL_SEED} is the evaluation seed); see sendg1/seeds.py"
    )
  return seed
