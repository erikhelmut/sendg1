"""Rough-terrain curriculum for the foot-tactile ablation.

Selection principle
-------------------
Foot tactile sensing supplies four things proprioception cannot: contact
timing, centre of pressure, whether part of the sole is unsupported, and slip
onset. All four live at the foot-ground interface at the instant of touchdown.
So a terrain discriminates between a blind and a tactile policy exactly when the
critical information is *at that interface* and is *not predictable from body
state*.

That rules a lot in and out. Smooth, slowly-varying terrain (slopes, waves,
Perlin) is fully described by projected gravity and joint angles, so both
conditions see the same problem -- mjlab's ``ROUGH_TERRAINS_CFG`` spends 40% of
its mix there. Sharp discontinuities (tilted tiles, discrete height steps,
stair edges) put the decisive information under the foot, where only tactile
reads it early.

The G1's sole is why this is sharp: its foot collision capsules span roughly
0.21 m long by 0.07 m wide (see ``g1.xml``). A long, narrow foot makes lateral
edge contact the dominant failure mode, and a 26-taxel array resolves heel/toe
and left/right edge easily.

Controls
--------
``flat`` and ``random_rough`` are included deliberately as *negative controls*:
terrain where we predict no tactile advantage. A result showing a gain on the
discriminators and no gain on the controls is far stronger than a single
aggregate number, because it rules out "tactile merely added capacity or
generic noise robustness". ``terrain_levels_vel`` already logs a per-sub-terrain
mean curriculum level, so this readout is free.

Difficulty
----------
Every range below is the value reached at the HARDEST row; row 0 is the
easiest and rows interpolate linearly. Several presets are toned down from
their mjlab defaults, which target a quadruped or a sighted policy and would
floor a blind humanoid. Floor effects are as useless as ceiling effects: the
comparison needs the baseline somewhere around 50-80% success, not 0%.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from mjlab.terrains.config import (
  box_random_grid,
  flat,
  pyramid_stairs,
  pyramid_stairs_inv,
  random_rough,
  random_spread_boxes,
  random_stairs,
  tilted_grid,
)
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg

Role = Literal["control", "discriminator"]


@dataclass(frozen=True)
class TerrainSpec:
  role: Role
  rationale: str


#: Column order here IS the terrain_type index order used for per-terrain
#: metrics, because curriculum mode assigns one column per sub-terrain.
TERRAIN_ROLES: dict[str, TerrainSpec] = {
  "flat": TerrainSpec(
    "control",
    "Curriculum floor. Both conditions should saturate; a difference here would "
    "indicate something other than terrain sensing.",
  ),
  "random_rough": TerrainSpec(
    "control",
    "Small-amplitude, statistically homogeneous noise. A blind policy handles "
    "this with a compliant gait, so tactile should add little.",
  ),
  "tilted_grid": TerrainSpec(
    "discriminator",
    "Primary discriminator. Every footfall lands on an unpredictable tilt; "
    "tactile reads it from CoP asymmetry at contact, blind only after the ankle "
    "deflects. Fails gracefully (destabilise, not fall into a pit).",
  ),
  "box_random_grid": TerrainSpec(
    "discriminator",
    "Discrete height steps between adjacent cells. Grid pitch is set below the "
    "foot length so the sole regularly straddles a discontinuity.",
  ),
  "random_stairs": TerrainSpec(
    "discriminator",
    "Irregular tread heights: the next step is not predictable from the last.",
  ),
  "pyramid_stairs": TerrainSpec(
    "discriminator",
    "Structured stairs, ascending away from the spawn platform. Up-steps are "
    "largely a toe-strike (a torque event proprioception already senses).",
  ),
  "pyramid_stairs_inv": TerrainSpec(
    "discriminator",
    "The same stairs descending away from spawn. Down-steps are the more "
    "tactile-relevant case: the ground arrives later than expected and touchdown "
    "timing is exactly what the taxels supply.",
  ),
  "random_spread_boxes": TerrainSpec(
    "discriminator",
    "Scattered low obstacles. Intermittent partial contact rather than a "
    "sustained regime; included at low weight for variety.",
  ),
}

CONTROL_TERRAINS = tuple(k for k, v in TERRAIN_ROLES.items() if v.role == "control")
DISCRIMINATOR_TERRAINS = tuple(
  k for k, v in TERRAIN_ROLES.items() if v.role == "discriminator"
)


TRAIN_TERRAIN_SEED = 0
"""Terrain layout seed used for training.

MUST be pinned. ``TerrainGeneratorCfg.seed`` defaults to ``None``, which draws
``np.random.randint(0, 10000)`` at build time -- so every process would generate
a different terrain and the blind and tactile policies would be trained, and
measured, on different ground. That is a direct confound in the one comparison
this repository exists to make. It is also invisible to the config fingerprint,
because ``seed=None`` hashes identically every time.

Evaluation deliberately uses a *different* pinned seed (see
``EvalProtocol.terrain_seed``), so policies are scored on terrain instances they
never trained on -- identical instances for every condition.
"""


def make_rough_terrain_cfg(
  num_rows: int = 10,
  seed: int | None = TRAIN_TERRAIN_SEED,
) -> TerrainGeneratorCfg:
  """Curriculum terrain: one column per type, difficulty rising with row.

  ``proportion`` in curriculum mode controls how environments are *spawned*
  across types, not the layout (each type still gets exactly one column).
  """
  return TerrainGeneratorCfg(
    seed=seed,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=num_rows,
    num_cols=len(TERRAIN_ROLES),  # ignored in curriculum mode; kept honest.
    curriculum=True,
    add_lights=True,
    sub_terrains={
      # -- controls ---------------------------------------------------------
      "flat": flat(proportion=0.05),
      "random_rough": random_rough(
        proportion=0.10,
        noise_range=(0.02, 0.08),
        noise_step=0.02,
      ),
      # -- discriminators ---------------------------------------------------
      "tilted_grid": tilted_grid(
        proportion=0.22,
        grid_width=1.0,
        # mjlab default 20 deg. The G1 ankle roll range plus a 0.07 m wide sole
        # makes that a near-certain rollover for a blind policy.
        tilt_range_deg=13.0,
        height_range=0.15,
        platform_width=1.5,
      ),
      "box_random_grid": box_random_grid(
        proportion=0.20,
        # Cell pitch is a geom-count decision as much as a difficulty one: this
        # terrain tiles the whole 8x8 m patch, so cost goes as 1/width^2. At
        # 0.35 m it alone produced 4,840 of 5,599 terrain geoms and the sim ran
        # out of GPU memory at 2,048 envs. At 0.55 m a 0.21 m foot still spans a
        # cell boundary on roughly a third of placements per axis, which is the
        # property that matters, for ~1/2.5 of the geoms.
        grid_width=0.55,
        # mjlab default tops out at 0.30 m between adjacent cells; that is a
        # stair riser appearing under one foot with no warning.
        grid_height_range=(0.0, 0.14),
        platform_width=1.5,
      ),
      "random_stairs": random_stairs(
        proportion=0.15,
        step_width=0.8,
        # mjlab default (0.10, 0.30). 0.30 m is above comfortable G1 step height.
        step_height_range=(0.04, 0.16),
        platform_width=1.5,
      ),
      "pyramid_stairs": pyramid_stairs(
        proportion=0.10,
        # mjlab default tread is 0.30 m against a 0.21 m foot - almost no margin.
        step_width=0.45,
        step_height_range=(0.0, 0.15),
        platform_width=2.5,
      ),
      "pyramid_stairs_inv": pyramid_stairs_inv(
        proportion=0.10,
        step_width=0.45,
        step_height_range=(0.0, 0.15),
        platform_width=2.5,
      ),
      "random_spread_boxes": random_spread_boxes(
        proportion=0.08,
        # mjlab default 80 boxes per patch; at 10 rows that is 800 extra geoms
        # in one column alone, and this run is already physics-bound.
        num_boxes=40,
        box_height_range=(0.03, 0.14),
        platform_width=1.5,
      ),
    },
  )


def terrain_names() -> tuple[str, ...]:
  """Sub-terrain names in column order (== terrain_type index order)."""
  return tuple(TERRAIN_ROLES)
