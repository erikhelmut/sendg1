"""Observation sets: which named groups feed the actor and which feed the critic.

An observation set is the study's independent variable. Everything else --
robot, rewards, events, terminations, terrain, network architecture -- is held
fixed across sets so that a difference in outcome is attributable to what the
actor can see.

Note the critic mapping is IDENTICAL in every set. That is deliberate: if the
critic changed alongside the actor, two things would differ at once and the
comparison would not isolate the actor's observation space. It also means the
Phase 1 baselines stay comparable when the tactile group later gains a real
sensor model, because the critic never touches that group.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ObsSet:
  """A mapping from rsl_rl observation *sets* to our observation *groups*."""

  actor: tuple[str, ...]
  critic: tuple[str, ...]
  description: str

  def obs_groups(self) -> dict[str, tuple[str, ...]]:
    """The dict assigned to ``RslRlBaseRunnerCfg.obs_groups``."""
    return {"actor": self.actor, "critic": self.critic}

  @property
  def group_names(self) -> tuple[str, ...]:
    seen: dict[str, None] = {}
    for name in (*self.actor, *self.critic):
      seen.setdefault(name, None)
    return tuple(seen)


_CRITIC: tuple[str, ...] = ("proprio", "privileged")

OBS_SETS: dict[str, ObsSet] = {
  "blind": ObsSet(
    actor=("proprio",),
    critic=_CRITIC,
    description=(
      "Baseline. Actor sees only what the real blind G1 measures: IMU angular "
      "velocity, projected gravity, joint positions and velocities, previous "
      "action, and the user command."
    ),
  ),
  "tactile": ObsSet(
    actor=("proprio", "tactile"),
    critic=_CRITIC,
    description=(
      "Adds the tactile slipper to the actor: per foot a 26-bit taxel touch "
      "map, load and centre of pressure, from the sensor model every task "
      "shares (sendg1.common.beam.tactile: gain error, dead taxels, noise, "
      "frame drops, latency; normal load only)."
    ),
  ),
}
"""A diagnostic ``contact_oracle`` set (noiseless foot contact and net contact
force in the actor) existed until 2026-11-15."""

DEFAULT_OBS_SET = "blind"


def research_obs_sets() -> tuple[str, ...]:
  """The sets a default sweep runs: all of them."""
  return tuple(OBS_SETS)


def uses_tactile(obs_set: str) -> bool:
  """Whether the scene needs the tactile sensors for this set. Other sets skip
  them: the sensors only feed the actor's ``tactile`` group, and the rays cost
  time on every step."""
  return "tactile" in OBS_SETS[obs_set].group_names
