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
  diagnostic: bool = False
  """Diagnostic sets may place ``oracle`` terms in the actor. They are excluded
  from the default sweep and must be requested explicitly, because their results
  are not claims about a deployable policy."""

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
  "contact_oracle": ObsSet(
    actor=("proprio", "contact_ideal"),
    critic=_CRITIC,
    diagnostic=True,
    description=(
      "Diagnostic upper bound, NOT a deployable policy. Adds noiseless, "
      "zero-latency per-foot contact and net contact force to the actor. If "
      "this does not beat 'blind', a noisy 26-taxel array will not either -- so "
      "it bounds the headroom available to real tactile sensing. Opt in "
      "explicitly; never included in a default sweep."
    ),
  ),
  "tactile_array": ObsSet(
    actor=("proprio", "tactile_array"),
    critic=_CRITIC,
    diagnostic=True,
    description=(
      "Adds per-taxel normal and shear force on both soles to the actor. "
      "Diagnostic while the sensor model is idealised (no "
      "noise, latency or calibration error): it measures what spatially "
      "resolved sole contact could add over contact_oracle's net force."
    ),
  ),
}

DEFAULT_OBS_SET = "blind"


def research_obs_sets() -> tuple[str, ...]:
  """Non-diagnostic sets -- what a default sweep runs."""
  return tuple(name for name, s in OBS_SETS.items() if not s.diagnostic)


def diagnostic_obs_sets() -> tuple[str, ...]:
  return tuple(name for name, s in OBS_SETS.items() if s.diagnostic)


def uses_tactile_array(obs_set: str) -> bool:
  """Whether the scene needs the tactile sensors for this set. Other sets skip
  them, which keeps their scenes -- and recorded fingerprints -- unchanged."""
  return "tactile_array" in OBS_SETS[obs_set].group_names
