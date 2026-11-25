"""Silence MuJoCo Warp's per-step solver iteration warnings.

MuJoCo Warp prints a line from inside the GPU kernel whenever the constraint
solver or its linesearch hits the iteration cap. That is routine for a contact
rich biped at the solver settings the tasks use (it is a cap, not a failure:
the step still returns the best iterate), but at 4096 envs it fires every step
and buried each cluster job's log under 2-5 GB of identical lines. Only the two
iteration flags are cleared; buffer overflows (nefc, contacts, broadphase,
tactile pairs) still warn, because those drop physics and must stay visible.

The flag is baked into the kernels at compile time (``wp.static``), so it has to
be set on the warp Option between ``put_model`` and the first step. mjlab's
``SimulationCfg.apply_wp_opt`` is called exactly there on every model path, so
it is wrapped here. Printing is the only thing the flag controls; the physics is
unchanged. Set ``SENDG1_SOLVER_WARNINGS=1`` to get the warnings back.
"""

import os

import mujoco_warp as mjwarp
from mjlab.sim.sim import SimulationCfg

_QUIET = mjwarp.OverflowType.ITERATIONS | mjwarp.OverflowType.LS_ITERATIONS


def silence_solver_iteration_warnings() -> None:
  if os.environ.get("SENDG1_SOLVER_WARNINGS") == "1":
    return
  original = SimulationCfg.apply_wp_opt
  if getattr(original, "_sendg1_quiet", False):
    return

  def apply_wp_opt(self: SimulationCfg, wp_opt: mjwarp.Option) -> None:
    original(self, wp_opt)
    wp_opt.warn_overflow = int(wp_opt.warn_overflow) & ~int(_QUIET)

  apply_wp_opt._sendg1_quiet = True  # type: ignore[attr-defined]
  SimulationCfg.apply_wp_opt = apply_wp_opt  # type: ignore[method-assign]
