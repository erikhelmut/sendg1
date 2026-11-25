"""Beam runner: checkpoints carry the per-env curriculum.

mjlab's runner restores only the step counter, so a resumed run (a requeued
scavenger job, a continuation to more iterations) put every env back on the
widest beam and spent its first thousands of iterations climbing again.
"""

from __future__ import annotations

from sendg1.common.beam import mdp
from sendg1.common.runner import Sendg1Runner


class BeamRunner(Sendg1Runner):
  def checkpoint_infos(self) -> dict:
    state = mdp.curriculum_state(self.env.unwrapped)
    return {"beam_curriculum": state} if state else {}

  def load(self, path: str, load_cfg=None, strict: bool = True, map_location=None) -> dict:
    infos = super().load(path, load_cfg, strict, map_location)
    if infos and "beam_curriculum" in infos:
      mdp.load_curriculum_state(self.env.unwrapped, infos["beam_curriculum"])
    return infos
