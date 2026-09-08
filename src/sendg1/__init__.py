"""Foot tactile sensing for Unitree G1 locomotion.

Research question: do tactile sensors on the feet of the Unitree G1 measurably
improve locomotion policies?

Every result is a paired comparison between a baseline policy and a tactile
policy on the same task, the same seeds and the same eval protocol. The repo is
organised so that (task x observation-set x seed) is the unit of work.
"""

from sendg1._pin import MJLAB_COMMIT, MJLAB_VERSION, check_mjlab_version

check_mjlab_version()

# Import mjlab eagerly, at a deterministic point.
#
# mjlab's __init__ loads the "mjlab.tasks" entry point group, which imports
# sendg1.tasks -- so mjlab importing us and us importing mjlab is inherently
# cyclic. The cycle is only safe if it always starts from the same place. If the
# first mjlab import instead happened lazily from deep inside, say,
# sendg1.common.obs.groups, that module would be re-entered while partially
# initialised and raise ImportError. Forcing it here means any entry point into
# the package resolves identically.
import mjlab as _mjlab  # noqa: E402,F401

__all__ = ["MJLAB_COMMIT", "MJLAB_VERSION", "check_mjlab_version"]
