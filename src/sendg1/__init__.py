"""Foot tactile sensing for Unitree G1 locomotion.

Research question: do tactile sensors on the feet of the Unitree G1 measurably
improve locomotion policies?

Every result is a paired comparison between a baseline policy and a tactile
policy on the same task, the same seeds and the same eval protocol. The repo is
organised so that (task x observation-set x seed) is the unit of work.
"""

import os

# Recurrent PPO (the beam tasks) splits every rollout into episode fragments, so
# tensor shapes change on every update. Under PyTorch's default CUDA caching
# allocator that fragments the cache: measured on the beam task, reserved memory
# grew from 2.1 to 9.3 GiB in 30 iterations while peak allocation stayed near
# 1.5 GiB, until MuJoCo-Warp could not allocate and training died. With
# expandable segments, reserved memory stayed within ~1.2x of the peak. The
# allocator reads this on first CUDA use, so it must be set before any tensor
# reaches the GPU; setdefault leaves an explicit user setting alone.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from sendg1._pin import MJLAB_COMMIT, MJLAB_VERSION, check_mjlab_version  # noqa: E402

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
