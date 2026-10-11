"""Straight-beam walking for the teacher-student study (``G1-Beam-Teacher`` /
``G1-Beam-Student``).

Independent of the older ``sendg1.common.beam`` task on purpose: nothing here
imports from it, so that task and its results stay frozen.

- ``terrain``   the beam tile: optional start/end platforms, a beam along a
                parametrised centreline path, ground below.
- ``geometry``  per-env lookup of the beam the robot is on, and projection of
                points onto its centreline.
- ``tactile``   per-foot tactile features built on ``sendg1.common.tactile``.
- ``mdp``       observations, rewards, terminations, events, curriculum.
- ``terms``     the observation registry with availability tags.

Like ``sendg1.common.beam``, this lives under ``common`` so that the observation
registry can import it without an import cycle through ``sendg1.tasks``.
"""
