"""Beam walking: a blind G1 follows a narrow, straight or curved beam by touch
(``G1-Beam-E2E``, ``G1-Beam-E2E-Curved``).

- ``terrain``   the beam tile: start/end platforms, a beam along a straight or
                bent centreline path, ground below.
- ``geometry``  per-env lookup of the beam the robot is on, and projection of
                points onto its centreline.
- ``tactile``   per-foot tactile features built on ``sendg1.common.tactile``.
- ``mdp``       observations, rewards, terminations, events, curriculum.
- ``terms``     the observation registry with availability tags.
- ``symmetry``  left-right mirror maps for the symmetric actor.

Lives under ``common`` so that the observation registry can import it without
an import cycle through ``sendg1.tasks``.
"""
