"""Pedalo riding: the G1 stands on a Pedalo Classic or Sport and drives it
forward by shifting weight from one foot to the other.

- ``model``     the Pedalo Classic as an MjSpec: dimensions, masses, the closed
                linkage (two wheelsets, two step boards).
- ``sport``     the Pedalo Sport: one Classic wheelset with two freely turning
                step blocks.
- ``variants``  the numbers and names the MDP needs from either.
- ``mdp``       observations, rewards, terminations, events.
- ``terms``     the observation registry with availability tags.

Lives under ``common`` for the same reason as ``common/beam``: the
observation registry imports it without a cycle through ``sendg1.tasks``.
"""
