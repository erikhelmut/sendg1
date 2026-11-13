"""Rolling-cylinder balance: the G1 stands across a cylinder lying free on the
floor and keeps it, and itself, from rolling away.

- ``model``     the cylinder as an MjSpec: radius, length, mass, friction.
- ``mdp``       observations, rewards, terminations, events.
- ``terms``     the observation registry with availability tags.

Lives under ``common`` for the same reason as ``common/beam``: the
observation registry imports it without a cycle through ``sendg1.tasks``.
"""
