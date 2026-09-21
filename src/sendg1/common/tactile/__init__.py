"""The tactile sole: taxel layout, the sensor model, and its observation term.

- ``layout``  where the taxels are on the sole (placeholder grid for now).
- ``model``   pure-torch force model: per-contact data -> per-taxel normal/shear.
- ``sensing`` reads the scene sensors and runs the model for a live env.
"""
