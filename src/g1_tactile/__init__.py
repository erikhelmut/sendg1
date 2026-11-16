"""Tactile slipper and sole sensor model for the Unitree G1 in mjlab.

- ``slipper``   the slipper's geometry and mass, and how it goes into the MJCF.
- ``robot``     mjlab's 29-DoF G1 wearing the slipper (``get_g1_cfg``).
- ``layout``    where the taxels sit on the sole (``slipper_v1``, from the CAD).
- ``model``     pure-torch force model: contacts + support -> per-taxel forces.
- ``sensors``   the scene sensors the model reads.
- ``sensing``   runs the model on a live env (``read_tactile``).
- ``features``  the policy observation (``tactile_touch``) and sensor faults.
- ``env_cfg``   ``wear_tactile_slipper(cfg)``: everything into an mjlab env config.
- ``viz``       matplotlib drawings of the sole and taxels.
"""
