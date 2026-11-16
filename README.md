# g1-tactile

A tactile slipper for the Unitree G1 in MuJoCo/mjlab: the slipper's geometry
and mass on mjlab's 29-DoF G1, the taxel layout of the sensor foil under it,
and a model that turns simulated foot contacts into per-taxel forces and a
compact observation for a policy.

How the model works: [docs/model.md](docs/model.md).

## Install

Python 3.11+, Linux, an NVIDIA GPU for mjlab/MuJoCo Warp. Tested against the
mjlab commit pinned in `pyproject.toml` (MuJoCo 4).

```bash
pip install -e ".[dev,viz]"     # or: uv pip install -e ".[dev,viz]"
pytest
python scripts/show_sole.py     # -> media/tactile_sole_layout.png
```

## Use in an mjlab env

One call on any mjlab env config that has the G1 as its `robot` entity:

```python
from g1_tactile.env_cfg import wear_tactile_slipper
from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg

cfg = wear_tactile_slipper(unitree_g1_flat_env_cfg())   # or your own G1 env config
```

It
- swaps the robot for the G1 wearing the slipper (mesh, 200 g per foot, one box
  collision sole per foot instead of the 7 capsules, spawned 8 mm higher);
- adds the tactile scene sensors;
- points terms that named the old foot capsules (mjlab's foot friction
  randomisation does) at the sole boxes;
- adds a `tactile` term to the `actor` observation group (58 values; per foot
  26 contact bits, load, centre of pressure x/y) and a `tactile_faults` reset
  event for the sensor fault model.

Options: `obs_group=None` (slipper and sensors only), `noisy=False` (clean
readings, no fault event), `strength=` (fault strength 0..1, default 0.5),
`secondary=` (what the feet stand on, default the terrain). The raw per-taxel
normal and shear forces: `g1_tactile.sensing.read_tactile(env, TactileParams())`.
The pieces can also be used one by one: `robot.get_g1_cfg()`,
`sensing.tactile_scene_sensors()`, `features.tactile_touch`,
`features.reset_tactile_noise`.

## Files

| Path | What |
|---|---|
| `src/g1_tactile/slipper.py` | The sole stack (3 mm TPU 95A, ~1 mm foil, 4 mm 75A rubber), 200 g per foot, the box collision sole, `add_slipper(spec)` |
| `src/g1_tactile/assets/g1_slipper.stl` | The slipper mesh in the `*_ankle_roll_link` frame, metres |
| `src/g1_tactile/robot.py` | mjlab's G1 wearing the slipper: `get_g1_spec()`, `get_g1_cfg()` |
| `src/g1_tactile/layout.py`, `layouts/slipper_v1.csv` | The 26 taxels of the foil, in the sole frame |
| `src/g1_tactile/model.py` | Pure-torch force model: contacts + support -> per-taxel normal/shear |
| `src/g1_tactile/sensors.py` | The contact and raycast scene sensors the model reads |
| `src/g1_tactile/sensing.py` | Runs the model on a live env: `tactile_scene_sensors()`, `read_tactile()` |
| `src/g1_tactile/features.py` | The observation term and the sensor-fault model |
| `src/g1_tactile/env_cfg.py` | `wear_tactile_slipper(cfg)`: all of the above into an mjlab env config |
| `src/g1_tactile/viz.py`, `scripts/show_sole.py` | Drawings of the sole and taxels |
| `scripts/make_slipper_asset.py` | Rebuild the slipper STL from a CAD export |
| `scripts/export_foot_cad.py` | The G1 foot in the sole frame (DXF/STL), to design around |
