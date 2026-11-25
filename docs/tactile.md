# Tactile sole

```bash
python scripts/make_slipper_asset.py <stl>  # slipper CAD -> robots/assets/g1_slipper.stl
python scripts/tactile_viewer.py     # MP4 of the robot walking + live taxel readings
python scripts/export_foot_cad.py    # foot as DXF + STL in the sole frame, mm (needs .[cad])
python scripts/sweep.py --tasks flat --obs-sets blind tactile
```

- **Layout** (`common/tactile/layout.py`): taxel patches in the sole frame
  (`*_ankle_roll_link`, sole plane at z = -0.043 m, the bottom of the slipper). The slipper sensor is one
  part used 1:1 on both feet, so taxel `i` sits at the same (x, y) on each
  (`mirror_right=False`).
  Currently `slipper_v1`: 26 patches of 12 mm, loaded from
  `common/tactile/layouts/slipper_v1.csv` (mm, left foot, exported from the
  slipper CAD in the frame of `scripts/export_foot_cad.py`). Taxel `i` is sensor
  id `i`. The loader rejects ids that are not 0..N-1, rotated patches, overlaps,
  and patches past the foot outline. A new layout is a new CSV plus one line in
  `LAYOUTS`; nothing downstream assumes a grid.
- **Slipper** (`common/robots/slipper.py`): the G1 wears the real slipper in
  every task, blind included. From the foot down: 3 mm TPU 95A slipper (its
  CAD, kyber blue), ~1 mm tactile foil (white), 4 mm Vibram 75A rubber (black),
  so the sole sits 8 mm lower and the robot spawns 8 mm higher. 200 g per foot
  is added to the ankle-roll link. Collision: mjlab's 7 capsules per foot are
  replaced by one box under the rubber outline (218 x 81 x 5 mm,
  `*_sole_collision`); its corners overhang the rounded toe and heel by up to
  ~1 cm.
- **Why not bin contact points**: the sim sole is one rigid box, and on flat
  ground MuJoCo contacts only its 4 corners. Binning would leave every inner
  taxel at zero while the foot is flat.
- **Model** (`common/tactile/model.py`, pure torch): the net force and centre of
  pressure come from the per-contact data (`sole_contacts` sensor); which
  patches have ground beneath them comes from 4 rays per taxel
  (`tactile_rays_{left,right}`), with 1 mm of compliance (the rubber stack
  barely gives; the margin covers MuJoCo's contact penetration); the normal force is
  split over the supported taxels as a plane of pressure (a rigid sole on a
  thin elastic layer) that reproduces the force and CoP exactly, dropping any
  taxel it puts in tension. Shear is shared in proportion to normal load.
- **Cost**: ~15% of training throughput on flat ground (147k vs 173k steps/s
  at 8192 envs).

## What the actor sees

One sensor model and one observation term for every task: `tactile_touch`
(`common/beam/tactile.py`, `touch_features`), in the beam, pedalo,
rolling-cylinder and slackline actors and in the `tactile` obs set of the
velocity and box-push tasks. Per foot, 29 dims, both feet 58:

- `bits[26]`: 1 where a taxel reads above 3 N, so which part of the sole touches;
- `load`: total normal force / body weight (343 N), clipped to [0, 1.5];
- `cop[2]`: centre of pressure from the taxel readings, scaled to the sole's
  half extents, 0 while the foot is not in contact (15 N or less in total).

Contact area and a contact flag were features until 2026-11-15.

The right foot is re-indexed into the left foot's mirror image, so +y is
lateral on both feet.

**Normal load only.** The real slipper reports per-taxel pressure, not force
vectors, so the actor gets no shear. The model's per-taxel shear is the net
tangential force shared in proportion to load, so it carries nothing the
per-foot total does not: the critics get clean per-taxel *normal* forces
(`taxel_normal_clean`) and each foot's net contact force, shear included. Most features are unaffected by a global gain error
(the bits and CoP); `load` is the one a calibration error
moves.

**Sensor faults** (`TactileNoiseState`, actor only; the critic's copies are
clean): per episode a per-foot gain, a per-taxel gain and dead taxels; per step
Gaussian noise per taxel and a frame drop that holds a foot's previous reading.
At strength 1: ±20 %, ±25 %, 8 %, 3 N, 5 %. One setting for every task, on
every curriculum level (`common/beam/tactile.py`):

- **Off by default** (`FAULTS_ENABLED = False`, since 2026-11-25): the actor
  reads clean taxels, with 0-1 steps of latency.
- **Strength 0.5 when on** (`FAULT_STRENGTH`). Per run:
  `train <task> --env.events.tactile-faults.params.enabled True
  [--env.events.tactile-faults.params.strength 0.5]`. Use the same setting for
  the blind run of a pair (it only draws numbers there, but keeps the configs
  identical).

The reset event `tactile_faults` is in every obs set's events, blind included.

**Joint and IMU noise** (`common/obs/terms.py`, one definition for every task,
actor only): per step uniform noise (gyro ±0.2 rad/s, gravity ±0.05, joint
position ±0.01 rad, joint velocity ±1.5 rad/s); per episode an IMU bias (gyro
±0.05 rad/s, gravity ±0.03) and a ±0.015 rad encoder offset.

**Sole friction**: the sole outranks the ground in contact priority, so every
foot contact uses the sole's friction, drawn per env from 0.3-1.2 in every task
(`common/events.py: FOOT_FRICTION`; box pushing samples the floor instead).

**Not modelled yet**: the real sensor's pressure response (nonlinearity,
saturation, hysteresis, drift), crosstalk, quantisation; a measured friction
coefficient of the rubber (the range above brackets it).

A diagnostic `contact_oracle` obs set, whose actor got noiseless foot contact
and net contact force as an upper bound, was removed on 2026-11-15.
