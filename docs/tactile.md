# Tactile sole

```bash
python scripts/show_sole.py          # the sole as simulated: mesh, capsules, taxels
python scripts/tactile_viewer.py     # MP4 of the robot walking + live taxel readings
python scripts/export_foot_cad.py    # foot as DXF + STL in the sole frame, mm (needs .[cad])
python scripts/sweep.py --tasks flat --obs-sets blind tactile --seeds 0
```

- **Layout** (`common/tactile/layout.py`): taxel patches in the sole frame
  (`*_ankle_roll_link`, sole plane at z = -0.035 m). The slipper sensor is one
  part used 1:1 on both feet, so taxel `i` sits at the same (x, y) on each
  (`mirror_right=False`).
  Currently `slipper_v1`: 26 patches of 12 mm, loaded from
  `common/tactile/layouts/slipper_v1.csv` (mm, left foot, exported from the
  slipper CAD in the frame of `scripts/export_foot_cad.py`). Taxel `i` is sensor
  id `i`. The loader rejects ids that are not 0..N-1, rotated patches, overlaps,
  and patches past the foot outline. A new layout is a new CSV plus one line in
  `LAYOUTS`; nothing downstream assumes a grid.
- **Why not bin contact points**: the sim foot is 7 rigid capsules, and on flat
  ground MuJoCo only contacts their end caps -- a line under the heel and one
  under the toe. Binning would leave the mid-foot taxels at zero while the foot
  is flat.
- **Model** (`common/tactile/model.py`, pure torch): the net force and centre of
  pressure come from the per-contact data (`sole_contacts` sensor); which
  patches have ground beneath them comes from 4 rays per taxel
  (`tactile_rays_{left,right}`), with 4 mm of compliance; the normal force is
  split over the supported taxels as a plane of pressure (a rigid sole on a
  thin elastic layer) that reproduces the force and CoP exactly, dropping any
  taxel it puts in tension. Shear is shared in proportion to normal load.
- **Cost**: ~15% of training throughput on flat ground (147k vs 173k steps/s
  at 8192 envs).

## What the actor sees

One sensor model and one observation term for every task: `tactile_touch`
(`common/beam/tactile.py`, `touch_features`), in the beam, pedalo,
rolling-cylinder and slackline actors and in the `tactile` obs set of the
velocity and box-push tasks. Per foot, 31 dims, both feet 62:

- `bits[26]`: 1 where a taxel reads above 3 N, so which part of the sole touches;
- `load`: total normal force / body weight (343 N), clipped to [0, 1.5];
- `cop[2]`: centre of pressure from the taxel readings, scaled to the sole's
  half extents, 0 while the foot is not in contact;
- `area`: fraction of taxels touching;
- `contact`: total normal force above 15 N.

The right foot is re-indexed into the left foot's mirror image, so +y is
lateral on both feet.

**Normal load only.** The real slipper reports per-taxel pressure, not force
vectors, so the actor gets no shear. The model's shear is the net tangential
force shared in proportion to load, and only the critic sees it, where a task's
critic carries it at all. Most features are unaffected by a global gain error
(the bits, CoP, area and contact flag); `load` is the one a calibration error
moves.

**Sensor faults** (`TactileNoiseState`, actor only; the critic's copies are
clean): per episode a per-foot gain (±20%), a per-taxel gain (±25%) and dead
taxels (8%); per step 3 N Gaussian noise per taxel and a 5% frame drop that
holds a foot's previous reading; on top, 0-1 steps of latency. All scale with a
strength: from 0.3 to 1.0 over the beam's curriculum levels, a fixed 0.5
(`FIXED_STRENGTH`) in every other task. The reset event `tactile_faults` is in
every obs set's events, blind included.

**Not modelled yet**: the real sensor's pressure response (nonlinearity,
saturation, hysteresis, drift), crosstalk, quantisation; the slipper's
thickness and friction.

## `contact_oracle` is diagnostic, and off by default

Its actor gets noiseless, zero-latency, perfectly-calibrated foot contact and net
contact force — an **upper bound** on what any real tactile array could deliver,
not a deployable policy. It exists to answer the question that gates the project:
*is there room for foot contact information to help this policy at all?* If it
does not beat `blind`, a noisy 26-taxel sensor will not either.

It is never in a default sweep. Request it explicitly:

```bash
python scripts/sweep.py --tasks flat --obs-sets contact_oracle --seeds 0
```
