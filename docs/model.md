# The tactile sole model

```bash
python scripts/demo_video.py                # demo clip with the tactile HUD -> media/tactile_demo.mp4
python scripts/make_slipper_asset.py <stl>  # slipper CAD -> src/g1_tactile/assets/g1_slipper.stl
python scripts/export_foot_cad.py           # foot as DXF + STL in the sole frame, mm (needs .[cad])
```

## Hardware

- **Slipper** (`slipper.py`): from the foot down, a 3 mm TPU 95A slipper (its
  CAD mesh, blue), a ~1 mm tactile foil (white) and 4 mm Vibram 75A rubber
  (black). The sole sits 8 mm lower and the robot spawns 8 mm higher
  (`robot.STANDING_BASE_Z`). 200 g per foot is added to the ankle-roll link.
  Collision: mjlab's 7 capsules per foot are replaced by one box under the
  rubber outline (218 x 81 x 5 mm, `*_sole_collision`); its corners overhang
  the rounded toe and heel by up to ~1 cm.
- **Layout** (`layout.py`): taxel patches in the sole frame
  (`*_ankle_roll_link`: x forward, y left, z up; sole plane at z = -0.043 m,
  the bottom of the rubber). The foil is one part used 1:1 on both feet, so
  taxel `i` sits at the same (x, y) on each (`mirror_right=False`).
  `slipper_v1`: 26 patches of 12 mm, loaded from `layouts/slipper_v1.csv`
  (mm, left foot, exported from the slipper CAD in the frame of
  `scripts/export_foot_cad.py`). Taxel `i` is sensor id `i`. The loader
  rejects ids that are not 0..N-1, rotated patches, overlaps, and patches past
  the foot outline. A new layout is a new CSV plus one line in `LAYOUTS`;
  nothing downstream assumes a grid.

## From contacts to taxel forces

- **Why not bin contact points**: the simulated sole is one rigid box, and on
  flat ground MuJoCo contacts only its 4 corners. Binning would leave every
  inner taxel at zero while the foot is flat.
- **Model** (`model.py`, pure torch): the net force and centre of pressure
  come from the per-contact data (`sole_contacts` sensor). Which patches have
  ground beneath them comes from 4 rays per taxel (`tactile_rays_{left,right}`),
  with 1 mm of compliance (the rubber stack barely gives; the margin covers
  MuJoCo's contact penetration). The normal force is split over the supported
  taxels as a plane of pressure (a rigid sole on a thin elastic layer) that
  reproduces the force and CoP exactly, dropping any taxel it puts in
  tension. Shear is shared in proportion to normal load.
- **Cost**: ~15% of simulation throughput on flat ground (147k vs 173k
  steps/s at 8192 envs on one GPU).

## Policy observation

`features.tactile_touch`: per foot 29 values, both feet 58.

- `bits[26]`: 1 where a taxel reads above 3 N, so which part of the sole touches;
- `load`: total normal force / body weight (343 N), clipped to [0, 1.5];
- `cop[2]`: centre of pressure from the taxel readings, scaled to the sole's
  half extents, 0 while the foot is not in contact (15 N or less in total).

The right foot is re-indexed into the left foot's mirror image, so +y is
lateral on both feet.

**Normal load only.** The real foil reports per-taxel pressure, not force
vectors, so the observation has no shear. The model's shear (from
`features.canonical_forces`) is the net tangential force shared in proportion
to load. The bits and CoP are unaffected by a global gain error; `load` is the
one feature a calibration error moves.

**Sensor faults** (`TactileNoiseState`): per episode a per-foot gain (±20%), a
per-taxel gain (±25%) and dead taxels (8%); per step 3 N Gaussian noise per
taxel and a 5% frame drop that holds a foot's previous reading. All scale with
a strength in [0, 1] (default 0.5), set by the `reset_tactile_noise` event.
`tactile_touch(env, noisy=False)` gives the clean features.

**Not modelled**: the real sensor's pressure response (nonlinearity,
saturation, hysteresis, drift), crosstalk, quantisation, latency; the rubber
sole's own friction (the sole box uses mjlab's 0.6).
