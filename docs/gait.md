# Gait analysis

Does feeling the soles change *how* a policy walks, beyond how well it tracks
speed? `scripts/gait_analysis.py` records a walking policy and measures its
gait the way a human gait lab would; `scripts/gait_report.py` compares tactile
against blind across seeds and writes an HTML report.

```bash
python scripts/gait_analysis.py --task Sendg1-Velocity-Flat-G1-LSTM-Tactile \
  --checkpoint <model.pt> --out results/gait/velocity_flat_lstm/tactile_s1_it20000_clean.json
python scripts/gait_analysis.py ... --condition perturbed
python scripts/gait_report.py results/gait/velocity_flat_lstm
```

File names follow `{tactile,blind}_s<seed>_<tag>_{clean,perturbed}.json`.

## Protocol

- Straight walking at fixed forward commands (0.5, 1.0, 1.5 m/s), 32 robots
  each, on the evaluation protocol's flat ground and seed: tactile and blind
  policies face the same robots and the same randomisation.
- 4 s warm-up, then 20 s measured.
- `clean`: no observation noise, no tactile sensor faults or noise (the
  noise-free touch features replace the faulted ones), no pushes.
- `perturbed`: training's noise, sensor faults and pushes.
- Everything is simulator ground truth. The blind task's scene gets the sole
  contact sensors for the measurement; they never reach its actor.

## Measures (`sendg1.eval.gait`)

- **Strides**: one foot's touchdown to its next, cut on the sole's true
  normal force (on above 30 N, off below 10 N, breaks under 60 ms ignored),
  resampled to 0-100 % of the cycle. Strides touching the warm-up or a fall are
  dropped.
- **Joint angles** per plane, with clinical signs:

  | Plane | View | Measures |
  |---|---|---|
  | Sagittal | side | hip flexion, knee flexion, ankle dorsiflexion, pelvis tilt |
  | Frontal | front | hip adduction, ankle inversion, pelvis obliquity |
  | Transverse | top | hip internal rotation, pelvis rotation |

  The G1 knee is a single hinge: no knee ab/adduction.
- **Kinetics**: actuator torque (Nm/kg) against each motor's limit (hip pitch
  and yaw 88 Nm, hip roll and knee 139 Nm, ankle 50 Nm), joint power, positive
  and negative work per stride, peak torque / limit, time saturated (>= 95 %),
  torque rate.
- **Foot contact**: centre of pressure along and across the sole over stance,
  vertical and fore-aft ground force, loading rate, touchdown speed, foot slide.
- **Spatio-temporal**: speed, cadence, stride length, step width, stance and
  double support, clearance, and stride-to-stride variability (CV).
- **Energy and smoothness**: mechanical cost of transport, leg joint
  acceleration RMS.

## Statistics

The seed is the sample: each policy's stride means count once. Scalars: Hedges'
g and a Welch t-test. Curves: a permutation test on the maximum Welch t over the
cycle (nonparametric statistical parametric mapping), exact over all
relabellings; it needs 4+ seeds per group to reach p < 0.05.

Limits: recorded at the 50 Hz policy rate, so loading rate and touchdown speed
are coarse. The real G1 ankle is a two-motor linkage and the simulator models
two serial joints, so ankle torques are approximate for hardware.

## Results

`results/gait/velocity_flat_lstm/` (report: `report.html`). As of 2026-11-16:
LSTM seeds 1-3 at 14k iterations, to be redone at 20k with seeds 1-5. Speed,
cadence, stride length and the joint-angle curves are close between the
conditions. At every speed and in both conditions the tactile policies change
their hip and ankle torques faster (hip pitch torque rate +15-60 %); in the
clean condition their stride timing and step width also vary more. Three seeds
are directional only.
