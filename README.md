# sendg1

Do tactile sensors on the feet of the Unitree G1 measurably improve locomotion
policies?

Every result here is a paired comparison: a baseline policy and a
tactile-augmented policy, same task, same seeds, same evaluation protocol. The
unit of work is **(task x observation-set x seed)**.

Built on [mjlab](https://github.com/mujocolab/mjlab), pinned to **1.6.0**
(commit `0fb8a681`).

## Status

Phase 1: velocity tracking on flat and rough terrain, **blind** policies.

Phase 2 has started: the tactile sole is modelled (`tactile_array`, see
[Tactile sole](#tactile-sole)) on the slipper's real 26-taxel layout. The
model is still idealised -- no noise,
latency or calibration error -- so the set is diagnostic for now.

## Quick start

```bash
conda activate sendg1
pip install -e ".[dev]"

./scripts/test.sh                       # or: pytest
list-envs --keyword sendg1              # mjlab's CLI sees our tasks
python scripts/sweep.py --dry-run       # what a default sweep would run

python scripts/sweep.py --tasks flat --seeds 0 --max-iterations 3000
python scripts/evaluate.py --task Sendg1-Velocity-Flat-G1-Blind --seed 0
python scripts/aggregate.py
```

## Registered tasks

| Task id | Actor sees | Actor dim |
|---|---|---|
| `Sendg1-Velocity-Flat-G1-Blind` | proprioception only | 96 |
| `Sendg1-Velocity-Rough-G1-Blind` | proprioception only | 96 |
| `Sendg1-Velocity-Flat-G1-ContactOracle` | + idealised foot contact | 104 |
| `Sendg1-Velocity-Rough-G1-ContactOracle` | + idealised foot contact | 104 |
| `Sendg1-Velocity-{Flat,Rough}-G1-TactileArray` | + per-taxel normal & shear | 252 |
| `Sendg1-BoxPush{,Curriculum,Patches}-G1-Blind` | proprioception, box pose, target | 100 |
| `Sendg1-BoxPush{,Curriculum,Patches}-G1-ContactOracle` | + idealised foot contact | 108 |
| `Sendg1-BoxPush{,Curriculum,Patches}-G1-TactileArray` | + per-taxel normal & shear | 256 |

Because the height scan is gone, flat and rough share a 96-dim actor input
space: one architecture serves both, and flat->rough transfer stays a
meaningful experiment.

### Beam walking

A blind G1 with the tactile slipper walks along a narrow beam it cannot see,
from a start platform to an end platform, at a joystick forward speed. Plain
PPO (`tasks/beam`, `common/beam`): four registered tasks, not obs-set variants.

| Task id | Beams | Actor sees | Actor dim |
|---|---|---|---|
| `G1-Beam-E2E` | straight | IMU, joints, previous action, forward command, tactile contact map | 160 |
| `G1-Beam-E2E-Curved` | straight, `bend` (joints up to 25°), `turn` (up to 60°) | same | 160 |
| `G1-Beam-Touch-Curved` | as `-Curved` | same, tactile as touch features; symmetric GRU | 156 |
| `G1-Beam-Touch-Curved-Plain` | as `-Curved` | same as `Touch-Curved`; plain GRU (no symmetry) | 156 |

- **Terrain** (`common/beam/terrain.py`): a 4 m beam between two 1 m platforms,
  0.15-0.30 m above the ground (one height per column). Rows are width levels:
  0.40, 0.30, 0.20, 0.15, 0.10, 0.07, 0.05 m. The curved task adds a `bend` and
  a `turn` column per height: straight pieces of 0.6-1.4 m joined at random
  heading changes, a disc filling each joint. Every bent tile has its own path.
- **Actor**: a GRU(256) -> MLP(256, 128), left-right equivariant by
  construction (`rl/symmetric_rnn.py`). It sees only `onboard` / `sensor`
  terms (`common/beam/terms.py`); registration fails otherwise. Tactile input:
  per foot a soft per-taxel contact map, load, CoP, contact area, edge
  loading, contact flag, with gain errors, dead taxels, noise, frame drops and
  0-1 step latency. The `Touch` tasks instead give per foot a 26-bit "taxel
  above 3 N" map, load / body weight, CoP (+-1, 0 in the air), contact area
  and a contact flag, under the same sensor faults. `Touch-Curved-Plain` uses
  rsl_rl's plain `RNNModel` of the same size, to test whether the symmetric
  actor helps.
- **Critic**: MLP on the actor's terms plus ground truth: base velocity, feet
  and base pose relative to the beam, beam progress, width, height and friction,
  the clean tactile map, contact forces, actuation lag; on curved beams also
  how far the beam turns 0.25-1.5 m ahead. The `Touch` critics also get the
  clean per-taxel normal and shear forces and the feet's slip velocity.
- **Curriculum** (per env): one width level narrower after a walking episode
  that reaches the end or covers 1 m of beam without falling, one wider after a
  fall; 20% of resets replay an easier level. A quarter of resets start
  mid-beam in a recorded heel-to-toe stance (`results/beam/v4_beam_states.pt`).
- **Reward** (one table in `tasks/beam/env_cfg.py`): forward progress with
  slack, reaching the end, feet and CoM over the centreline, light first
  touches (`load_off_support`, `early_contact_overload`), heel-to-toe steps
  paid per metre gained (`feet_in_line`), posture with shoulder roll left free
  for balancing with the arms out, and the walking regularisers.

```bash
train G1-Beam-E2E --env.scene.num-envs 3072 --agent.logger tensorboard
train G1-Beam-Touch-Curved-Plain --env.scene.num-envs 3072 --agent.run-name touch_plain_s0
train G1-Beam-Curved-Blind --env.scene.num-envs 3072 --agent.run-name blind_s0
python scripts/beam_eval.py --task G1-Beam-Touch-Curved-Plain-Eval --checkpoint <pt> --out results/beam/<name>_eval.json
python scripts/beam_eval.py --checkpoint runs/beam_e2e/<run>/model_X.pt --out results/beam/<name>.json
python scripts/beam_eval.py --task G1-Beam-E2E-Curved --checkpoint <pt> --out results/beam/<name>_curved.json
python scripts/beam_video.py --task G1-Beam-E2E-Curved --path bend --checkpoint <pt> --width 0.20
python scripts/render_terrain.py --terrain-set beam --prefix beam --row 2
```

Best straight-beam policy so far (`e2e_v3b`, `results/beam/e2e_v3b*.json`):
success 79/69/56/52/0/0/0 % from 0.40 to 0.05 m.

### Box pushing under unknown floor friction

A **scripted high level** walks the robot around behind the box and pushes
along the box-to-target line; the **learned policy is the low level**, which
executes that walking command while leaning into a heavy box on a floor of
unknown friction. The split, and the reward shape, follow VOFA
(arXiv:2605.01518); the high level is scripted rather than learned because the
box and target poses are known in simulation, and because a fixed high level is
a control variable -- byte-identical across observation sets, so it cannot
absorb any of the difference between them.

The actor therefore sees exactly what the velocity tasks' actor sees: 96 dims,
with the pilot's (vx, vy, wz) in place of the operator's command. It is never
told where the box is.

- **Difficulty is the traction margin, not the mass.** Each episode draws
  `margin = 1 - required/available` push force, then derives the box mass:
  `m = (1 - margin) * mu_floor * m_robot / mu_box`. Sampling mass and friction
  independently left only 11% of episodes near the limit against 23%
  infeasible; drawing the margin makes the near-limit share a design choice.
  At the shipped ranges (margin U[-0.05, 0.40], floor mu U[0.25, 0.7], box mu
  U[0.35, 0.7], mass clipped to 8-50 kg): **10% infeasible, 45% near the limit,
  45% comfortable**, median push 125 N (38% of the robot's weight).
- **The box is wide and low** (0.7-0.9 m deep, 0.6-0.8 m tall) so that it
  slides rather than tips: a box tips first when the push height exceeds
  `depth / (2 mu_box)`, which at the first version's 0.4-0.6 m depth was 72% of
  boxes at a 0.7 m push. Now ~3%.
- **Friction model.** MuJoCo max-mixes friction for equal priorities, and the
  G1 foot capsules carry priority 1, so priorities are set so each contact
  takes exactly one sampled value: foot/floor -> floor mu, box/floor -> box mu,
  hand/box -> the ball hand's fixed 0.8. Verified per contact in the solver by
  `tests/test_box_push.py`.
- **Ball hands.** A 70 mm ball replaces each passive rubber hand
  (`get_g1_ball_hands_cfg`): one contact per hand, its own friction, and no
  hooking the box's top edge. Models a printed end-effector, not the stock
  robot; `ball_hands=False` restores the rubber hands.
- **Rewards** (one table in `tasks/box_push/env_cfg.py`): command tracking
  (+2.0/+2.0) and `exp(-d_box_target^2/2.5^2)` (+2.5) are dense and positive,
  as in VOFA; then a success bonus (+20, box in the disc for 2 s), hand contact
  on the push face (+0.5), and penalties for foot slip (-2.0), falls (-10),
  box tipped (-10), box tilt (-5), feet on the box (-1), torque and action
  rate. The first version had no positive per-step term, and both
  2,000-iteration checks learned to END the episode instead: first by tipping
  the box (99% of episodes), then by falling immediately.
- **Critic** adds floor mu under each foot, box mass/mu/size, box pose,
  velocity and tilt, sole slip velocities and the push force.
- **`tactile_array` here also carries 12 aggregate channels**: per foot the net
  normal force, shear, centre of pressure and the shear/normal ratio -- the
  foot's distance from its friction cone. Implied by the per-taxel array, but
  given explicitly so the policy need not learn a division. Actor: 96 + 144 +
  12 = 252.
- **Per-episode log** (`Episode_Metrics/*`; one CSV row per episode in
  evaluation): success, time to target, full-foot slip events (a foot in
  contact whose heel and toe both slide > 0.1 m/s for >= 60 ms), falls split
  into before/after first box contact (VOFA's pre/post-contact failures),
  command tracking error, traction margin, box tipped, mean push force, share
  of push impulse in double support, floor mu, box mass, box mu.
- **Flags** are separate registered tasks: `Sendg1-BoxPushCurriculum-G1-*`
  (difficulty alpha: margin and floor mu grow from the easy corner when >= 50%
  of a 512-episode window succeeds) and `Sendg1-BoxPushPatches-G1-*` (1-3
  patches of mu 0.1-0.3 on the path -- the sharpest tactile test, since the
  friction changes underfoot without warning).
- **Throughput:** 4,096 envs, ~125k steps/s blind and ~110k with the tactile
  array on the 5070. The box's derived inertia constants are written
  analytically per reset; mjlab's generic `set_const` recomputes every world on
  every step and was 2.4x slower.

```bash
python scripts/box_push_feasibility.py                 # margins and tipping of the ranges
python scripts/box_push_video.py --random --episodes 4 # scene check
python scripts/sweep.py --tasks box_push --obs-sets blind contact_oracle tactile_array --seeds 0 1 2
python scripts/evaluate_box_push.py --task Sendg1-BoxPush-G1-TactileArray --seed 0
python scripts/plot_box_push_curves.py runs/box_push__tactile_array/<run> --out curves.png
```

Evaluation (`eval/box_push.py`): floor mu {0.25..0.70} x mass {10..40} kg, 32
episodes per cell. Every episode's world (box mu, size, gap, target) is drawn
from the protocol seed and pinned, so every policy meets the same 640 episodes.

## The three design rules

**1. The actor sees only what the real blind G1 measures.**
IMU angular velocity, projected gravity (from the IMU site sensor, not
ground-truth body orientation), joint positions (encoder-biased) and velocities,
previous action, user command. No `base_lin_vel` (needs a state estimator), no
`height_scan` (needs an elevation-mapping stack).

Enforced two ways: `tests/test_actor_allowlist.py`, and an import-time check in
`sendg1/tasks/_register.py` so `train` refuses to start rather than failing only
under pytest. Every term declares its availability in
`sendg1/common/obs/terms.py`; an *undeclared* term is also a failure, so there is
no bypass by forgetting to register one. Six negative tests prove the guard
actually fires.

**2. Only the actor changes between conditions.**
The critic, rewards, events, terminations, terrain and network architecture are
identical across observation sets — verified by
`tests/test_config_fingerprint.py`. If two things differed at once, a measured
difference could not be attributed to the actor's observation space.

**3. Configs are fingerprinted.**
A pin stops mjlab moving; the fingerprints tell you *what* moved when you
deliberately upgrade. When one fails:

```bash
python scripts/show_fingerprints.py --task <id> --dump   # canonical config
python scripts/show_fingerprints.py --write              # re-record
```
Re-record in the same commit as the change, and re-run every baseline that moved.

## Observation sets

Defined in `sendg1/common/obs/obs_sets.py` as mappings from rsl_rl observation
*sets* to our named *groups*:

```python
"blind":          actor=("proprio",)                  critic=("proprio", "privileged")
"contact_oracle": actor=("proprio", "contact_ideal")  critic=("proprio", "privileged")
```

- `proprio` — deployable, 96-dim, actor and critic.
- `privileged` — critic only, permanently. Holds `base_lin_vel`, `height_scan`,
  unbiased joint positions, and the *idealised* foot contact terms.
- `contact_ideal` — oracle contact, diagnostic only.
- `tactile_array` — per-taxel normal and shear force on both soles, actor only.
  Diagnostic while the sensor model is idealised.

The critic's contact information is deliberately the stable idealised version.
When `tactile_array` gains a realistic sensor model, the critic is untouched, so
Phase 1 baselines stay comparable. The tactile sensors are added to a scene only
for sets that use the group, so every other variant's scene -- and recorded
fingerprint -- is unchanged.

## Tactile sole

```bash
python scripts/show_sole.py          # the sole as simulated: mesh, capsules, taxels
python scripts/tactile_viewer.py     # MP4 of the robot walking + live taxel readings
python scripts/export_foot_cad.py    # foot as DXF + STL in the sole frame, mm (needs .[cad])
python scripts/sweep.py --tasks flat --obs-sets tactile_array --seeds 0
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
- **Observation**: per foot `[normal(26), shear_x(26), shear_y(26)]` in N,
  sole frame, `sign(x)·log1p(|x|)` -- 156 dims.
- **Cost**: ~15% of training throughput on flat ground (147k vs 173k steps/s
  at 8192 envs).
- **Not modelled yet**: sensor noise, latency, per-taxel gain/offset,
  saturation, crosstalk, dead taxels; the slipper's thickness and friction;
  torsional shear about the CoP.

### `contact_oracle` is diagnostic, and off by default

Its actor gets noiseless, zero-latency, perfectly-calibrated foot contact and net
contact force — an **upper bound** on what any real tactile array could deliver,
not a deployable policy. It exists to answer the question that gates the project:
*is there room for foot contact information to help this policy at all?* If it
does not beat `blind`, a noisy 26-taxel sensor will not either.

It is never in a default sweep. Request it explicitly:

```bash
python scripts/sweep.py --tasks flat --obs-sets contact_oracle --seeds 0
```

## Evaluation videos

Every policy video goes through `sendg1.video.EvalVideo` (`record_video.py`,
`beam_video.py`, `box_push_video.py`, `evaluate_box_push.py`), so clips of
paired policies look the same:

- **Theme** (`--theme`): `studio` (default; dark grid floor fading into a dark
  sky, terrain in greys), `gray`, or `mujoco` (mjlab's stock look). The theme
  only touches the renderer's private model copy; physics and sensors never see it.
- **Caption**, top left: a badge (`TACTILE`, `BLIND`, `CONTACT ORACLE`, read
  off the actor's observation terms), the task, and the script's lines.
- **Tactile feet card**, bottom right: both soles from above, toe up, with
  per-taxel normal force, the centre of pressure, per-foot load and a 3 s trace
  of the load on a newton axis, with the robot's body weight as a dashed reference.
  It is live when the actor reads the tactile sole and all zero otherwise:
  a blind policy gets nothing from it, and the clip shows that.

A new video script needs `prepare_video_cfg(env_cfg, w, h)` before the env is
built, `EvalVideo(env, rl_cfg, title=...)` after, `video.capture(lines)` per
step and `video.save(path)`.

## Adding things

**A task** — new directory under `src/sendg1/tasks/`, one `register_variants`
call, one import line in `tasks/__init__.py`. It fans across every observation
set automatically.

**An observation set** — one entry in `OBS_SETS`, one branch in
`build_groups`. It fans across every existing task automatically. The allowlist
test polices it from the moment it appears.

## Layout

```
src/sendg1/
  common/          shared library: robot, obs terms/groups, rewards, events, sensors
  tasks/           task packages; _register.py fans each across all obs sets
  eval/            the measurement protocol (versioned; it is the product)
  fingerprint.py   stable config hashing
tests/             allowlist, fingerprints, registration, import order
scripts/           sweep, evaluate, aggregate, fingerprints, num_envs tuner
runs/              training output (gitignored)
results/           eval JSON, one per (experiment, seed) — committed
```

## Decisions on record

- **29-DoF G1, all joints actuated.** mjlab ships only this variant. Upper-body
  motion is a confound for a foot study, but every reward weight upstream was
  tuned for 29 DoF, and un-actuating a joint does not freeze it — the compliant
  position actuators let un-actioned arms drift ~0.5 rad. `DofSet` in
  `common/robots/g1.py` keeps `legs12` available as a robustness check.
- **Import mjlab, own the assembly.** We use mjlab's MDP term library, sensors
  and terrain generator, but build our own configs rather than mutating
  `unitree_g1_flat_env_cfg`. The numbers that define the baseline are ours.
- **Contact-dependent rewards are kept.** They cannot leak into a deployed
  policy (rewards are training-only). They do make the *baseline* stronger, since
  it must infer contact from proprioception to collect them — so any tactile gain
  measured is conservative, which is the honest comparison.
- **Primary metric fixed in advance**: linear velocity tracking error. Declared
  in `eval/metrics.py` so it cannot be chosen after seeing results.

## Known environment issue

This machine has ROS 2 Jazzy on `PYTHONPATH`, leaking Python 3.12 packages into
this 3.11 env. It injects 7 pytest plugins that crash on import.
`pyproject.toml` disables them by name; `scripts/test.sh` removes the cause with
`unset PYTHONPATH`, which is more robust.

## Open

- The sensor's rate, latency, noise and units. The layout is in (`slipper_v1`).
- Seeds: 1 for now (directional only). 5 before any reported claim.
- `num_envs`: per-task, tune with `scripts/tune_num_envs.py`. On a 12 GB 5070,
  throughput saturates around 8192 (~220k steps/s); larger fits but buys nothing.
  Safe to vary across tasks: the command curriculum advances on a counter that
  ticks once per `env.step()` regardless of batch size, so its stages land at
  fixed iteration counts either way.
