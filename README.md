# sendg1

Do tactile sensors on the feet of the Unitree G1 measurably improve locomotion
policies?

Every result here is a paired comparison: a baseline policy and a
tactile-augmented policy, same task, same seeds, same evaluation protocol. The
unit of work is **(task x observation-set x seed)**.

Built on [mjlab](https://github.com/mujocolab/mjlab), pinned to **1.6.0**
(commit `0fb8a681`).

## Status

Phase 1: velocity tracking on flat and rough terrain, **blind** policies, no
tactile modelling yet. The 26-taxel normal+shear sensor is not modelled; the
plumbing for it exists and is enforced by tests.

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

Because the height scan is gone, flat and rough share a 96-dim actor input
space: one architecture serves both, and flat->rough transfer stays a
meaningful experiment.

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
- `tactile` — Phase 2, not yet declared.

The critic's contact information is deliberately the stable idealised version.
When `tactile` gains a real sensor model, the critic is untouched, so Phase 1
baselines stay comparable.

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

- The 26-taxel sensor spec (layout, rate, latency, noise) — arriving in 1-2 weeks.
  MuJoCo's `mjSENS_TOUCH` is **not** exposed by mjlab 1.6.0's `BuiltinSensorCfg`,
  so the path is a per-geom `ContactSensorCfg` over the 7 collision capsules per
  foot (`G1_FOOT_GEOMS`).
- Seeds: 1 for now (directional only). 5 before any reported claim.
- `num_envs`: per-task, tune with `scripts/tune_num_envs.py`. On a 12 GB 5070,
  throughput saturates around 8192 (~220k steps/s); larger fits but buys nothing.
  Safe to vary across tasks: the command curriculum advances on a counter that
  ticks once per `env.step()` regardless of batch size, so its stages land at
  fixed iteration counts either way.
