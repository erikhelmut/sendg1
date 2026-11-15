# Tasks and observation sets

## Registered tasks

| Task id | Actor sees | Actor dim |
|---|---|---|
| `Sendg1-Velocity-Flat-G1-Blind` | proprioception only | 96 |
| `Sendg1-Velocity-Rough-G1-Blind` | proprioception only | 96 |
| `Sendg1-Velocity-{Flat,Rough}-G1-Tactile` | + tactile slipper (`tactile_touch`) | 154 |
| `Sendg1-VelocityPush-G1-{Blind,Tactile}` | as flat velocity, plus a hand-force curriculum | 96 / 154 |
| `Sendg1-BoxPush{,Curriculum,Patches}-G1-Blind` | proprioception, pilot command | 96 |
| `Sendg1-BoxPush{,Curriculum,Patches}-G1-Tactile` | + tactile slipper (`tactile_touch`) | 154 |

Because the height scan is gone, flat and rough share a 96-dim actor input
space: one architecture serves both, and flat->rough transfer stays a
meaningful experiment.

Beam walking, box pushing, rolling-cylinder balance and the slackline have
their own pages:
[experiments/beam.md](experiments/beam.md),
[experiments/box-push.md](experiments/box-push.md),
[experiments/rolling-cylinder.md](experiments/rolling-cylinder.md),
[experiments/slackline.md](experiments/slackline.md).

## Observation sets

Defined in `sendg1/common/obs/obs_sets.py` as mappings from rsl_rl observation
*sets* to our named *groups*:

```python
"blind":   actor=("proprio",)            critic=("proprio", "privileged")
"tactile": actor=("proprio", "tactile")  critic=("proprio", "privileged")
```

- `proprio` — deployable, 96-dim, actor and critic.
- `privileged` — critic only, permanently. Holds `base_lin_vel`, `height_scan`,
  unbiased joint positions, and the *idealised* foot contact terms.
- `tactile` — the tactile slipper, actor only: the `tactile_touch` term, the
  same sensor model every task uses ([tactile.md](tactile.md#what-the-actor-sees)).

A third, diagnostic `contact_oracle` set (noiseless foot contact in the actor)
was removed on 2026-11-15.

The critic's contact information is deliberately the stable idealised version,
so changes to the tactile sensor model never touch it. The tactile sensors are
added to a scene only for sets that use the group; the sensor-fault reset event
(`tactile_faults`) is in every set's events, so the sets differ in the actor's
input alone.

## Adding things

**A task** — new directory under `src/sendg1/tasks/`, one `register_variants`
call, one import line in `tasks/__init__.py`. It fans across every observation
set automatically.

**An observation set** — one entry in `OBS_SETS`, one branch in
`build_groups`. It fans across every existing task automatically. The allowlist
test polices it from the moment it appears.
