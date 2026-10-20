# Tasks and observation sets

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

Beam walking and box pushing have their own pages:
[experiments/beam.md](experiments/beam.md),
[experiments/box-push.md](experiments/box-push.md).

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

## Adding things

**A task** — new directory under `src/sendg1/tasks/`, one `register_variants`
call, one import line in `tasks/__init__.py`. It fans across every observation
set automatically.

**An observation set** — one entry in `OBS_SETS`, one branch in
`build_groups`. It fans across every existing task automatically. The allowlist
test polices it from the moment it appears.
