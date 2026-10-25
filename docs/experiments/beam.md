# Beam walking

A blind G1 with the tactile slipper walks along a narrow beam it cannot see,
from a start platform to an end platform, at a joystick forward speed. Plain
PPO (`tasks/beam`, `common/beam`): ten registered tasks, not obs-set variants,
each with an evaluation-only `<task>-Eval` twin (held-out beams, 35 s episodes;
same observations, so its checkpoints load).

| Task id | Beams | Actor sees | Actor dim |
|---|---|---|---|
| `G1-Beam-E2E` | straight | IMU, joints, previous action, forward command, tactile contact map | 160 |
| `G1-Beam-E2E-Curved` | straight, `bend` (joints up to 25°), `turn` (up to 60°) | same | 160 |
| `G1-Beam-Touch-Curved` | as `-Curved` | same, tactile as touch features; symmetric GRU | 156 |
| `G1-Beam-Touch-Curved-Plain` | as `-Curved` | same as `Touch-Curved`; plain GRU (no symmetry) | 156 |
| `G1-Beam-Curved-Blind` | as `-Curved` | as `Touch-Curved-Plain` without tactile input (blind baseline) | 94 |
| `G1-Beam-Curved-Blind-Sym` | as `-Curved` | as `Curved-Blind`; symmetric GRU (blind `Touch-Curved`) | 94 |
| `G1-Beam-Touch-Curved-MLP-Sym` | as `-Curved` | touch features; symmetric MLP, no memory | 156 |
| `G1-Beam-Touch-Curved-MLP-Plain` | as `-Curved` | touch features; plain MLP, no memory | 156 |
| `G1-Beam-Curved-Blind-MLP-Sym` | as `-Curved` | no tactile; symmetric MLP, no memory | 94 |
| `G1-Beam-Curved-Blind-MLP-Plain` | as `-Curved` | no tactile; plain MLP, no memory | 94 |

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
  actor helps. The `-MLP-*` tasks drop the memory: an MLP(512, 256, 128) on the
  current observation only (joints, previous action, IMU, command, tactile),
  mirror-equivariant by construction (`rl/symmetric_mlp.py`) or plain.
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

`beam_eval.py` also reports the gait per width: real swings (air phases of at
least 0.1 s) per metre, their median and p90 apex, the share of lift-offs that
are contact chatter, and stance slide in cm per metre travelled. Success alone
hides a skating gait (no swings, high slide), and such a gait depends on
friction on hardware, so read both.

## Reward study (touch plain MLP)

The first touch-MLP run skated: almost no swings, 1.3-2 m of foot slide per
metre walked. `G1-Beam-Touch-Curved-MLP-Plain-R-<Variant>` changes only the
reward (`env_cfg.REWARD_VARIANTS`, test-enforced), built from six parts:

| part | change |
|---|---|
| `slip` | `foot_slip` (-0.1, \|v\|^2) -> `stance_slip` (-1.0, \|v\| of feet in contact) |
| `clear` | `swing_clearance` +5 per metre gained by steps with >= 5 cm apex |
| `air` | `feet_air_time` 0.25 -> 1.0 |
| `soft` | `touchdown_force_rate` -5 -> -2, swing-foot speed cap 1.2 -> 1.6 m/s |
| `arms` | posture on legs + waist only, `arm_posture` / `arm_joint_vel` -0.02 -> -0.005 |
| `tandem` | `feet_in_line` gap std 8 -> 15 cm (partial credit for a narrower stance) |

Variants: `base`, `slip`, `clear`, `step` (slip + clear), `step_air`,
`step_soft`, `arms`, `tandem`, `step_arms`, `step_tandem`, `all`. Seed 0 runs
on the normal QoS, seed 1 on `--qos=scavenger`. A preempted scavenger job is
requeued and resumes from its newest checkpoint (`train.sbatch`), so it
saves every 250 iterations.

Best straight-beam policy so far (`e2e_v3b`, `results/beam/e2e_v3b*.json`):
success 79/69/56/52/0/0/0 % from 0.40 to 0.05 m.
