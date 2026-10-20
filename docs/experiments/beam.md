# Beam walking

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
python scripts/beam_eval.py --checkpoint runs/beam_e2e/<run>/model_X.pt --out results/beam/<name>.json
python scripts/beam_eval.py --task G1-Beam-E2E-Curved --checkpoint <pt> --out results/beam/<name>_curved.json
python scripts/beam_video.py --task G1-Beam-E2E-Curved --path bend --checkpoint <pt> --width 0.20
python scripts/render_terrain.py --terrain-set beam --prefix beam --row 2
```

Best straight-beam policy so far (`e2e_v3b`, `results/beam/e2e_v3b*.json`):
success 79/69/56/52/0/0/0 % from 0.40 to 0.05 m.
