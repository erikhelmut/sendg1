# Beam walking

A blind G1 with the tactile slipper walks along a narrow beam it cannot see,
from a start platform to an end platform, at a joystick forward speed. Plain
PPO (`tasks/beam`, `common/beam`): registered tasks, not obs-set variants,
each with an evaluation-only `<task>-Eval` twin (held-out beams; same
observations and episode length, so its checkpoints load).

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
| `G1-Beam-Touch-Curved-Transformer` | as `-Curved` | touch features, last 64 frames; causal transformer | 64 x 156 |
| `G1-Beam-Touch-Curved-Plain-R64` | as `-Curved` | as `Touch-Curved-Plain`, 64-step rollouts | 156 |
| `G1-Beam-Touch-Curved-Plain-R2-35s` | round-two widths | as `Touch-Curved-Plain`, round-two rules, 35 s episodes | 156 |
| `G1-Beam-Touch-Curved-Plain-R2-35s-R64` | round-two widths | as `-R2-35s`, 64-step rollouts | 156 |
| `G1-Beam-Touch-Curved-LSTM-R2-35s` | round-two widths | as `-R2-35s`, LSTM(256) instead of the GRU | 156 |
| `G1-Beam-Touch-Curved-LSTM-R2-35s-R64` | round-two widths | as `-LSTM-R2-35s`, 64-step rollouts | 156 |

- **Terrain** (`common/beam/terrain.py`): a 4 m beam between two 1 m platforms,
  0.15-0.30 m above the ground (one height per column). Rows are width levels:
  0.40, 0.30, 0.25, 0.20, 0.175, 0.15, 0.125, 0.10, 0.085, 0.07, 0.05 m
  (seven levels, 0.40-0.05 m, before round three). The curved task adds a `bend` and
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
  `-Transformer` (`rl/transformer.py`): the actor group carries the last 64
  frames (1.28 s); every 4 frames form one token, 2 pre-norm layers (64 dim,
  4 heads, causal mask) run over the 16 tokens, the newest token's output
  feeds an MLP (256, 128). One token per frame made the PPO update ~25x the
  GRU's and did not fit a 24 GB GPU. `-R64` is the plain GRU with 64-step
  rollouts (in 8 minibatches, so each keeps the other runs' size): its
  gradients span 1.28 s instead of 0.64 s, under one step of the slow beam gait.
- **Critic**: MLP on the actor's terms plus ground truth: base velocity, feet
  and base pose relative to the beam, beam progress, width, height and friction,
  the clean tactile map, contact forces, actuation lag; on curved beams also
  how far the beam turns 0.25-1.5 m ahead. The `Touch` critics also get the
  clean per-taxel normal and shear forces and the feet's slip velocity.
- **Curriculum** (per env): each env keeps its last 3 results on its level, a
  success (a walking episode that reaches the end or covers 2 m of beam
  without falling) or a fall; 2 successes move it one width narrower, 2 falls
  one wider. 20% of resets replay an easier level and 15% try the next harder
  one, neither moving the level. A quarter of resets start mid-beam in a
  recorded heel-to-toe stance (`results/beam/v4_beam_states.pt`).
  Checkpoints carry the per-env levels (`tasks/beam/runner.py`), so a resumed
  run continues where it was. Before round three: one success (1 m) or one fall
  moved the level, no harder tries, and a resume restarted at the widest beam.
- **Episodes**: 35 s, training and evaluation alike (training used 20 s
  before), ending 2 s after the robot reaches the end platform (a time-out,
  not a fall). A fall ends one early: tilt over 40 deg, pelvis low, a non-foot
  contact, or a sole 10 cm below the beam top (a foot that has dropped off; 3
  cm before, which also ended episodes where a sole only brushed the beam side
  and now costs `foot_side_contact`).
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
| `walk` | `stance_slip` -0.3 on top of `foot_slip`, `swing_clearance` +2 for >= 3 cm apex |

Variants: `base`, `slip`, `clear`, `step` (slip + clear), `step_air`,
`step_soft`, `arms`, `tandem`, `step_arms`, `step_tandem`, `all`, and from
round two `walk`, `walk_arms`. Seed 0 runs
on the normal QoS, seed 1 on `--qos=scavenger`. A preempted scavenger job is
requeued and resumes from its newest checkpoint (`train.sbatch`), so it
saves every 250 iterations.

Round one (seed 0, held-out, 20k): every `slip`/`clear` variant steps (5-10
swings/m, 3-6 cm apex, half the slide) but loses narrow-beam success; best
`step_arms` 57/61/26/2 % at 0.40-0.15 m vs base 86/85/59/21 %, and base itself
switches to low (1.4 cm) steps between 15k and 20k (95/91/72/30 % at 15k while
it still skated). `step_air`, `step_tandem` and `all` never leave the widest
beam. Round two tries the gentler `walk` and continues base, `step`,
`step_arms` and `tandem` to 40k.

Round two (held-out, 40k, seeds 0/1): `step` keeps its 6-7 cm steps and
recovers, 89/86/79/60 % and 82/75/56/17 % at 0.40-0.15 m, the best 0.15 m
result so far and still rising at 40k. Base gets worse with its own late
switch to low steps (51/44/1/0, 50/39/0/0). `step_arms` splits by seed
(96/83/73/38 vs 19/13/0/0), `tandem` stays near 55/60/5/0, and `walk`
(20k) steps lower (4 cm) for 65/61/10/0 and 82/83/41/1 %.

## Round three: memory over a gait cycle

Unchanged base reward, touch input, seed 0, under the round-three episodes,
terminations and curriculum above: `-Transformer` against `-Plain-R64`. Their
results are not comparable with earlier rounds' (different widths, episodes
and curriculum); for a reference, re-evaluate old checkpoints with the current
`-Eval` tasks.

Both climbed to a mean level of ~2.2-2.7 (0.25 m) by 6-7k iterations
(transformer; R64 at the same samples) and fell back to ~0.3-0.4 by 10k, while
the action std rose 0.32 -> 0.39 and tilt falls grew. Since both architectures
did it, the cause is likely shared: `-R2-35s` and `-R2-35s-R64`
(`make_beam_env_cfg(round2=True)`) keep round two's widths, one-episode
curriculum (1 m success, no harder tries), 3 cm foot-off and reward, and change
only the episodes to 35 s.

They collapsed too, later: at ~23k (32 steps) and ~25k (64 steps,
32-step-equivalent iterations), after peaking near 0.27 m; the round-two GRU
(20 s) peaked near 0.22 m and held to 20k. Every collapse looks the same: the
action std jumps from ~0.34 to 0.37-0.39 within ~1k iterations, tilt falls
rise and the curriculum drains to the wide beams. So the 35 s episodes alone
destabilise training and the other round-three changes bring it forward. Next:
`-R2-35s` with `entropy_coef` 0.001 instead of 0.005 (`--agent.algorithm.entropy-coef`),
and the `-LSTM-R2-35s` pair against the GRU pair.

Held-out eval (seed 0, success % at 0.40/0.30/0.20/0.15 m; iterations in each
run's own units; `results/beam/{transformer,gru_r64,gru_r2_35s,gru_r2_35s_r64}_s0_it*_eval.json`):
transformer 7.5k 88/86/53/17, 30k 80/74/45/6; R64 7.5k 95/90/69/23, 15k
47/39/2/0; R2-35s 10k 94/84/61/18, 25k 26/18/2/0; R2-35s-R64 5k 96/88/69/21,
15k 30/29/0/0. The best checkpoints come early, and every decline comes with
the gait turning from skating (0-6 swings/m, 70-210 cm slide per metre) to
low steps (9-11 swings/m, 2-3 cm apex, 25-60 cm slide): the same switch as the
round-one base MLP between 15k and 20k.

Best straight-beam policy so far (`e2e_v3b`, `results/beam/e2e_v3b*.json`):
success 79/69/56/52/0/0/0 % from 0.40 to 0.05 m.
