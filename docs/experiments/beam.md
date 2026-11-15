# Beam walking

A blind G1 with the tactile slipper walks along a narrow beam it cannot see,
from a start platform to an end platform, at a joystick forward speed. Plain
PPO (`tasks/beam`, `common/beam`): registered tasks, not obs-set variants,
each with an evaluation-only `<task>-Eval` twin (held-out beams; same
observations and episode length, so its checkpoints load).

| Task id | Beams | Actor sees | Actor dim |
|---|---|---|---|
| `G1-Beam-Touch-Curved-Plain` | straight, `bend` (joints up to 25°), `turn` (up to 60°) | IMU, joints, previous action, forward command, tactile touch features; GRU | 152 |
| `G1-Beam-Curved-Blind` | same | as `Touch-Curved-Plain` without tactile input (blind baseline) | 94 |
| `G1-Beam-Touch-Curved-MLP-Plain` | same | touch features; plain MLP, no memory | 152 |
| `G1-Beam-Curved-Blind-MLP-Plain` | same | no tactile; plain MLP, no memory | 94 |
| `G1-Beam-Touch-Curved-LSTM` | same | touch features; LSTM(256), 64-step rollouts, LR 4e-4 at start (adaptive) | 152 |
| `G1-Beam-Curved-Blind-LSTM` | same | no tactile; as `Touch-Curved-LSTM` | 94 |

Removed on 2026-11-15: `G1-Beam-E2E{,-Curved}` (the old soft contact-map
tactile input) and the round-three experiments `-Plain-R64`, `-Plain-R2-35s`,
`-Plain-R2-35s-R64`, `-LSTM-R2-35s`, `-LSTM-R2-35s-R64`. Their results below
stay as a record; their checkpoints no longer load.

- **Terrain** (`common/beam/terrain.py`): a 4 m beam between two 1 m platforms,
  0.15-0.30 m above the ground (one height per column). Rows are width levels:
  0.40, 0.30, 0.25, 0.20, 0.175, 0.15, 0.125, 0.10, 0.085, 0.07, 0.05 m
  (seven levels, 0.40-0.05 m, before round three). Next to the straight beams
  every task has a `bend` and a `turn` column per height: straight pieces of 0.6-1.4 m joined at random
  heading changes, a disc filling each joint. Every bent tile has its own path.
- **Actor**: rsl_rl's `RNNModel`, a GRU(256) -> MLP(256, 128). It sees only
  `onboard` / `sensor` terms (`common/beam/terms.py`); registration fails
  otherwise. Tactile input (`Touch` tasks): per foot a 26-bit "taxel above
  3 N" map, load / body weight and CoP (+-1, 0 in the air), with gain errors,
  dead taxels, noise, frame drops and 0-1 step latency: the model every task
  shares ([tactile.md](../tactile.md#what-the-actor-sees)). The `-MLP-Plain`
  tasks drop the memory: rsl_rl's `MLPModel` (512, 256, 128) on the current
  observation only. `rl_cfg.ActorKind` also offers an LSTM(256).
- **Removed actors** (2026-11-14): the mirror-equivariant GRU and MLP
  (`G1-Beam-Touch-Curved`, `-Curved-Blind-Sym`, `-MLP-Sym`) and the causal
  transformer (`-Transformer`: 64 frames, 4 frames per token, 2 layers). Their
  results below stay as a record; their checkpoints no longer load.
- **Critic** (the same for touch and blind): MLP on the actor's
  proprioception plus ground truth: base velocity, feet and base pose relative
  to the beam, beam progress, width, height and friction, the clean touch
  features, clean per-taxel normal and shear forces, the feet's slip velocity,
  contact forces, actuation lag, and how far the beam turns 0.25-1.5 m ahead.
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
  cm before, which also ended episodes where a sole only brushed the beam side;
  that is now neither ended nor penalised).
- **Reward**, `beam_reward` (one table, `build_beam_reward` in
  `tasks/beam/env_cfg.py`; first trained in the `touch_plain_s0` runs): forward progress with
  slack, reaching the end, feet and CoM over the centreline, light first
  touches (`load_off_support`, `early_contact_overload`), heel-to-toe steps
  paid per metre gained (`feet_in_line`), posture with shoulder roll left free
  for balancing with the arms out, and the walking regularisers.

```bash
train G1-Beam-Touch-Curved-Plain --env.scene.num-envs 3072 --agent.run-name touch_plain_s0
train G1-Beam-Curved-Blind --env.scene.num-envs 3072 --agent.run-name blind_s0
python scripts/beam_eval.py --task G1-Beam-Touch-Curved-Plain-Eval --checkpoint <pt> --out results/beam/<name>_eval.json
python scripts/beam_video.py --task G1-Beam-Touch-Curved-Plain --path bend --checkpoint <pt> --width 0.20
python scripts/render_terrain.py --terrain-set beam --prefix beam --row 2
```

`beam_eval.py` also reports the gait per width: real swings (air phases of at
least 0.1 s) per metre, their median and p90 apex, the share of lift-offs that
are contact chatter, and stance slide in cm per metre travelled. Success alone
hides a skating gait (no swings, high slide), and such a gait depends on
friction on hardware, so read both.

## Reward study (touch plain MLP, removed)

The first touch-MLP run skated (almost no swings, 1.3-2 m of foot slide per
metre walked), so rounds one and two tried reward variants on the touch MLP
(`-R-<Variant>` tasks): a linear stance-slip cost, pay per metre gained by
lifted steps (`swing_clearance`), more air time, softer touchdowns, freer
arms, a wider heel-to-toe reward, and combinations. Best at 40k (held-out,
seeds 0/1): `step` (slip + clearance), 89/86/79/60 % and 82/75/56/17 % at
0.40-0.15 m; base 51/44/1/0 and 50/39/0/0. The study and the variants were
removed on 2026-11-15: every beam task uses `beam_reward`, the reward of the
`touch_plain_s0` GRU runs (2026-10-17/20), without the later
`foot_side_contact` cost.

## Round three: memory over a gait cycle (tasks removed)

A record: the tasks below were removed on 2026-11-15.

`beam_reward`, touch input, seed 0, under the round-three episodes,
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
