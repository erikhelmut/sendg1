# Rolling-cylinder balance

The G1 stands across a cylinder lying free on the floor and has to stay on it
for a 20 s episode without letting it roll away. Each sole touches the
cylinder along a single line. Where that line falls under the sole is the
balance cue: the tactile slipper resolves it directly, while a blind robot
has to infer it from its joints and IMU. Plain PPO (`tasks/rolling_cylinder`,
`common/rolling_cylinder`), set up as a paired comparison from the start.

| Task id | Actor sees | Actor dim |
|---|---|---|
| `G1-RollingCylinder-Blind` | IMU, joints, previous action | 93 |
| `G1-RollingCylinder-Touch` | the same plus the tactile touch features | 155 |

Everything except the actor's inputs is the same in both tasks: scene
(tactile sensors included, the critic reads them), critic, reward, events,
terminations, PPO and network. `tests/test_rolling_cylinder.py` checks this
by fingerprint.

- **Cylinder** (`common/rolling_cylinder/model.py`): rigid, 30 cm diameter,
  80 cm long, 8 kg, friction 0.9. These numbers are chosen, not sourced. The
  radius is the main difficulty knob. Startup DR: mass x0.6-1.5 with the
  inertia scaled to match, friction 0.6-1.0.
- **Spawn**: the cylinder at rest at a random yaw, the robot in its default
  stance facing the rolling direction, sole centres on the top line within
  ±2 cm, dropped from 1 cm. With zero actions it falls off in ~0.65 s.
- **Actor and PPO**: rsl_rl's plain GRU `RNNModel` (GRU 256 -> MLP 256, 128),
  no mirror symmetry; the beam task's PPO settings (`tasks/beam/rl_cfg.py`),
  10k iterations. No command: the task is balance only.
- **Critic**: the blind actor's terms plus ground truth: base velocity,
  cylinder motion (rolling/sliding speed, yaw and spin rate, drift from
  spawn), each sole relative to the top line, base and CoM relative to the
  cylinder, clean touch features and per-taxel forces, foot slip on the
  surface, foot forces, actuation lag.
- **Terminations**: a foot on the floor, any other body on the floor, base
  below 0.8 m, torso tilt over 40°.

## Reward

At equilibrium the CoM, the line where the soles touch the cylinder and the
line where the cylinder touches the floor are stacked vertically, and the
cylinder is at rest. Any fore-aft CoM error rolls the cylinder, and the
cylinder carries the feet with it. The table (`build_rewards` in
`tasks/rolling_cylinder/env_cfg.py`) is ordered by that:

| Group | Terms (weight) |
|---|---|
| Survival | `alive` (1.0), `termination` (-100, once) |
| Equilibrium | `cylinder_still` (1.0, std 0.1 m/s), `com_over_support` (1.0, std 3 cm fore-aft, 8 cm along the axis), `stay_near_spawn` (0.5, std 0.3 m) |
| Posture with control authority | `feet_on_top` (0.5, std 4 cm), `upright` (0.5), `posture` (0.3), `foot_yaw` (0.3), `foot_flat` (0.3), `heading_to_cylinder` (0.3) |
| Costs | `foot_off_cylinder` (-0.5), `foot_past_end` (-10/m), cylinder sliding (-1.0) and turning (-0.5), foot slip (-0.5), stamping (-1.0), elbow and wrist windows (-5, from the pedalo), beam regularisers |

Every positive term is in [0, 1], 5.7 per step at best, so surviving always
pays. Soles centred on the top line give the most control authority: the
support line can then roll toward the toes or the heel by the same amount.

## Training

One run per condition and seed, locally or on the cluster
([../cluster.md](../cluster.md)):

```bash
scripts/cluster/submit.sh ~/sendg1-rolling-cylinder G1-RollingCylinder-Blind \
  --agent.run-name blind_s0 --agent.seed 0
scripts/cluster/submit.sh ~/sendg1-rolling-cylinder G1-RollingCylinder-Touch \
  --agent.run-name touch_s0 --agent.seed 0
```

Seeds 0-4 for anything reported. Picture of the spawn pose:
`python scripts/rolling_cylinder_render.py`. One episode of a policy as a
video (EvalVideo, rendered on the workstation, never on the cluster):
`python scripts/rolling_cylinder_video.py --task G1-RollingCylinder-Touch
--checkpoint <pt>`.

## Status

Seed 0 of both conditions training on the cluster since 2026-10-20.

Known limitation: MuJoCo makes one contact per capsule-cylinder pair, so each
sole capsule touches the cylinder at one point.
