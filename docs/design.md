# Design rules and decisions

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

## Open

- The sensor's rate, latency, noise and units. The layout is in (`slipper_v1`).
- Seeds: 1 for now (directional only). 5 before any reported claim.
- `num_envs`: per-task, tune with `scripts/tune_num_envs.py`. On a 12 GB 5070,
  throughput saturates around 8192 (~220k steps/s); larger fits but buys nothing.
  Safe to vary across tasks: the command curriculum advances on a counter that
  ticks once per `env.step()` regardless of batch size, so its stages land at
  fixed iteration counts either way.
