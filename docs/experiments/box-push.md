# Box pushing under unknown floor friction

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
