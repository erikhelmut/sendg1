"""Heavy-box pushing under unknown floor friction, Unitree G1.

Three task families, each fanned across every observation set:

  Sendg1-BoxPush-G1-*             the task as specified: full ranges from the
                                  first episode, plain floor
  Sendg1-BoxPushCurriculum-G1-*   light boxes / high friction first, widened
                                  toward the full ranges by success rate
  Sendg1-BoxPushPatches-G1-*      plus 1-3 low-friction floor patches on the path

The flags are factory arguments rather than CLI overrides, so each combination
is its own registered (and fingerprinted) task.
"""

from functools import partial

from sendg1.tasks._register import register_variants
from sendg1.tasks.box_push.env_cfg import make_g1_box_push_env_cfg
from sendg1.tasks.velocity.rl_cfg import build_ppo_cfg

BOX_PUSH_ENTROPY_COEF = 0.005
"""Half the velocity tasks' 0.01, as on the beam task and for the same reason:
at 0.01 the action std drifts up (1.0 -> 1.19 by iteration 4,000 in an
overnight run), the policy stops being able to stay upright, and training
collapses into falling immediately -- 100% of episodes ended as falls before
the robot ever touched the box."""

_ppo = partial(build_ppo_cfg, entropy_coef=BOX_PUSH_ENTROPY_COEF)

register_variants(
  "Sendg1-BoxPush-G1",
  make_g1_box_push_env_cfg,
  _ppo,
  experiment_prefix="box_push",
)

register_variants(
  "Sendg1-BoxPushCurriculum-G1",
  partial(make_g1_box_push_env_cfg, curriculum=True),
  _ppo,
  experiment_prefix="box_push_curriculum",
)

register_variants(
  "Sendg1-BoxPushPatches-G1",
  partial(make_g1_box_push_env_cfg, low_friction_patches=True),
  _ppo,
  experiment_prefix="box_push_patches",
)
