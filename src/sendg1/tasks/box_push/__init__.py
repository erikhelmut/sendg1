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

register_variants(
  "Sendg1-BoxPush-G1",
  make_g1_box_push_env_cfg,
  build_ppo_cfg,
  experiment_prefix="box_push",
)

register_variants(
  "Sendg1-BoxPushCurriculum-G1",
  partial(make_g1_box_push_env_cfg, curriculum=True),
  build_ppo_cfg,
  experiment_prefix="box_push_curriculum",
)

register_variants(
  "Sendg1-BoxPushPatches-G1",
  partial(make_g1_box_push_env_cfg, low_friction_patches=True),
  build_ppo_cfg,
  experiment_prefix="box_push_patches",
)
