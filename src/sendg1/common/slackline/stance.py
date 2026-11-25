"""The tandem slackline stance, solved by ``scripts/slackline_stance.py``.

Generated; do not edit by hand. Per front foot: leg and waist joint angles
(rad; joints not listed stay at the default pose), and the base's offset
along the line (``__base_x__``) and height over the soles (``__base_z__``)
with the stance centred on the origin."""

STANCES: dict[str, dict[str, float]] = {
  "left": {
    "left_hip_pitch_joint": -0.53434,
    "left_hip_roll_joint": -0.15280,
    "left_hip_yaw_joint": 0.12989,
    "left_knee_joint": 0.66647,
    "left_ankle_pitch_joint": -0.14209,
    "left_ankle_roll_joint": 0.20021,
    "right_hip_pitch_joint": 0.09433,
    "right_hip_roll_joint": -0.10916,
    "right_hip_yaw_joint": -0.77957,
    "right_knee_joint": 0.60734,
    "right_ankle_pitch_joint": -0.64822,
    "right_ankle_roll_joint": 0.02066,
    "waist_yaw_joint": 0.00266,
    "waist_roll_joint": -0.05468,
    "waist_pitch_joint": 0.02619,
    "__base_x__": -0.02615,
    "__base_z__": 0.74178,
  },
  "right": {
    "left_hip_pitch_joint": 0.09440,
    "left_hip_roll_joint": 0.10911,
    "left_hip_yaw_joint": 0.77957,
    "left_knee_joint": 0.60718,
    "left_ankle_pitch_joint": -0.64814,
    "left_ankle_roll_joint": -0.02066,
    "right_hip_pitch_joint": -0.53428,
    "right_hip_roll_joint": 0.15281,
    "right_hip_yaw_joint": -0.12987,
    "right_knee_joint": 0.66636,
    "right_ankle_pitch_joint": -0.14206,
    "right_ankle_roll_joint": -0.20020,
    "waist_yaw_joint": -0.00286,
    "waist_roll_joint": 0.05732,
    "waist_pitch_joint": 0.02614,
    "__base_x__": -0.02614,
    "__base_z__": 0.74179,
  },
}
