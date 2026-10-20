"""The tandem slackline stance, solved by ``scripts/slackline_stance.py``.

Generated; do not edit by hand. Per front foot: leg and waist joint angles
(rad; joints not listed stay at the default pose), and the base's offset
along the line (``__base_x__``) and height over the soles (``__base_z__``)
with the stance centred on the origin."""

STANCES: dict[str, dict[str, float]] = {
  "left": {
    "left_hip_pitch_joint": -0.51504,
    "left_hip_roll_joint": -0.15443,
    "left_hip_yaw_joint": 0.12626,
    "left_knee_joint": 0.63204,
    "left_ankle_pitch_joint": -0.12680,
    "left_ankle_roll_joint": 0.19915,
    "right_hip_pitch_joint": 0.10935,
    "right_hip_roll_joint": -0.09527,
    "right_hip_yaw_joint": -0.78102,
    "right_knee_joint": 0.56269,
    "right_ankle_pitch_joint": -0.62394,
    "right_ankle_roll_joint": 0.02130,
    "waist_yaw_joint": 0.00237,
    "waist_roll_joint": -0.04869,
    "waist_pitch_joint": 0.02696,
    "__base_x__": -0.02510,
    "__base_z__": 0.73764,
  },
  "right": {
    "left_hip_pitch_joint": 0.10937,
    "left_hip_roll_joint": 0.09527,
    "left_hip_yaw_joint": 0.78102,
    "left_knee_joint": 0.56267,
    "left_ankle_pitch_joint": -0.62394,
    "left_ankle_roll_joint": -0.02131,
    "right_hip_pitch_joint": -0.51503,
    "right_hip_roll_joint": 0.15443,
    "right_hip_yaw_joint": -0.12626,
    "right_knee_joint": 0.63204,
    "right_ankle_pitch_joint": -0.12681,
    "right_ankle_roll_joint": -0.19915,
    "waist_yaw_joint": -0.00257,
    "waist_roll_joint": 0.05134,
    "waist_pitch_joint": 0.02691,
    "__base_x__": -0.02509,
    "__base_z__": 0.73764,
  },
}
