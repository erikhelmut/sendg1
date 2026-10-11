"""Observation terms of the beam teacher-student task, each with its availability.

Same contract as ``sendg1.common.obs.terms``: every term declares whether the
real robot can measure it, and ``tasks/beam_ts`` refuses to register a student
that observes anything else (``assert_student_is_deployable``).

``onboard``   on the G1 itself. Checked against the unitree_sdk2 low-level state
              (``unitree_hg::msg::dds_::LowState_``): ``imu_state`` carries the
              quaternion, gyroscope, accelerometer and rpy; ``motor_state[]``
              carries q, dq, ddq and tau_est per joint. There is no base linear
              velocity, no foot contact and no terrain information in it.
``sensor``    the tactile slipper: hardware the robot will carry, but not part of
              the stock G1. Modelled with gain error, dead taxels, noise and
              frame drops. Legal in the student.
``privileged`` simulation ground truth. Teacher actor and critics only.

``real`` group (student, teacher, critic): base_ang_vel, projected_gravity,
joint_pos, joint_vel, actions, command_vx, tactile.
``privileged`` group (teacher, critic): everything else.
"""

from __future__ import annotations

from typing import Callable, Literal

from mjlab.envs import mdp as envs_mdp
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.tasks.velocity import mdp as vel_mdp
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.utils.noise.noise_cfg import NoiseModelWithAdditiveBiasCfg

from sendg1.common.beam_ts import mdp
from sendg1.common.obs.terms import TermSpec

Availability = Literal["onboard", "sensor", "privileged"]

IMU_GYRO_BIAS = 0.05
"""rad/s, uniform +/-, per episode."""
IMU_GRAVITY_BIAS = 0.03
"""Per component of the unit gravity vector, per episode (~1.7 deg of tilt)."""


def _ang_vel() -> ObservationTermCfg:
  return ObservationTermCfg(
    func=envs_mdp.builtin_sensor,
    params={"sensor_name": "robot/imu_ang_vel"},
    noise=NoiseModelWithAdditiveBiasCfg(
      noise_cfg=Unoise(n_min=-0.2, n_max=0.2),
      bias_noise_cfg=Unoise(n_min=-IMU_GYRO_BIAS, n_max=IMU_GYRO_BIAS),
    ),
  )


def _gravity() -> ObservationTermCfg:
  return ObservationTermCfg(
    func=envs_mdp.projected_gravity_from_sensor,
    params={"sensor_name": "robot/imu_upvector"},
    noise=NoiseModelWithAdditiveBiasCfg(
      noise_cfg=Unoise(n_min=-0.05, n_max=0.05),
      bias_noise_cfg=Unoise(n_min=-IMU_GRAVITY_BIAS, n_max=IMU_GRAVITY_BIAS),
    ),
  )


def _joint_pos() -> ObservationTermCfg:
  return ObservationTermCfg(
    func=envs_mdp.joint_pos_rel, params={"biased": True}, noise=Unoise(n_min=-0.01, n_max=0.01)
  )


def _joint_vel() -> ObservationTermCfg:
  return ObservationTermCfg(func=envs_mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))


def _plain(func, noise=None) -> Callable[[], ObservationTermCfg]:
  return lambda: ObservationTermCfg(func=func, noise=noise)


def _tactile_noisy() -> ObservationTermCfg:
  # Sensor faults live inside the term (they are per-taxel and stateful); one
  # step of latency on top, as for a sensor read over a bus.
  return ObservationTermCfg(func=mdp.tactile_features_noisy, delay_min_lag=0, delay_max_lag=1)


TERMS: dict[str, TermSpec] = {
  # -- real ------------------------------------------------------------------
  "base_ang_vel": TermSpec(
    _ang_vel, "onboard", "LowState_.imu_state.gyroscope; noise and a per-episode bias."
  ),
  "projected_gravity": TermSpec(
    _gravity,
    "onboard",
    "From LowState_.imu_state.quaternion; read off the IMU site so IMU pose DR "
    "applies; noise and a per-episode bias.",
  ),
  "joint_pos": TermSpec(
    _joint_pos, "onboard", "LowState_.motor_state[].q, with encoder bias DR."
  ),
  "joint_vel": TermSpec(_joint_vel, "onboard", "LowState_.motor_state[].dq."),
  "actions": TermSpec(_plain(envs_mdp.last_action), "onboard", "The policy's previous output."),
  "command_vx": TermSpec(
    _plain(mdp.command_vx), "onboard", "Operator forward-speed command (joystick)."
  ),
  "tactile": TermSpec(
    _tactile_noisy,
    "sensor",
    "Tactile slipper, per-foot features (sendg1.common.beam_ts.tactile) from the "
    "faulted taxel readings, 0-1 step latency.",
  ),
  # -- privileged --------------------------------------------------------------
  "base_lin_vel": TermSpec(
    lambda: ObservationTermCfg(
      func=envs_mdp.builtin_sensor, params={"sensor_name": "robot/imu_lin_vel"}
    ),
    "privileged",
    "True body-frame linear velocity. Not in LowState_; needs an estimator.",
  ),
  "feet_beam_pose": TermSpec(
    _plain(mdp.feet_beam_pose),
    "privileged",
    "Each foot's lateral offset, edge margin, height and yaw relative to the beam.",
  ),
  "base_beam_pose": TermSpec(
    _plain(mdp.base_beam_pose),
    "privileged",
    "Base offset, heading and height relative to the beam; CoM offset.",
  ),
  "beam_progress": TermSpec(
    _plain(mdp.beam_progress_obs), "privileged", "Distance to the beam start/end."
  ),
  "beam_params": TermSpec(
    _plain(mdp.beam_params_obs), "privileged", "Beam width and height, contact friction."
  ),
  "tactile_clean": TermSpec(
    _plain(mdp.tactile_features_clean), "privileged", "Noise-free tactile features."
  ),
  "foot_contact_forces": TermSpec(
    lambda: ObservationTermCfg(
      func=vel_mdp.foot_contact_forces, params={"sensor_name": mdp.FOOT_CONTACT_SENSOR}
    ),
    "privileged",
    "Net contact force per foot from the simulator.",
  ),
  "action_lag": TermSpec(
    _plain(mdp.action_lag_obs), "privileged", "This episode's actuation latency."
  ),
}

REAL_TERMS: tuple[str, ...] = (
  "base_ang_vel",
  "projected_gravity",
  "joint_pos",
  "joint_vel",
  "actions",
  "command_vx",
  "tactile",
)
PRIVILEGED_TERMS: tuple[str, ...] = (
  "base_lin_vel",
  "feet_beam_pose",
  "base_beam_pose",
  "beam_progress",
  "beam_params",
  "tactile_clean",
  "foot_contact_forces",
  "action_lag",
)
STUDENT_LEGAL: frozenset[str] = frozenset(
  n for n, s in TERMS.items() if s.availability in ("onboard", "sensor")
)
