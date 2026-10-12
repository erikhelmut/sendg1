"""Straight-beam walking, teacher-student: ``G1-Beam-Teacher``, ``G1-Beam-Student``.

The teacher is trained with mjlab's ``train``; the student with
``scripts/beam_ts_distill.py``, which loads a teacher checkpoint first.

Probing variant: ``G1-Beam-Probe-Teacher`` sees the beam geometry only once a
foot has felt a beam edge (``mdp.beam_revealed``), so it has to find the beam
by touch like the student. It is distilled into ``G1-Beam-Probe-Student``
(tactile) and ``G1-Beam-Probe-Student-Blind`` (no tactile).
"""

from __future__ import annotations

from mjlab.tasks.registry import register_mjlab_task

from sendg1.common.beam_ts.terms import STUDENT_LEGAL, TERMS
from sendg1.rl.dagger import DistillationRunner
from sendg1.tasks.beam_ts.env_cfg import make_beam_ts_env_cfg
from sendg1.tasks.beam_ts.rl_cfg import (
  BLIND_STUDENT_OBS,
  PROBE_TEACHER_OBS,
  student_distill_cfg,
  teacher_ppo_cfg,
)

TEACHER_TASK = "G1-Beam-Teacher"
STUDENT_TASK = "G1-Beam-Student"
PROBE_TEACHER_TASK = "G1-Beam-Probe-Teacher"
PROBE_STUDENT_TASK = "G1-Beam-Probe-Student"
PROBE_BLIND_STUDENT_TASK = "G1-Beam-Probe-Student-Blind"


def assert_student_is_deployable(observations: dict, groups: tuple[str, ...]) -> None:
  """Refuse to register a student that can see privileged state."""
  for g in groups:
    for term in observations[g].terms:
      if term not in TERMS:
        raise KeyError(f"student group {g!r} has undeclared term {term!r}")
      if term not in STUDENT_LEGAL:
        raise ValueError(
          f"student group {g!r} contains {TERMS[term].availability} term {term!r}: "
          f"{TERMS[term].rationale}"
        )


_env = make_beam_ts_env_cfg()
_play = make_beam_ts_env_cfg(play=True)
_student = student_distill_cfg()
assert_student_is_deployable(_env.observations, _student.obs_groups["student"])

register_mjlab_task(
  task_id=TEACHER_TASK, env_cfg=_env, play_env_cfg=_play, rl_cfg=teacher_ppo_cfg()
)
register_mjlab_task(
  task_id=STUDENT_TASK,
  env_cfg=make_beam_ts_env_cfg(),
  play_env_cfg=make_beam_ts_env_cfg(play=True),
  rl_cfg=_student,
  runner_cls=DistillationRunner,
)

_probe_students = {
  PROBE_STUDENT_TASK: student_distill_cfg(
    teacher_obs=PROBE_TEACHER_OBS, experiment_name="beam_ts_probe_student"
  ),
  PROBE_BLIND_STUDENT_TASK: student_distill_cfg(
    student_obs=BLIND_STUDENT_OBS,
    teacher_obs=PROBE_TEACHER_OBS,
    experiment_name="beam_ts_probe_student_blind",
  ),
}
_probe_env = make_beam_ts_env_cfg(probe=True)
for _cfg in _probe_students.values():
  assert_student_is_deployable(_probe_env.observations, _cfg.obs_groups["student"])

register_mjlab_task(
  task_id=PROBE_TEACHER_TASK,
  env_cfg=_probe_env,
  play_env_cfg=make_beam_ts_env_cfg(play=True, probe=True),
  rl_cfg=teacher_ppo_cfg(actor_obs=PROBE_TEACHER_OBS, experiment_name="beam_ts_probe_teacher"),
)
for _task, _cfg in _probe_students.items():
  register_mjlab_task(
    task_id=_task,
    env_cfg=make_beam_ts_env_cfg(probe=True),
    play_env_cfg=make_beam_ts_env_cfg(play=True, probe=True),
    rl_cfg=_cfg,
    runner_cls=DistillationRunner,
  )
