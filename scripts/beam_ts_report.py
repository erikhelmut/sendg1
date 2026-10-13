#!/usr/bin/env python
"""Build the self-contained HTML report of the beam teacher-student study.

    python scripts/beam_ts_report.py --out results/beam_ts/report.html

Four policies: the privileged teacher, the touch (probing, felt-only) teacher,
and the tactile and blind students distilled from the touch teacher. Reads
their evaluations (``EVALS``, scripts/beam_ts_eval.py), training curves
(``CURVES``, scripts/beam_ts_curves.py), the teachers' input sensitivities
(``SENS``, scripts/beam_ts_sensitivity.py) and the clips in media/beam_ts/*.mp4
(+ .json sidecar, scripts/beam_ts_video.py; the file name starts with the
policy key).
Reward weights and observation dims are read from the live task config, so the
tables cannot drift from the code.
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RES = REPO / "results" / "beam_ts"

REWARD_DOC = {
  "track_forward_vel": "exp(-(v_cmd - v_along)^2 / 0.1^2): forward speed along the beam matches the command (0 = stand).",
  "alive": "Constant per step alive; keeps every living step net-positive so ending the episode never pays.",
  "reach_end": "One-off bonus on the step both feet and the base are on the end platform; the command then drops to 0.",
  "foot_centreline": "Feet in contact: exp(-d^2/sigma^2), d = lateral sole offset from the centreline; sigma tightens with level.",
  "foot_yaw": "Feet in contact: exp(-yaw_err^2/0.25^2) of the foot against the beam direction.",
  "cop_centred": "Feet in contact: exp(-off^2/0.015^2), off = lateral distance of the CoP from the centroid of the sole area that has support under it.",
  "com_over_beam": "exp(-d^2/sigma^2), d = lateral offset of the whole-body CoM projection from the centreline (level sigma).",
  "base_lateral_vel": "Squared base velocity across the beam.",
  "base_roll": "Squared sideways gravity component in the base frame (~roll^2) + 0.05 x roll rate^2.",
  "passing_step": "At each alternating touchdown: clip(lead / 0.5 m, -1, 1), lead = landing foot ahead of the other along the beam. A step-to shuffle scores nothing; stepping one foot past the other scores every step.",
  "foot_flat": "Feet in contact: exp(-tilt^2/0.15^2) of the sole. A flat sole loads many taxels instead of one edge.",
  "swing_foot_speed": "Feet in the air: relu(|v_foot| - 1.2 m/s)^2. Stops the fast swing-leg flick.",
  "touchdown_force_rate": "Squared one-step rise of each foot's vertical contact force / body weight. Favours a light first touch -- the physical shape of probing.",
  "arm_posture": "Arm joints away from a slightly bent default pose (squared, scaled).",
  "leg_joint_acc": "Squared leg joint accelerations.",
  "arm_joint_vel": "Squared arm joint velocities.",
  "upright": "Torso upright (walking reward term).",
  "posture": "Joint posture near the default pose, walking tolerances x1.5 (allows hip adduction / tandem stance).",
  "feet_air_time": "Swing phases of 0.05-0.4 s while a walking command is active (relaxed from the walking reward).",
  "action_rate_l2": "Squared change of the action between steps.",
  "joint_torques_l2": "Squared joint torques.",
  "dof_pos_limits": "Joint positions beyond the soft limits.",
  "body_ang_vel": "Torso angular velocity (xy).",
  "angular_momentum": "Whole-body angular momentum.",
  "foot_slip": "Foot velocity while in contact, when walking.",
  "self_collisions": "Self-contacts above 10 N.",
}

TERMINATION_DOC = {
  "nonfoot_contact": "Any robot body other than the feet touches the terrain.",
  "foot_off_beam": "A sole touches the ground or a side face of the beam/platform (contact > 3 cm below the top).",
  "base_too_low": "Pelvis less than 0.5 m above the beam top.",
  "tilt": "Torso tilt above 40 deg.",
  "nan_detection": "Physics state went NaN.",
}


def b64(path: Path) -> str:
  return base64.b64encode(path.read_bytes()).decode()


def load(res: Path, name: str):
  p = res / name
  return json.loads(p.read_text()) if p.exists() else None


def reward_rows():
  import sendg1.tasks  # noqa: F401
  from mjlab.tasks.registry import load_env_cfg

  cfg = load_env_cfg("G1-Beam-Probe-Teacher")
  return [(name, term.weight, REWARD_DOC.get(name, "")) for name, term in cfg.rewards.items()]


def videos(media: Path):
  out = []
  for mp4 in sorted(media.glob("*.mp4")):
    meta = json.loads(mp4.with_suffix(".json").read_text()) if mp4.with_suffix(".json").exists() else {}
    out.append({"name": mp4.stem, "src": "data:video/mp4;base64," + b64(mp4), **meta})
  return out


EVALS = {
  "priv": {"default": "teacher_v4_10000.json", "wide": "teacher_v4_10000_wide.json", "push": "teacher_v4_10000_push.json"},
  "touch": {"default": "probe_teacher_felt.json", "wide": "probe_teacher_felt_wide.json", "push": "probe_teacher_felt_push.json"},
  "tactile": {"default": "probe_student_tactile.json", "wide": "probe_student_tactile_wide.json", "push": "probe_student_tactile_push.json"},
  "blind": {"default": "probe_student_blind.json", "wide": "probe_student_blind_wide.json", "push": "probe_student_blind_push.json"},
}
CURVES = {
  "priv": ["teacher_v4_curves.json"],
  # Probing teacher (0-10k), its continuation (stopped at 12.5k), felt fine-tune (12.5k on).
  "touch": ["probe_teacher_curves_a.json", "probe_teacher_curves_b.json", "probe_teacher_curves_c.json"],
  "tactile": ["probe_student_tactile_curves.json"],
  "blind": ["probe_student_blind_curves.json"],
}
SENS = {"priv": "sens_teacher_v4.json", "touch": "sens_probe_teacher.json"}
CURVE_PREFIXES = ("Curriculum/beam_levels/mean", "Curriculum/beam_levels/success", "Loss/behavior", "Loss/beta")


def merged_curves(res: Path, files: list[str]) -> dict | None:
  """Concatenate back-to-back runs; a later run overrides from its first step on."""
  out: dict[str, list] = {}
  for f in files:
    c = load(res, f)
    if not c:
      continue
    for tag, pts in c["curves"].items():
      if not tag.startswith(CURVE_PREFIXES) or not pts:
        continue
      first = pts[0][0]
      out[tag] = [p for p in out.get(tag, []) if p[0] < first] + pts
  return out or None


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--res-dir", default=str(RES))
  ap.add_argument("--out", default=str(RES / "report.html"))
  ap.add_argument("--media-dir", default=str(REPO / "media" / "beam_ts"))
  ap.add_argument("--notes", default=str(RES / "video_notes.json"))
  args = ap.parse_args()

  res = Path(args.res_dir)
  evals = {}
  for pol, kinds in EVALS.items():
    evals[pol] = {}
    for kind, f in kinds.items():
      r = load(res, f)
      if r:
        r.pop("episodes", None)
      evals[pol][kind] = r
  sens = {pol: (load(res, f) or {}).get("action_change", {}) for pol, f in SENS.items()}
  # Inputs the students do not get (names as in the teachers' actor groups).
  sens["student_missing"] = [
    "actions", "base_lin_vel", "feet_beam_pose", "base_beam_pose", "beam_progress", "beam_params",
    "beam_revealed", "tactile_clean", "foot_contact_forces", "action_lag",
  ]
  data = {
    "evals": evals,
    "curves": {pol: merged_curves(res, files) for pol, files in CURVES.items()},
    "sens": sens,
    "rewards": [{"name": n, "weight": w, "doc": d} for n, w, d in reward_rows()],
    "terminations": [{"name": k, "doc": v} for k, v in TERMINATION_DOC.items()],
    "videos": videos(Path(args.media_dir)),
    "notes": json.loads(Path(args.notes).read_text()) if Path(args.notes).exists() else {},
  }

  template = (REPO / "scripts" / "beam_ts_report_template.html").read_text()
  page = template.replace("/*__DATA__*/null", json.dumps(data))
  Path(args.out).parent.mkdir(parents=True, exist_ok=True)
  Path(args.out).write_text(page)
  print(f"wrote {args.out} ({len(page) / 1e6:.1f} MB)")


if __name__ == "__main__":
  main()
