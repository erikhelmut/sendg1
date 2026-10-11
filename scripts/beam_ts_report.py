#!/usr/bin/env python
"""Build the self-contained HTML report of the beam teacher-student study.

    python scripts/beam_ts_report.py --out results/beam_ts/report.html

Reads results/beam_ts/{teacher,student}.json (scripts/beam_ts_eval.py),
results/beam_ts/{teacher,student}_curves.json (scripts/beam_ts_curves.py) and
the clips in media/beam_ts/*.mp4 (+ .json sidecar, scripts/beam_ts_video.py).
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

  cfg = load_env_cfg("G1-Beam-Teacher")
  return [(name, term.weight, REWARD_DOC.get(name, "")) for name, term in cfg.rewards.items()]


def videos(media: Path):
  out = []
  for mp4 in sorted(media.glob("*.mp4")):
    meta = json.loads(mp4.with_suffix(".json").read_text()) if mp4.with_suffix(".json").exists() else {}
    out.append({"name": mp4.stem, "src": "data:video/mp4;base64," + b64(mp4), **meta})
  return out


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--res-dir", default=str(RES))
  ap.add_argument("--out", default=str(RES / "report.html"))
  ap.add_argument("--media-dir", default=str(REPO / "media" / "beam_ts"))
  ap.add_argument("--notes", default=str(RES / "video_notes.json"))
  args = ap.parse_args()

  res = Path(args.res_dir)
  rows = reward_rows()
  data = {
    "teacher": load(res, "teacher.json"),
    "student": load(res, "student.json"),
    "teacher_push": load(res, "teacher_push.json"),
    "student_push": load(res, "student_push.json"),
    "teacher_curves": load(res, "teacher_curves.json"),
    "student_curves": load(res, "student_curves.json"),
    "rewards": [{"name": n, "weight": w, "doc": d} for n, w, d in rows],
    "terminations": [{"name": k, "doc": v} for k, v in TERMINATION_DOC.items()],
    "videos": videos(Path(args.media_dir)),
    "notes": json.loads(Path(args.notes).read_text()) if Path(args.notes).exists() else {},
  }
  for k in ("teacher", "student", "teacher_push", "student_push"):
    if data[k]:
      data[k].pop("episodes", None)
  for k in ("teacher_curves", "student_curves"):
    c = data[k]
    if c:
      keep = {}
      for tag, pts in c["curves"].items():
        if tag.startswith(("Curriculum/beam_levels/", "Episode_Termination/", "Loss/", "Train/mean_reward", "Train/mean_episode_length")):
          keep[tag] = pts
      c["curves"] = keep

  template = (REPO / "scripts" / "beam_ts_report_template.html").read_text()
  page = template.replace("/*__DATA__*/null", json.dumps(data))
  Path(args.out).parent.mkdir(parents=True, exist_ok=True)
  Path(args.out).write_text(page)
  print(f"wrote {args.out} ({len(page) / 1e6:.1f} MB)")


if __name__ == "__main__":
  main()
