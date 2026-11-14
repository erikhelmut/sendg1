"""Compare the line models (``sm.SURFACES`` x ``sm.CONTACTS``) with the G1 on
them: how deep the soles sink, which way the contacts push, what the tactile
slipper reads. No policy: each spawn holds the solved stance as joint targets
for the first ``--seconds`` (it topples soon after, on any model).

  uv run python scripts/slackline_surfaces.py --stats      # numbers, 256 spawns each
  uv run python scripts/slackline_surfaces.py --media      # stills + one clip

Stills and the clip go through ``EvalVideo`` (tactile card live).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch

VARIANTS = (
  # (surface, contact, label)
  ("box", "soft", "boxes, default contact (current)"),
  ("box", "stiff", "boxes, stiff contact"),
  ("ribbon", "stiff", "flex ribbon"),
  ("cable", "stiff", "flex cable (round rope)"),
)
VIEWS = {
  # name: (tracked body, distance, azimuth, elevation); "{front}" is the front foot
  "feet": ("{front}_ankle_roll_link", 1.0, 140.0, -30.0),
  "side": ("{front}_ankle_roll_link", 0.55, 90.0, -4.0),
  "wide": ("torso_link", 2.6, 130.0, -10.0),
}
BODY_WEIGHT_N = 343.0
SLOW = 4


def make_env(
  surface: str,
  contact: str,
  num_envs: int,
  render: tuple[int, int] | None = None,
  view: tuple[str, float, float, float] | None = None,
):
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.termination_manager import TerminationTermCfg
  from mjlab.tasks.velocity import mdp as vel_mdp

  import sendg1.tasks  # noqa: F401
  from sendg1.common.slackline import mdp
  from sendg1.tasks.slackline.env_cfg import make_slackline_env_cfg
  from sendg1.video import prepare_video_cfg

  cfg = make_slackline_env_cfg(
    play=True, tactile=True, surface=surface, contact=contact, num_envs=num_envs
  )
  cfg.events["line_level"].params = {"fixed_level": mdp.TOP_LEVEL}
  # Only the clock ends an episode: the point is the contact, not the fall.
  cfg.terminations = {"time_out": TerminationTermCfg(func=vel_mdp.time_out, time_out=True)}
  if render is not None:
    cfg.events["reset_on_line"].params = {
      "joint_noise": 0.0,
      "heading_noise": 0.0,
      "along_range": (-0.3, -0.3),
    }
    prepare_video_cfg(cfg, *render)
    if view is not None:
      cfg.viewer.body_name = view[0]
      cfg.viewer.distance, cfg.viewer.azimuth, cfg.viewer.elevation = view[1:]
  return ManagerBasedRlEnv(
    cfg=cfg, device="cuda:0", render_mode="rgb_array" if render else None
  )


class Probe:
  """Per-world contact statistics of soles on the line, from the raw
  contact list (works for boxes and flexes alike)."""

  def __init__(self, env) -> None:
    m = env.sim.mj_model
    dev = env.device
    names = [m.geom(g).name for g in range(m.ngeom)]
    self.foot = torch.tensor(
      [n.startswith("robot/") and n.endswith("_sole_collision") for n in names], device=dev
    )
    self.line = torch.tensor(
      [n.startswith("slackline/webbing_") and bool(m.geom_contype[g]) for g, n in enumerate(names)],
      device=dev,
    )
    self.env = env

  def __call__(self) -> dict[str, torch.Tensor]:
    from sendg1.common.sensors import taxel_ray_sensor_name
    from sendg1.common.tactile.layout import SIDES
    from sendg1.common.tactile.sensing import TactileParams, cached_reading

    env = self.env
    c = env.sim.data.contact
    n = int(env.sim.data.nacon[0])
    geom, dist, frame = c.geom[:n], c.dist[:n], c.frame[:n]
    world = c.worldid[:n].long()
    valid = geom >= 0
    is_foot = (self.foot[geom.clamp_min(0)] & valid).any(-1)
    on_line = (self.line[geom.clamp_min(0)] & valid).any(-1)
    if c.flex.numel():
      on_line |= (c.flex[:n] >= 0).any(-1)
    sel = is_foot & on_line
    b = env.num_envs
    count = torch.zeros(b, device=env.device).index_add_(0, world[sel], torch.ones_like(dist[sel]))
    depth = torch.zeros(b, device=env.device).index_add_(0, world[sel], -dist[sel])
    up = torch.zeros(b, device=env.device).index_add_(
      0, world[sel], (frame[sel][:, 0, 2].abs() > 0.9).float()
    )
    deepest = torch.zeros(b, device=env.device).index_reduce_(
      0, world[sel], -dist[sel], "amax", include_self=False
    )
    r = cached_reading(env, TactileParams())
    hits = torch.cat(
      [env.scene[taxel_ray_sensor_name(s)].data.distances for s in SIDES], -1
    )
    return {
      "contacts": count,
      "depth_mm": 1000 * depth / count.clamp_min(1),
      "deepest_mm": 1000 * deepest,
      "up": up / count.clamp_min(1),
      "load_bw": r.net_force[..., 2].sum(-1) / BODY_WEIGHT_N,
      "taxel_bw": r.normal.sum((-1, -2)) / BODY_WEIGHT_N,
      "support": r.support.mean((-1, -2)),
      "ray_hits": (hits >= 0).float().mean(-1),
    }


def hold_action(env):
  from sendg1.common.slackline import mdp

  st = mdp.LineState.get(env)
  stances = mdp._stance_tensors(env)
  robot = env.scene["robot"]
  scale = env.action_manager.get_term("joint_pos")._scale
  q = torch.where(st.front_left[:, None], stances["left"], stances["right"])
  return (q - robot.data.default_joint_pos) / scale


def stats(num_envs: int, seconds: float) -> dict:
  out = {}
  for surface, contact, label in VARIANTS:
    env = make_env(surface, contact, num_envs)
    try:
      probe = Probe(env)
      env.reset()
      rows = []
      with torch.inference_mode():
        act = hold_action(env)
        for k in range(round(seconds / env.step_dt)):
          env.step(act)
          if k >= 5:  # loaded, past the drop
            rows.append(probe())
      mean = {
        key: float(torch.stack([r[key] for r in rows]).nanmean())
        for key in rows[0]
      }
      out[f"{surface}:{contact}"] = {"label": label, **mean}
      print(
        f"{label:34s} sinks {mean['depth_mm']:4.1f} mm (deepest {mean['deepest_mm']:4.1f})"
        f" | up {100 * mean['up']:3.0f}% | {mean['contacts']:4.1f} contacts"
        f" | sole load {mean['load_bw']:+.2f} BW, taxels {mean['taxel_bw']:.2f} BW"
        f" | support {mean['support']:.2f}, rays hit {100 * mean['ray_hits']:3.0f}%"
      )
    finally:
      env.close()
  return out


def media(
  out: Path,
  seconds: float,
  width: int,
  height: int,
  variants: tuple = VARIANTS,
  spawns: int = 1,
  front: str = "left",
) -> None:
  """Stills of every variant and view; one clip with every view of every
  variant, ``spawns`` episodes each, in ``SLOW`` x slow motion."""
  import mediapy
  from mjlab.tasks.registry import load_rl_cfg
  from PIL import Image

  from sendg1.common.slackline import mdp
  from sendg1.tasks.slackline import TOUCH_TASK
  from sendg1.video import EvalVideo

  clip: list[np.ndarray] = []
  stills: dict[str, list[np.ndarray]] = {v: [] for v in VIEWS}
  for surface, contact, label in variants:
    for view, (body, dist, az, el) in VIEWS.items():
      env = make_env(
        surface, contact, 1, render=(width, height), view=(body.format(front=front), dist, az, el)
      )
      try:
        st = mdp.LineState.get(env)
        probe = Probe(env)
        video = EvalVideo(env, load_rl_cfg(TOUCH_TASK), title=label)
        for spawn in range(spawns):
          # Reset in inference mode too: buffers first written in a step are
          # inference tensors.
          with torch.inference_mode():
            env.reset()
            for _ in range(50 if spawn == 0 else 0):  # the asked-for front foot, facing hi
              if bool(st.front_left[0]) == (front == "left") and float(st.direction[0]) > 0:
                break
              env.reset()
          video.new_episode()
          with torch.inference_mode():
            act = hold_action(env)
            for k in range(round(seconds / env.step_dt)):
              env.step(act)
              p = {key: float(v[0]) for key, v in probe().items()}
              frame = video.capture(
                [
                  (
                    f"no policy: stance held as joint targets, spawn {spawn + 1}/{spawns}, "
                    f"t = {k * env.step_dt:4.2f} s, {SLOW}x slow motion"
                  ),
                  (
                    f"soles sink {p['depth_mm']:4.1f} mm into the line (deepest "
                    f"{p['deepest_mm']:4.1f} mm); {100 * p['up']:3.0f}% of contacts push up"
                  ),
                  f"sole load {p['load_bw']:+.2f} body weight; taxels read {p['taxel_bw']:.2f}",
                ]
              )
              assert frame is not None
              if k == 4 and spawn == 0:  # 0.1 s: loaded, still upright
                stills[view].append(frame)
              clip += [frame] * SLOW
      finally:
        env.close()
  out.mkdir(parents=True, exist_ok=True)
  for view, frames in stills.items():
    for (surface, contact, _), f in zip(variants, frames, strict=True):
      Image.fromarray(f).save(out / f"surface_{surface}_{contact}_{view}.png")
    if len(frames) == 4:
      top = np.concatenate(frames[:2], 1)
      bottom = np.concatenate(frames[2:], 1)
      grid = Image.fromarray(np.concatenate([top, bottom], 0))
      grid.resize((grid.width // 2, grid.height // 2)).save(out / f"surfaces_{view}.png")
  name = (
    "surfaces" if len(variants) > 1 else f"surface_{variants[0][0]}_{variants[0][1]}"
  )
  mediapy.write_video(str(out / f"{name}.mp4"), clip, fps=50, qp=26)
  print(f"wrote {out}/{name}.mp4 ({len(clip) / 50:.1f} s) and stills")


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--stats", action="store_true")
  ap.add_argument("--media", action="store_true")
  ap.add_argument("--num-envs", type=int, default=256)
  ap.add_argument("--seconds", type=float, default=0.3)
  ap.add_argument("--clip-seconds", type=float, default=0.8)
  ap.add_argument("--out", type=Path, default=Path("results/slackline/surfaces"))
  ap.add_argument(
    "--variants", nargs="+", default=None, help="surface:contact, e.g. box:stiff (default: all)"
  )
  ap.add_argument("--spawns", type=int, default=1, help="episodes per view in the clip")
  ap.add_argument("--width", type=int, default=1280)
  ap.add_argument("--height", type=int, default=720)
  args = ap.parse_args()
  if args.stats:
    result = stats(args.num_envs, args.seconds)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "stats.json").write_text(json.dumps(result, indent=2))
  if args.media:
    variants = VARIANTS
    if args.variants:
      variants = tuple(v for v in VARIANTS if f"{v[0]}:{v[1]}" in args.variants)
    media(args.out, args.clip_seconds, args.width, args.height, variants, args.spawns)


if __name__ == "__main__":
  main()
