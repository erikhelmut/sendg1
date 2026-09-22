#!/usr/bin/env python
"""Watch the simulated tactile soles while the robot walks.

    python scripts/tactile_viewer.py                      # blind flat policy, seed 0
    python scripts/tactile_viewer.py --no-policy          # zero actions: stand/sag
    python scripts/tactile_viewer.py --task Sendg1-Velocity-Flat-G1-TactileArray

Writes an MP4 with the robot on the left and both soles on the right: taxel
colour is normal force, arrows are shear, the dot is the centre of pressure,
and a taxel's outline is dark when there is ground beneath it. Below, total
normal force per foot over the last few seconds. The per-step readings are
saved next to the video as .npz.

Any velocity task works as the driver: a task whose scene lacks the tactile
sensors (e.g. the Blind variants) gets them added -- they only observe, so the
policy's behaviour is unchanged.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent

# (seconds, vx, vy, wz, label)
SCHEDULE = [
  (2.0, 0.0, 0.0, 0.0, "stand"),
  (3.0, 0.6, 0.0, 0.0, "forward"),
  (2.5, 0.6, 0.0, 0.6, "forward + turn left"),
  (2.5, 0.6, 0.0, -0.6, "forward + turn right"),
  (2.5, 0.0, 0.4, 0.0, "strafe left"),
  (2.0, -0.4, 0.0, 0.0, "reverse"),
  (1.5, 0.0, 0.0, 0.0, "stand"),
]


NET_SHEAR_M_PER_N = 0.001
"""Whole-foot shear arrow length: 1 mm per newton (1 cm = 10 N)."""


class SolePanel:
  """The right-hand half of every frame, built once and updated in place."""

  def __init__(self, layout, width_px: int, height_px: int, vmax: float,
               history_s: float, dt: float):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.patheffects as pe
    import matplotlib.pyplot as plt
    from matplotlib.collections import PatchCollection
    from matplotlib.colors import Normalize

    from sendg1.common.tactile import viz

    self.viz = viz
    self.layout = layout
    self.fig = plt.figure(figsize=(width_px / 100, height_px / 100), dpi=100,
                          facecolor=viz.SURFACE)
    gs = self.fig.add_gridspec(2, 2, height_ratios=[1.25, 1], hspace=0.45,
                               wspace=0.08, left=0.07, right=0.97, top=0.9, bottom=0.11)
    cmap = viz.force_cmap()
    cmap.set_under(viz.UNLOADED)
    self.norm = Normalize(vmin=0.05, vmax=vmax)
    self.cmap = cmap
    self.patches, self.quivers, self.cops, self.titles = [], [], [], []
    self.net_quivers = []
    for col, side in enumerate(("left", "right")):
      ax = self.fig.add_subplot(gs[0, col])
      viz.draw_mesh_plan(ax, side)
      pc = PatchCollection(viz.taxel_patches(layout, side), cmap=cmap, norm=self.norm,
                           lw=1.0, zorder=3)
      pc.set_array(np.zeros(layout.num_taxels))
      ax.add_collection(pc)
      c = np.array(layout.for_side(side))
      q = ax.quiver(c[:, 0], c[:, 1], np.zeros(len(c)), np.zeros(len(c)),
                    angles="xy", scale_units="xy", scale=1.0 / 0.0005,
                    color=viz.INK, width=0.006, zorder=5)
      (cop,) = ax.plot([], [], "o", ms=7, mfc=viz.CAPSULE, mec=viz.SURFACE, mew=1.5,
                       zorder=6)
      # Whole-foot shear, drawn from the CoP: per-taxel arrows are ~1/26 of it.
      net_q = ax.quiver([0.0], [0.0], [0.0], [0.0], angles="xy", scale_units="xy",
                        scale=1.0 / NET_SHEAR_M_PER_N, color=viz.CAPSULE, width=0.012,
                        zorder=8, edgecolor=viz.SURFACE, linewidth=0.8)
      self.net_quivers.append(net_q)
      # Taxel ids, readable on any fill: the same number must sit at the same
      # place on both feet (one sensor part, not mirrored).
      for i, ((x, y), (hx, hy)) in enumerate(zip(layout.for_side(side), layout.half_sizes)):
        ax.text(x - hx + 0.0006, y + hy - 0.0006, str(i), ha="left", va="top",
                fontsize=5.5, color=viz.INK, zorder=7,
                path_effects=[pe.withStroke(linewidth=1.6, foreground=viz.SURFACE)])
      viz.style_plan_axes(ax)
      ax.set_xlabel("")
      ax.set_ylabel("")
      ax.set_xticks([])
      ax.set_yticks([])
      # Orientation: view from above, toe to the right, robot's left up.
      outer_up = side == "left"
      ax.text(0.99, 0.03, "toe →", transform=ax.transAxes, ha="right", va="bottom",
              fontsize=7, color=viz.INK_2)
      ax.text(0.01, 0.03, "← heel", transform=ax.transAxes, ha="left", va="bottom",
              fontsize=7, color=viz.INK_2)
      ax.text(0.5, 0.97, ("outer" if outer_up else "inner") + " edge  ·  robot's left ↑",
              transform=ax.transAxes, ha="center", va="top", fontsize=7, color=viz.INK_2)
      ax.text(0.5, 0.03, ("inner" if outer_up else "outer") + " edge",
              transform=ax.transAxes, ha="center", va="bottom", fontsize=7, color=viz.INK_2)
      self.titles.append(ax.set_title(side, loc="left", fontsize=10, color=viz.INK))
      self.patches.append(pc)
      self.quivers.append(q)
      self.cops.append(cop)
    cb = self.fig.colorbar(self.patches[1], ax=self.fig.axes[:2], orientation="horizontal",
                           fraction=0.05, pad=0.05, aspect=40, extend="min")
    cb.set_label("normal force per taxel (N)  ·  dot: centre of pressure  ·  "
                 f"orange arrow: foot's total shear, 1 cm = {0.01 / NET_SHEAR_M_PER_N:.0f} N",
                 fontsize=8, color=viz.INK_2)
    cb.ax.tick_params(labelsize=7, colors=viz.INK_2)
    cb.outline.set_edgecolor(viz.MESH)

    self.n_hist = int(history_s / dt)
    self.t = np.arange(-self.n_hist + 1, 1) * dt
    self.hist = np.full((2, self.n_hist), np.nan)
    ax = self.fig.add_subplot(gs[1, :])
    self.lines = [
      ax.plot(self.t, self.hist[i], color=color, lw=2)[0]
      for i, color in enumerate(("#2a78d6", "#eb6834"))
    ]
    ax.legend(self.lines, ["left foot", "right foot"], loc="lower left", fontsize=8,
              frameon=False, labelcolor=viz.INK_2, ncol=2, bbox_to_anchor=(0, 1.0))
    ax.set_xlim(self.t[0], 0)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("time (s)", fontsize=8, color=viz.INK_2)
    ax.set_ylabel("total normal force (N)", fontsize=8, color=viz.INK_2)
    ax.grid(axis="y", color=viz.MESH, lw=0.6)
    ax.set_facecolor(viz.SURFACE)
    ax.tick_params(labelsize=7, colors=viz.INK_2)
    for s in ax.spines.values():
      s.set_visible(False)
    self.ts_ax = ax
    self.ymax = 1.0

  def update(self, normal, shear, support, cop, fn) -> np.ndarray:
    """All inputs are numpy for one env: normal [2,T], shear [2,T,2],
    support [2,T], cop [2,2], fn [2]."""
    viz = self.viz
    for f in range(2):
      self.patches[f].set_array(normal[f])
      edge = np.where(support[f][:, None] > 0.5, (0.04, 0.04, 0.04, 1.0),
                      (0.75, 0.75, 0.73, 1.0))
      self.patches[f].set_edgecolor(edge)
      self.quivers[f].set_UVC(shear[f, :, 0], shear[f, :, 1])
      if fn[f] > 1.0:
        self.cops[f].set_data([cop[f, 0]], [cop[f, 1]])
        total = shear[f].sum(0)
        self.net_quivers[f].set_offsets([cop[f]])
        self.net_quivers[f].set_UVC([total[0]], [total[1]])
      else:
        self.cops[f].set_data([], [])
        self.net_quivers[f].set_UVC([0.0], [0.0])
      side = ("left", "right")[f]
      self.titles[f].set_text(f"{side} foot  ·  {fn[f]:5.0f} N")
    self.hist = np.roll(self.hist, -1, axis=1)
    self.hist[:, -1] = fn
    for f in range(2):
      self.lines[f].set_ydata(self.hist[f])
    peak = np.nanmax(self.hist) if np.isfinite(self.hist).any() else 1.0
    if peak > self.ymax:
      self.ymax = 1.25 * peak
      self.ts_ax.set_ylim(0, self.ymax)
    self.fig.canvas.draw()
    rgba = np.asarray(self.fig.canvas.buffer_rgba())
    return rgba[..., :3].copy()


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--task", default="Sendg1-Velocity-Flat-G1-Blind")
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--checkpoint", default=None)
  ap.add_argument("--no-policy", action="store_true",
                  help="zero actions: the robot holds its default pose and sags")
  ap.add_argument("--log-root", default=str(REPO / "runs"))
  ap.add_argument("--out", default=None)
  ap.add_argument("--height", type=int, default=560)
  ap.add_argument("--vmax", type=float, default=60.0, help="colour scale top (N)")
  ap.add_argument("--history", type=float, default=3.0, help="seconds in the trace")
  ap.add_argument("--fps", type=int, default=50)
  args = ap.parse_args()

  import mediapy
  import torch
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
  from mjlab.utils.os import get_checkpoint_path
  from mjlab.utils.torch import configure_torch_backends

  import sendg1.tasks  # noqa: F401
  from sendg1.common.sensors import SOLE_CONTACT_SENSOR
  from sendg1.common.tactile.layout import LAYOUTS
  from sendg1.common.tactile.sensing import (
    TactileParams,
    read_tactile,
    tactile_scene_sensors,
  )

  configure_torch_backends()
  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  params = TactileParams()

  env_cfg = load_env_cfg(args.task, play=True)
  env_cfg.scene.num_envs = 1
  sensors = tuple(env_cfg.scene.sensors or ())
  if not any(s.name == SOLE_CONTACT_SENSOR for s in sensors):
    sensors += tactile_scene_sensors(params)
  for s in sensors:
    if hasattr(s, "debug_vis"):
      s.debug_vis = False
  env_cfg.scene.sensors = sensors
  for group in env_cfg.observations.values():
    group.enable_corruption = False
  robot_w = int(args.height * 0.9)
  env_cfg.viewer.width = robot_w
  env_cfg.viewer.height = args.height
  env_cfg.viewer.distance = 2.4
  env_cfg.viewer.elevation = -10.0

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode="rgb_array")
  rl_cfg = load_rl_cfg(args.task)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=rl_cfg.clip_actions)

  policy = None
  if not args.no_policy:
    ckpt = (
      Path(args.checkpoint)
      if args.checkpoint
      else get_checkpoint_path(
        Path(args.log_root) / rl_cfg.experiment_name, f".*seed{args.seed}$", "model_.*.pt"
      )
    )
    print(f"[tactile] checkpoint: {ckpt}")
    runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
    runner = runner_cls(wrapped, asdict(rl_cfg), device=device)
    runner.load(str(ckpt), load_cfg={"actor": True}, strict=True, map_location=device)
    policy = runner.get_inference_policy(device=device)

  layout = LAYOUTS[params.layout]
  panel = SolePanel(layout, width_px=int(args.height * 1.25), height_px=args.height,
                    vmax=args.vmax, history_s=args.history, dt=env.step_dt)
  cmd_term = env.command_manager.get_term("twist") if "twist" in env_cfg.commands else None

  frames, log = [], {k: [] for k in ("normal", "shear", "support", "net_force", "cop")}
  obs, _ = wrapped.reset()
  with torch.inference_mode():
    for secs, vx, vy, wz, _label in SCHEDULE:
      cmd = torch.tensor([[vx, vy, wz]], device=device)
      for _ in range(int(secs / env.step_dt)):
        if cmd_term is not None:
          cmd_term.vel_command_b[:] = cmd
        if policy is None:
          action = torch.zeros(1, env.action_manager.total_action_dim, device=device)
        else:
          action = policy(obs)
        obs, _, dones, _ = wrapped.step(action)
        if policy is not None:
          policy.reset(dones.bool())

        r = read_tactile(env, params)
        rec = {k: getattr(r, k)[0].cpu().numpy() for k in log}
        for k in log:
          log[k].append(rec[k])
        side = panel.update(rec["normal"], rec["shear"], rec["support"], rec["cop"],
                            rec["net_force"][:, 2].clip(min=0))
        robot = env.render()
        if robot is not None:
          frames.append(np.concatenate([robot, side[: robot.shape[0]]], axis=1))
  env.close()

  name = "tactile_" + (args.task.split("-", 1)[1].lower().replace("-", "_"))
  if args.no_policy:
    name += "_nopolicy"
  out = Path(args.out or REPO / "media" / f"{name}.mp4")
  out.parent.mkdir(parents=True, exist_ok=True)
  mediapy.write_video(str(out), frames, fps=args.fps)
  np.savez_compressed(out.with_suffix(".npz"), dt=env.step_dt,
                      **{k: np.stack(v) for k, v in log.items()})
  print(f"[tactile] wrote {out} ({len(frames)} frames) and {out.with_suffix('.npz').name}")


if __name__ == "__main__":
  main()
