"""Figure: how the tactile sole is simulated, on one foot standing over a beam edge.

Left, a render: the slipper (see-through), its single collision box, the
taxel rays (2 x 2 per taxel, cast down from 20 mm above the sole), and the
contacts MuJoCo makes. Right, the plan view the model works in: which ray
samples found ground, the net force's centre of pressure, and the per-taxel
normal force that ``tactile.model.distribute`` assigns.

Everything runs the real pieces: ``slipper.sole_box`` collision, the layout's
``sample_points``, ``support_from_gaps``, ``net_force_and_cop`` and
``distribute``. Only the scene is plain MuJoCo (mj_ray, data.contact) rather
than mjlab's sensors, which wrap the same calls.

  MUJOCO_GL=egl uv run python scripts/render_tactile_model.py
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
import torch

from sendg1.common.robots.slipper import slipper_mesh, sole_box, sole_outline
from sendg1.common.sensors import TAXEL_RAY_LIFT
from sendg1.common.tactile import model as tmodel
from sendg1.common.tactile.layout import FOIL, G1_SOLE_Z, LAYOUTS, SLIPPER_FLOOR, SOLE_Z
from sendg1.common.tactile.sensing import TactileParams
from sendg1.common.tactile.viz import _g1_xml_dir

KYBER = "#4cc9f0"
YELLOW = "#fac055"
SLATE = "#1f2833"
INK = "#e8eef2"
INK_2 = "#8a9aa6"
KYBER_RGB = np.array([0x4C, 0xC9, 0xF0]) / 255
YELLOW_RGB = np.array([0xFA, 0xC0, 0x55]) / 255

PARAMS = TactileParams()
LAYOUT = LAYOUTS[PARAMS.layout]
MAX_DIST = TAXEL_RAY_LIFT + 0.05  # as in sensors.taxel_ray_sensor
FOOT_MASS = 17.5  # about half the G1's weight on this foot
EDGE_Y = 0.012  # beam edge (world y); the sole's left ~40 % hangs over it


def prism(o, z_top, z_bot):
    k = len(o)
    v = np.concatenate(
        [np.column_stack([o, np.full(k, z_top)]), np.column_stack([o, np.full(k, z_bot)])]
    )
    return v


def build() -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    spec.option.timestep = 0.001
    spec.visual.global_.offwidth = 2400
    spec.visual.global_.offheight = 2400
    spec.visual.quality.offsamples = 16
    spec.visual.headlight.active = 0
    spec.visual.map.znear = 0.001
    spec.add_texture(
        name="sky",
        type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_FLAT,
        rgb1=(0, 0, 0),
        rgb2=(0, 0, 0),
        width=8,
        height=8,
    )

    # The beam: top at z = 0, edge along x at y = EDGE_Y.
    spec.add_material(name="beam", rgba=(0.17, 0.21, 0.26, 1), specular=0.3)
    spec.worldbody.add_geom(
        name="beam",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(0.22, 0.08, 0.04),
        pos=(0.03, EDGE_Y - 0.08, -0.04),
        material="beam",
    )

    spec.add_mesh(
        name="foot", file=str(_g1_xml_dir() / "assets" / "left_ankle_roll_link.STL")
    )
    sv, sf = slipper_mesh()
    m = spec.add_mesh(name="slipper")
    m.uservert = sv.astype(np.float32).ravel().tolist()
    m.userface = sf.astype(np.int32).ravel().tolist()
    z = G1_SOLE_Z - SLIPPER_FLOOR
    spec.add_mesh(name="foil", uservert=prism(sole_outline(), z, z - FOIL).ravel())
    spec.add_mesh(name="rubber", uservert=prism(sole_outline(), z - FOIL, SOLE_Z).ravel())

    body = spec.worldbody.add_body(name="foot", pos=(0, 0, -SOLE_Z + 0.002))
    body.add_freejoint()
    center, half = sole_box()
    # Centre of mass a little inboard of the box centre, over the beam.
    body.explicitinertial = True
    body.mass = FOOT_MASS
    body.ipos = (center[0], -0.006, center[2] + 0.03)
    body.inertia = (0.02, 0.06, 0.06)
    for name, rgba in (
        ("foot", (0.35, 0.35, 0.38, 0.30)),
        ("slipper", (*KYBER_RGB, 0.22)),
        ("foil", (0.95, 0.95, 0.95, 0.35)),
        ("rubber", (0.05, 0.05, 0.06, 0.30)),
    ):
        body.add_geom(
            name=name,
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname=name,
            rgba=rgba,
            contype=0,
            conaffinity=0,
            group=2,
            density=0,
        )
    # The one collision geom per foot: a box under the rubber outline.
    body.add_geom(
        name="sole_collision",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=half,
        pos=center,
        rgba=(*YELLOW_RGB, 0.18),
        group=0,
        density=0,
        friction=(1.0, 0.005, 0.0001),
    )

    for pos, rgb in (
        ((0.4, -0.6, 0.6), (0.7, 0.7, 0.72)),
        ((-0.5, 0.4, 0.5), (0.35, 0.4, 0.45)),
        ((0.3, 0.6, -0.3), (0.25, 0.28, 0.32)),
    ):
        p = np.array(pos)
        lt = spec.worldbody.add_light(pos=p, dir=-p / np.linalg.norm(p))
        lt.diffuse, lt.specular, lt.cutoff, lt.castshadow = rgb, (0.2,) * 3, 60, False
    spec.visual.headlight.ambient = (0.15, 0.15, 0.17)
    return spec


def settle(model, data):
    mujoco.mj_resetData(model, data)
    for _ in range(3000):
        data.qvel[:] *= 0.995  # light damping so it comes to rest quickly
        mujoco.mj_step(model, data)


def cast_rays(model, data):
    """Origins, hit points and gaps for the 2x2 samples of every taxel."""
    bid = model.body("foot").id
    R = data.xmat[bid].reshape(3, 3)
    p0 = data.xpos[bid]
    pts = np.array(LAYOUT.sample_points("left", PARAMS.rays_per_axis))
    local = np.column_stack([pts, np.full(len(pts), SOLE_Z + TAXEL_RAY_LIFT)])
    origins = p0 + local @ R.T
    down = -R[:, 2]
    geomgroup = np.array([1, 0, 0, 0, 0, 0], np.uint8)
    dist = np.empty(len(pts))
    gid = np.zeros(1, np.int32)
    for i, o in enumerate(origins):
        d = mujoco.mj_ray(model, data, o, down, geomgroup, 1, bid, gid)
        dist[i] = d if 0 <= d <= MAX_DIST else -1
    gap = np.where(dist >= 0, dist - TAXEL_RAY_LIFT, np.inf)
    ends = origins + down[None] * np.where(dist >= 0, dist, MAX_DIST)[:, None]
    return pts, origins, ends, gap


def contacts(model, data):
    """Contact points and the ground's force on the sole, in the sole frame."""
    bid = model.body("foot").id
    R = data.xmat[bid].reshape(3, 3)
    p0 = data.xpos[bid]
    sole = model.geom("sole_collision").id
    pos, force, world = [], [], []
    f6 = np.zeros(6)
    for i in range(data.ncon):
        c = data.contact[i]
        if sole not in (c.geom1, c.geom2):
            continue
        mujoco.mj_contactForce(model, data, i, f6)
        fw = c.frame.reshape(3, 3).T @ f6[:3]  # force geom1 exerts on geom2, world
        if c.geom1 == sole:
            fw = -fw
        world.append(c.pos.copy())
        pos.append(R.T @ (c.pos - p0))
        force.append(R.T @ fw)
    return np.array(pos), np.array(force), np.array(world)


def model_reading(pos_s, force_s, gap):
    t = lambda x: torch.tensor(np.asarray(x), dtype=torch.float32)
    net, cop = tmodel.net_force_and_cop(
        t(pos_s)[None], t(force_s)[None], torch.ones(1, len(pos_s), dtype=torch.bool)
    )
    per = PARAMS.rays_per_axis**2
    sample_support = tmodel.support_from_gaps(
        t(gap).view(1, LAYOUT.num_taxels, per), PARAMS.compliance_depth
    )
    support = sample_support.mean(-1)
    normal, _ = tmodel.distribute(
        net,
        cop,
        support,
        t(LAYOUT.for_side("left"))[None],
        t(LAYOUT.areas()),
        fallback_sigma=min(min(h) for h in LAYOUT.half_sizes) * 2,
    )
    return (
        net[0].numpy(),
        cop[0].numpy(),
        sample_support[0].numpy().ravel(),
        support[0].numpy(),
        normal[0].numpy(),
    )


def add_line(scn, a, b, width, rgba):
    if scn.ngeom >= scn.maxgeom:
        return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(
        g,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        np.zeros(3),
        np.zeros(3),
        np.zeros(9),
        np.asarray(rgba, np.float32),
    )
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, width, a, b)
    scn.ngeom += 1


def add_sphere(scn, p, r, rgba):
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(
        g,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([r, r, r]),
        p,
        np.eye(3).ravel(),
        np.asarray(rgba, np.float32),
    )
    scn.ngeom += 1


def render_3d(model, data, origins, ends, sample_support, cpos_w, cforce_w_mag, size):
    cam = mujoco.MjvCamera()
    cam.lookat[:] = (0.035, 0.01, -0.02)
    cam.distance, cam.azimuth, cam.elevation = 0.34, -118, -16
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = 0
    opt.geomgroup[0] = opt.geomgroup[2] = 1
    with mujoco.Renderer(model, size, size) as r:
        r.update_scene(data, camera=cam, scene_option=opt)
        scn = r.scene
        for o, e, s in zip(origins, ends, sample_support):
            if s > 0:
                add_line(scn, o, e, 0.0007, (*KYBER_RGB, 1))
                add_sphere(scn, e, 0.0016, (*KYBER_RGB, 1))
            else:
                add_line(scn, o, e, 0.0006, (0.75, 0.78, 0.82, 0.8))
        fmax = max(cforce_w_mag.max(), 1e-6)
        for p, f in zip(cpos_w, cforce_w_mag):
            add_sphere(scn, p, 0.0028, (*YELLOW_RGB, 1))
            add_line(
                scn,
                p,
                p + np.array([0, 0, 0.004 + 0.016 * f / fmax]),
                0.0009,
                (*YELLOW_RGB, 1),
            )
        return r.render()


def plan_view(ax, pts, sample_support, edge_y_s, cpos_s, cop, normal):
    from matplotlib.collections import PatchCollection
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.patches import Polygon, Rectangle

    ax.set_facecolor("black")
    o = sole_outline()
    center, half = sole_box()
    # Beam (ground under the sole) and its edge.
    ax.add_patch(Rectangle((-0.115, -0.1), 0.4, edge_y_s + 0.1, color=SLATE, zorder=0))
    ax.axhline(edge_y_s, color=INK_2, lw=1, ls=(0, (4, 3)), zorder=1)
    ax.text(-0.112, edge_y_s + 0.002, "beam edge", color=INK_2, fontsize=10, va="bottom")
    ax.text(-0.112, edge_y_s - 0.012, "beam", color=INK_2, fontsize=10, va="top")
    ax.add_patch(Polygon(o, closed=True, fill=False, ec=INK, lw=1.4, zorder=3))
    ax.add_patch(
        Rectangle(
            center[:2] - half[:2],
            *(2 * half[:2]),
            fill=False,
            ec=YELLOW,
            lw=1.2,
            ls=(0, (2, 2)),
            zorder=3,
        )
    )

    cmap = LinearSegmentedColormap.from_list("k", [SLATE, "#2a6f8a", KYBER, "#d6f3fc"])
    norm = Normalize(0, max(normal.max(), 1e-6))
    patches, colors = [], []
    for (cx, cy), (hx, hy), f in zip(LAYOUT.for_side("left"), LAYOUT.half_sizes, normal):
        patches.append(Rectangle((cx - hx, cy - hy), 2 * hx, 2 * hy))
        colors.append(cmap(norm(f)) if f > 0.05 else (0.07, 0.08, 0.1, 1))
    ax.add_collection(PatchCollection(patches, fc=colors, ec="#5d6d78", lw=0.8, zorder=4))
    for (cx, cy), f in zip(LAYOUT.for_side("left"), normal):
        if f > 0.5:
            ax.text(
                cx,
                cy,
                f"{f:.0f}",
                ha="center",
                va="center",
                fontsize=7.5,
                zorder=6,
                color="black" if norm(f) > 0.55 else INK,
            )
    sup = sample_support > 0
    ax.scatter(*pts[sup].T, s=7, color=KYBER, zorder=5, lw=0)
    ax.scatter(*pts[~sup].T, s=9, facecolor="none", edgecolor=INK_2, lw=0.7, zorder=5)
    ax.scatter(*cpos_s[:, :2].T, s=70, marker="x", color=YELLOW, lw=1.8, zorder=7)
    ax.scatter(*cop, s=160, facecolor="none", edgecolor=YELLOW, lw=2, zorder=7)
    ax.scatter(*cop, s=12, color=YELLOW, zorder=7)
    ax.set_xlim(-0.115, 0.175)
    ax.set_ylim(-0.062, 0.062)
    ax.set_aspect("equal")
    ax.axis("off")
    return cmap, norm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out", type=Path, default=Path("results/picture/tactile_sim_model.png")
    )
    ap.add_argument("--size", type=int, default=1800)
    args = ap.parse_args()

    model = build().compile()
    data = mujoco.MjData(model)
    settle(model, data)
    pts, origins, ends, gap = cast_rays(model, data)
    cpos_s, cforce_s, cpos_w = contacts(model, data)
    net, cop, sample_support, _support, normal = model_reading(cpos_s, cforce_s, gap)
    bid = model.body("foot").id
    edge_y_s = (
        data.xmat[bid].reshape(3, 3).T @ (np.array([0, EDGE_Y, 0]) - data.xpos[bid])
    )[1]
    print(
        f"contacts {len(cpos_s)}, net force {net.round(1)} N, cop {cop.round(4)} m, "
        f"supported samples {int((sample_support > 0).sum())}/{len(sample_support)}, "
        f"taxel sum {normal.sum():.1f} N, max {normal.max():.1f} N"
    )

    img = render_3d(
        model,
        data,
        origins,
        ends,
        sample_support,
        cpos_w,
        np.linalg.norm(cforce_s, axis=1),
        args.size,
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.lines import Line2D

    plt.rcParams["font.family"] = "DejaVu Sans"
    fig = plt.figure(figsize=(16, 9), dpi=150, facecolor="black")
    ax3 = fig.add_axes((0.0, 0.04, 0.5, 0.86))
    ax3.imshow(img)
    ax3.axis("off")
    axp = fig.add_axes((0.52, 0.22, 0.46, 0.6))
    cmap, norm = plan_view(axp, pts, sample_support, edge_y_s, cpos_s, cop, normal)

    fig.text(0.03, 0.93, "Simulated tactile sole", color=INK, fontsize=22, weight="bold")
    fig.text(
        0.03,
        0.895,
        f"one foot over a beam edge  ·  {LAYOUT.num_taxels} taxels  ·  "
        f"{PARAMS.rays_per_axis}×{PARAMS.rays_per_axis} rays per taxel  ·  "
        f"{net[2]:.0f} N on this foot",
        color=INK_2,
        fontsize=12,
    )
    fig.text(
        0.53,
        0.75,
        "Plan view (sole frame, toe right, lateral up)",
        color=INK_2,
        fontsize=12,
    )

    handles = [
        Line2D([], [], color=YELLOW, lw=1.2, ls=(0, (2, 2)), label="collision box"),
        Line2D([], [], color=KYBER, marker="o", ls="", ms=4, label="ray hits ground"),
        Line2D(
            [], [], color=INK_2, marker="o", mfc="none", ls="", ms=4, label="ray misses"
        ),
        Line2D(
            [], [], color=YELLOW, marker="x", ls="", ms=8, mew=1.8, label="MuJoCo contact"
        ),
        Line2D(
            [],
            [],
            color=YELLOW,
            marker="o",
            mfc="none",
            ls="",
            ms=10,
            mew=2,
            label="centre of pressure",
        ),
    ]
    leg = fig.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(0.53, 0.07),
        ncol=3,
        frameon=False,
        fontsize=11,
        labelcolor=INK,
        handletextpad=0.4,
        columnspacing=1.6,
    )
    leg.set_zorder(10)
    cax = fig.add_axes((0.62, 0.2, 0.26, 0.016))
    cb = fig.colorbar(
        ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal"
    )
    cb.set_label("taxel normal force (N)", color=INK_2, fontsize=11)
    cb.ax.tick_params(colors=INK_2, labelsize=10)
    cb.outline.set_edgecolor(INK_2)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, facecolor="black")
    print(args.out)


if __name__ == "__main__":
    main()
