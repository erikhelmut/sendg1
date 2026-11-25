"""Hero shot of the G1 foot wearing the tactile slipper, on black.

Black rubber on a black background reads through lighting: blue rim lights
graze the sole edge and a soft under-light gives the rubber a specular sheen.
"""

import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from PIL import Image, ImageFilter

from sendg1.common.robots.slipper import slipper_mesh, sole_outline
from sendg1.common.tactile.layout import FOIL, G1_SOLE_Z, LAYOUTS, SLIPPER_FLOOR, SOLE_Z
from sendg1.common.tactile.sensing import TactileParams
from sendg1.common.tactile.viz import _g1_xml_dir

KYBER = np.array([0x4C, 0xC9, 0xF0]) / 255


def euler_quat(roll, pitch, yaw):
    q = np.zeros(4)
    mujoco.mju_euler2Quat(q, np.deg2rad([roll, pitch, yaw]), "xyz")
    return q


def taxel_rgb(a) -> np.ndarray:
    h = a.taxel_color.lstrip("#")
    return np.array([int(h[i : i + 2], 16) for i in (0, 2, 4)]) / 255


def flat_prism(o, z_top, z_bot):
    """Prism with unshared vertices per face, so MuJoCo can't smooth normals
    across the sole's edge (the shared-vertex prism shades like a cone)."""
    k = len(o)
    V, F = [], []

    def face(pts, tris):
        b = len(V)
        V.extend(pts)
        F.extend([(b + i, b + j, b + l) for i, j, l in tris])

    face([(*p, z_top) for p in o], [(0, i + 1, i) for i in range(1, k - 1)])
    face([(*p, z_bot) for p in o], [(0, i, i + 1) for i in range(1, k - 1)])
    for i in range(k):
        j = (i + 1) % k
        face(
            [(*o[i], z_top), (*o[i], z_bot), (*o[j], z_bot), (*o[j], z_top)],
            [(0, 1, 2), (0, 2, 3)],
        )
    return np.array(V), np.array(F)


def build(a) -> mujoco.MjModel:
    spec = mujoco.MjSpec()
    spec.visual.global_.offwidth = a.w
    spec.visual.global_.offheight = a.h
    spec.visual.quality.offsamples = 16
    spec.visual.quality.shadowsize = 8192
    spec.visual.headlight.active = 0
    spec.visual.headlight.ambient = (0.0, 0.0, 0.0)
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
    spec.add_mesh(
        name="foot", file=str(_g1_xml_dir() / "assets" / "left_ankle_roll_link.STL")
    )
    sv, sf = slipper_mesh()
    o = sole_outline()
    z = G1_SOLE_Z - SLIPPER_FLOOR
    for name, (v, f) in {
        "slipper": (sv, sf),
        "foil": flat_prism(o, z, z - FOIL),
        "rubber": flat_prism(o, z - FOIL, SOLE_Z),
    }.items():
        m = spec.add_mesh(name=name)
        m.uservert = v.astype(np.float32).ravel().tolist()
        m.userface = f.astype(np.int32).ravel().tolist()

    mats = {
        "foot": {"rgba": (0.30, 0.30, 0.33, 1), "specular": 0.7, "shininess": 0.5},
        "slipper": {"rgba": (*KYBER, 1), "specular": 0.25, "shininess": 0.3},
        "foil": {"rgba": (0.95, 0.95, 0.95, 1), "specular": 0.8, "shininess": 0.9},
        "rubber": {"rgba": (0.07, 0.07, 0.08, 1), "specular": 0.5, "shininess": 0.15},
    }
    for n, kw in mats.items():
        spec.add_material(name=n, **kw)

    body = spec.worldbody.add_body(
        name="foot", pos=(0, 0, 0), quat=euler_quat(a.roll, a.pitch, a.yaw)
    )
    if a.taxels:
        # Smoked-glass rubber so the foil's taxels show through it.
        spec.material("rubber").rgba = (0.03, 0.03, 0.04, a.rubber_alpha)
        spec.material("foil").rgba = (0.80, 0.82, 0.85, 1)
        spec.add_material(
            name="taxel",
            rgba=(*taxel_rgb(a), 1),
            emission=a.taxel_glow,
            specular=0.2,
            shininess=0.3,
        )
    for n in mats:
        body.add_geom(
            name=n,
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname=n,
            material=n,
            contype=0,
            conaffinity=0,
            group=1 if n == "rubber" else 0,
        )

    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, euler_quat(a.roll, a.pitch, a.yaw))
    if a.taxels:
        lay = LAYOUTS[TactileParams().layout]
        zt = z - FOIL - 0.0004
        for i, ((cx, cy), (hx, hy)) in enumerate(
            zip(lay.for_side("left"), lay.half_sizes)
        ):
            body.add_geom(
                name=f"taxel{i}",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=(hx * a.taxel_scale, hy * a.taxel_scale, 0.00035),
                pos=(cx, cy, zt),
                material="taxel",
                contype=0,
                conaffinity=0,
            )
    if a.gap is not None:
        vv = (R.reshape(3, 3) @ flat_prism(o, z - FOIL, SOLE_Z)[0].T).T
        a.floor = vv[:, 2].min() - a.gap
    if a.floor is not None:
        spec.add_material(
            name="floor", rgba=(0, 0, 0, 1), specular=0, shininess=0, reflectance=a.refl
        )
        spec.worldbody.add_geom(
            name="floor",
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=(3, 3, 0.05),
            pos=(0, 0, a.floor),
            material="floor",
            contype=0,
            conaffinity=0,
        )

    # Lights (world frame; foot centre is near the origin).
    def light(pos, rgb, spec_=0.3, shadow=False, cutoff=40):
        p = np.array(pos, float)
        l = spec.worldbody.add_light(pos=p, dir=-p / np.linalg.norm(p))
        l.diffuse, l.specular, l.castshadow = rgb, (spec_,) * 3, shadow
        l.cutoff, l.exponent = cutoff, 2
        return l

    light((0.25, 0.35, 0.5), (0.85, 0.83, 0.80), 0.5, True)  # key, warm, top front-left
    light((-0.45, -0.1, 0.15), KYBER * 1.1, 0.6)  # blue rim from behind
    light((-0.3, 0.4, -0.25), KYBER * 0.9, 0.9)  # blue under-rim on the sole
    light((0.35, -0.35, -0.3), (0.35, 0.37, 0.42), 1.0)  # cool under-fill: rubber sheen
    light((0.0, 0.5, -0.05), (0.25, 0.25, 0.28), 0.6)  # side fill

    n = -R.reshape(3, 3)[:, 2]  # sole normal (points away from foot)
    dcam = np.array(
        [
            np.cos(np.deg2rad(a.cel)) * np.cos(np.deg2rad(a.caz)),
            np.cos(np.deg2rad(a.cel)) * np.sin(np.deg2rad(a.caz)),
            np.sin(np.deg2rad(a.cel)),
        ]
    )
    mirror = 2 * (dcam @ n) * n - dcam
    light(0.5 * mirror + np.array(a.target), (0.10, 0.11, 0.13), a.sheen, cutoff=60)

    cam = spec.worldbody.add_camera(name="shot", fovy=a.fovy)
    d = np.array(
        [
            np.cos(np.deg2rad(a.cel)) * np.cos(np.deg2rad(a.caz)),
            np.cos(np.deg2rad(a.cel)) * np.sin(np.deg2rad(a.caz)),
            np.sin(np.deg2rad(a.cel)),
        ]
    )
    target = np.array(a.target)
    cam.pos = target + a.dist * d
    fwd = -d
    right = np.cross(fwd, (0, 0, 1))
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    cam.alt.type = mujoco.mjtOrientation.mjORIENTATION_XYAXES
    cam.alt.xyaxes = np.concatenate([right, up])
    return spec.compile()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="foot.png")
    ap.add_argument("--w", type=int, default=3840)
    ap.add_argument("--h", type=int, default=2160)
    ap.add_argument("--roll", type=float, default=0)
    ap.add_argument("--pitch", type=float, default=-28)  # toe up (heel strike)
    ap.add_argument("--yaw", type=float, default=-35)
    ap.add_argument("--caz", type=float, default=20)
    ap.add_argument("--cel", type=float, default=-6)
    ap.add_argument("--dist", type=float, default=0.42)
    ap.add_argument("--fovy", type=float, default=35)
    ap.add_argument("--target", type=float, nargs=3, default=(0.03, 0.0, -0.02))
    ap.add_argument("--floor", type=float, default=None)
    ap.add_argument("--refl", type=float, default=0.3)
    ap.add_argument("--rgain", type=float, default=0.8)
    ap.add_argument("--fade", type=float, default=0.3)
    ap.add_argument("--taxels", action="store_true")
    ap.add_argument("--rubber-alpha", type=float, default=0.55)
    ap.add_argument("--taxel-color", default="#4cc9f0")
    ap.add_argument("--taxel-glow", type=float, default=0.6)
    ap.add_argument("--bloom", type=float, default=0.35)
    ap.add_argument("--taxel-scale", type=float, default=0.85)
    ap.add_argument("--gap", type=float, default=None)
    ap.add_argument("--pool", type=float, default=0.0)
    ap.add_argument("--sheen", type=float, default=0.3)
    ap.add_argument("--rim", type=float, default=1.0)
    ap.add_argument("--glow", type=float, default=0.35)
    a = ap.parse_args()

    model = build(a)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    with mujoco.Renderer(model, a.h, a.w) as r:
        r.update_scene(data, camera="shot")
        r.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = True
        img = r.render().astype(np.float32) / 255
        r.enable_segmentation_rendering()
        r.update_scene(data, camera="shot")
        seg = r.render()[..., 0]
        fl = model.geom("floor").id if a.floor is not None else -2
        rub = (seg == model.geom("rubber").id) | (seg == model.geom("foil").id)
        mask = ((seg >= 0) & (seg != fl)).astype(np.float32)
        # Taxel mask from a pass without the (translucent) rubber in front.
        opt = mujoco.MjvOption()
        opt.geomgroup[1] = 0
        r.update_scene(data, camera="shot", scene_option=opt)
        seg_t = r.render()[..., 0]
        taxel_ids = [
            model.geom(f"taxel{i}").id
            for i in range(model.ngeom)
            if f"taxel{i}" in {model.geom(j).name for j in range(model.ngeom)}
        ]
        tx = np.isin(seg_t, taxel_ids) & rub

    # Kyber rim on the sole's outline: edge pixels of the rubber that touch the
    # background, blurred into a thin glow, drawn over everything.
    from scipy.ndimage import binary_dilation, gaussian_filter

    bgm = mask < 0.5
    k = max(2, a.w // 600)
    edge = rub & binary_dilation(bgm, iterations=k)
    rim = gaussian_filter(edge.astype(np.float32), a.w / 1500)
    rim = np.clip(rim / (rim.max() + 1e-6) * 1.6, 0, 1) * a.rim
    # Soft kyber halo behind the silhouette so the black sole separates from
    # the black slide.
    m = Image.fromarray((mask * 255).astype(np.uint8))
    halo = np.asarray(m.filter(ImageFilter.GaussianBlur(a.w / 30)), np.float32) / 255
    halo2 = np.asarray(m.filter(ImageFilter.GaussianBlur(a.w / 9)), np.float32) / 255
    bg = (a.glow * (0.55 * halo + 0.45 * halo2))[..., None] * KYBER * 0.6
    # Floor pixels carry the mirrored foot; fade it out with distance below the sole.
    ys = np.arange(a.h, dtype=np.float32)[:, None, None]
    # Fade from the foot's lowest pixel per column, so it starts under the sole.
    rows = np.flatnonzero(mask.any(1))
    top = rows[-1] - a.h * 0.06
    fade = np.clip(1 - (ys - top) / (a.h * a.fade), 0, 1) ** 1.6
    if a.pool > 0:
        cols = np.flatnonzero(mask.any(0))
        rws = np.flatnonzero(mask.any(1))
        cx, cy = cols.mean(), rws[-1] + 0.02 * a.h
        rx, ry = (cols[-1] - cols[0]) * 0.85, a.h * 0.13
        yy, xx = np.mgrid[0 : a.h, 0 : a.w].astype(np.float32)
        r2 = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2
        pool = np.exp(-1.6 * r2)[..., None] * np.array([0x1F, 0x28, 0x33]) / 255 * a.pool
        bg = bg + pool
    out = img * mask[..., None] + np.maximum(bg, a.rgain * img * fade) * (
        1 - mask[..., None]
    )
    out = out + rim[..., None] * (KYBER * 1.15 - out) * 0.9
    if a.taxels and a.bloom > 0:
        # Taxels seen through the rubber: bright, saturated blue inside the sole.
        glow = gaussian_filter(tx.astype(np.float32), a.w / 250)
        glow = np.clip(glow / (glow.max() + 1e-6), 0, 1) ** 0.8 * a.bloom
        out = out + glow[..., None] * taxel_rgb(a) * (1 - tx[..., None])
    fg = img * mask[..., None]
    fg = fg + rim[..., None] * (KYBER * 1.15 - fg) * 0.9
    alpha = np.clip(np.maximum(mask, rim), 0, 1)
    rgba = np.dstack([np.clip(fg / np.maximum(alpha, 1e-3)[..., None], 0, 1), alpha])
    Image.fromarray((rgba * 255).astype(np.uint8), "RGBA").save(
        a.out.replace(".png", "_alpha.png")
    )
    Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8)).save(a.out)
    print(a.out)


if __name__ == "__main__":
    main()
