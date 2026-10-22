"""Half prosthesis user, half G1: natural colours, no labels."""
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageEnhance

import common as cf

D = cf.D
human = cf.trim(D + "osl34_180_d.png")
# The prosthesis model's bone texture is greenish; desaturate and lift it so it
# matches the plain full-body skeleton (the slate prosthesis is near-neutral anyway).
a = human.getchannel("A")
rgb = ImageEnhance.Color(human.convert("RGB")).enhance(0.0)
arr = np.asarray(rgb).astype(np.float32) / 255
arr = arr ** 0.8  # gentle brighten
human = Image.fromarray((arr * 255).astype(np.uint8)).convert("RGBA")
human.putalpha(a)
g1 = cf.trim(D + "front_g1_plain.png")
H = 2200
hs = human.resize((int(human.width * H / human.height), H), Image.LANCZOS)
gs = g1.resize((int(g1.width * H / g1.height), H), Image.LANCZOS)


def centre_col(img):
  # Body midline from the upper body (pelvis to shoulders): the prosthesis makes
  # the legs asymmetric, so the leg silhouette would shift the split.
  a = np.asarray(img)[int(img.height * 0.15): int(img.height * 0.45), :, 3] > 8
  cols = np.where(a.any(0))[0]
  return (cols.min() + cols.max()) // 2


ch, cg = centre_col(hs), centre_col(gs)
half = max(ch, hs.width - ch, cg, gs.width - cg)
canvas = Image.new("RGBA", (2 * half, H), (255, 255, 255, 0))
canvas.alpha_composite(hs.crop((ch - half, 0, ch, H)), (0, 0))
canvas.alpha_composite(gs.crop((cg, 0, cg + half, H)), (half, 0))
canvas.save(D + "prosthesis_split_body_transparent.png")

fig = plt.figure(figsize=(9, 11.5), facecolor="white")
ax = fig.add_axes([0.1, 0.29, 0.8, 0.69])
ax.imshow(canvas)
ax.axvline(half, color=cf.KYBER, lw=2.5, ls=(0, (6, 4)))
ax.set_xlim(0, canvas.width)
ax.axis("off")
sax = fig.add_axes([0.35, 0.02, 0.30, 0.26])
cf.sole_axes(sax)
lclip = plt.Rectangle((-1, -1), 1, 2, transform=sax.transData)
rclip = plt.Rectangle((0, -1), 1, 2, transform=sax.transData)
cf.draw_human_sole(sax, clip=lclip, q=False)
cf.draw_g1_sole(sax, cf.STAND, clip=rclip, cop=False)
sax.axvline(0, color=cf.KYBER, lw=2.5, ls=(0, (6, 4)))
sax.text(-0.022, 0.07, "?", ha="center", va="center", fontsize=34, color=cf.DASH,
         fontweight="bold")
fig.savefig(D + "prosthesis_split.png", dpi=300, facecolor="white")
fig.savefig(D + "prosthesis_split.pdf", facecolor="white")
print("ok")
