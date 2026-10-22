"""Face-off: prosthesis user (MyoSuite + Open-Source Leg) vs G1, same camera."""
import matplotlib.pyplot as plt

import common as cf

D = cf.D
human = cf.trim(D + "osl34_155.png")
g1 = cf.trim(D + "front_g1_turn.png")

fig = plt.figure(figsize=(14, 9), facecolor="white")
scale = 1 / max(human.height, g1.height)  # both rendered with the same camera: true scale
for k, (img, x0) in enumerate(((human, 0.06), (g1, 0.70))):
  h = 0.62 * img.height * scale
  wd = h * img.width / img.height * 9 / 14
  ax = fig.add_axes([x0 + (0.24 - wd) / 2, 0.27, wd, h])
  ax.imshow(img)
  ax.axis("off")
  fig.add_artist(plt.Line2D([x0, x0 + 0.24], [0.268, 0.268], color="#d5d9de", lw=3))
  sax = fig.add_axes([x0 + 0.07, 0.02, 0.10, 0.22])
  if k == 0:
    cf.draw_human_sole(sax)
  else:
    cf.draw_g1_sole(sax, cf.STAND)
  cf.sole_axes(sax)
fig.text(0.18, 0.99, "Prosthesis user", ha="center", va="top", fontsize=24, fontweight="bold",
         color=cf.INK)
fig.text(0.82, 0.99, "G1 robot", ha="center", va="top", fontsize=24, fontweight="bold",
         color=cf.INK)
# Curved arrows: the top arcs up, the bottom arcs down, labels outside the arcs.
cf.arrow(fig, (0.33, 0.62), (0.67, 0.62), "", cf.INK_2, rad=-0.3)
fig.text(0.5, 0.735, "what should a\nprosthetic foot feel?", ha="center", va="bottom",
         fontsize=18, color=cf.INK)
cf.arrow(fig, (0.67, 0.46), (0.33, 0.46), "", cf.KYBER, rad=-0.3)
fig.text(0.5, 0.345, "tactile sole ·\ncontact prediction", ha="center", va="top",
         fontsize=18, color=cf.INK)
fig.savefig(D + "prosthesis_faceoff.png", dpi=300, facecolor="white")
fig.savefig(D + "prosthesis_faceoff.pdf", facecolor="white")
print("ok")
