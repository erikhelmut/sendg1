"""Braille-art banner of the README from ``results/picture/g1_kick.png``.

Writes ``docs/assets/banner.txt`` (plain, pasted into the README) and
``docs/assets/banner.ansi`` (taxels in kyber blue; ``cat`` it in a terminal).

Each character is a 2x4 braille dot cell. The robot is drawn as outlines (it is
white on white, so shading carries little), the sole as an outline with its
taxels filled in, as on the video HUD.

    python scripts/make_banner.py [--cols 78] [--edge 60]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
BLUE = "\x1b[38;2;76;201;240m"  # kyber #4cc9f0
RESET = "\x1b[0m"
DOTS = ((0, 0, 0x01), (1, 0, 0x02), (2, 0, 0x04), (0, 1, 0x08),
        (1, 1, 0x10), (2, 1, 0x20), (3, 0, 0x40), (3, 1, 0x80))
"""(row, col, bit) of each dot in a braille cell."""


def dots(image: Path, cols: int, edge: float) -> tuple[np.ndarray, np.ndarray]:
  """Dot mask [rows*4, cols*2] and which pixels are taxel blue."""
  a = np.asarray(Image.open(image).convert("RGB"))
  ys, xs = np.nonzero((255 - a.min(-1).astype(int)) > 12)
  a = np.pad(a[ys.min():ys.max() + 1, xs.min():xs.max() + 1], ((12, 12), (12, 12), (0, 0)),
             constant_values=255)
  rows = round(cols * 2 * a.shape[0] / a.shape[1] / 4)
  img = Image.fromarray(a).resize((cols * 2, rows * 4), Image.LANCZOS)
  rgb = np.asarray(img).astype(float)
  grey = rgb.mean(-1)
  blue = (rgb[..., 2] - rgb[..., 0]) > 40
  dark = grey < 90
  # the sole: the largest dark-or-taxel region
  lab, _ = ndimage.label(ndimage.binary_closing(dark | blue, iterations=2))
  sole = lab == np.bincount(lab.ravel())[1:].argmax() + 1
  sole_in = ndimage.binary_erosion(sole)
  e = np.asarray(img.convert("L").filter(ImageFilter.FIND_EDGES)).astype(float)
  e[:2] = e[-2:] = 0
  e[:, :2] = e[:, -2:] = 0
  fg = ndimage.binary_opening((255 - rgb.min(-1)) > 10)
  on = (e > edge) & ~sole_in
  on |= fg & ~ndimage.binary_erosion(fg)  # silhouette
  on |= dark & ~sole  # head, hands, the other foot
  on |= (blue & sole) | (sole & ~sole_in)  # taxels, sole outline
  return on, blue


def render(on: np.ndarray, blue: np.ndarray) -> tuple[str, str]:
  rows, cols = on.shape[0] // 4, on.shape[1] // 2
  cell_blue = blue.reshape(rows, 4, cols, 2).any((1, 3))
  plain, ansi = [], []
  for r in range(rows):
    p = a = ""
    for c in range(cols):
      v = sum(b for dy, dx, b in DOTS if on[r * 4 + dy, c * 2 + dx])
      ch = chr(0x2800 + v) if v else " "
      p += ch
      a += f"{BLUE}{ch}{RESET}" if v and cell_blue[r, c] else ch
    plain.append(p.rstrip())
    ansi.append(a.rstrip())
  return "\n".join(plain) + "\n", "\n".join(ansi) + "\n"


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument("--image", type=Path, default=ROOT / "results/picture/g1_kick.png")
  ap.add_argument("--cols", type=int, default=78)
  ap.add_argument("--edge", type=float, default=60, help="edge threshold, 0-255")
  ap.add_argument("--out", type=Path, default=ROOT / "docs/assets")
  args = ap.parse_args()
  plain, ansi = render(*dots(args.image, args.cols, args.edge))
  args.out.mkdir(parents=True, exist_ok=True)
  (args.out / "banner.txt").write_text(plain)
  (args.out / "banner.ansi").write_text(ansi)
  print(ansi, end="")


if __name__ == "__main__":
  main()
