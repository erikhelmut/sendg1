"""2D overlays drawn onto rendered frames: the tactile-sole card and captions.

Plain PIL, drawn at 2x and downsampled, so a frame costs a few milliseconds
rather than a matplotlib redraw. Soles are shown from above with the toe up,
the robot's left foot on the left: the view of someone standing behind the
robot looking down at its footprints.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

from sendg1.common.tactile.layout import SIDES, SOLE_EXTENT_X, SOLE_EXTENT_Y, TaxelLayout

SS = 2
"""Supersampling factor."""

CARD = (11, 13, 17, 205)
CARD_EDGE = (255, 255, 255, 28)
INK = (232, 236, 242, 255)
INK_2 = (138, 146, 158, 255)
SOLE_FILL = (30, 34, 41, 255)
SOLE_EDGE = (78, 86, 100, 255)
UNLOADED = (31, 40, 51, 255)  # == HOLOCRON_RAMP at 0
TACTILE_ACCENT = (76, 201, 240, 255)  # holocron blue #4cc9f0
BLIND_ACCENT = (150, 158, 170, 255)
FOOT_COLORS = ((76, 201, 240, 255), (250, 192, 85, 255))  # kyber #4cc9f0, #fac055
"""Left, right: the force traces."""


@lru_cache(maxsize=8)
def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
  try:
    import matplotlib

    ttf = matplotlib.get_data_path() + "/fonts/ttf/"
    return ImageFont.truetype(ttf + ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"), size)
  except Exception:
    return ImageFont.load_default()


HOLOCRON_RAMP = ((0.0, "#1f2833"), (0.5, "#2f6f95"), (1.0, "#4cc9f0"))
"""Taxel colour ramp: dark slate at zero, rising to the holocron (kyber) accent
at full scale, so the most loaded taxel is the brightest thing on the card."""


@lru_cache(maxsize=1)
def _force_lut() -> np.ndarray:
  """256 RGB colours along ``HOLOCRON_RAMP``."""
  pos = np.array([p for p, _ in HOLOCRON_RAMP])
  rgb = np.array([[int(h[i : i + 2], 16) for i in (1, 3, 5)] for _, h in HOLOCRON_RAMP])
  t = np.linspace(0.0, 1.0, 256)
  return np.stack([np.interp(t, pos, rgb[:, c]) for c in range(3)], -1).astype(np.uint8)


def force_color(value: float, vmax: float) -> tuple[int, int, int, int]:
  if value < 0.5:
    return UNLOADED
  idx = int(np.clip(value / vmax, 0.0, 1.0) * 255)
  return (*(int(c) for c in _force_lut()[idx]), 255)


@lru_cache(maxsize=4)
def _sole_silhouette(side: str, scale: float) -> Image.Image:
  """The foot mesh seen from below, rasterised once: RGBA, toe up, sized so
  1 m of sole = ``scale`` px, origin at the sole extent's corner."""
  from sendg1.common.tactile.viz import foot_mesh

  w = int((SOLE_EXTENT_Y[1] - SOLE_EXTENT_Y[0]) * scale) + 4
  h = int((SOLE_EXTENT_X[1] - SOLE_EXTENT_X[0]) * scale) + 4
  mask = Image.new("L", (w, h), 0)
  d = ImageDraw.Draw(mask)
  m = foot_mesh(side)  # type: ignore[arg-type]
  for tri in m.vertices[m.faces][:, :, :2]:
    d.polygon([_to_px(x, y, scale) for x, y in tri], fill=255)
  edge = ImageChops.subtract(mask, mask.filter(ImageFilter.MinFilter(5)))
  img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
  img.paste(Image.new("RGBA", (w, h), SOLE_FILL), mask=mask)
  img.paste(Image.new("RGBA", (w, h), SOLE_EDGE), mask=edge)
  return img


def _to_px(x: float, y: float, scale: float) -> tuple[float, float]:
  """Sole frame (x forward, y left) -> silhouette px (toe up, left on the left)."""
  return (SOLE_EXTENT_Y[1] - y) * scale + 2, (SOLE_EXTENT_X[1] - x) * scale + 2


class TactileCard:
  """Bottom-right card: both soles' taxels, per-foot load, a force trace."""

  def __init__(
    self,
    layout: TaxelLayout,
    frame_hw: tuple[int, int],
    *,
    live: bool,
    vmax: float,
    history: int,
    history_s: float,
    weight: float | None = None,
  ):
    self.layout = layout
    self.live = live
    self.vmax = vmax
    fh, fw = frame_hw
    self.u = max(fh / 720, 0.5)  # 1x layout unit: designed at 720 p
    u = self.u * SS
    self.scale = 820 * u  # px per metre of sole
    self.margin = int(18 * self.u)
    sole = _sole_silhouette("left", self.scale)
    self.sole_wh = sole.size
    self.pad = int(16 * u)
    self.gap = int(40 * u)
    self.header = int(46 * u)
    self.footer = int(170 * u)
    self.w = 2 * sole.size[0] + self.gap + 2 * self.pad + int(70 * u)
    self.h = self.header + sole.size[1] + self.footer
    self.trace = np.zeros((2, history))
    self.history_s = history_s
    self.weight = weight
    self.reset()

  def reset(self) -> None:
    self.trace[:] = 0.0
    self.ymax, self.ystep = _nice_axis(max(400.0, 1.3 * (self.weight or 0.0)))

  def _axis(self) -> tuple[float, float]:
    """Trace y range: only ever grows within an episode, in round steps, so the
    gridlines do not jump around while the robot walks."""
    peak = float(self.trace.max()) * 1.1
    if peak > self.ymax:
      self.ymax, self.ystep = _nice_axis(peak)
    return self.ymax, self.ystep

  def draw(self, frame: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """``normal`` [2, T] newtons, feet (left, right)."""
    u = self.u * SS
    totals = normal.sum(-1)
    self.trace = np.roll(self.trace, -1, axis=1)
    self.trace[:, -1] = totals

    card = Image.new("RGBA", (self.w, self.h), (0, 0, 0, 0))
    d = ImageDraw.Draw(card)
    d.rounded_rectangle([0, 0, self.w - 1, self.h - 1], radius=int(14 * u), fill=CARD,
                        outline=CARD_EDGE, width=max(1, int(1 * u)))

    # header: name + status chip
    d.text((self.pad, int(14 * u)), "TACTILE FEET", font=font(int(15 * u), True), fill=INK)
    chip, color = ("LIVE", TACTILE_ACCENT) if self.live else ("BLIND · zero", BLIND_ACCENT)
    f = font(int(11 * u), True)
    tw = d.textlength(chip, font=f)
    x1 = self.w - self.pad
    text_x = int((18 if self.live else 9) * u)  # the live chip has a dot first
    x0 = x1 - text_x - tw - int(9 * u)
    d.rounded_rectangle([x0, int(13 * u), x1, int(33 * u)], radius=int(10 * u),
                        outline=color, width=max(1, int(1.5 * u)))
    if self.live:
      d.ellipse([x0 + int(7 * u), int(20 * u), x0 + int(13 * u), int(26 * u)], fill=color)
    d.text((x0 + text_x, int(16 * u)), chip, font=f, fill=color)

    # soles
    sw, sh = self.sole_wh
    x_soles = (self.w - (2 * sw + self.gap)) // 2
    glow = Image.new("RGBA", card.size, (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    tops = []
    for f_idx, side in enumerate(SIDES):
      ox = x_soles + f_idx * (sw + self.gap)
      oy = self.header
      tops.append((ox, oy))
      card.alpha_composite(_sole_silhouette(side, self.scale), (ox, oy))
      for (cx, cy), (hx, hy), n in zip(
        self.layout.for_side(side), self.layout.half_sizes, normal[f_idx]
      ):
        px, py = _to_px(cx, cy, self.scale)
        box = [ox + px - hy * self.scale, oy + py - hx * self.scale,
               ox + px + hy * self.scale, oy + py + hx * self.scale]
        col = force_color(float(n), self.vmax)
        d.rounded_rectangle(box, radius=int(2.5 * u), fill=col)
        if col is not UNLOADED:
          a = int(255 * min(1.0, n / self.vmax) ** 0.5)
          grow = 4 * u
          gd.rounded_rectangle([box[0] - grow, box[1] - grow, box[2] + grow, box[3] + grow],
                               radius=int(6 * u), fill=(*col[:3], a))
      # centre of pressure: a ring, so the taxel under it stays readable (a
      # foot often carries its whole load on one or two taxels)
      if totals[f_idx] > 5.0:
        c = np.asarray(self.layout.for_side(side))
        cop = (normal[f_idx, :, None] * c).sum(0) / totals[f_idx]
        px, py = _to_px(cop[0], cop[1], self.scale)
        r = 7 * u
        box = [ox + px - r, oy + py - r, ox + px + r, oy + py + r]
        d.ellipse(box, outline=(11, 13, 17, 255), width=int(4 * u))
        d.ellipse(box, outline=(255, 255, 255, 255), width=int(2 * u))
    card = Image.alpha_composite(glow.filter(ImageFilter.GaussianBlur(7 * u)), card)
    d = ImageDraw.Draw(card)

    # per-foot labels
    y = self.header + sh + int(6 * u)
    for f_idx, (ox, _) in enumerate(tops):
      label = ("L", "R")[f_idx]
      d.text((ox + sw / 2, y), f"{label}  {totals[f_idx]:4.0f} N", font=font(int(13 * u), True),
             fill=FOOT_COLORS[f_idx], anchor="ma")

    # colour bar
    y = self.header + sh + int(30 * u)
    x0, x1 = self.pad, self.w - self.pad
    bar = _force_lut()[np.linspace(0, 255, int(x1 - x0)).astype(int)]
    bar_img = Image.fromarray(np.repeat(bar[None], int(6 * u), 0), "RGB").convert("RGBA")
    card.alpha_composite(bar_img, (x0, y))
    small = font(int(10 * u))
    d.text((x0, y + int(9 * u)), "0", font=small, fill=INK_2)
    d.text(((x0 + x1) / 2, y + int(9 * u)), "taxel normal force", font=small, fill=INK_2,
           anchor="ma")
    d.text((x1, y + int(9 * u)), f"{self.vmax:.0f} N", font=small, fill=INK_2, anchor="ra")

    # force trace, with a newton axis and the robot's weight for reference
    t0 = y + int(48 * u)
    t1 = self.h - self.pad
    ymax, ystep = self._axis()
    d.text((x0, y + int(26 * u)), f"total load per foot [N], last {self.history_s:g} s",
           font=small, fill=INK_2)
    gx0 = x0 + int(30 * u)  # tick-label gutter
    to_y = lambda v: t1 - (v / ymax) * (t1 - t0)  # noqa: E731
    for v in np.arange(0.0, ymax + 1e-6, ystep):
      yy = to_y(v)
      d.line([gx0, yy, x1, yy], fill=(255, 255, 255, 60 if v == 0 else 22), width=max(1, int(u)))
      d.text((gx0 - int(6 * u), yy), f"{v:.0f}", font=small, fill=INK_2, anchor="rm")
    if self.weight:
      yy = to_y(self.weight)
      dash = 6 * u
      for xa in np.arange(gx0, x1, 2 * dash):
        d.line([xa, yy, min(xa + dash, x1), yy], fill=(*INK[:3], 150), width=max(1, int(u)))
      label = f"body weight {self.weight:.0f} N"
      bx = d.textbbox((x1, yy - int(3 * u)), label, font=small, anchor="rd")
      d.rectangle([bx[0] - int(4 * u), bx[1] - int(2 * u), bx[2], bx[3] + int(1 * u)],
                  fill=CARD[:3] + (255,))
      d.text((x1, yy - int(3 * u)), label, font=small, fill=INK, anchor="rd")
    n = self.trace.shape[1]
    xs = np.linspace(gx0, x1, n)
    for f_idx in range(2):
      ys = to_y(self.trace[f_idx])
      d.line(list(zip(xs, ys)), fill=FOOT_COLORS[f_idx], width=int(2 * u), joint="curve")

    card = card.resize((self.w // SS, self.h // SS), Image.LANCZOS)
    img = Image.fromarray(frame).convert("RGBA")
    pos = (img.width - card.width - self.margin, img.height - card.height - self.margin)
    img.alpha_composite(card, pos)
    return np.asarray(img.convert("RGB"))


def _nice_axis(peak: float) -> tuple[float, float]:
  """A round axis top >= ``peak`` and a step giving 2-4 gridlines."""
  for step in (100.0, 200.0, 250.0, 500.0, 1000.0, 2000.0):
    if peak / step <= 4:
      break
  return float(np.ceil(peak / step) * step), step


def caption(frame: np.ndarray, title: str, badge: str, live: bool, lines) -> np.ndarray:
  """Top-left card: a policy badge and title, then free-form caption lines."""
  u = max(frame.shape[0] / 720, 0.5) * SS
  big, small, bf = font(int(17 * u), True), font(int(14 * u)), font(int(12 * u), True)
  lines = [str(s) for s in lines if s]
  probe = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
  pad = int(14 * u)
  badge_w = probe.textlength(badge, font=bf) + int(20 * u)
  w = max([badge_w + int(10 * u) + probe.textlength(title, font=big)]
          + [probe.textlength(s, font=small) for s in lines]) + 2 * pad
  h = pad + int(26 * u) + int(22 * u) * len(lines) + pad - int(4 * u)
  card = Image.new("RGBA", (int(w), int(h)), (0, 0, 0, 0))
  d = ImageDraw.Draw(card)
  d.rounded_rectangle([0, 0, card.width - 1, card.height - 1], radius=int(14 * u), fill=CARD,
                      outline=CARD_EDGE, width=max(1, int(u)))
  color = TACTILE_ACCENT if live else BLIND_ACCENT
  d.rounded_rectangle([pad, pad, pad + badge_w, pad + int(22 * u)], radius=int(11 * u),
                      fill=color)
  d.text((pad + badge_w / 2, pad + int(11 * u)), badge, font=bf, fill=(11, 13, 17, 255),
         anchor="mm")
  d.text((pad + badge_w + int(10 * u), pad + int(11 * u)), title, font=big, fill=INK,
         anchor="lm")
  for i, s in enumerate(lines):
    d.text((pad, pad + int(32 * u) + int(22 * u) * i), s, font=small, fill=INK_2)
  card = card.resize((card.width // SS, card.height // SS), Image.LANCZOS)
  img = Image.fromarray(frame).convert("RGBA")
  m = int(18 * u / SS)
  img.alpha_composite(card, (m, m))
  return np.asarray(img.convert("RGB"))


@lru_cache(maxsize=4)
def vignette(h: int, w: int) -> np.ndarray:
  """Multiplier that darkens the corners a little, [h, w, 1]."""
  yy, xx = np.mgrid[0:h, 0:w]
  r = np.hypot((xx - w / 2) / (w / 2), (yy - h / 2) / (h / 2)) / np.sqrt(2)
  return (1.0 - 0.35 * np.clip(r - 0.35, 0, None) ** 1.5 / 0.65**1.5)[..., None].astype(
    np.float32
  )
