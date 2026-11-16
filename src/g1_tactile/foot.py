"""The bare G1 foot, straight from mjlab's G1 MJCF: its visual mesh and its 7
stock collision capsules, in the ankle-roll-link (sole) frame."""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from g1_tactile.layout import Side


def _g1_xml_dir() -> Path:
  import mjlab

  return Path(os.path.dirname(mjlab.__file__)) / "asset_zoo/robots/unitree_g1/xmls"


@dataclass(frozen=True)
class Capsule:
  name: str
  p0: tuple[float, float, float]
  p1: tuple[float, float, float]
  radius: float


@lru_cache(maxsize=2)
def foot_capsules(side: Side) -> tuple[Capsule, ...]:
  """The 7 collision capsules of one foot, straight from the G1 MJCF."""
  root = ET.parse(_g1_xml_dir() / "g1.xml").getroot()
  radius = None
  for d in root.iter("default"):
    if d.get("class") == "foot_capsule":
      radius = float(d.find("geom").get("size"))  # type: ignore[union-attr]
  assert radius is not None
  out = []
  for g in root.iter("geom"):
    name = g.get("name", "")
    if name.startswith(f"{side}_foot") and name.endswith("_collision"):
      v = [float(t) for t in g.get("fromto", "").split()]
      out.append(Capsule(name, tuple(v[:3]), tuple(v[3:]), radius))  # type: ignore[arg-type]
  return tuple(sorted(out, key=lambda c: c.name))


@lru_cache(maxsize=2)
def foot_mesh(side: Side):
  """The visual foot mesh as a trimesh, in the ankle-roll-link frame."""
  import trimesh

  path = next(_g1_xml_dir().rglob(f"{side}_ankle_roll_link.STL"))
  return trimesh.load(path)
