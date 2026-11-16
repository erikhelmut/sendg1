"""Put the tactile slipper into an existing mjlab G1 env config in one call."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.sensor import ContactMatch

from g1_tactile.features import DEFAULT_STRENGTH, reset_tactile_noise, tactile_touch
from g1_tactile.robot import G1_FOOT_GEOMS, get_g1_cfg
from g1_tactile.sensing import DEFAULT_PARAMS, TactileParams, tactile_scene_sensors
from g1_tactile.sensors import TERRAIN_MATCH

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnvCfg

_STOCK_FOOT_GEOM = re.compile(r"(left|right)_foot\d+_collision")


def _retarget_foot_geoms(cfg: ManagerBasedRlEnvCfg) -> list[str]:
  """Terms that name mjlab's foot capsules by geom (mjlab's own G1 velocity
  config does, for foot friction randomisation) get the sole boxes instead:
  the capsules no longer exist, and the env would fail to resolve them."""
  changed = []
  for group in ("events", "rewards", "terminations", "curriculum"):
    for name, term in (getattr(cfg, group, None) or {}).items():
      for value in (getattr(term, "params", None) or {}).values():
        geoms = getattr(value, "geom_names", None)
        if isinstance(geoms, str):
          geoms = (geoms,)
        if geoms and any(_STOCK_FOOT_GEOM.fullmatch(g) for g in geoms):
          value.geom_names = G1_FOOT_GEOMS
          changed.append(f"{group}.{name}")
  return changed


def wear_tactile_slipper(
  cfg: ManagerBasedRlEnvCfg,
  obs_group: str | None = "actor",
  noisy: bool = True,
  strength: float = DEFAULT_STRENGTH,
  params: TactileParams = DEFAULT_PARAMS,
  secondary: ContactMatch | None = TERRAIN_MATCH,
  robot: str = "robot",
) -> ManagerBasedRlEnvCfg:
  """Make the G1 in ``cfg`` wear the tactile slipper, in place.

  - the robot entity becomes :func:`g1_tactile.robot.get_g1_cfg` (slipper mesh,
    200 g per foot, box soles, spawned 8 mm higher);
  - the scene gets the tactile sensors; ``secondary`` is what the soles stand on
    (default the terrain; ``None``: anything);
  - terms that named the stock foot capsules now name the sole boxes;
  - with ``obs_group``, that observation group gets a ``tactile`` term (58
    values) and the events a ``tactile_faults`` reset event at ``strength``
    (only when ``noisy``).

  The robot must be named ``robot``: the tactile sensors attach to that entity.
  """
  if robot != "robot":
    raise ValueError("the tactile sensors expect the G1 entity to be named 'robot'")
  cfg.scene.entities[robot] = get_g1_cfg()
  names = {s.name for s in cfg.scene.sensors or ()}
  new = tuple(
    s for s in tactile_scene_sensors(params, secondary=secondary) if s.name not in names
  )
  cfg.scene.sensors = (*(cfg.scene.sensors or ()), *new)
  _retarget_foot_geoms(cfg)
  if obs_group is not None:
    cfg.observations[obs_group].terms["tactile"] = ObservationTermCfg(
      func=tactile_touch, params={"params": params, "noisy": noisy}
    )
    if noisy:
      cfg.events["tactile_faults"] = EventTermCfg(
        func=reset_tactile_noise,
        mode="reset",
        params={"strength": strength, "params": params},
      )
  return cfg
