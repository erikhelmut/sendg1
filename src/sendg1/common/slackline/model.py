"""The slackline as an MjSpec: a tensioned strip of webbing between two anchors.

A slackline's stiffness comes from its tension, not from its material: pushed
sideways or down at a point, a line under tension ``T`` pushes back with
``T`` times the kink it makes there. The model is that, discretised. The span
is cut into ``NODE_SPACING`` segments; each interior node is a short piece of
webbing that may move sideways (``y``), up and down (``z``) and roll about the
line (``x``), and every pair of neighbours (and each end node with its anchor)
is joined by a fixed tendon ``q_i - q_{i+1}`` with stiffness ``T / dx``. The
energy, sum of ``T / (2 dx) (q_i - q_{i+1})^2``, is exactly a taut string's,
so a point load ``F`` at ``x`` along a span ``L`` sinks it by
``F x (L - x) / (T L)``, softest at mid-span, stiff by the anchors. Tension is
assumed constant (no stretch): fine at the slopes a person makes, ~0.1.

Roll is the webbing twisting about its own length. A flat strip under tension
resists twist by stretching its edges, ``T w^2 / 12`` per unit length, which
for 5 cm webbing is nearly nothing: under a sole it rolls freely and the sole
has to hold it flat, as on a real line.

Stability at the 5 ms physics step: the webbing weighs grams, so its own
modes are far too fast to integrate explicitly. Each node gets a little
armature, and every tendon a damping on the *relative* velocity of its two
nodes. ``implicitfast`` integrates that damping implicitly, so the fast
neighbour-against-neighbour modes are overdamped and stable, while the slow
mode the robot rides (the whole line swinging together, where neighbours move
alike) is barely damped, as on a real line.

The numbers: 4 m is a short park line; 1.5 kN is a typical tension for a
person's line of that length, and with the G1's 35 kg it sags ~23 cm at
mid-span (a person's line sags ~10 % of its length). Webbing 5 cm wide is the
beginner ("2 inch") line; the G1 sole is ~7 cm across, so the line is
narrower than the foot. The anchors are 1.0 m up, so the loaded line stays
~75 cm off the floor.

Frame: x along the line (anchor to anchor), y to its left, z up; origin on
the floor below the middle of the span.
"""

from __future__ import annotations

import mujoco

SPAN = 4.0
"""Anchor to anchor (m)."""
HEIGHT = 1.0
"""Top of the unloaded webbing above the floor (m)."""
TENSION = 1500.0
"""N. The nominal line; ``TENSION_RANGE`` is randomised per world."""
TENSION_RANGE = (1000.0, 2000.0)
WIDTH = 0.05
THICKNESS = 0.004
COLLISION_RGBA = (0.0, 0.0, 0.0, 0.002)
"""Near-invisible, not invisible: raycasts skip alpha-0 geoms (mujoco_warp
``ray.py``), and the taxel rays must see the webbing."""
COLLISION_DEPTH = 0.08
"""The contact box reaches this far below the webbing's top face: 4 mm of
real webbing is thinner than a loaded contact sinks, and a foot would fall
through it. Only the top face is ever touched. At 2 cm a loaded sole sank
past the box's mid-plane and MuJoCo resolved the contact through the bottom
face: normals flipped and the line pulled the foot down."""
NODE_SPACING = 0.2
SEGMENT_OVERLAP = 0.01
"""Each piece's box reaches this far past its half of the spacing at both ends,
so neighbours overlap by 2 cm: where two pieces meet at an angle there is no
exposed end face for a sole to catch on. The pieces never collide with each
other (contype 2, conaffinity 1)."""
NUM_NODES = round(SPAN / NODE_SPACING) - 1
"""Interior nodes; the two ends are the anchors."""

NODE_MASS = 0.007
"""70 g/m webbing."""
NODE_ARMATURE = 0.3
"""Extra inertia (kg) on each slide: slows the webbing's own modes to what the
step can resolve. 2 kg over the span, against the robot's 35."""
ROLL_ARMATURE = 5e-3
"""kg m^2 on each roll hinge, likewise."""
NODE_DAMPING = 2.0
"""N s/m on each slide against the world: air and the webbing's hysteresis."""
SEGMENT_DAMPING = 300.0
"""N s/m on each neighbour pair's relative velocity. Integrated implicitly,
it keeps the webbing's own fast modes stable at the 5 ms step; the slow mode
the robot rides moves neighbours nearly together, so it is left lightly
damped (~0.1 b over the span, a few % of critical)."""
ROLL_STIFFNESS_PER_M = TENSION * WIDTH**2 / 12.0
"""N m^2: the tension's resistance to twisting a flat strip, per unit length."""
ROLL_DAMPING = 0.05
PITCH_ARMATURE = 2e-3
"""kg m^2 on each pitch and yaw hinge; the slope equalities drive them."""

FRICTION = 0.8
"""Polyester/nylon webbing on a rubber sole."""
CONTYPE = 2
CONAFFINITY = 1
"""Collides with the robot, never with itself (as the cylinder)."""

ROOT_BODY = "slackline"
NODE_BODY = "node_{:02d}"
NODE_GEOM = "webbing_{:02d}"
LOOK_GEOM = "webbing_look_{:02d}"
SLIDE_Y = "node_{:02d}_y"
SLIDE_Z = "node_{:02d}_z"
ROLL = "node_{:02d}_roll"
PITCH = "node_{:02d}_pitch"
YAW = "node_{:02d}_yaw"

WEBBING_RGBA = (0.298, 0.788, 0.941, 1.0)
"""Kyber accent (#4cc9f0)."""
ANCHOR_RGBA = (0.98, 0.75, 0.33, 1.0)
"""#fac055 posts."""
POST_RGBA = (0.16, 0.19, 0.24, 1.0)


def node_x(i: int) -> float:
  """x of interior node ``i`` (0-based) relative to mid-span."""
  return -SPAN / 2 + (i + 1) * NODE_SPACING


def static_sag(x: float, load_n: float, tension: float = TENSION) -> list[float]:
  """Node z offsets (m, negative down) of the line carrying ``load_n`` at
  ``x``: the taut string's triangle, ``F a b / (T L)`` under the load."""
  a = x + SPAN / 2
  out = []
  for i in range(NUM_NODES):
    s = node_x(i) + SPAN / 2
    out.append(-load_n * (min(s, a) * (SPAN - max(s, a))) / (tension * SPAN))
  return out


def _visual(geom) -> None:
  geom.contype = 0
  geom.conaffinity = 0
  geom.mass = 0.0
  geom.group = 1


def _coupling(
  spec: mujoco.MjSpec, name: str, joints: list[tuple[str, float]], k: float, d: float = 0.0
) -> None:
  ten = spec.add_tendon(name=name, stiffness=k, damping=d, springlength=(0.0, 0.0))
  for joint, coef in joints:
    ten.wrap_joint(joint, coef)


SURFACES = ("box", "ribbon", "cable")
"""What the feet touch, over the same node chain (so the same sag and sway):

- ``box``: a deep box per 20 cm piece (the original model).
- ``ribbon``: a MuJoCo flex, a seamless strip of triangles with two vertices
  on every piece, rolling with it; thickness ``FLEX_RADIUS`` either side.
- ``cable``: a 1D flex through the pieces' centres, a round rope of radius
  ``CABLE_RADIUS`` (no flat face, so roll means nothing to the sole).

In the flex variants the boxes stay as thin, non-colliding strips: the taxel
rays (mujoco_warp's geom raycast) do not see flexes."""
CONTACTS = ("soft", "stiff")
"""``soft``: MuJoCo's default contact (the original model). ``stiff``: the
line's contacts get ``STIFF_SOLREF`` / ``STIFF_SOLIMP`` (by priority over the
sole's) and each piece ``STIFF_TURN_ARMATURE`` on its pitch and yaw hinges.
MuJoCo scales a soft contact's stiffness by the effective mass at it, and a
20 cm piece of webbing pivoting on those hinges weighs almost nothing: with
the defaults a 35 kg sole on capsules sinks through the line. Roll is left
alone, so the webbing rolls as freely as before."""
STIFF_SOLREF = (0.01, 1.0)
STIFF_SOLIMP = (0.95, 0.99, 0.001, 0.5, 2.0)
STIFF_PRIORITY = 2
STIFF_TURN_ARMATURE = 20 * 2e-3
FLEX_RADIUS = 0.004
CABLE_RADIUS = 0.0125
FLEX_NAME = "webbing_flex"


def _add_flex(spec: mujoco.MjSpec, surface: str, width: float, stiff: bool) -> None:
  """Skin the node chain with a flex, from anchor to anchor. Its top surface is
  the node frame (as the box's top face is), so vertices sit one radius below."""
  radius = FLEX_RADIUS if surface == "ribbon" else CABLE_RADIUS
  bodies: list[str] = []
  verts: list[float] = []
  rows = (width / 2, -width / 2) if surface == "ribbon" else (0.0,)
  stations = [(ROOT_BODY, -SPAN / 2)] + [(NODE_BODY.format(i), None) for i in range(NUM_NODES)]
  stations.append((ROOT_BODY, SPAN / 2))
  for body, x in stations:
    for y in rows:
      bodies.append(body)
      # The anchors' vertices live in the root frame, the pieces' in their own.
      verts += [x, y, HEIGHT - radius] if x is not None else [0.0, y, -radius]
  elem: list[int] = []
  n = len(stations)
  for i in range(n - 1):
    if surface == "ribbon":
      lo, ro, ln, rn = 2 * i, 2 * i + 1, 2 * i + 2, 2 * i + 3
      elem += [lo, ro, rn, lo, rn, ln]  # both wound with the normal up
    else:
      elem += [i, i + 1]
  f = spec.add_flex()
  f.name = FLEX_NAME
  f.dim = 2 if surface == "ribbon" else 1
  f.vertbody = bodies
  f.vert = verts
  f.elem = elem
  f.radius = radius
  f.contype = CONTYPE
  f.conaffinity = CONAFFINITY
  f.friction = (FRICTION, 0.005, 0.0001)
  f.selfcollide = mujoco.mjtFlexSelf.mjFLEXSELF_NONE
  f.internal = False
  f.young = 0.0  # the string's springs are the tendons below; the flex is a skin
  f.rgba = WEBBING_RGBA
  f.group = 0
  if stiff:
    f.solref, f.solimp, f.priority = STIFF_SOLREF, STIFF_SOLIMP, STIFF_PRIORITY


def slackline_spec(
  tension: float = TENSION, surface: str = "box", contact: str | None = None
) -> mujoco.MjSpec:
  """``contact`` defaults to ``soft`` for the box, ``stiff`` for the flexes
  (which a sole otherwise passes straight through)."""
  if surface not in SURFACES:
    raise ValueError(f"surface {surface!r} not in {SURFACES}")
  flex = surface != "box"
  contact = contact or ("stiff" if flex else "soft")
  if contact not in CONTACTS:
    raise ValueError(f"contact {contact!r} not in {CONTACTS}")
  stiff = contact == "stiff"
  turn_armature = STIFF_TURN_ARMATURE if stiff else PITCH_ARMATURE
  spec = mujoco.MjSpec()
  root = spec.worldbody.add_body(name=ROOT_BODY)
  half_seg = NODE_SPACING / 2 + SEGMENT_OVERLAP
  for i in range(NUM_NODES):
    node = root.add_body(name=NODE_BODY.format(i), pos=(node_x(i), 0.0, HEIGHT))
    for name, jtype, axis, arm, damp in (
      (SLIDE_Y, mujoco.mjtJoint.mjJNT_SLIDE, (0, 1, 0), NODE_ARMATURE, NODE_DAMPING),
      (SLIDE_Z, mujoco.mjtJoint.mjJNT_SLIDE, (0, 0, 1), NODE_ARMATURE, NODE_DAMPING),
      (YAW, mujoco.mjtJoint.mjJNT_HINGE, (0, 0, 1), turn_armature, 0.0),
      (PITCH, mujoco.mjtJoint.mjJNT_HINGE, (0, 1, 0), turn_armature, 0.0),
      (ROLL, mujoco.mjtJoint.mjJNT_HINGE, (1, 0, 0), ROLL_ARMATURE, ROLL_DAMPING),
    ):
      node.add_joint(name=name.format(i), type=jtype, axis=axis, armature=arm, damping=damp)
    # The node frame sits on the webbing's top face, on its centreline.
    depth = THICKNESS if flex else COLLISION_DEPTH
    g = node.add_geom(
      name=NODE_GEOM.format(i),
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=(half_seg, WIDTH / 2, depth / 2),
      pos=(0.0, 0.0, -depth / 2),
      mass=NODE_MASS,
      rgba=COLLISION_RGBA,
    )
    g.contype = 0 if flex else CONTYPE
    g.conaffinity = 0 if flex else CONAFFINITY
    g.friction = (FRICTION, 0.005, 0.0001)
    g.group = 0  # the taxel rays look for group-0 geoms
    if stiff and not flex:
      g.solref, g.solimp, g.priority = STIFF_SOLREF, STIFF_SOLIMP, STIFF_PRIORITY
    look = node.add_geom(
      name=LOOK_GEOM.format(i),
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=(half_seg, WIDTH / 2, THICKNESS / 2),
      # A hair proud of the contact box's top face, so the two never z-fight.
      pos=(0.0, 0.0, -THICKNESS / 2 + 0.0005),
      rgba=WEBBING_RGBA,
    )
    _visual(look)
    if flex:
      look.rgba = (0, 0, 0, 0)  # the flex draws itself

  # The string: each neighbour pair, and each end node against its anchor.
  k = tension / NODE_SPACING
  k_roll = ROLL_STIFFNESS_PER_M / NODE_SPACING
  for axis, fmt, kk, dd in (
    ("y", SLIDE_Y, k, SEGMENT_DAMPING),
    ("z", SLIDE_Z, k, SEGMENT_DAMPING),
    ("roll", ROLL, k_roll, ROLL_DAMPING),
  ):
    _coupling(spec, f"anchor_lo_{axis}", [(fmt.format(0), 1.0)], kk, dd)
    for i in range(NUM_NODES - 1):
      _coupling(spec, f"seg_{i:02d}_{axis}", [(fmt.format(i), 1.0), (fmt.format(i + 1), -1.0)], kk, dd)
    _coupling(spec, f"anchor_hi_{axis}", [(fmt.format(NUM_NODES - 1), 1.0)], kk, dd)

  # Each piece turned to the line's local slope (central difference over its
  # neighbours, the anchors at zero), so the webbing is a smooth chain rather
  # than a staircase of level pieces. Yaw about +z turns +x towards +y,
  # pitch about +y tips +x down: ``2 dx yaw - (y_next - y_prev) = 0`` and
  # ``2 dx pitch + (z_next - z_prev) = 0``. Equalities, not springs: the
  # solver holds them without a stiffness the 5 ms step would have to resolve.
  for i in range(NUM_NODES):
    for angle, slide, sign in ((YAW, SLIDE_Y, -1.0), (PITCH, SLIDE_Z, 1.0)):
      terms = [(angle.format(i), 2 * NODE_SPACING)]
      if i + 1 < NUM_NODES:
        terms.append((slide.format(i + 1), sign))
      if i > 0:
        terms.append((slide.format(i - 1), -sign))
      name = f"slope_{i:02d}_{'yaw' if angle == YAW else 'pitch'}"
      _coupling(spec, name, terms, 0.0)
      spec.add_equality(type=mujoco.mjtEq.mjEQ_TENDON, name1=name, objtype=mujoco.mjtObj.mjOBJ_TENDON)

  if flex:
    _add_flex(spec, surface, WIDTH, stiff)

  # Posts and anchor blocks: visual only (the task ends before a foot gets there).
  for side, x in (("lo", -SPAN / 2), ("hi", SPAN / 2)):
    post = root.add_geom(
      name=f"post_{side}",
      type=mujoco.mjtGeom.mjGEOM_CYLINDER,
      size=(0.04, (HEIGHT + 0.1) / 2, 0),
      pos=(x + (-0.05 if x < 0 else 0.05), 0.0, (HEIGHT + 0.1) / 2),
      rgba=POST_RGBA,
    )
    _visual(post)
    block = root.add_geom(
      name=f"anchor_{side}",
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=(0.03, 0.045, 0.03),
      pos=(x, 0.0, HEIGHT),
      rgba=ANCHOR_RGBA,
    )
    _visual(block)
  return spec
