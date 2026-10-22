# Poster figures: prosthesis

Figures for the LokoAssist retreat poster (collaboration box): a prosthesis
user next to the G1, both with a sole underneath. The finished figures are in
`results/poster/`:

- `prosthesis_faceoff.{png,pdf}`: prosthesis user and G1 face to face at true
  scale, curved arrows between them, a "?" footprint vs the G1 sensor sole.
- `prosthesis_split.{png,pdf}`: one figure, left half prosthesis user, right
  half G1, with a split sole below. `prosthesis_split_body_transparent.png` is
  the body alone.
- `prosthesis_{120,60}*.png`: the prosthesis skeleton on its own.

The human is the MyoSuite skeleton with the Open-Source Leg prosthesis
(`myolegs_osl` from [myo_sim](https://github.com/MyoHub/myo_sim), Apache-2.0),
standing in its default pose. Credit both on the poster. The G1 stands still
in a walking rollout of the blind flat-ground policy; its sole map is the
simulated tactile reading at that instant.

## Regenerating

Intermediate renders go to `POSTER_DIR` (default `media/poster/`, gitignored).
myo_sim is not vendored: clone it next to the repo first.

```bash
git clone --depth 1 https://github.com/MyoHub/myo_sim.git ../myo_sim
export MUJOCO_GL=egl
cd scripts/poster
D=../../media/poster

uv run python record_walk.py                            # walk.npz + walk.mjb
uv run python -I build_human.py ../../../myo_sim $D/human.mjb
PYTHONPATH=. uv run python render_fronts.py             # front_human, front_g1, front_g1_turn
NO_TINT=1 SUFFIX=_plain PYTHONPATH=. uv run python render_fronts.py
uv run python -I osl_render.py ../../../myo_sim 155     # osl34_155.png (kyber prosthetic foot)
FULLBODY_LIGHT=1 NO_TINT=1 AMB=0.3 DIF=0.6 SUFFIX=_d \
  uv run python -I osl_render.py ../../../myo_sim 180   # osl34_180_d.png
PYTHONPATH=. uv run python prosthesis_faceoff.py
PYTHONPATH=. uv run python prosthesis_split.py
```

The prosthesis model's bone texture is greenish; `prosthesis_split.py`
desaturates it to match the plain MyoSuite skeleton.
