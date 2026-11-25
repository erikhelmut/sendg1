# Install

The environment is locked in `uv.lock`: every package at an exact version,
torch from the CUDA 13.0 index (the RTX 50 series needs CUDA >= 12.8, and
torch 2.14 has no cu128 build; driver >= 580). mjlab comes from a git commit,
the first with MuJoCo 4 (`src/sendg1/_pin.py`).

Results up to 2026-11-14 were produced on mjlab 1.6.0 (commit `0fb8a681`),
MuJoCo / mujoco-warp 3.11, torch 2.11 cu128 and rsl-rl 5.4.2: the conda env,
then the first `uv.lock`. They are not directly comparable with runs on the
current stack (different physics engine); re-run a baseline before pairing.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # once per machine
uv sync --frozen --extra dev            # .venv exactly as locked (~8 GB)

uv run ./scripts/test.sh                # or: uv run pytest
uv run list-envs --keyword sendg1       # mjlab's CLI sees our tasks
uv run python scripts/sweep.py --dry-run   # what a default sweep would run

uv run python scripts/sweep.py --tasks flat --max-iterations 3000
uv run python scripts/evaluate.py --task Sendg1-Velocity-Flat-G1-Blind --seed 1
uv run python scripts/aggregate.py
```

`source .venv/bin/activate` instead of `uv run` works too. `--frozen` installs
the lock as is and fails rather than re-resolving; change versions only on
purpose, with `uv lock --upgrade-package <name>`, and commit `uv.lock`
together with a note on which baselines the change can move. The
`--extra cad` extra adds `ezdxf` for `scripts/export_foot_cad.py`.

For the IAS cluster setup and workflow see [cluster.md](cluster.md).
On a cluster node: the NVIDIA driver must support CUDA 12.8 (driver >= 570),
and headless rendering needs the driver's EGL library on the node.

The old conda env (`conda activate sendg1`, `pip install -e ".[dev]"`) still
works but is not locked.

Headless rendering needs `MUJOCO_GL=egl`; the video scripts set it by default.

## Known environment issue

This machine has ROS 2 Jazzy on `PYTHONPATH`, leaking Python 3.12 packages into
this 3.11 env. It injects 7 pytest plugins that crash on import.
`pyproject.toml` disables them by name; `scripts/test.sh` removes the cause with
`unset PYTHONPATH`, which is more robust. For anything else, run
`unset PYTHONPATH` in the shell first (or `env -u PYTHONPATH uv run ...`).
