# Cluster

Training runs on the IAS cluster (`mn.ias.informatik.tu-darmstadt.de`, user
`helmut`, Slurm). Everything else stays on the workstation: videos are never
rendered on the cluster, and results come back here before anyone looks at them.

## Layout

The cluster mirrors the workstation's checkouts, one directory per worktree:

| workstation | cluster |
|---|---|
| `~/sendg1` (main) | `~/sendg1`, which also holds the only `.venv` |
| `~/sendg1-<name>` (worktree) | `~/sendg1-<name>` (a git worktree of `~/sendg1`) |

Every worktree shares `~/sendg1/.venv` (`uv sync --frozen --extra dev`, built
from main's `uv.lock`) and runs its own code with `PYTHONPATH=<worktree>/src`,
as on the workstation. The mirrors are never edited on the cluster;
`sync.sh` overwrites them.

## Workflow

```bash
scripts/cluster/sync.sh ~/sendg1-beam          # code, uncommitted edits included
scripts/cluster/submit.sh ~/sendg1-beam G1-Beam-E2E-Curved \
  --agent.run-name blind_s0 --agent.seed 0       # sync + sbatch one `train` run
ssh mn.ias.informatik.tu-darmstadt.de squeue --me
scripts/cluster/fetch.sh ~/sendg1-beam         # logs/, runs/, results/, slurm/
```

Then evaluate and render videos here, from the fetched checkpoints.

- `sync.sh` pushes the branch to `refs/remotes/hamming/<branch>` on the
  cluster, creates the worktree on first use, resets its HEAD to that ref, and
  rsyncs the files `git ls-files` lists, minus `logs/`, `runs/`, checkpoints and
  videos. `git status` on the cluster then shows the same uncommitted diff as
  here. Files deleted locally are not deleted there.
- `submit.sh` passes its arguments to mjlab's `train` unchanged. Default
  resources (`train.sbatch`): one GPU with at least 24 GB (RTX 3090, A5000, A6000,
  RTX 6000 Ada), 8 CPUs, 32 GB, 24 h; the V100 node is excluded because the
  cu128 torch build does not support it. Override with
  `--sbatch "--time=3-00:00:00"`. A `--video` argument is refused.
- `fetch.sh` copies only files newer than the local copy and skips `*.mp4`.
- wandb is logged in on the cluster; runs land in the same project.
