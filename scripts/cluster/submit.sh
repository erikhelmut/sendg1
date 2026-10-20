#!/usr/bin/env bash
# Sync a checkout to the cluster and queue one training run there.
#
#   scripts/cluster/submit.sh [--sbatch "<sbatch flags>"] <checkout> <train args...>
#   scripts/cluster/submit.sh ~/sendg1-beam G1-Beam-E2E-Curved \
#     --agent.run-name blind_s0 --agent.seed 0
#
# <train args> go to mjlab's `train` unchanged; the job runs in the cluster
# worktree with its src on PYTHONPATH and writes logs/ there as it does here.
set -euo pipefail
SBATCH_FLAGS=""
if [ "${1:-}" = "--sbatch" ]; then SBATCH_FLAGS="$2"; shift 2; fi
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/common.sh"
resolve_worktree "$1"; shift
"$HERE/sync.sh" "$LOCAL_WT"
# The job script travels with each submit, so it is the one from this checkout
# of main, not whatever the worktree's branch happens to contain.
ssh "$CLUSTER_HOST" "mkdir -p ~/$WT_NAME/slurm"
rsync -a "$HERE/train.sbatch" "$CLUSTER_HOST:$WT_NAME/slurm/train.sbatch"
ssh "$CLUSTER_HOST" "cd ~/$WT_NAME && sbatch --job-name=$WT_NAME $SBATCH_FLAGS \
  slurm/train.sbatch $WT_NAME $(printf '%q ' "$@")"
