#!/usr/bin/env bash
# Sync a checkout to the cluster and queue one training run there.
#
#   scripts/cluster/submit.sh [--sbatch "<sbatch flags>"] <checkout> <train args...>
#   scripts/cluster/submit.sh ~/sendg1 G1-Beam-Touch-Curved-Plain \
#     --agent.run-name touch_s1 --agent.seed 1
#
# <train args> go to mjlab's `train` unchanged; the job runs in the cluster
# worktree with its src on PYTHONPATH and writes logs/ there as it does here.
set -euo pipefail
SBATCH_FLAGS=""
if [ "${1:-}" = "--sbatch" ]; then SBATCH_FLAGS="$2"; shift 2; fi
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/common.sh"
resolve_worktree "$1"; shift
# Training seeds are 1-5 only (src/sendg1/seeds.py); without --agent.seed the
# task's default, seed 1, is used. Seed 0 is the evaluation seed.
prev=""
for a in "$@"; do
  seed=""
  case "$a" in --agent.seed=*) seed="${a#*=}" ;; esac
  [ "$prev" = --agent.seed ] && seed="$a"
  if [ -n "$seed" ]; then
    case "$seed" in 1|2|3|4|5) ;; *)
      echo "training seed $seed is not one of the study's seeds 1-5 (src/sendg1/seeds.py)" >&2
      exit 2 ;;
    esac
  fi
  prev="$a"
done
"$HERE/sync.sh" "$LOCAL_WT"
# The job script travels with each submit, so it is the one from this checkout
# of main, not whatever the worktree's branch happens to contain.
ssh "$CLUSTER_HOST" "mkdir -p ~/$WT_NAME/slurm"
rsync -a "$HERE/train.sbatch" "$CLUSTER_HOST:$WT_NAME/slurm/train.sbatch"
ssh "$CLUSTER_HOST" "cd ~/$WT_NAME && sbatch --job-name=$WT_NAME $SBATCH_FLAGS \
  slurm/train.sbatch $WT_NAME $(printf '%q ' "$@")"
