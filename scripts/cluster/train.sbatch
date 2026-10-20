#!/usr/bin/env bash
# Runs on a cluster node; submitted by scripts/cluster/submit.sh.
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --constraint=vram24gb
# The V100s (sm_70) are not in the cu128 torch build.
#SBATCH --exclude=dgx-station
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm/%x-%j.out
set -euo pipefail
WT="$1"; shift
# Training only: videos are rendered on the workstation from fetched
# checkpoints (docs/cluster.md), so recording during training is refused.
for a in "$@"; do
  case "$a" in --video*) echo "no videos on the cluster" >&2; exit 2 ;; esac
done
cd ~/"$WT"
unset PYTHONPATH
export PYTHONPATH="$HOME/$WT/src"
echo "[job] $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader) wt=$WT"
exec ~/sendg1/.venv/bin/train "$@"
