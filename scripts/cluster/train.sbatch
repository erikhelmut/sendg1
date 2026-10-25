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
# A requeued job (preempted under --qos=scavenger) resumes from the newest
# checkpoint of its --agent.run-name, in a new run directory, for the
# iterations that are left of --agent.max-iterations (default 20000).
if [ "${SLURM_RESTART_COUNT:-0}" -gt 0 ]; then
  run="" total=20000 prev="" keep=()
  for a in "$@"; do
    case "$prev" in
      --agent.run-name) run="$a" ;;
      --agent.max-iterations) total="$a" ;;
    esac
    # The original --agent.max-iterations is replaced by what is left.
    [ "$a" = --agent.max-iterations ] || [ "$prev" = --agent.max-iterations ] || keep+=("$a")
    prev="$a"
  done
  [ -n "$run" ] || { echo "[job] requeued without --agent.run-name: cannot resume" >&2; exit 3; }
  ck=$(ls -d logs/rsl_rl/*/*_"$run"/model_*.pt 2>/dev/null \
    | awk -F'model_' '{split($NF, n, "."); print n[1], $0}' | sort -n | tail -1 | cut -d" " -f2 || true)
  if [ -n "$ck" ]; then
    done_it=$(basename "$ck" .pt | sed 's/model_//')
    left=$(( total - done_it ))
    [ "$left" -gt 0 ] || { echo "[job] $run already at $done_it/$total"; exit 0; }
    rundir=$(basename "$(dirname "$ck")")
    echo "[job] restart $SLURM_RESTART_COUNT: resuming $rundir/$(basename "$ck"), $left iterations left"
    set -- "${keep[@]}" --agent.resume True --agent.load-run "^${rundir}\$" \
      --agent.load-checkpoint "^$(basename "$ck")\$" --agent.max-iterations "$left"
  fi
fi
exec ~/sendg1/.venv/bin/train "$@"
