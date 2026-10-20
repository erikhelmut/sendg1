#!/usr/bin/env bash
# Copy a checkout's training output back from the cluster: logs/, runs/,
# results/ and the slurm/ job logs. Only files newer than the local copy are
# taken; videos never exist there, and are rendered here afterwards.
#
#   scripts/cluster/fetch.sh [~/sendg1-beam]
set -euo pipefail
source "$(dirname "$0")/common.sh"
resolve_worktree "${1:-}"
for d in logs runs results slurm; do
  if ssh "$CLUSTER_HOST" "test -d ~/$WT_NAME/$d"; then
    rsync -a --update --exclude='*.mp4' --exclude='videos/' \
      "$CLUSTER_HOST:$WT_NAME/$d/" "$LOCAL_WT/$d/"
  fi
done
echo "[fetch] $CLUSTER_HOST:~/$WT_NAME -> $LOCAL_WT"
