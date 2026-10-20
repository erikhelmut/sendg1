#!/usr/bin/env bash
# Copy a local checkout's code to the cluster, uncommitted edits included.
#
#   scripts/cluster/sync.sh [~/sendg1-beam]
#
# The cluster copy is a mirror, never edited there. The branch goes to
# refs/remotes/hamming/<branch> (a push into a checked-out branch would be
# refused once the mirror holds uncommitted edits); the cluster worktree is
# created from it on first use and its HEAD moved to it on every sync, so
# `git status` there shows exactly the local uncommitted diff. Files are
# rsynced from `git ls-files`, minus training output, so checkpoints, logs and
# videos never travel. Files deleted locally stay on the cluster.
set -euo pipefail
source "$(dirname "$0")/common.sh"
resolve_worktree "${1:-}"

git -C "$LOCAL_WT" push -q -f cluster "$BRANCH:refs/remotes/hamming/$BRANCH"
ssh "$CLUSTER_HOST" bash -s -- "$WT_NAME" "$BRANCH" <<'REMOTE'
set -e
if [ ! -d ~/"$1" ]; then
  git -C ~/sendg1 worktree add -q -B "$2" ~/"$1" "hamming/$2"
else
  git -C ~/"$1" reset -q "hamming/$2"
fi
REMOTE
git -C "$LOCAL_WT" ls-files -co --exclude-standard -z \
  | grep -zvE '^(logs|runs|slurm|wandb|media|videos)/|\.(pt|mp4|log)$|MUJOCO_LOG' \
  | rsync -a --from0 --files-from=- "$LOCAL_WT/" "$CLUSTER_HOST:$WT_NAME/"
echo "[sync] $LOCAL_WT ($BRANCH) -> $CLUSTER_HOST:~/$WT_NAME"
