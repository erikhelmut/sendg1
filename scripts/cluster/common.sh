# Sourced by the cluster scripts. A local checkout /home/erik/<name> maps to
# ~/<name> on the cluster; every worktree shares the cluster's ~/sendg1/.venv
# and swaps in its own code with PYTHONPATH=<worktree>/src (as on this PC).
CLUSTER_HOST="${CLUSTER_HOST:-mn.ias.informatik.tu-darmstadt.de}"

# $1: a local checkout (default: the one this shell is in).
resolve_worktree() {
  LOCAL_WT="$(git -C "${1:-.}" rev-parse --show-toplevel)" || exit 1
  WT_NAME="$(basename "$LOCAL_WT")"
  BRANCH="$(git -C "$LOCAL_WT" branch --show-current)"
}
