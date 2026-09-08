#!/usr/bin/env bash
# Run the test suite in a clean interpreter environment.
#
# This machine has ROS 2 Jazzy on PYTHONPATH, which leaks Python 3.12 packages
# (and 7 pytest plugins) into this 3.11 conda env. pyproject.toml disables the
# plugins by name; this script removes the cause instead, which is more robust.
set -euo pipefail
unset PYTHONPATH
exec python -m pytest "$@"
