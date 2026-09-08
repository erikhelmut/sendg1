#!/usr/bin/env python
"""Print (or record) config fingerprints for every registered variant.

    python scripts/show_fingerprints.py            # show
    python scripts/show_fingerprints.py --write    # write into the test file
    python scripts/show_fingerprints.py --task <id> --dump   # canonical config
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TEST_FILE = REPO / "tests" / "test_config_fingerprint.py"


def main() -> None:
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

  import sendg1.tasks  # noqa: F401
  from sendg1.fingerprint import config_report, fingerprint
  from sendg1.tasks._register import variants

  ap = argparse.ArgumentParser()
  ap.add_argument("--task", default=None)
  ap.add_argument("--dump", action="store_true", help="print the canonical config")
  ap.add_argument("--write", action="store_true", help="record into the test file")
  args = ap.parse_args()

  if args.dump:
    if not args.task:
      raise SystemExit("--dump needs --task")
    print(config_report(load_env_cfg(args.task)))
    return

  rows = {
    t: (fingerprint(load_env_cfg(t)), fingerprint(load_rl_cfg(t)))
    for t in sorted(variants())
  }
  for task_id, (env_fp, rl_fp) in rows.items():
    print(f"{task_id:44s} env={env_fp} rl={rl_fp}")

  if args.write:
    body = "\n".join(
      f'  "{t}": ("{e}", "{r}"),' for t, (e, r) in rows.items()
    )
    src = TEST_FILE.read_text()
    new = re.sub(
      r"EXPECTED: dict\[str, tuple\[str, str\]\] = \{.*?\n\}",
      "EXPECTED: dict[str, tuple[str, str]] = {\n"
      "  # task_id: (env_fingerprint, rl_fingerprint)\n"
      f"{body}\n}}",
      src,
      flags=re.S,
    )
    TEST_FILE.write_text(new)
    print(f"\nrecorded {len(rows)} fingerprints into {TEST_FILE}")


if __name__ == "__main__":
  main()
