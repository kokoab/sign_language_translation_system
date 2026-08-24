#!/usr/bin/env python3
"""Compatibility entry point for active.v17.train_stage_1_v17."""

from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from active.v17.train_stage_1_v17 import *  # noqa: F401,F403
from active.v17.train_stage_1_v17 import main


if __name__ == "__main__":
    main()
