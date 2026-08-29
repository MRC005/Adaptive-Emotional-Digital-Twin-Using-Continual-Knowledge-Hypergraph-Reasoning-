#!/usr/bin/env python3
"""Review-2 demonstration.

    python scripts/run_demo.py --dataset synthetic
    python scripts/run_demo.py --dataset studentlife --root /path/to/archive

Exit codes:
    0  completed
    2  DECISION REQUIRED - a scientific decision is unspecified; nothing guessed
    3  no eligible participants
    4  placebo not runnable - the primary is blocked
    5  placebo FAILED - the primary is NOT run; that is the finding
    6  REAL DATA UNAVAILABLE - no synthetic substitute was used
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aedt.demo import main

if __name__ == "__main__":
    raise SystemExit(main())
