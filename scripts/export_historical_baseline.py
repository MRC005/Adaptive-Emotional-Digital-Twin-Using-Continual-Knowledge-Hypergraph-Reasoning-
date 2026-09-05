#!/usr/bin/env python3
"""Extract the ORIGINALLY REPORTED results from git, so they can be cited.

The first study reported a K=80 twin macro-F1 of 0.2847. Rerunning the
committed pipeline on the digest-verified archive gives 0.2793. Both numbers
are real and neither replaces the other: 0.2847 is what the study reported,
0.2793 is what the code reproduces. The website and the README must be able to
show both, and neither may be typed by hand.

This script reads the historical block straight out of the git object that
published it -- ``38f8785:frontend/src/data/findings.json`` -- and writes
``frontend/src/data/historical_findings.json``. Its provenance is therefore a
commit hash, and ``tests/regression/test_historical_baseline.py`` verifies the
file still matches that object byte for byte.

    python3 scripts/export_historical_baseline.py

Exit codes:
    0  wrote frontend/src/data/historical_findings.json
    6  the git object is unreachable (shallow clone, or not a repository)
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: The commit that published the original results to the site. Chosen because
#: it is the LAST commit in which findings.json carried hand-entered values;
#: everything after it is generated.
HISTORICAL_COMMIT = "38f8785"
HISTORICAL_PATH = "frontend/src/data/findings.json"

OUT = ROOT / "frontend" / "src" / "data" / "historical_findings.json"


def read_git_object(commit: str, path: str) -> str | None:
    try:
        r = subprocess.run(["git", "show", f"{commit}:{path}"],
                           capture_output=True, text=True, timeout=30, cwd=ROOT)
    except Exception:
        return None
    return r.stdout if r.returncode == 0 else None


def main() -> int:
    raw = read_git_object(HISTORICAL_COMMIT, HISTORICAL_PATH)
    if raw is None:
        print(f"cannot read {HISTORICAL_COMMIT}:{HISTORICAL_PATH} -- is this a "
              "full clone of the repository?", file=sys.stderr)
        return 6

    historical = json.loads(raw)
    payload = {
        "_what_this_is": (
            "The results EXACTLY AS ORIGINALLY REPORTED by the first study. "
            "These are the published historical figures. They are NOT "
            "reproducible from the current committed pipeline and must never "
            "be presented as if they were, nor overwritten with the "
            "regenerated values."),
        "_generated_by": "scripts/export_historical_baseline.py",
        "_provenance": {
            "source": f"git object {HISTORICAL_COMMIT}:{HISTORICAL_PATH}",
            "commit": HISTORICAL_COMMIT,
            "verified_by": "tests/regression/test_historical_baseline.py",
            "status": "HISTORICAL REPORTED - not reproducible",
            "see": "docs/statistic_provenance.md",
        },
        "cohort": historical.get("cohort"),
        "headline": historical.get("headline"),
        "learning_curve": historical.get("learning_curve"),
        "ablation": historical.get("ablation"),
        "ceiling": historical.get("ceiling"),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    twin = payload["headline"]["models"]["T_twin"]["macro_f1"]
    print(f"wrote {OUT}")
    print(f"  historical K=80 twin macro-F1: {twin:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
