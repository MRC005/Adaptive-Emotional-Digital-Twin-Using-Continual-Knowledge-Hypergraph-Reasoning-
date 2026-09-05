"""The historical figures must stay historical, and stay traceable to git.

Two numbers coexist for the same quantity. 0.2847 is what the first study
reported; 0.2793 is what the committed pipeline reproduces on the
digest-verified archive. Keeping both honest needs two guarantees:

  1. the historical file is a faithful copy of the git object that published
     it, not a hand-typed remembering of it;
  2. nothing has quietly replaced the historical value with the regenerated
     one, or vice versa.

Both are asserted here. The first skips only when git history is unavailable
(a shallow clone), and says so rather than passing quietly.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
HISTORICAL = ROOT / "frontend" / "src" / "data" / "historical_findings.json"
REGENERATED = ROOT / "frontend" / "src" / "data" / "findings.json"

HISTORICAL_COMMIT = "38f8785"
HISTORICAL_PATH = "frontend/src/data/findings.json"

#: The two values that must never be conflated, at the precision they are cited.
HISTORICAL_TWIN_K80 = 0.2847
REGENERATED_TWIN_K80 = 0.2793


def _git_object(commit: str, path: str) -> str | None:
    try:
        r = subprocess.run(["git", "show", f"{commit}:{path}"],
                           capture_output=True, text=True, timeout=30, cwd=ROOT)
    except Exception:
        return None
    return r.stdout if r.returncode == 0 else None


def test_the_historical_file_matches_the_git_object_that_published_it():
    raw = _git_object(HISTORICAL_COMMIT, HISTORICAL_PATH)
    if raw is None:
        pytest.skip(
            f"git object {HISTORICAL_COMMIT}:{HISTORICAL_PATH} is unreachable, "
            "so the historical figures CANNOT be verified against their source. "
            "A full clone is required. Re-run "
            "scripts/export_historical_baseline.py once history is available.")
    published = json.loads(raw)
    stored = json.loads(HISTORICAL.read_text(encoding="utf-8"))
    for block in ("cohort", "headline", "learning_curve", "ablation", "ceiling"):
        assert stored[block] == published[block], (
            f"historical_findings.json::{block} no longer matches "
            f"{HISTORICAL_COMMIT}. The historical record has been edited; it is "
            "a transcript of a commit and must not be changed.")


def test_the_historical_twin_value_is_still_the_historical_one():
    d = json.loads(HISTORICAL.read_text(encoding="utf-8"))
    got = d["headline"]["models"]["T_twin"]["macro_f1"]
    assert round(got, 4) == HISTORICAL_TWIN_K80, (
        f"the historical K=80 twin figure reads {got!r}. It must remain "
        f"{HISTORICAL_TWIN_K80} — the value the study actually reported. "
        "Overwriting it with the regenerated figure would rewrite the record.")


def test_the_regenerated_twin_value_is_not_the_historical_one():
    d = json.loads(REGENERATED.read_text(encoding="utf-8"))
    got = d["headline"]["models"]["T_twin"]["macro_f1"]
    assert round(got, 4) == REGENERATED_TWIN_K80, (
        f"the regenerated K=80 twin figure reads {got!r}, expected "
        f"{REGENERATED_TWIN_K80}. Either the pipeline changed, or the "
        "historical value has been copied over the generated one.")
    assert round(got, 4) != HISTORICAL_TWIN_K80, (
        "the regenerated file now carries the historical value; the two "
        "records have been conflated")


def test_the_historical_file_says_it_is_not_reproducible():
    d = json.loads(HISTORICAL.read_text(encoding="utf-8"))
    prov = d["_provenance"]
    assert "not reproducible" in prov["status"].lower(), (
        "the historical file must state that its figures are not reproducible "
        "from the committed pipeline")
    assert prov["commit"] == HISTORICAL_COMMIT
    assert "statistic_provenance" in prov["see"]


def test_the_two_files_disagree_where_they_are_expected_to():
    """A guard against a well-meaning 'fix' that silently unifies them."""
    h = json.loads(HISTORICAL.read_text(encoding="utf-8"))
    r = json.loads(REGENERATED.read_text(encoding="utf-8"))
    hm, rm = h["headline"]["models"], r["headline"]["models"]

    # the five baselines reproduce exactly -- that is the whole reason the twin
    # discrepancy is attributable to the twin's own path
    for name in ("B0_majority", "B1_persistence", "B2_global",
                 "B3_static_prior", "B4_calibrated"):
        assert round(hm[name]["macro_f1"], 4) == round(rm[name]["macro_f1"], 4), (
            f"{name} no longer reproduces; the discrepancy is no longer "
            "confined to the twin and docs/statistic_provenance.md is stale")

    assert round(hm["T_twin"]["macro_f1"], 4) != round(rm["T_twin"]["macro_f1"], 4)
