"""Every new experiment must be able to explain itself afterwards.

The first study's K=80 twin figure is unexplainable today because its run
recorded nothing about itself, while the machinery to record it already existed
and was not wired in. These tests pin the pieces that make that impossible to
repeat: the version list must cover the packages that can move a number, the
threading must be captured, and the record must carry the dataset's own
digests rather than a restatement of them.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import pytest

from aedt.reporting.experiment_record import (build_experiment_record,
                                              dataset_provenance,
                                              write_experiment_record)
from aedt.reporting.metadata import package_versions, thread_environment

ROOT = Path(__file__).resolve().parents[2]
PINS = ROOT / "requirements-experiment.txt"
FROZEN = ROOT / "requirements.txt"

#: Packages whose version can change a fitted model or a reported number.
#: scikit-learn is the one whose absence made the first investigation hard.
MUST_TRACK = ("numpy", "scipy", "pandas", "scikit-learn", "joblib",
              "threadpoolctl")


def test_version_capture_covers_every_package_that_can_move_a_number():
    v = package_versions()
    missing = [p for p in MUST_TRACK if p not in v]
    assert not missing, (
        f"package_versions() does not record {missing}. scikit-learn was "
        "absent from this list until 2026-09-05, which is exactly why the "
        "first study's environment could not be reconstructed.")
    for name in MUST_TRACK:
        assert v[name] != "not installed", f"{name} is required and absent"


def test_versions_are_reported_under_their_install_names():
    v = package_versions()
    assert "scikit-learn" in v and "sklearn" not in v, (
        "report the name used to install the package, so the record can be "
        "acted on directly")


def test_thread_environment_is_captured():
    t = thread_environment()
    assert "env" in t and "cpu_count" in t and "threadpools" in t
    assert "OMP_NUM_THREADS" in t["env"], (
        "the OpenMP thread count must be recorded: a gradient-boosted fit can "
        "differ between thread counts")
    assert isinstance(t["cpu_count"], int) and t["cpu_count"] >= 1


def test_dataset_provenance_carries_the_recorded_digests():
    prov = dataset_provenance(ROOT / "data" / "raw" / "college-experience")
    assert prov["digests"], "the dataset's recorded digests are missing"
    assert "EMA/general_ema.csv" in prov["digests"]
    assert prov["doi"], "the DOI must travel with the result"


def test_dataset_provenance_reports_a_missing_record_rather_than_inventing_one(
        tmp_path):
    prov = dataset_provenance(tmp_path)
    assert prov["digests"] is None
    assert "NO PROVENANCE" in prov["status"]


def test_a_written_record_carries_everything_the_protocol_requires(tmp_path):
    p = write_experiment_record(
        tmp_path, experiment="unit-test", dataset="college-experience",
        seed=20260828, config={"K": 20, "routers": ["R0", "R4", "R6"]},
        started=time.time() - 1.0,
        data_root=ROOT / "data" / "raw" / "college-experience")
    r = json.loads(p.read_text(encoding="utf-8"))

    for key in ("experiment", "git_commit", "python_version",
                "package_versions", "thread_environment",
                "dataset_provenance", "seed", "config", "platform",
                "python_executable", "elapsed_seconds"):
        assert key in r, f"{key} missing from the run record"

    assert r["seed"] == 20260828
    assert r["config"]["K"] == 20
    assert r["run"]["seed"] == 20260828, (
        "the seed must reach the RunMetadata block too, not only the wrapper")
    for name in MUST_TRACK:
        assert name in r["package_versions"]


def test_the_seed_override_reaches_run_metadata_without_a_config():
    """The frozen scripts carry their own seed; a Config is not always present."""
    rec = build_experiment_record(
        experiment="x", dataset="d", seed=12345, config={}, started=time.time())
    assert rec["run"]["seed"] == 12345


# ------------------------------------------------------------------- pinning
def test_the_experiment_requirements_pin_exact_versions():
    assert PINS.exists(), "requirements-experiment.txt is missing"
    lines = [ln.strip() for ln in PINS.read_text(encoding="utf-8").splitlines()
             if ln.strip() and not ln.strip().startswith("#")]
    assert lines, "no pins found"
    for ln in lines:
        assert re.fullmatch(r"[A-Za-z0-9_.\-]+==[0-9][^\s]*", ln), (
            f"'{ln}' is not an exact pin; this file exists precisely because "
            "lower bounds were not enough")
    pinned = {ln.split("==")[0] for ln in lines}
    missing = [p for p in MUST_TRACK if p not in pinned]
    assert not missing, f"unpinned but able to move a number: {missing}"


def test_the_pins_match_the_environment_actually_running():
    lines = [ln.strip() for ln in PINS.read_text(encoding="utf-8").splitlines()
             if ln.strip() and not ln.strip().startswith("#")]
    pinned = dict(ln.split("==", 1) for ln in lines)
    live = package_versions()
    for name in MUST_TRACK:
        assert pinned[name] == live[name], (
            f"{name}: pinned {pinned[name]}, running {live[name]}. Either the "
            "environment drifted or the pins are stale; both are reportable.")


def test_the_frozen_requirements_file_is_untouched():
    """requirements.txt is part of the historical record."""
    text = FROZEN.read_text(encoding="utf-8")
    assert "scikit-learn>=1.3" in text, (
        "requirements.txt has been edited; it pins lower bounds by historical "
        "fact and must stay as it was")
    assert "==" not in text, "exact pins belong in requirements-experiment.txt"
