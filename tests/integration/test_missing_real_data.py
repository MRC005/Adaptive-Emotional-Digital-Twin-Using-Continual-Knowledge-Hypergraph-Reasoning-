"""MANDATORY TEST D: if the real files are missing, the system must report

    REAL DATA UNAVAILABLE

and must NOT silently use synthetic data.

This is the single most important safety property in the project.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from aedt.constants import DataStatus
from aedt.errors import RealDataUnavailable
from aedt.io import get_adapter
from aedt.pipeline import run_pipeline

REAL = ["studentlife", "pmdata", "relax", "wesad"]
ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("name", REAL)
def test_audit_reports_unavailable_and_never_guesses(name):
    a = get_adapter(name).audit("/definitely/not/a/real/archive")
    assert a.local_files_available is False
    assert a.data_status is DataStatus.PLANNED
    assert "REAL DATA UNAVAILABLE" in a.source_status
    assert name.upper() in a.source_status
    # nothing is invented in place of the absent facts
    assert a.participant_count is None
    assert a.observation_count is None
    assert a.eligible_for_primary_analysis is None
    assert a.acquisition_instructions


@pytest.mark.parametrize("name", REAL)
def test_load_raises_rather_than_returning_synthetic_data(name):
    with pytest.raises(RealDataUnavailable, match="REAL DATA UNAVAILABLE"):
        get_adapter(name).load("/definitely/not/a/real/archive")


@pytest.mark.parametrize("name", REAL)
def test_pipeline_returns_no_frame_and_a_planned_status(name):
    res = run_pipeline(name, root="/definitely/not/a/real/archive")
    assert res.frame is None
    assert res.data_status is DataStatus.PLANNED
    assert res.blocking_reasons
    assert not res.validated
    assert res.primary is None


@pytest.mark.parametrize("name", REAL)
def test_strict_real_mode_raises_exit_code_6(name):
    with pytest.raises(RealDataUnavailable) as e:
        run_pipeline(name, root="/definitely/not/a/real/archive",
                     strict_real=True)
    assert e.value.exit_code == 6


def test_demo_prints_the_message_and_exits_6():
    r = subprocess.run(
        [sys.executable, "scripts/run_demo.py", "--dataset", "studentlife",
         "--root", "/definitely/not/a/real/archive"],
        capture_output=True, text=True, cwd=ROOT, timeout=300)
    assert r.returncode == 6, r.stdout[-2000:]
    assert "REAL DATA UNAVAILABLE - STUDENTLIFE AUDIT NOT RUN" in r.stdout
    assert "No synthetic substitute was used" in r.stdout
    # and it must not have quietly produced a result
    assert "PRIMARY" not in r.stdout.split("REAL DATA UNAVAILABLE")[-1].upper()


def test_a_fixture_directory_can_never_be_stamped_real(studentlife_fixture):
    """A dataset-shaped SYNTHETIC fixture must produce SYNTHETIC results even
    though the adapter's own load() reports REAL for a present archive."""
    res = run_pipeline("studentlife", root=str(studentlife_fixture),
                       n_resamples=99, build_twins=False,
                       halt_on_placebo_failure=False)
    assert res.data_status is DataStatus.SYNTHETIC
    assert not res.validated, "a fixture must never be called validated"
    assert res.audit.data_status is DataStatus.REAL, (
        "the adapter correctly reports that files were opened; it is the "
        "PIPELINE that must downgrade the result status for a fixture")


def test_studentlife_fixture_reproduces_the_documented_weak_association():
    """Round 16 ran this pipeline on a StudentLife-shaped fixture and found a
    weak sensor-report association: many participants excluded for an
    indeterminate slope or a sign flip, and a uselessly wide interval despite a
    perfectly calibrated estimator.

    That finding is the reason diagnostic [9b] exists and is read FIRST on real
    data. This test pins the behaviour so the lesson cannot be lost.
    """
    from aedt.io.fixtures import make_studentlife_fixture
    import tempfile
    root = make_studentlife_fixture(
        Path(tempfile.mkdtemp()) / "sl", n_participants=24, days=60)
    res = run_pipeline("studentlife", root=str(root), n_resamples=199,
                       build_twins=False, halt_on_placebo_failure=False)
    reasons = [r for e in res.eligibility for r in e.reasons]
    assert any("below the floor" in r for r in reasons), (
        "expected indeterminate-slope exclusions on the weak-association "
        "fixture")
    assert any("flips sign" in r for r in reasons), (
        "expected sign-flip exclusions on the weak-association fixture")
    assert res.n_eligible < res.frame["pid"].nunique()


def test_synthetic_dataset_is_never_stamped_real():
    res = run_pipeline("synthetic", n_resamples=99, build_twins=False)
    assert res.data_status is DataStatus.SYNTHETIC
    assert not res.validated
    assert res.primary.data_status is DataStatus.SYNTHETIC
    assert res.placebo.data_status is DataStatus.SYNTHETIC
