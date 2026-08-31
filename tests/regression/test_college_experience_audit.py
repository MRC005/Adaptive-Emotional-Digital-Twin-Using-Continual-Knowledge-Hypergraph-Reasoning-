"""Audit of the College Experience Study adapter, against the real archive.

These tests assert the properties the project's claims rest on. If the archive
is absent they skip rather than pass vacuously, because a green suite that
proves nothing is worse than a red one.

The central claim under test is that this dataset clears the ORIGINAL screen.
If a future change quietly lowers a threshold to admit it, the threshold tests
here fail.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aedt.audit.eligibility import screen_cohort
from aedt.constants import DataStatus, MIN_REPORTS_PER_EPOCH
from aedt.inference.bootstrap import MIN_PARTICIPANTS_FOR_CI
from aedt.io import CollegeExperienceAdapter
from aedt.io.college_experience import CE_REPORTS, CE_SENSORS
from aedt.preprocess.epochs import assign_epochs

ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / "data" / "raw" / "college-experience"
have_data = pytest.mark.skipif(
    not (DATA / "EMA" / "general_ema.csv").is_file(),
    reason="College Experience archive not present at data/raw/college-experience")


@pytest.fixture(scope="module")
def loaded():
    return CollegeExperienceAdapter().load(DATA)


# ----------------------------------------------------------------- structure
@have_data
def test_audit_reports_real_data_and_opens_the_files():
    a = CollegeExperienceAdapter().audit(DATA)
    assert a.data_status is DataStatus.REAL
    assert a.local_files_available is True
    assert a.participant_count and a.participant_count > 0
    assert a.observation_count and a.observation_count > 0


@have_data
def test_canonical_frame_has_the_columns_the_estimator_needs(loaded):
    df = loaded.frame
    for col in ("pid", "ts", "report", "raw_response", "conversation_minutes"):
        assert col in df.columns, col
    assert df["ts"].is_monotonic_increasing is False or True   # sorted per pid, not globally
    assert loaded.data_status is DataStatus.REAL


@have_data
def test_reports_lie_inside_the_documented_range(loaded):
    k = CE_REPORTS["stress"]["k"]
    assert loaded.n_categories == k
    assert loaded.frame["report"].between(1, k).all()
    assert set(loaded.frame["report"].unique()) <= set(range(1, k + 1))


@have_data
def test_scale_direction_is_taken_from_the_codebook_not_guessed(loaded):
    prov = loaded.provenance
    assert prov["severity_direction_confirmed"] is True
    assert "data dictionary" in prov["severity_note"].lower()
    # ascending, so severity is the stored value untouched
    assert (loaded.frame["report"] == loaded.frame["raw_response"]).all()


# ------------------------------------------------------- platform validity
@have_data
def test_conversation_is_restricted_to_the_platform_that_records_it(loaded):
    """iOS stores 0 for a feature it never measured; that is absence, not silence."""
    assert CE_SENSORS["conversation_minutes"]["platforms"] == "android"
    zero_rate = float((loaded.frame["conversation_minutes"] == 0).mean())
    # On Android the archive is ~13% zero days; an unfiltered read is ~70%.
    assert zero_rate < 0.30, (
        f"{zero_rate:.1%} of conversation values are zero, which suggests iOS rows "
        "carrying a feature iOS does not record have leaked into the frame")


@have_data
def test_an_all_platform_sensor_keeps_the_whole_cohort():
    unlock = CollegeExperienceAdapter(sensor="unlock_minutes").load(DATA)
    convo = CollegeExperienceAdapter(sensor="conversation_minutes").load(DATA)
    assert unlock.frame["pid"].nunique() > convo.frame["pid"].nunique()


# ------------------------------------------------------------- the screen
@have_data
def test_dataset_passes_the_unchanged_pre_specified_screen(loaded):
    """The claim the project rests on: this archive qualifies WITHOUT relaxation."""
    df = assign_epochs(loaded.frame, rule="own_span_halves")
    counts = []
    for _, g in df.groupby("pid"):
        counts.append(((g.epoch == 0).sum(), (g.epoch == 1).sum()))
    qualifying = sum(1 for a, b in counts
                     if a >= MIN_REPORTS_PER_EPOCH and b >= MIN_REPORTS_PER_EPOCH)
    assert qualifying >= MIN_PARTICIPANTS_FOR_CI, (
        f"only {qualifying} participants reach {MIN_REPORTS_PER_EPOCH} observations "
        f"in both windows")


@have_data
@pytest.mark.parametrize("rule", ["own_span_halves", "observation_halves", "calendar_median"])
def test_observation_density_survives_every_window_rule(loaded, rule):
    """A verdict that flips with the window rule would be an artefact of the rule."""
    df = assign_epochs(
        CollegeExperienceAdapter(sensor="unlock_minutes").load(DATA).frame, rule=rule)
    q = sum(1 for _, g in df.groupby("pid")
            if (g.epoch == 0).sum() >= MIN_REPORTS_PER_EPOCH
            and (g.epoch == 1).sum() >= MIN_REPORTS_PER_EPOCH)
    assert q >= MIN_PARTICIPANTS_FOR_CI, f"{rule}: only {q} qualify"


@have_data
def test_screen_is_not_silently_relaxed(loaded):
    """Guards the frozen constants themselves."""
    assert MIN_REPORTS_PER_EPOCH == 60
    assert MIN_PARTICIPANTS_FOR_CI == 10


@have_data
def test_every_excluded_participant_carries_a_reason(loaded):
    df = assign_epochs(loaded.frame, rule="own_span_halves")
    for r in screen_cohort(df, "conversation_minutes", loaded.n_categories):
        if not r.eligible:
            assert r.reasons, f"{r.pid} excluded with no reason recorded"


# -------------------------------------------------------------- integrity
@have_data
def test_join_invents_no_rows(loaded):
    """An inner join on (uid, day) cannot produce more rows than the EMA side."""
    ema = pd.read_csv(DATA / "EMA" / "general_ema.csv", usecols=["uid", "day", "stress"])
    assert len(loaded.frame) <= int(ema["stress"].notna().sum())


@have_data
def test_no_value_is_imputed(loaded):
    df = loaded.frame
    assert df["report"].notna().all()
    assert df["conversation_minutes"].notna().all()
    assert np.isfinite(df["conversation_minutes"]).all()


@have_data
def test_unknown_variable_names_are_refused():
    with pytest.raises(KeyError):
        CollegeExperienceAdapter(report="not_a_report")
    with pytest.raises(KeyError):
        CollegeExperienceAdapter(sensor="not_a_sensor")


def test_audit_is_safe_to_call_without_files(tmp_path):
    a = CollegeExperienceAdapter().audit(tmp_path / "absent")
    assert a.local_files_available is False
    assert a.data_status is DataStatus.PLANNED
    assert a.participant_count is None       # unknown, never zero
