"""Leakage tests for the personalised prediction task.

These are the tests that decide whether any result from this experiment means
anything. Each one encodes a way the experiment could silently cheat.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aedt.twin.prediction_data import (MAX_GAP_DAYS, assert_no_leakage,
                                       build_prediction_frame)

ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / "data" / "raw" / "college-experience"
have_data = pytest.mark.skipif(
    not (DATA / "EMA" / "general_ema.csv").is_file(),
    reason="College Experience archive not present")


@pytest.fixture(scope="module")
def pd_data():
    return build_prediction_frame(DATA)


# --------------------------------------------------------------- temporal
@have_data
def test_temporal_contract_holds_for_every_row(pd_data):
    assert_no_leakage(pd_data.frame)


@have_data
def test_target_is_strictly_in_the_future(pd_data):
    f = pd_data.frame
    assert (f["target_time"] > f["prediction_time"]).all()


@have_data
def test_features_are_never_timestamped_after_the_prediction(pd_data):
    f = pd_data.frame
    assert (f["feature_time"] <= f["prediction_time"]).all()


@have_data
def test_gap_respects_the_pre_registered_horizon(pd_data):
    gap = (pd_data.frame["target_time"] - pd_data.frame["prediction_time"]).dt.days
    assert gap.min() > 0
    assert gap.max() <= MAX_GAP_DAYS


@have_data
def test_history_features_never_contain_the_target(pd_data):
    """prev_1 is the value BEFORE the current one, not the one being predicted."""
    f = pd_data.frame
    both = f[["prev_1", "target"]].dropna()
    # they may coincide by chance; they must not coincide systematically
    assert (both["prev_1"] == both["target"]).mean() < 0.6


@have_data
def test_history_is_causal_within_each_participant(pd_data):
    """hist_n must be non-decreasing in time for a participant."""
    f = pd_data.frame.sort_values(["participant_id", "prediction_time"])
    for pid, g in f.groupby("participant_id"):
        h = g["hist_n"].dropna().to_numpy()
        assert (np.diff(h) >= 0).all(), f"{pid}: history count went backwards"


@have_data
def test_participant_identity_is_not_a_feature(pd_data):
    assert "participant_id" not in pd_data.feature_columns
    for c in pd_data.feature_columns:
        assert "uid" not in c and "id" != c


# ------------------------------------------------------------- the splits
@have_data
def test_participant_splits_are_disjoint():
    from scripts.run_twin_experiment import make_splits
    tr, va, te = make_splits(build_prediction_frame(DATA), seed=20260828)
    assert set(tr) & set(va) == set()
    assert set(tr) & set(te) == set()
    assert set(va) & set(te) == set()
    assert len(tr) + len(va) + len(te) == 217


@have_data
def test_warmup_and_evaluation_rows_never_overlap():
    from scripts.run_twin_experiment import split_warmup
    d = build_prediction_frame(DATA)
    g = d.frame[d.frame.participant_id == d.frame.participant_id.iloc[0]]
    warm, ev = split_warmup(g, K=20)
    assert set(warm.index) & set(ev.index) == set()
    # and warm-up must be strictly earlier
    if len(warm) and len(ev):
        assert warm["prediction_time"].max() < ev["prediction_time"].min()


@have_data
def test_evaluation_rows_come_after_the_warmup_cutoff():
    from scripts.run_twin_experiment import split_warmup
    d = build_prediction_frame(DATA)
    for pid, g in list(d.frame.groupby("participant_id"))[:20]:
        warm, ev = split_warmup(g, K=10)
        if len(warm) and len(ev):
            assert (ev["prediction_time"] > warm["prediction_time"].max()).all()


# ------------------------------------------------- synthetic leakage probe
def test_a_deliberately_leaky_frame_is_rejected():
    """The detector must actually fire; a check that never fails is worthless."""
    bad = pd.DataFrame({
        "participant_id": ["a", "a"],
        "feature_time": pd.to_datetime(["2026-01-02", "2026-01-03"]),
        "prediction_time": pd.to_datetime(["2026-01-01", "2026-01-02"]),
        "target_time": pd.to_datetime(["2026-01-03", "2026-01-04"]),
        "target": [1, 2],
    })
    with pytest.raises(AssertionError, match="after its prediction"):
        assert_no_leakage(bad)


def test_a_target_in_the_past_is_rejected():
    bad = pd.DataFrame({
        "participant_id": ["a"],
        "feature_time": pd.to_datetime(["2026-01-01"]),
        "prediction_time": pd.to_datetime(["2026-01-05"]),
        "target_time": pd.to_datetime(["2026-01-03"]),
        "target": [1],
    })
    with pytest.raises(AssertionError):
        assert_no_leakage(bad)


def test_an_over_horizon_gap_is_rejected():
    bad = pd.DataFrame({
        "participant_id": ["a"],
        "feature_time": pd.to_datetime(["2026-01-01"]),
        "prediction_time": pd.to_datetime(["2026-01-01"]),
        "target_time": pd.to_datetime(["2026-02-01"]),
        "target": [1],
    })
    with pytest.raises(AssertionError, match="gap"):
        assert_no_leakage(bad)
