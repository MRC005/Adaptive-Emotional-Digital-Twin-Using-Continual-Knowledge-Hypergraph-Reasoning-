"""MANDATORY FUTURE-LEAKAGE TESTS (test C).

Insert future information; the causal pipeline must REJECT it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aedt.alignment.align import (CausalWindow, align_sensor_to_reports,
                                  assert_no_leakage)
from aedt.constants import DataStatus
from aedt.errors import DecisionRequired
from aedt.knowledge.store import ContinualKnowledgeStore
from aedt.schemas import SensorWindow
from aedt.twin.state import new_twin
from aedt.twin.update import observe


def _reports():
    return pd.DataFrame({"pid": ["a", "a", "a"],
                         "ts": pd.to_datetime(["2026-01-02 12:00",
                                               "2026-01-03 12:00",
                                               "2026-01-04 12:00"])})


def test_alignment_excludes_samples_after_the_report():
    samples = pd.DataFrame({
        "pid": ["a"] * 4,
        "ts": pd.to_datetime(["2026-01-02 08:00",    # before report 1  -> used
                              "2026-01-02 18:00",    # AFTER report 1   -> not
                              "2026-01-03 09:00",    # before report 2  -> used
                              "2026-01-03 11:00"]),  # before report 2  -> used
        "v": [1.0, 999.0, 3.0, 4.0]})
    out = align_sensor_to_reports(_reports(), samples, "v",
                                  CausalWindow(lookback_hours=24))
    assert out.loc[0, "v"] == 1.0, "a future sample leaked into report 1"
    assert 999.0 not in set(out["v"].dropna())


def test_planted_future_window_is_rejected_hard():
    out = align_sensor_to_reports(
        _reports(),
        pd.DataFrame({"pid": ["a"], "ts": pd.to_datetime(["2026-01-02 06:00"]),
                      "v": [1.0]}),
        "v", CausalWindow(lookback_hours=24))
    # plant a window that ends AFTER the report it feeds
    out.loc[0, "v_window_end"] = out.loc[0, "ts"] + pd.Timedelta(hours=1)
    with pytest.raises(DecisionRequired, match="FUTURE LEAKAGE"):
        assert_no_leakage(out, "v")


def test_lag_pushes_the_window_further_into_the_past():
    samples = pd.DataFrame({
        "pid": ["a"] * 2,
        "ts": pd.to_datetime(["2026-01-03 11:00", "2026-01-03 02:00"]),
        "v": [50.0, 7.0]})
    out = align_sensor_to_reports(_reports(), samples, "v",
                                  CausalWindow(lookback_hours=24, lag_hours=6))
    # the 11:00 sample is inside the 6-hour exclusion gap before the 12:00 report
    assert out.loc[1, "v"] == 7.0


def test_sensor_window_schema_refuses_a_future_window():
    with pytest.raises(DecisionRequired, match="Future leakage"):
        SensorWindow(pid="a", ts=pd.Timestamp("2026-01-02 12:00"),
                     window_start=pd.Timestamp("2026-01-02 11:00"),
                     window_end=pd.Timestamp("2026-01-02 13:00"),
                     features={"v": 1.0})


def test_twin_refuses_an_observation_dated_in_its_past():
    twin = new_twin("a", "synthetic", DataStatus.SYNTHETIC)
    later = pd.Series({"ts": pd.Timestamp("2026-02-01"), "report": 3,
                       "epoch": 0, "conversation_minutes": 40.0})
    earlier = pd.Series({"ts": pd.Timestamp("2026-01-01"), "report": 2,
                         "epoch": 0, "conversation_minutes": 60.0})
    observe(twin, later, sensor="conversation_minutes", n_categories=5)
    with pytest.raises(DecisionRequired, match="Temporal leakage"):
        observe(twin, earlier, sensor="conversation_minutes", n_categories=5)


def test_knowledge_store_refuses_a_back_dated_node():
    store = ContinualKnowledgeStore(pid="a")
    store.append("state_history", {"x": 1},
                 valid_from=pd.Timestamp("2026-02-01"), provenance="t")
    with pytest.raises(DecisionRequired, match="Causal violation"):
        store.append("state_history", {"x": 0},
                     valid_from=pd.Timestamp("2026-01-01"), provenance="t")


def test_context_discretisation_uses_epoch1_cutpoints_only(small_frame):
    """Pooled quantiles would leak epoch-2 structure into the definition of
    'the same situation' and partly absorb the drift under test."""
    from aedt.contexts.discretise import discretise_epoch1
    pid = small_frame["pid"].iloc[0]
    g = small_frame[small_frame["pid"] == pid].copy()
    before = discretise_epoch1(g, ["conversation_minutes"])
    e0 = before["epoch"] == 0

    tampered = g.copy()
    m = tampered["epoch"] == 1
    tampered.loc[m, "conversation_minutes"] += 1000.0
    after = discretise_epoch1(tampered, ["conversation_minutes"])

    assert (before.loc[e0, "b_conversation_minutes"].to_numpy()
            == after.loc[e0, "b_conversation_minutes"].to_numpy()).all()


def test_synthetic_context_trend_feature_never_sees_the_present(small_frame):
    """recent_sensor_trend is a shift(1) rolling mean; at each occasion it must
    equal a function of STRICTLY EARLIER values only."""
    for _pid, g in small_frame.groupby("pid"):
        g = g.sort_values("ts")
        v = g["conversation_minutes"].to_numpy(float)
        trend = g["recent_sensor_trend"].to_numpy(float)
        assert np.isnan(trend[0]), "the first observation has no past"
        for i in range(1, len(v)):
            expected = np.mean(v[max(0, i - 3):i])
            assert trend[i] == pytest.approx(expected, abs=1e-9)
