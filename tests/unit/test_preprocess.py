"""Preprocessing: explicit rules, a missingness ledger, and NO imputation."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aedt.preprocess.clean import clean_long_frame
from aedt.preprocess.reports import category_usage_table


def _frame():
    return pd.DataFrame({
        "pid": ["a"] * 6 + ["b"] * 4,
        "ts": list(pd.date_range("2026-01-01", periods=6, freq="D"))
              + list(pd.date_range("2026-02-01", periods=4, freq="D")),
        "report": [1, 2, np.nan, 4, 5, 1, 2, 3, 4, 5],
        "conversation_minutes": [10.0, np.nan, 30.0, 40.0, 50.0, 60.0,
                                 11.0, 22.0, 33.0, 44.0]})


def test_missing_outcome_is_dropped_and_counted_never_imputed():
    out, led = clean_long_frame(_frame(), "conversation_minutes")
    assert out["report"].notna().all()
    assert any("outcome NEVER imputed" in k for k in led.removed)
    assert led.removed[[k for k in led.removed if "outcome" in k][0]] == 1


def test_missing_covariate_is_dropped_and_counted_never_imputed():
    out, led = clean_long_frame(_frame(), "conversation_minutes")
    assert out["conversation_minutes"].notna().all()
    assert any("covariate NEVER imputed" in k for k in led.removed)


def test_ledger_accounts_for_every_row():
    out, led = clean_long_frame(_frame(), "conversation_minutes")
    assert led.n_input == 10
    assert led.n_output == len(out) == 8
    assert led.n_removed == 2
    assert led.retention_rate == pytest.approx(0.8)


def test_duplicate_occasions_are_removed_and_counted():
    d = pd.concat([_frame().dropna(), _frame().dropna()], ignore_index=True)
    _out, led = clean_long_frame(d, "conversation_minutes")
    assert any("duplicate" in k for k in led.removed)


def test_outliers_are_flagged_not_dropped():
    d = _frame().dropna().copy()
    d.loc[d.index[0], "conversation_minutes"] = 1e6
    out, led = clean_long_frame(d, "conversation_minutes",
                               sensor_outlier_z=3.0)
    assert len(out) == len(d), "an outlier was dropped instead of flagged"
    assert "sensor_outlier_flag" in out.columns
    assert out["sensor_outlier_flag"].sum() >= 1


def test_participants_lost_entirely_are_named():
    d = _frame()
    d.loc[d["pid"] == "b", "report"] = np.nan
    _out, led = clean_long_frame(d, "conversation_minutes")
    assert led.participants_lost == ["b"]


def test_column_missing_rates_are_recorded():
    _out, led = clean_long_frame(_frame(), "conversation_minutes")
    assert led.column_missing_rate["report"] == pytest.approx(0.1)
    assert led.column_missing_rate["conversation_minutes"] == pytest.approx(0.1)


def test_ledger_is_serialisable():
    _out, led = clean_long_frame(_frame(), "conversation_minutes")
    d = led.to_dict()
    assert d["n_input"] == 10 and d["n_output"] == 8


def test_category_usage_counts_per_epoch(small_frame):
    u = category_usage_table(small_frame, 5)
    assert set(u["epoch"]) == {0, 1}
    assert len(u) == 2 * small_frame["pid"].nunique()
    for _i, r in u.iterrows():
        assert sum(r[f"n_cat{k}"] for k in range(1, 6)) == r["n"]
        assert 0.0 <= r["floor_rate"] <= 1.0
        assert 1 <= r["categories_used"] <= 5
