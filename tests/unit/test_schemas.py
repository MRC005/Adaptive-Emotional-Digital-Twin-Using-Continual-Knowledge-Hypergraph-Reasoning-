"""Canonical schemas validate themselves; no anonymous dicts for science."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aedt.constants import DataStatus, DatasetRole
from aedt.errors import DecisionRequired
from aedt.schemas import (DatasetAudit, Hyperedge, Observation, Participant,
                          SelfReport, UncertaintyResult, frame_to_participants,
                          validate_long_frame)


def _ok_frame():
    return pd.DataFrame({
        "pid": ["a"] * 4,
        "ts": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03",
                              "2026-01-04"]),
        "report": [1, 2, 3, 4],
        "epoch": [0, 0, 1, 1],
        "conversation_minutes": [10.0, 20.0, 30.0, 40.0]})


def test_valid_frame_passes():
    assert validate_long_frame(_ok_frame(), "conversation_minutes",
                               require_epoch=True, n_categories=5) is not None


@pytest.mark.parametrize("mutate,match", [
    (lambda d: d.drop(columns=["report"]), "missing required columns"),
    (lambda d: d.assign(ts=["a", "b", "c", "d"]), "not datetime"),
    (lambda d: d.assign(report=[1, 2, 3, np.nan]), "non-numeric"),
    (lambda d: d.assign(report=[1.5, 2, 3, 4]), "not integers"),
    (lambda d: d.assign(report=[0, 2, 3, 4]), "1-based severity"),
    (lambda d: d.assign(report=[1, 2, 3, 99]), "exceeds K"),
    (lambda d: d.assign(epoch=[0, 1, 2, 3]), "outside"),
    (lambda d: d.iloc[0:0], "empty"),
])
def test_contract_breaches_raise_decision_required(mutate, match):
    with pytest.raises(DecisionRequired, match=match):
        validate_long_frame(mutate(_ok_frame()), "conversation_minutes",
                            require_epoch=True, n_categories=5)


def test_missing_sensor_column_is_named():
    with pytest.raises(DecisionRequired, match="absent from the LongFrame"):
        validate_long_frame(_ok_frame(), "no_such_sensor")


def test_participant_requires_a_pid():
    with pytest.raises(DecisionRequired, match="empty pid"):
        Participant(pid="  ", dataset="d", data_status=DataStatus.SYNTHETIC)


def test_self_report_severity_must_be_in_range():
    with pytest.raises(DecisionRequired, match="outside 1..5"):
        SelfReport(pid="a", ts=pd.Timestamp("2026-01-01"), severity=7,
                   n_categories=5)


def test_observation_epoch_must_be_0_or_1():
    with pytest.raises(DecisionRequired, match="Epoch must be 0 or 1"):
        Observation(pid="a", ts=pd.Timestamp("2026-01-01"), report=3,
                    n_categories=5, features={}, epoch=2)


def test_audit_cannot_claim_real_without_files():
    with pytest.raises(DecisionRequired, match="requires audited files"):
        DatasetAudit(dataset_name="x", role=DatasetRole.PRIMARY_LONGITUDINAL,
                     data_status=DataStatus.REAL, source_status="s",
                     local_files_available=False)


def test_hyperedge_arity_and_occupancy():
    e = Hyperedge(key="a=low|b=high", vertices=("a=low", "b=high"), pid="p",
                  n_epoch0=3, n_epoch1=0)
    assert e.arity == 2
    assert not e.occupied_both_epochs
    assert Hyperedge(key="k", vertices=("a",), pid="p", n_epoch0=1,
                     n_epoch1=1).occupied_both_epochs


def test_uncertainty_excludes_null_logic():
    assert UncertaintyResult("m", 20, 100, 0.9, 0.85, 0.95).excludes_null
    assert not UncertaintyResult("m", 20, 100, 0.98, 0.9, 1.05).excludes_null
    assert not UncertaintyResult("m", 20, 0, 0.9, float("nan"),
                                 float("nan")).excludes_null


def test_serialisation_is_json_safe():
    u = UncertaintyResult("m", 20, 100, 0.9, float("nan"), 0.95)
    d = u.to_dict()
    assert d["ci_low"] is None                 # NaN becomes null, not "NaN"
    assert d["data_status"] == "SYNTHETIC"
    import json
    json.loads(u.to_json())


def test_frame_to_participants(small_frame):
    ps = frame_to_participants(small_frame, "synthetic", DataStatus.SYNTHETIC)
    assert len(ps) == small_frame["pid"].nunique()
    assert all(p.span_days > 0 for p in ps)
