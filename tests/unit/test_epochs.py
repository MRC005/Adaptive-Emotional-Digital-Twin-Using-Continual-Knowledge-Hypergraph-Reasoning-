"""Epoch assignment: halves of each participant's OWN span (FROZEN)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aedt.errors import DecisionRequired
from aedt.preprocess.epochs import assign_epochs, epoch_definitions
from aedt.schemas import EpochDefinition


def _frame():
    # two participants with DELIBERATELY different enrolment windows
    a = pd.DataFrame({"pid": "a",
                      "ts": pd.date_range("2026-01-01", periods=20, freq="D"),
                      "report": 3})
    b = pd.DataFrame({"pid": "b",
                      "ts": pd.date_range("2026-03-01", periods=100, freq="D"),
                      "report": 3})
    return pd.concat([a, b], ignore_index=True)


def test_epochs_split_each_participants_own_span():
    out = assign_epochs(_frame())
    for pid, g in out.groupby("pid"):
        assert set(pd.unique(g["epoch"])) == {0, 1}
        # roughly half each, since the occasions are evenly spaced
        assert abs((g["epoch"] == 0).sum() - (g["epoch"] == 1).sum()) <= 1
    # a shared calendar midpoint would have put ALL of 'a' in epoch 0
    assert (out.loc[out["pid"] == "a", "epoch"] == 1).sum() > 0


def test_epoch_2_never_precedes_epoch_1_in_time():
    out = assign_epochs(_frame())
    for _pid, g in out.groupby("pid"):
        assert g.loc[g["epoch"] == 0, "ts"].max() <= g.loc[g["epoch"] == 1,
                                                           "ts"].min()


def test_calendar_split_is_available_but_is_not_the_default():
    out = assign_epochs(_frame(), rule="calendar_median")
    assert out.attrs["epoch_rule"] == "calendar_median"
    # under a shared midpoint, participant 'a' falls entirely in epoch 0
    assert set(pd.unique(out.loc[out["pid"] == "a", "epoch"])) == {0}
    assert assign_epochs(_frame()).attrs["epoch_rule"] == "own_span_halves"


def test_unknown_rule_is_refused():
    with pytest.raises(DecisionRequired, match="frozen rule"):
        assign_epochs(_frame(), rule="whatever_looks_best")


def test_epoch_definitions_are_typed_and_consistent():
    out = assign_epochs(_frame())
    defs = epoch_definitions(out)
    assert len(defs) == 2
    for d in defs:
        assert isinstance(d, EpochDefinition)
        assert d.start <= d.midpoint <= d.end
        assert d.n_epoch0 > 0 and d.n_epoch1 > 0


def test_epoch_definition_refuses_an_impossible_midpoint():
    with pytest.raises(DecisionRequired, match="outside the observed span"):
        EpochDefinition(pid="a", rule="own_span_halves",
                        start=pd.Timestamp("2026-01-10"),
                        midpoint=pd.Timestamp("2026-01-01"),
                        end=pd.Timestamp("2026-01-20"),
                        n_epoch0=1, n_epoch1=1)


def test_generator_epochs_agree_with_rederived_epochs(small_frame):
    """The simulator labels its own two blocks; assign_epochs re-derives them
    from timestamps alone. They must agree -- a real cross-check."""
    original = small_frame[["pid", "ts", "epoch"]].copy()
    rederived = assign_epochs(small_frame.drop(columns=["epoch"]))
    merged = original.merge(rederived[["pid", "ts", "epoch"]],
                            on=["pid", "ts"], suffixes=("_gen", "_derived"))
    agreement = (merged["epoch_gen"] == merged["epoch_derived"]).mean()
    assert agreement > 0.99, f"only {agreement:.1%} of epoch labels agree"
