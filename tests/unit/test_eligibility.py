"""The eligibility screen: thresholds fixed before any data, reasons recorded."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aedt import constants as C
from aedt.audit.eligibility import (ELIGIBILITY_THRESHOLDS, filter_eligible,
                                    screen_cohort, screen_participant)
from aedt.constants import DataStatus
from aedt.schemas import EligibilityResult


def test_frozen_thresholds_match_the_specification():
    """These values are quoted in the report and the deck. If one changes, this
    test fails, which is the point."""
    assert ELIGIBILITY_THRESHOLDS == {
        "MIN_REPORTS_PER_EPOCH": 60, "MIN_CATEGORIES_USED": 2,
        "MIN_SENSOR_SD": 0.10, "VAR_RATIO_LO": 0.25, "VAR_RATIO_HI": 4.0,
        "MIN_ABS_BETA": 0.02}
    assert C.REQUIRE_MATCHING_SIGN is True
    assert C.SEED == 20260828
    assert C.BOOTSTRAP_B == 2000


def test_healthy_participant_passes(small_frame):
    g = small_frame[small_frame["pid"] == "p00"]
    r = screen_participant(g, "conversation_minutes", 5, pid="p00")
    assert r.eligible, r.reasons


def test_too_few_reports_is_excluded_with_a_named_reason(small_frame):
    g = small_frame[small_frame["pid"] == "p00"]
    short = pd.concat([g[g["epoch"] == 0].head(10), g[g["epoch"] == 1]])
    r = screen_participant(short, "conversation_minutes", 5, pid="p00")
    assert not r.eligible
    assert any("10 reports < 60" in x for x in r.reasons)


def test_single_category_is_excluded_as_an_A5_violation(small_frame):
    g = small_frame[small_frame["pid"] == "p00"].copy()
    g["report"] = 2
    r = screen_participant(g, "conversation_minutes", 5, pid="p00")
    assert not r.eligible
    assert any("categories used (A5)" in x for x in r.reasons)


def test_no_sensor_variation_is_excluded(small_frame):
    g = small_frame[small_frame["pid"] == "p00"].copy()
    g["conversation_minutes"] = 42.0
    r = screen_participant(g, "conversation_minutes", 5, pid="p00")
    assert not r.eligible
    assert any("no sensor variation" in x for x in r.reasons)


def test_unstable_variance_is_excluded_as_an_A3_violation(small_frame):
    g = small_frame[small_frame["pid"] == "p00"].copy()
    m = g["epoch"] == 1
    mu = g.loc[m, "conversation_minutes"].mean()
    g.loc[m, "conversation_minutes"] = (
        mu + (g.loc[m, "conversation_minutes"] - mu) * 8.0)
    r = screen_participant(g, "conversation_minutes", 5, pid="p00")
    assert not r.eligible
    assert any("A3" in x for x in r.reasons)


def test_an_ineligible_result_must_carry_a_reason():
    with pytest.raises(ValueError, match="must carry a reason"):
        EligibilityResult(pid="x", eligible=False, reasons=())


def test_screen_cohort_reports_every_participant(small_frame):
    rs = screen_cohort(small_frame, "conversation_minutes", 5)
    assert len(rs) == small_frame["pid"].nunique()
    assert {r.pid for r in rs} == set(small_frame["pid"].astype(str))


def test_filter_eligible_keeps_only_passing_participants(small_frame):
    rs = screen_cohort(small_frame, "conversation_minutes", 5)
    kept = filter_eligible(small_frame, rs)
    assert set(kept["pid"]) == {r.pid for r in rs if r.eligible}


def test_a_config_override_is_honoured_and_visible(small_frame):
    """A pre-specified sensitivity analysis may raise a threshold; doing so must
    actually change the outcome, so the deviation cannot be cosmetic."""
    g = small_frame[small_frame["pid"] == "p00"]
    strict = screen_participant(g, "conversation_minutes", 5, pid="p00",
                               thresholds={"MIN_REPORTS_PER_EPOCH": 10_000})
    assert not strict.eligible
    assert any("< 10000" in x for x in strict.reasons)
