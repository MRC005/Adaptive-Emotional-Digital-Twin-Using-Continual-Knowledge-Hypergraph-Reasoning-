"""MANDATORY KNOWN-ANSWER TESTS for the slope-ratio estimator (ROUND-17 §W)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aedt.constants import SEED, TOL_NULL_RATIO
from aedt.errors import DecisionRequired
from aedt.estimators.slope_ratio import (estimate_rho_star, fit_person_epochs,
                                         person_log_ratio,
                                         standardise_within_epoch)
from aedt.schemas import EstimatorResult


def test_null_ratio_is_one_within_tolerance(null_frame):
    """rho* = 1.00 +/- 0.03 under the TRUE NULL on the floor-heavy generator
    (P=50, n=300). A method that reports scale change where there is none is
    unusable."""
    res = estimate_rho_star(null_frame, "conversation_minutes", 5,
                            n_resamples=399)
    assert abs(res.rho_star - 1.0) < TOL_NULL_RATIO, (
        f"null rho* = {res.rho_star:.4f}; the estimator is not calibrated")


def test_attenuated_never_inflated(shift_frame):
    """At true rho = 0.85 the estimate must lie in (0.85, 1.00): attenuated
    toward the null, never inflated. This is what makes 1 - rho* a LOWER
    BOUND."""
    res = estimate_rho_star(shift_frame, "conversation_minutes", 5,
                            n_resamples=399)
    assert 0.85 < res.rho_star < 1.00, (
        f"rho* = {res.rho_star:.4f} is outside (0.85, 1.00): the attenuation "
        "argument is violated")


def test_null_ci_covers_one_and_shift_ci_excludes_it(null_frame, shift_frame):
    n = estimate_rho_star(null_frame, "conversation_minutes", 5, n_resamples=999)
    s = estimate_rho_star(shift_frame, "conversation_minutes", 5, n_resamples=999)
    assert not n.uncertainty.excludes_null, "false positive under the null"
    assert s.uncertainty.excludes_null, "no detection of a real 15% shift"


def test_standardisation_is_epoch_local(small_frame):
    """A standardiser fitted on epoch 1 must NEVER touch epoch 2."""
    for _pid, g in small_frame.groupby("pid"):
        for e in (0, 1):
            s = g.loc[g["epoch"] == e, "conversation_minutes"].to_numpy(float)
            x, m, sd = standardise_within_epoch(s)
            assert abs(x.mean()) < 1e-9
            assert abs(x.std(ddof=1) - 1.0) < 1e-9
            assert abs(m - s.mean()) < 1e-9


def test_epoch_fits_are_independent(small_frame):
    """MANDATORY: changing epoch-2 rows must not alter epoch-1 fitted values."""
    pid = small_frame["pid"].iloc[0]
    g = small_frame[small_frame["pid"] == pid].copy()
    before = fit_person_epochs(g, "conversation_minutes", 5, pid=pid)

    tampered = g.copy()
    m = tampered["epoch"] == 1
    tampered.loc[m, "conversation_minutes"] = (
        tampered.loc[m, "conversation_minutes"] * 3.0 + 500.0)
    tampered.loc[m, "report"] = 5
    after = fit_person_epochs(tampered, "conversation_minutes", 5, pid=pid)

    assert after[0].beta == pytest.approx(before[0].beta, abs=1e-12)
    assert list(after[0].cutpoints) == pytest.approx(list(before[0].cutpoints),
                                                     abs=1e-12)
    assert after[0].standardiser_mean == pytest.approx(
        before[0].standardiser_mean, abs=1e-12)


def test_sign_flip_participants_are_excluded_by_name(small_frame):
    """A participant whose association reverses between epochs has a negative,
    uninterpretable ratio and must be excluded by pre-stated rule."""
    pid = small_frame["pid"].iloc[0]
    g = small_frame[small_frame["pid"] == pid].copy()
    m = g["epoch"] == 1
    # invert the epoch-2 relationship
    g.loc[m, "conversation_minutes"] = -g.loc[m, "conversation_minutes"]
    v, why, _f = person_log_ratio(g, "conversation_minutes", 5, pid=pid)
    assert np.isnan(v)
    assert "flips sign" in why


def test_result_refuses_a_wrong_estimand():
    with pytest.raises(DecisionRequired, match="rho_star"):
        EstimatorResult(estimand="rho", rho_star=0.9, log_rho_star=-0.1,
                        uncertainty=None, n_participants_used=1,
                        n_participants_screened=1)


def test_additive_component_can_never_be_populated():
    r = EstimatorResult(estimand="rho_star", rho_star=0.9, log_rho_star=-0.1,
                        uncertainty=None, n_participants_used=1,
                        n_participants_screened=1)
    assert r.additive_component is None
    assert r.additive_component_status == "NOT IDENTIFIED"


def test_lower_bound_property(shift_frame):
    res = estimate_rho_star(shift_frame, "conversation_minutes", 5,
                            bootstrap=False)
    assert res.lower_bound_on_recalibration == pytest.approx(1 - res.rho_star)


def test_exclusions_are_recorded_with_reasons(small_frame):
    """Nothing is dropped silently: every unused participant carries a reason."""
    df = small_frame.copy()
    pid = df["pid"].iloc[0]
    df.loc[df["pid"] == pid, "report"] = 3        # one category only
    res = estimate_rho_star(df, "conversation_minutes", 5, bootstrap=False)
    assert pid in res.exclusions
    assert res.exclusions[pid]
