"""The ceiling statistics must recover answers we planted, and refuse to guess.

These numbers are the evidential basis for the project's follow-up direction,
and until now they existed only as literals in an exporter. A test that merely
executed the code would prove nothing, so every case here either plants a known
value and requires it back, or feeds deliberately degenerate input and requires
an EXCLUSION rather than a number.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aedt.audit.ceiling import (MIN_OBS_PER_PERSON, ceiling_statistics,
                                consecutive_pairs, early_late_stability,
                                icc_one_way, per_person_autocorrelations,
                                person_autocorrelation, pooled_autocorrelation)


# --------------------------------------------------------------- generators
def ar1(rho: float, n: int, seed: int, level: float = 0.0) -> np.ndarray:
    """A stationary AR(1) series with lag-1 autocorrelation exactly ``rho``."""
    rng = np.random.default_rng(seed)
    e = rng.normal(0, np.sqrt(1 - rho ** 2), n)
    v = np.empty(n)
    v[0] = rng.normal(0, 1)
    for i in range(1, n):
        v[i] = rho * v[i - 1] + e[i]
    return v + level


def frame_from(series: dict[str, np.ndarray], step: int = 1) -> pd.DataFrame:
    """Long frame with one row per (person, day), days ``step`` apart."""
    rows = []
    for pid, v in series.items():
        for i, x in enumerate(v):
            rows.append((pid, i * step, float(x)))
    return pd.DataFrame(rows, columns=["participant_id", "day_ord", "stress"])


# ------------------------------------------------------------------- basics
@pytest.mark.parametrize("rho", [0.0, 0.25, 0.5, 0.8])
def test_person_autocorrelation_recovers_a_planted_rho(rho):
    v = ar1(rho, 4000, seed=11)
    got = person_autocorrelation(v, days=np.arange(4000))
    assert abs(got - rho) < 0.04, f"planted {rho}, recovered {got:.4f}"


def test_variance_explained_is_r_squared():
    f = frame_from({f"p{i}": ar1(0.6, 400, seed=i) for i in range(20)})
    s = ceiling_statistics(f, value_col="stress", day_col="day_ord")
    assert abs(s.variance_explained - s.within_person_autocorrelation ** 2) < 1e-12


def test_pooled_autocorrelation_is_not_inflated_by_person_level_differences():
    """Two cohorts, same rho, wildly different means.

    Pooling raw pairs would absorb the between-person spread and report a much
    larger correlation. The pooled statistic centres each participant first, so
    it must stay near the planted value.
    """
    rho = 0.4
    series = []
    for i in range(30):
        level = -20.0 if i % 2 else 20.0        # enormous between-person spread
        v = ar1(rho, 500, seed=100 + i, level=level)
        series.append((v, np.arange(500)))
    pooled = pooled_autocorrelation(series)
    assert abs(pooled - rho) < 0.05, f"planted {rho}, pooled {pooled:.4f}"

    # the naive alternative really would have been inflated -- show it
    x = np.concatenate([s[0][:-1] for s in series])
    y = np.concatenate([s[0][1:] for s in series])
    naive = float(np.corrcoef(x, y)[0, 1])
    assert naive > pooled + 0.4, (
        "the person-centring is not doing anything; this test would not catch "
        f"the bug it exists for (naive {naive:.3f}, pooled {pooled:.3f})")


# ------------------------------------------------------------- refusal cases
def test_a_constant_series_yields_no_number():
    assert np.isnan(person_autocorrelation(np.full(100, 3.0),
                                           days=np.arange(100)))


def test_a_flat_participant_is_excluded_and_counted_never_imputed():
    f = frame_from({"flat": np.full(200, 3.0),
                    "varies": ar1(0.5, 200, seed=7)})
    rows, excluded = per_person_autocorrelations(
        f, value_col="stress", day_col="day_ord")
    assert [r.participant_id for r in rows] == ["varies"]
    assert excluded["no_variance_or_too_few_pairs"] == 1
    assert all(np.isfinite(r.r) for r in rows)


def test_a_short_participant_is_excluded_and_counted():
    short = MIN_OBS_PER_PERSON - 1
    f = frame_from({"short": ar1(0.5, short, seed=3),
                    "long": ar1(0.5, 200, seed=4)})
    rows, excluded = per_person_autocorrelations(
        f, value_col="stress", day_col="day_ord")
    assert [r.participant_id for r in rows] == ["long"]
    assert excluded["too_few_observations"] == 1


def test_too_few_points_gives_nan_rather_than_a_confident_zero():
    assert np.isnan(person_autocorrelation([1.0, 2.0], days=[0, 1]))


# ---------------------------------------------------------------- gap rule
def test_pairs_beyond_the_horizon_are_dropped():
    v = np.arange(10, dtype=float)
    days = np.array([0, 1, 2, 30, 31, 32, 60, 61, 62, 63], dtype=float)
    x, _ = consecutive_pairs(v, days, max_gap_days=7)
    assert len(x) == 7, "the two 28-day jumps should have been excluded"
    x_all, _ = consecutive_pairs(v, days, max_gap_days=None)
    assert len(x_all) == 9


def test_the_horizon_changes_the_answer_so_the_definition_matters():
    """Guards the reason both definitions are reported side by side."""
    rng = np.random.default_rng(5)
    v, days, t = [], [], 0
    for block in range(40):
        base = rng.normal(0, 3)                  # each block sits at its own level
        for i in range(10):
            v.append(base + rng.normal(0, 0.2))
            days.append(t)
            t += 1
        t += 40                                   # a long gap between blocks
    within = person_autocorrelation(v, days, max_gap_days=7)
    across = person_autocorrelation(v, days, max_gap_days=None)
    assert within > across + 0.05


# --------------------------------------------------------------------- ICC
def test_icc_recovers_a_planted_variance_split():
    rng = np.random.default_rng(2)
    var_b, var_w = 1.0, 1.0                       # ICC = 0.5 by construction
    groups = [rng.normal(rng.normal(0, np.sqrt(var_b)), np.sqrt(var_w), 120)
              for _ in range(200)]
    got = icc_one_way(groups)
    assert abs(got - 0.5) < 0.05, f"planted 0.5, recovered {got:.4f}"


def test_icc_is_near_zero_when_everyone_shares_one_distribution():
    rng = np.random.default_rng(3)
    groups = [rng.normal(0, 1, 120) for _ in range(200)]
    assert icc_one_way(groups) < 0.03


def test_icc_is_near_one_when_people_barely_vary_within_themselves():
    rng = np.random.default_rng(4)
    groups = [rng.normal(rng.normal(0, 5), 0.05, 60) for _ in range(120)]
    assert icc_one_way(groups) > 0.95


def test_icc_refuses_with_fewer_than_two_groups():
    assert np.isnan(icc_one_way([np.arange(10, dtype=float)]))


# ------------------------------------------------------- early/late stability
def test_early_late_correlation_is_positive_when_predictability_is_a_trait():
    """Each person keeps their own rho across both halves."""
    rng = np.random.default_rng(21)
    series = {}
    for i in range(120):
        rho = float(rng.uniform(-0.1, 0.75))
        series[f"p{i}"] = np.concatenate([ar1(rho, 120, seed=1000 + i),
                                          ar1(rho, 120, seed=5000 + i)])
    f = frame_from(series)
    rows, _ = per_person_autocorrelations(f, value_col="stress",
                                          day_col="day_ord")
    r, n = early_late_stability(rows)
    assert n >= 100 and r > 0.3, f"stability {r:.3f} over {n} participants"


def test_early_late_correlation_is_near_zero_when_predictability_is_not_a_trait():
    """Each person's rho is redrawn between the halves — the honest null."""
    rng = np.random.default_rng(22)
    series = {}
    for i in range(160):
        a = float(rng.uniform(-0.1, 0.75))
        b = float(rng.uniform(-0.1, 0.75))
        series[f"p{i}"] = np.concatenate([ar1(a, 120, seed=2000 + i),
                                          ar1(b, 120, seed=6000 + i)])
    f = frame_from(series)
    rows, _ = per_person_autocorrelations(f, value_col="stress",
                                          day_col="day_ord")
    r, n = early_late_stability(rows)
    assert n >= 120 and abs(r) < 0.25, (
        f"found stability {r:.3f} where none was planted — the split is leaking")


# ----------------------------------------------------------------- assembly
def test_full_block_is_complete_and_internally_consistent():
    rng = np.random.default_rng(31)
    series = {f"p{i}": ar1(float(rng.uniform(0.0, 0.7)), 160, seed=300 + i)
              for i in range(80)}
    f = frame_from(series)
    s = ceiling_statistics(f, value_col="stress", day_col="day_ord")
    d = s.to_dict()

    for key in ("within_person_autocorrelation", "variance_explained",
                "icc_between_person", "per_person_r_median", "per_person_r_iqr",
                "per_person_r_range", "frac_near_unpredictable",
                "frac_well_predictable", "early_late_r",
                "n_participants_analysed", "definition"):
        assert key in d, f"{key} missing from the exported block"

    lo, hi = d["per_person_r_iqr"]
    rlo, rhi = d["per_person_r_range"]
    assert rlo <= lo <= d["per_person_r_median"] <= hi <= rhi
    assert 0.0 <= d["frac_near_unpredictable"] <= 1.0
    assert 0.0 <= d["frac_well_predictable"] <= 1.0
    assert d["n_participants_analysed"] == 80
    assert d["definition"]["min_obs_per_person"] == MIN_OBS_PER_PERSON


def test_the_computation_is_deterministic():
    f = frame_from({f"p{i}": ar1(0.45, 150, seed=i) for i in range(25)})
    a = ceiling_statistics(f, value_col="stress", day_col="day_ord").to_dict()
    b = ceiling_statistics(f, value_col="stress", day_col="day_ord").to_dict()
    assert a == b


def test_row_order_does_not_change_the_answer():
    f = frame_from({f"p{i}": ar1(0.45, 150, seed=i) for i in range(25)})
    shuffled = f.sample(frac=1.0, random_state=9).reset_index(drop=True)
    a = ceiling_statistics(f, value_col="stress", day_col="day_ord").to_dict()
    b = ceiling_statistics(shuffled, value_col="stress",
                           day_col="day_ord").to_dict()
    assert a["within_person_autocorrelation"] == pytest.approx(
        b["within_person_autocorrelation"], abs=1e-12), (
        "the statistic depends on input row order, so it is not sorting within "
        "participant before pairing")
