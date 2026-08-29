"""The hypergraph layer: structure, honesty about its role, and the ablation."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aedt.constants import DataStatus
from aedt.hypergraph.ablation import (ABLATION_CRITERIA, ablation_verdict,
                                      run_context_ablation)
from aedt.hypergraph.structure import (MAX_FACTORS, build_hypergraph,
                                       nary_context)

CTX = ["conversation_minutes", "activity_level", "recent_sensor_trend"]


def test_hyperedges_are_conjunctive_and_exact(small_frame):
    g = small_frame[small_frame["pid"] == "p00"]
    hg = build_hypergraph(g, CTX)
    assert hg.n_edges > 0
    for e in hg.edges:
        assert e.arity == 3, "a 3-factor context must give a 3-ary hyperedge"
        assert all("=" in v for v in e.vertices)
        assert e.key == "|".join(e.vertices)
        assert set(e.vertices) <= set(hg.vertices)


def test_arity_is_capped_so_the_matcher_stays_usable(small_frame):
    g = small_frame[small_frame["pid"] == "p00"]
    hg = build_hypergraph(g, CTX + ["hour_of_day", "sensed_level"])
    assert hg.mean_arity() <= MAX_FACTORS == 3


def test_occupancy_overlap_is_a_jaccard_over_epochs(small_frame):
    g = small_frame[small_frame["pid"] == "p00"]
    hg = build_hypergraph(g, CTX)
    ov = hg.occupancy_overlap()
    assert 0.0 <= ov <= 1.0
    a = {e.key for e in hg.edges if e.n_epoch0 > 0}
    b = {e.key for e in hg.edges if e.n_epoch1 > 0}
    assert ov == pytest.approx(len(a & b) / len(a | b))


def test_empty_context_yields_an_empty_graph_not_a_crash(small_frame):
    hg = build_hypergraph(small_frame[small_frame["pid"] == "p00"], [])
    assert hg.n_edges == 0 and hg.n_vertices == 0
    assert hg.mean_arity() == 0.0


def test_nary_context_labels_each_occasion(small_frame):
    out, graphs = nary_context(small_frame, CTX)
    assert "context_hyperedge" in out.columns
    assert set(graphs) == set(small_frame["pid"].astype(str))
    labelled = out["context_hyperedge"].notna()
    assert labelled.mean() > 0.8
    assert out.loc[labelled, "context_hyperedge"].str.count(r"\|").eq(2).all()


def test_ablation_runs_all_three_arms_on_the_same_participants(small_frame):
    tab = run_context_ablation(small_frame, "conversation_minutes", 5,
                               ctx_cols=CTX, true_rho=0.85, n_resamples=99)
    assert list(tab["representation"]) == ["continuous", "feature_vector",
                                           "nary_hyperedge"]
    assert tab.loc[tab["representation"] == "nary_hyperedge",
                   "uses_hypergraph"].item()
    assert not tab.loc[tab["representation"] == "continuous",
                       "uses_hypergraph"].item()
    for c in ABLATION_CRITERIA:
        assert c in tab.columns, f"judgement criterion {c} is not reported"


def test_ablation_verdict_is_honest_when_the_hypergraph_loses():
    tab = pd.DataFrame([
        {"representation": "continuous", "rho_star": 0.93, "ci_width": 0.10,
         "placebo_rejects": False},
        {"representation": "feature_vector", "rho_star": 0.98, "ci_width": 0.30,
         "placebo_rejects": False},
        {"representation": "nary_hyperedge", "rho_star": 0.99, "ci_width": 0.40,
         "placebo_rejects": False},
    ])
    v = ablation_verdict(tab)
    assert "did not improve" in v or "not the methodological contribution" in v


def test_ablation_verdict_credits_the_hypergraph_when_it_wins():
    tab = pd.DataFrame([
        {"representation": "continuous", "rho_star": 0.93, "ci_width": 0.40,
         "placebo_rejects": False},
        {"representation": "feature_vector", "rho_star": 0.98, "ci_width": 0.30,
         "placebo_rejects": False},
        {"representation": "nary_hyperedge", "rho_star": 0.94, "ci_width": 0.10,
         "placebo_rejects": False},
    ])
    assert "earns its place" in ablation_verdict(tab)


def test_an_arm_whose_placebo_fires_is_disqualified_however_narrow_its_ci():
    """CI width never rescues an arm that detects a change where none can
    exist."""
    tab = pd.DataFrame([
        {"representation": "continuous", "rho_star": 0.93, "ci_width": 0.20,
         "placebo_rejects": False},
        {"representation": "feature_vector", "rho_star": 0.98, "ci_width": 0.30,
         "placebo_rejects": False},
        {"representation": "nary_hyperedge", "rho_star": 0.94, "ci_width": 0.02,
         "placebo_rejects": True},
    ])
    v = ablation_verdict(tab)
    assert "DISQUALIFIED" in v
    assert "its placebo fired" in v
    assert "nary_hyperedge" in v
    assert "earns its place" not in v


def test_verdict_tolerates_a_table_without_a_placebo_column():
    tab = pd.DataFrame([
        {"representation": "continuous", "rho_star": 0.93, "ci_width": 0.10},
        {"representation": "nary_hyperedge", "rho_star": 0.99, "ci_width": 0.40},
    ])
    assert "narrowest interval" in ablation_verdict(tab)


def test_the_primary_estimator_does_not_depend_on_the_hypergraph(small_frame):
    """The decisive honesty test: rho* is identified by the ratio construction,
    NOT by the hypergraph. Removing every context column must leave the primary
    estimate untouched."""
    from aedt.estimators.slope_ratio import estimate_rho_star
    full = estimate_rho_star(small_frame, "conversation_minutes", 5,
                             bootstrap=False)
    stripped = small_frame.drop(columns=["activity_level",
                                         "recent_sensor_trend"])
    without = estimate_rho_star(stripped, "conversation_minutes", 5,
                                bootstrap=False)
    assert full.rho_star == pytest.approx(without.rho_star, abs=1e-12)


def test_spread_ratio_needs_enough_occupied_edges(small_frame):
    """A finding, not a nuisance: at low observation density the conjunctive
    contexts simply do not recur, and the estimator says so by name."""
    from aedt.estimators.spread_ratio import person_spread_log_ratio
    g = small_frame[small_frame["pid"] == "p00"].copy()
    g["context_hyperedge"] = "only|one|edge"
    v, why = person_spread_log_ratio(g, "context_hyperedge")
    assert np.isnan(v)
    assert "occupied hyperedges" in why


def test_null_calibration_is_checked_and_gates_the_verdict(small_frame):
    """"Effect retention AT MATCHED CALIBRATION" means calibration is checked
    FIRST. An arm that does not hold its size when the truth is rho = 1 cannot
    be ranked on CI width."""
    tab = pd.DataFrame([
        {"representation": "continuous", "rho_star": 0.91, "ci_width": 0.10,
         "effect_retention": 0.57, "placebo_rejects": False,
         "null_calibrated": True},
        {"representation": "nary_hyperedge", "rho_star": 1.06,
         "ci_width": 0.02, "effect_retention": -0.41,
         "placebo_rejects": False, "null_calibrated": False},
    ])
    v = ablation_verdict(tab)
    assert "DISQUALIFIED" in v
    assert "nary_hyperedge" in v
    assert "hold its size" in v
    assert "WRONG DIRECTION" in v
    assert "earns its place" not in v


def test_the_placebo_alone_does_not_catch_the_spread_ratio_failure():
    """A documented, non-obvious finding worth keeping.

    The split-half placebo builds both pseudo-epochs from within epoch 1, so
    they contain no genuine change in the construct. The n-ary spread-ratio
    form confounds genuine construct change with scale change, so it PASSES the
    placebo and still fails null calibration. This is why the ablation runs a
    separate null-cohort check.
    """
    from aedt.hypergraph.ablation import null_calibration_check
    out = null_calibration_check(
        n_participants=40, n_per_epoch=250,
        ctx_cols=CTX, sensor="conversation_minutes", K=5, seed=20260828,
        n_resamples=299, n_bins=3)
    assert out["continuous"] is True, (
        "the frozen primary must hold its size when the truth is rho = 1")
    assert out["nary_hyperedge"] is False, (
        "the n-ary spread-ratio form is documented as NOT null-calibrated; if "
        "this now passes, re-examine the finding rather than deleting the test")


def test_ablation_reports_all_five_judgement_criteria(small_frame):
    tab = run_context_ablation(small_frame, "conversation_minutes", 5,
                               ctx_cols=CTX, true_rho=0.85, n_resamples=99,
                               check_null_calibration=False)
    for c in ABLATION_CRITERIA:
        assert c in tab.columns, f"judgement criterion {c} is not reported"
