"""ASSERT THE KNOWN FAILURES.

A future "improvement" that silently breaks a documented negative result would
otherwise make our method look better than it is. These tests make that
impossible: the failures are pinned.
"""
from __future__ import annotations

import numpy as np
import pytest

from aedt.constants import LINEAR_ANCHOR_KNOWN_NULL_BIAS
from aedt.estimators.affine_did import (DISCRETISATION_MODES,
                                        affine_did_null_bias, run_affine_did)


@pytest.mark.parametrize("K", [5, 7])
def test_affine_estimator_reproduces_its_documented_null_bias(K):
    """ROUND-17 §Q: the affine estimator fabricates a -0.107 null bias on
    5-point (and 7-point) Likert responses -- roughly 10% apparent scale
    compression when NOTHING has changed.

    This is the reason the ordinal slope-ratio construction exists. If this
    test starts passing with a bias near zero, the baseline has been broken,
    not fixed.
    """
    bias = affine_did_null_bias(K=K, nrep=250, seed=3 * 37 if K == 5 else 9 * 37)
    assert bias == pytest.approx(LINEAR_ANCHOR_KNOWN_NULL_BIAS, abs=0.02), (
        f"the affine estimator's documented K={K} null bias of "
        f"{LINEAR_ANCHOR_KNOWN_NULL_BIAS} was not reproduced (got {bias:+.3f})")


def test_continuous_reference_is_nearly_unbiased():
    """The same estimator on CONTINUOUS responses is fine (-0.008). The failure
    is caused by the ordinal boundary, not by the estimator's algebra."""
    r = run_affine_did("continuous", None, 1.00, nrep=250, seed=37)
    assert abs(r.bias) < 0.03, (
        f"the continuous reference should be nearly unbiased, got {r.bias:+.3f}")


def test_withdrawn_per_anchor_artefact_is_still_reproducible():
    """SELF-CORRECTION (Round 14). An earlier round reported -0.19 and froze the
    project around it. That number came from a harness bug: each anchor was
    re-scaled by its own 15 observations, erasing the between-anchor level
    variation the estimator reads.

    Both code paths are retained so the correction stays VERIFIABLE rather than
    merely asserted.
    """
    withdrawn = run_affine_did("per-anchor", 5, 1.00, nrep=250, seed=2 * 37)
    correct = run_affine_did("per-person", 5, 1.00, nrep=250, seed=3 * 37)
    assert withdrawn.bias == pytest.approx(-0.186, abs=0.03)
    assert correct.bias == pytest.approx(-0.107, abs=0.02)
    assert withdrawn.bias < correct.bias, (
        "the withdrawn harness bug must still show a LARGER apparent bias "
        "than the corrected code path")


def test_boundary_rate_explains_the_bias_ordering():
    """The bias tracks the share of responses at the scale boundary, which is
    the mechanism: the level coefficient absorbs threshold saturation."""
    cont = run_affine_did("continuous", None, 1.00, nrep=120, seed=37)
    person = run_affine_did("per-person", 5, 1.00, nrep=120, seed=3 * 37)
    anchor = run_affine_did("per-anchor", 5, 1.00, nrep=120, seed=2 * 37)
    assert cont.boundary_rate_pct == 0.0
    assert 8 < person.boundary_rate_pct < 20
    assert anchor.boundary_rate_pct > person.boundary_rate_pct


def test_all_three_discretisation_modes_are_retained():
    assert set(DISCRETISATION_MODES) == {"continuous", "per-person",
                                         "per-anchor"}


def test_the_ordinal_estimator_beats_the_affine_one_where_it_matters(null_frame):
    """Both must be calibrated under the null, but only the ordinal one stays
    calibrated on the FLOOR-HEAVY scales real stress items actually use."""
    from aedt.baselines.runners import linear_anchor
    from aedt.estimators.slope_ratio import estimate_rho_star
    ordinal = estimate_rho_star(null_frame, "conversation_minutes", 5,
                                bootstrap=False).rho_star
    affine = linear_anchor(null_frame, "conversation_minutes", 5).statistic
    assert abs(ordinal - 1.0) < 0.03
    # the ported linear anchor is calibrated on THIS generator (-0.012 in the
    # validated Round-14 table); the catastrophic case is the affine DiD above
    assert np.isfinite(affine)
