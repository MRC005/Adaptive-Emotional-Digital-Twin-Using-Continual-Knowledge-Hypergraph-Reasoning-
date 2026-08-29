"""MANDATORY KNOWN-ANSWER TEST: the ordinal probit recovers a known slope.

ROUND-17 §W: "ordinal_probit recovers a known slope within 5% on 2,000
simulated observations."
"""
from __future__ import annotations

import numpy as np
import pytest

from aedt.constants import SEED, TOL_SLOPE_RECOVERY
from aedt.models.ordinal import (cutpoints_from_params, ordinal_probit,
                                 ordinal_probit_fit, predict_cumulative,
                                 predict_expected_category)


def _make(n=2000, beta=0.8, K=5, seed=SEED):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n)
    # residual SD is 1, so the identified probit slope IS beta
    latent = beta * x + rng.normal(0, 1, n)
    cuts = np.quantile(latent, np.linspace(0, 1, K + 1)[1:K])
    return x, np.searchsorted(cuts, latent) + 1


@pytest.mark.parametrize("beta", [0.4, 0.8, 1.2])
def test_recovers_known_slope_within_5_percent(beta):
    x, y = _make(n=4000, beta=beta)
    fit = ordinal_probit_fit(x, y, 5)
    assert fit.converged, fit.reason
    rel = abs(fit.beta - beta) / beta
    assert rel < TOL_SLOPE_RECOVERY, (
        f"recovered {fit.beta:.4f} for a true slope of {beta}: "
        f"{rel:.1%} error exceeds the {TOL_SLOPE_RECOVERY:.0%} tolerance")


def test_recovers_negative_slope():
    """Sign is NOT assumed. Self-correction 26: conversation minutes fall as
    stress rises, so the true slope for the primary feature is NEGATIVE."""
    x, y = _make(beta=-0.8, n=4000)
    fit = ordinal_probit_fit(x, y, 5)
    assert fit.converged
    assert fit.beta < 0
    assert abs(fit.beta - (-0.8)) / 0.8 < TOL_SLOPE_RECOVERY


def test_cutpoints_are_strictly_ordered():
    x, y = _make()
    fit = ordinal_probit_fit(x, y, 5)
    assert len(fit.cutpoints) == 4
    assert list(fit.cutpoints) == sorted(fit.cutpoints)
    assert np.all(np.diff(fit.cutpoints) > 0)


def test_cutpoint_parameterisation_enforces_order():
    par = np.array([-1.0, np.log(0.5), np.log(0.5), np.log(0.5), 0.3])
    cuts = cutpoints_from_params(par, 5)
    assert np.all(np.diff(cuts) > 0)


def test_degenerate_inputs_do_not_silently_return_a_number():
    x = np.random.default_rng(0).normal(0, 1, 100)
    one_category = ordinal_probit_fit(x, np.ones(100, int), 5)
    assert not one_category.converged
    assert "categories" in one_category.reason
    assert np.isnan(one_category.beta)

    no_variation = ordinal_probit_fit(np.zeros(100), np.arange(100) % 5 + 1, 5)
    assert not no_variation.converged
    assert "variation" in no_variation.reason


def test_slope_below_floor_is_rejected_not_returned():
    rng = np.random.default_rng(SEED)
    x = rng.normal(0, 1, 1500)
    y = rng.integers(1, 6, 1500)          # report unrelated to x
    fit = ordinal_probit_fit(x, y, 5, min_abs_beta=0.5)
    assert not fit.converged
    assert "below the floor" in fit.reason
    assert np.isnan(ordinal_probit(x, y, 5, min_abs_beta=0.5))


def test_predicted_probabilities_are_monotone_and_valid():
    x, y = _make()
    fit = ordinal_probit_fit(x, y, 5)
    grid = np.linspace(-3, 3, 50)
    cum = predict_cumulative(fit, grid)
    assert cum.shape == (50, 4)
    assert np.all((cum >= 0) & (cum <= 1))
    # P(R<=1) <= P(R<=2) <= ... at every x
    assert np.all(np.diff(cum, axis=1) >= -1e-9)
    exp = predict_expected_category(fit, grid)
    assert np.all((exp >= 1) & (exp <= 5))
    # a positive slope means the expected category RISES with x
    assert exp[-1] > exp[0]


def test_predict_refuses_a_nonconvergent_fit():
    bad = ordinal_probit_fit(np.zeros(10), np.ones(10, int), 5)
    with pytest.raises(ValueError):
        predict_cumulative(bad, np.array([0.0]))
