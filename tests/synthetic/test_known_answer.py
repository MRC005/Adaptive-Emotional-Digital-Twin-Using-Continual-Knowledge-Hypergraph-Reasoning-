"""MANDATORY TEST A: SYNTHETIC KNOWN-ANSWER RECOVERY.

Generate data with a known rho_star and verify recovery within a
PRE-SPECIFIED tolerance. The tolerances live in ``aedt/constants.py`` and were
fixed before these tests were written.
"""
from __future__ import annotations

import numpy as np
import pytest

from aedt.constants import SEED, TOL_NULL_RATIO
from aedt.estimators.slope_ratio import estimate_rho_star
from aedt.simulate.generator import (THRESHOLD_PLACEMENTS, cohort_to_long_frame,
                                     simulate_cohort)
from aedt.simulate.scenarios import SCENARIOS, run_scenario


def _rho_star(rho, *, placement="skewed", phi=0.4, link="identity",
              P=50, n=300, seed=SEED):
    df = cohort_to_long_frame(simulate_cohort(
        rho, n_participants=P, n_per_epoch=n, seed=seed, placement=placement,
        phi=phi, link=link))
    return estimate_rho_star(df, "conversation_minutes", 5,
                             bootstrap=False).rho_star


@pytest.mark.parametrize("placement", sorted(THRESHOLD_PLACEMENTS))
def test_null_calibration_across_threshold_placements(placement):
    """The decisive property: under the TRUE NULL the ratio is 1, including on
    floor-heavy scales where ~68% of responses sit in category 1."""
    est = _rho_star(1.00, placement=placement)
    assert abs(est - 1.0) < TOL_NULL_RATIO, (
        f"null rho* = {est:.4f} on '{placement}' usage; the estimator reports "
        "scale change where none exists")


@pytest.mark.parametrize("phi", [0.0, 0.6])
def test_null_calibration_under_serial_dependence(phi):
    """AR(1) sensor noise affects the VARIANCE, not the point estimate."""
    est = _rho_star(1.00, phi=phi)
    assert abs(est - 1.0) < TOL_NULL_RATIO, f"null rho* = {est:.4f} at phi={phi}"


@pytest.mark.parametrize("rho", [0.85, 0.70])
def test_attenuation_is_conservative_never_inflated(rho):
    """rho* must lie in (rho, 1.00): attenuated toward the null. This is what
    makes 1 - rho* a LOWER bound rather than a point estimate of rho."""
    est = _rho_star(rho)
    assert rho < est < 1.00, (
        f"true rho = {rho} gave rho* = {est:.4f}, outside ({rho}, 1.00)")


def test_ordering_is_monotone_in_the_truth():
    """A stronger true recalibration must give a smaller rho*."""
    ests = [_rho_star(r) for r in (1.00, 0.85, 0.70)]
    assert ests[0] > ests[1] > ests[2], ests


def test_saturating_link_bias_stays_within_the_documented_limit():
    """A1 violation. Residual curvature beyond the threshold structure biases
    the null by up to -0.068 (ROUND-14 §7). That number is a stated limitation,
    so the test pins it rather than pretending it is zero."""
    est = _rho_star(1.00, link="tanh", phi=0.0)
    assert abs(est - 1.0) < 0.09, (
        f"saturating-link null rho* = {est:.4f}; the documented envelope is "
        "about -0.068")


def test_every_scenario_is_runnable_and_labelled_synthetic():
    for name in SCENARIOS:
        out = run_scenario(name, 1.00, n_participants=20, n_per_epoch=120,
                           bootstrap=False, n_resamples=None)
        assert out["data_status"] == "SYNTHETIC"
        assert np.isfinite(out["rho_star"]), f"{name} produced no estimate"
        assert out["tests_assumption"], f"{name} does not say what it tests"


def test_weak_association_widens_the_interval_without_breaking_calibration():
    """Calibration and usefulness are DIFFERENT properties. This is the lesson
    the Round-16 fixture taught, encoded as a test."""
    clean = run_scenario("clean_sensor", 1.00, n_participants=40,
                         n_per_epoch=250, n_resamples=399)
    noisy = run_scenario("noisy_sensor", 1.00, n_participants=40,
                         n_per_epoch=250, n_resamples=399)
    assert abs(clean["rho_star"] - 1.0) < 0.05
    assert abs(noisy["rho_star"] - 1.0) < 0.08
    assert ((noisy["ci_high"] - noisy["ci_low"])
            > (clean["ci_high"] - clean["ci_low"])), (
        "a weaker sensor-report association must widen the interval")


def test_generator_is_deterministic_under_the_frozen_seed():
    a = cohort_to_long_frame(simulate_cohort(0.85, n_participants=5,
                                             n_per_epoch=60, seed=SEED))
    b = cohort_to_long_frame(simulate_cohort(0.85, n_participants=5,
                                             n_per_epoch=60, seed=SEED))
    assert a.equals(b)
    c = cohort_to_long_frame(simulate_cohort(0.85, n_participants=5,
                                             n_per_epoch=60, seed=SEED + 1))
    assert not a["report"].equals(c["report"])


def test_floor_heavy_placement_actually_produces_a_floor():
    """The generator must reproduce the regime it claims to: ~45% in category 1
    for 'skewed' and ~65% for 'extreme_floor'."""
    for placement, lo, hi in (("balanced", 0.10, 0.30),
                              ("skewed", 0.35, 0.55),
                              ("extreme_floor", 0.55, 0.75)):
        df = cohort_to_long_frame(simulate_cohort(
            1.00, n_participants=20, n_per_epoch=300, seed=SEED,
            placement=placement))
        e0 = df[df["epoch"] == 0]
        floor = float((e0["report"] == 1).mean())
        assert lo < floor < hi, f"{placement}: floor rate {floor:.1%}"
