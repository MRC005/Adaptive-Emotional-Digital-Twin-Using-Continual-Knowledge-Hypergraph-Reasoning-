"""Bias envelope and the pre-specified baselines."""
from __future__ import annotations

import numpy as np
import pytest

from aedt.audit.envelope import (ENVELOPE_SCENARIOS, bias_envelope,
                                 measured_properties)
from aedt.baselines import BASELINES, run_baselines
from aedt.constants import SEED


@pytest.fixture(scope="module")
def envelope():
    return bias_envelope(n_participants=20, n_per_epoch=150, n_replications=3,
                         scenarios=("balanced", "skewed", "extreme_floor",
                                    "skewed_ar1"))


def test_envelope_brackets_the_null(envelope):
    assert envelope.envelope_low < 1.0 < envelope.envelope_high, (
        f"the null envelope [{envelope.envelope_low:.3f}, "
        f"{envelope.envelope_high:.3f}] does not contain rho = 1")


def test_envelope_reports_every_scenario_it_was_given(envelope):
    assert set(envelope.rho_star_by_scenario) == set(envelope.scenarios)
    assert all(np.isfinite(v) for v in envelope.rho_star_by_scenario.values())


def test_envelope_scenarios_are_enumerated_in_advance():
    """They are NOT chosen to optimise the result."""
    assert len(ENVELOPE_SCENARIOS) == 9
    from aedt.simulate.scenarios import SCENARIOS
    assert set(ENVELOPE_SCENARIOS) <= set(SCENARIOS)


def test_envelope_interpretation_states_what_it_means(envelope):
    t = envelope.interpretation
    assert "TRUE NULL" in t
    assert "cannot be distinguished from an artefact" in t
    assert envelope.data_status.value == "SYNTHETIC"


def test_measured_properties_describe_the_analysed_data(small_frame):
    m = measured_properties(small_frame, "conversation_minutes", 5)
    assert 0.0 <= m["median_floor_rate"] <= 1.0
    assert m["median_obs_per_epoch"] > 0
    assert m["n_participants"] == small_frame["pid"].nunique()
    assert np.isfinite(m["median_var_ratio"])


def test_unknown_envelope_scenario_is_refused():
    with pytest.raises(KeyError):
        bias_envelope(scenarios=("not_a_scenario",), n_replications=1)


# --------------------------------------------------------------- baselines
def test_all_six_prespecified_baselines_run(null_frame):
    t = run_baselines(null_frame, "conversation_minutes", 5, n_resamples=99)
    assert set(t["name"]) == set(BASELINES)
    assert len(t) == 6


def test_no_baseline_claims_to_estimate_rho_star(null_frame):
    """Only OUR estimator targets rho*. A baseline that claimed to would be a
    misleading comparison."""
    t = run_baselines(null_frame, "conversation_minutes", 5, n_resamples=99)
    assert not t["estimates_rho_star"].any()


def test_every_baseline_says_what_it_represents_and_its_limitation(null_frame):
    t = run_baselines(null_frame, "conversation_minutes", 5, n_resamples=99)
    for _i, r in t.iterrows():
        assert r["represents"], r["name"]
        assert len(r["interpretation"]) > 60, r["name"]


def test_epoch_normalisation_is_exactly_zero_by_construction(null_frame):
    """The common ad-hoc fix removes the very information the estimand needs."""
    t = run_baselines(null_frame, "conversation_minutes", 5,
                      which=["epoch_normalised"], n_resamples=99)
    assert abs(t.iloc[0]["statistic"]) < 1e-9
    assert "Identically zero by construction" in t.iloc[0]["interpretation"]


def test_a_failing_baseline_is_reported_not_swallowed(null_frame):
    t = run_baselines(null_frame, "conversation_minutes", 5,
                      which=["koren_drift"], n_resamples=99)
    assert len(t) == 1
    assert "data_status" in t.columns


def test_unknown_baseline_is_refused(null_frame):
    with pytest.raises(KeyError):
        run_baselines(null_frame, "conversation_minutes", 5,
                      which=["make_us_look_good"])
