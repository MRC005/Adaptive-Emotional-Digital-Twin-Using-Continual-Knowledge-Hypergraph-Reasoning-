"""MANDATORY: the bootstrap resamples PARTICIPANTS, not observations.

ROUND-17 §W rule 6. Resampling observations treats repeated measures as
independent participants and understates the interval. This is a bug, not an
option, and the schema refuses it.
"""
from __future__ import annotations

import numpy as np
import pytest

from aedt.constants import BOOTSTRAP_B, SEED
from aedt.errors import DecisionRequired
from aedt.inference.bootstrap import (MIN_PARTICIPANTS_FOR_CI,
                                      bootstrap_participants)
from aedt.schemas import UncertaintyResult


def test_schema_refuses_any_other_resampling_unit():
    with pytest.raises(DecisionRequired, match="resample participants"):
        UncertaintyResult(method="bad", n_participants=10, n_resamples=100,
                          point=1.0, ci_low=0.9, ci_high=1.1,
                          resampling_unit="observation")


def test_resamples_participants_not_observations():
    """With P participants the bootstrap mean must only ever be an average of
    P DRAWS FROM THOSE P VALUES -- so it can never leave their range."""
    logs = np.log(np.array([0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]))
    u = bootstrap_participants(logs, n_resamples=500, seed=SEED)
    assert u.resampling_unit == "participant"
    assert u.n_participants == 10
    assert np.exp(logs.min()) <= u.ci_low <= u.ci_high <= np.exp(logs.max())


def test_deterministic_under_the_frozen_seed():
    logs = np.log(np.linspace(0.8, 1.2, 25))
    a = bootstrap_participants(logs, n_resamples=400, seed=SEED)
    b = bootstrap_participants(logs, n_resamples=400, seed=SEED)
    assert (a.ci_low, a.ci_high) == (b.ci_low, b.ci_high)
    c = bootstrap_participants(logs, n_resamples=400, seed=SEED + 1)
    assert (c.ci_low, c.ci_high) != (a.ci_low, a.ci_high)


def test_too_few_participants_yields_no_interval_rather_than_a_fake_one():
    u = bootstrap_participants(np.log([0.9, 1.0, 1.1]), n_resamples=200)
    assert u.n_participants == 3 < MIN_PARTICIPANTS_FOR_CI
    assert u.n_resamples == 0
    assert np.isnan(u.ci_low) and np.isnan(u.ci_high)
    assert not u.excludes_null          # an absent interval never "rejects"


def test_default_resample_count_is_the_frozen_2000():
    u = bootstrap_participants(np.log(np.linspace(0.9, 1.1, 12)))
    assert u.n_resamples == BOOTSTRAP_B == 2000


def test_non_finite_values_are_dropped_not_propagated():
    logs = [0.0, 0.1, np.nan, -0.1, np.inf] + [0.02] * 10
    u = bootstrap_participants(logs, n_resamples=200, seed=SEED)
    assert u.n_participants == 13
    assert np.isfinite(u.point)


def test_ci_narrows_as_participants_increase():
    rng = np.random.default_rng(SEED)
    wide = bootstrap_participants(rng.normal(0, 0.2, 12), n_resamples=800,
                                  seed=SEED)
    narrow = bootstrap_participants(rng.normal(0, 0.2, 400), n_resamples=800,
                                    seed=SEED)
    assert (narrow.ci_high - narrow.ci_low) < (wide.ci_high - wide.ci_low)
