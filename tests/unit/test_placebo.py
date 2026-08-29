"""MANDATORY PLACEBO BEHAVIOUR TESTS.

ROUND-17 §W: "placebo rejects at <= 10% on data generated WITH a true rho = 0.70
shift" -- that is, the control must NOT fire merely because temporal structure
exists. It is validated in three regimes.
"""
from __future__ import annotations

import numpy as np
import pytest

from aedt.constants import SEED
from aedt.inference.placebo import placebo_split_half
from aedt.schemas import PlaceboResult
from aedt.simulate.generator import cohort_to_long_frame, simulate_cohort


@pytest.mark.parametrize("true_rho", [1.00, 0.85, 0.70])
def test_placebo_does_not_fire_even_when_a_real_shift_is_present(true_rho):
    """The decisive property: a contiguous epoch-1 split-half contains NO
    response shift by construction, so the control must not reject even when
    the full series does contain a genuine 30% recalibration."""
    df = cohort_to_long_frame(simulate_cohort(
        true_rho, n_participants=48, n_per_epoch=300, seed=SEED))
    p = placebo_split_half(df, "conversation_minutes", 5, n_resamples=999)
    assert p.runnable
    assert not p.rejected, (
        f"placebo fired at true rho = {true_rho} (rho*={p.rho_star:.3f}, "
        f"CI [{p.ci_low:.3f}, {p.ci_high:.3f}]); it is detecting temporal "
        "structure rather than recalibration")


def test_placebo_gates_the_primary_when_it_rejects():
    p = PlaceboResult(n_participants=30, rho_star=0.80, ci_low=0.70,
                      ci_high=0.90, rejected=True, verdict="REJECTS")
    assert p.gates_primary


def test_placebo_gates_the_primary_when_it_cannot_run():
    p = PlaceboResult(n_participants=2, rho_star=float("nan"),
                      ci_low=float("nan"), ci_high=float("nan"),
                      rejected=False, verdict="NOT RUNNABLE", runnable=False)
    assert p.gates_primary


def test_placebo_reports_unrunnable_rather_than_guessing():
    """Epoch 1 must hold TWICE the minimum before it can be halved."""
    df = cohort_to_long_frame(simulate_cohort(
        1.00, n_participants=20, n_per_epoch=70, seed=SEED))
    p = placebo_split_half(df, "conversation_minutes", 5, n_resamples=199)
    assert not p.runnable
    assert p.gates_primary
    assert "NOT RUNNABLE" in p.verdict
    assert np.isnan(p.rho_star)


def test_placebo_split_is_contiguous_not_shuffled(monkeypatch):
    """Contiguous halves preserve serial dependence; shuffling would destroy it
    and make the control pass trivially."""
    seen = {}
    import aedt.inference.placebo as mod
    real = mod.person_log_ratio if hasattr(mod, "person_log_ratio") else None

    from aedt.estimators import slope_ratio as sr
    original = sr.person_log_ratio

    def spy(g, sensor, K, **kw):
        a = g[g["epoch"] == 0]["ts"]
        b = g[g["epoch"] == 1]["ts"]
        seen["contiguous"] = bool(a.max() <= b.min())
        seen["ordered"] = bool(a.is_monotonic_increasing
                               and b.is_monotonic_increasing)
        return original(g, sensor, K, **kw)

    monkeypatch.setattr(sr, "person_log_ratio", spy)
    df = cohort_to_long_frame(simulate_cohort(
        1.00, n_participants=12, n_per_epoch=300, seed=SEED))
    placebo_split_half(df, "conversation_minutes", 5, n_resamples=99)
    assert seen.get("contiguous") is True
    assert seen.get("ordered") is True
