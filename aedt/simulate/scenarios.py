"""The thirteen misspecification scenarios validated in Rounds 14-15.

Each scenario is a named perturbation of the frozen data-generating process.
They are declared here rather than inline so that the bias envelope, the
known-answer tests and the report tables all draw on the SAME definitions and
cannot drift apart.

Every scenario output is SYNTHETIC.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..constants import SEED, DataStatus

__all__ = ["Scenario", "SCENARIOS", "run_scenario", "scenario_table"]


@dataclass(frozen=True)
class Scenario:
    """One named misspecification condition."""

    name: str
    description: str
    kwargs: dict[str, Any] = field(default_factory=dict)
    tests_assumption: str = ""


SCENARIOS: dict[str, Scenario] = {
    "balanced": Scenario(
        "balanced", "All 5 categories used equally -- the best case.",
        {"placement": "balanced", "phi": 0.0}, "A5 (category use)"),
    "balanced_ar1": Scenario(
        "balanced_ar1", "Balanced usage with AR(1) sensor noise phi=0.6.",
        {"placement": "balanced", "phi": 0.6}, "A4 (stationary sensor error)"),
    "balanced_saturating": Scenario(
        "balanced_saturating", "Balanced usage, tanh (saturating) reporting.",
        {"placement": "balanced", "phi": 0.0, "link": "tanh"},
        "A1 (affine latent reporting)"),
    "skewed": Scenario(
        "skewed", "Floor-heavy usage: ~45% of epoch-1 reports in category 1.",
        {"placement": "skewed", "phi": 0.0}, "A5 (category use)"),
    "skewed_ar1": Scenario(
        "skewed_ar1", "Floor-heavy usage with AR(1) sensor noise phi=0.6.",
        {"placement": "skewed", "phi": 0.6}, "A4"),
    "skewed_saturating": Scenario(
        "skewed_saturating", "Floor-heavy usage, saturating reporting.",
        {"placement": "skewed", "phi": 0.0, "link": "tanh"}, "A1"),
    "skewed_ar1_saturating": Scenario(
        "skewed_ar1_saturating", "Floor-heavy, AR(1) AND saturating together.",
        {"placement": "skewed", "phi": 0.6, "link": "tanh"}, "A1 + A4"),
    "extreme_floor": Scenario(
        "extreme_floor", "Hard floor: ~68% of epoch-1 reports in category 1.",
        {"placement": "extreme_floor", "phi": 0.0}, "A5"),
    "extreme_floor_ar1": Scenario(
        "extreme_floor_ar1", "Hard floor with AR(1) sensor noise phi=0.6.",
        {"placement": "extreme_floor", "phi": 0.6}, "A4 + A5"),
    "extreme_floor_saturating": Scenario(
        "extreme_floor_saturating", "Hard floor, saturating reporting.",
        {"placement": "extreme_floor", "phi": 0.0, "link": "tanh"}, "A1 + A5"),
    "noisy_sensor": Scenario(
        "noisy_sensor", "Weak sensor-report association (sigma_p = 1.6).",
        {"placement": "skewed", "phi": 0.4, "sigma_p": 1.6},
        "association strength ([9b])"),
    "clean_sensor": Scenario(
        "clean_sensor", "Strong sensor-report association (sigma_p = 0.4).",
        {"placement": "skewed", "phi": 0.4, "sigma_p": 0.4},
        "association strength ([9b])"),
    "noisy_report": Scenario(
        "noisy_report", "Noisy reporting channel (sigma_r = 1.0).",
        {"placement": "skewed", "phi": 0.4, "sigma_r": 1.0},
        "attenuation of rho* toward 1"),
}


def run_scenario(name: str, rho: float, *, n_participants: int = 48,
                 n_per_epoch: int = 200, seed: int = SEED,
                 n_resamples: int | None = 399, bootstrap: bool = True) -> dict:
    """Run one scenario at a known rho and return the estimator summary."""
    from ..estimators.slope_ratio import estimate_rho_star
    from .generator import cohort_to_long_frame, simulate_cohort

    if name not in SCENARIOS:
        raise KeyError(f"unknown scenario {name!r}; known: {sorted(SCENARIOS)}")
    sc = SCENARIOS[name]
    cohort = simulate_cohort(rho, n_participants=n_participants,
                             n_per_epoch=n_per_epoch, seed=seed, **sc.kwargs)
    df = cohort_to_long_frame(cohort)
    K = cohort[0].n_categories
    res = estimate_rho_star(df, "conversation_minutes", K, seed=seed,
                            bootstrap=bootstrap, n_resamples=n_resamples,
                            data_status=DataStatus.SYNTHETIC)
    unc = res.uncertainty
    return {
        "scenario": sc.name,
        "description": sc.description,
        "tests_assumption": sc.tests_assumption,
        "true_rho": rho,
        "rho_star": res.rho_star,
        "error": res.rho_star - rho,
        "ci_low": unc.ci_low if unc else float("nan"),
        "ci_high": unc.ci_high if unc else float("nan"),
        "n_used": res.n_participants_used,
        "n_screened": res.n_participants_screened,
        "convergence_rate": (res.n_participants_used /
                             max(res.n_participants_screened, 1)),
        "data_status": DataStatus.SYNTHETIC.value,
    }


def scenario_table(rhos=(1.00, 0.85), **kw) -> list[dict]:
    """Every scenario at every rho. This is the E3 misspecification table."""
    rows = []
    for i, name in enumerate(SCENARIOS):
        for j, rho in enumerate(rhos):
            rows.append(run_scenario(name, rho, seed=SEED + 97 * i + j, **kw))
    return rows
