"""MODULE 14 -- UNCERTAINTY ESTIMATION (participant-cluster bootstrap).

Purpose  A confidence interval that respects the clustering of observations
         within participants.
Input    One log-ratio per PARTICIPANT.
Output   UncertaintyResult with a 95% percentile interval on rho*.
Algorithm Nonparametric bootstrap resampling PARTICIPANTS with replacement,
         2000 resamples, percentile interval, exponentiated back to the rho*
         scale.
Status   STANDARD / EXISTING technique; the choice of clustering unit is
         load-bearing and pre-specified.

NON-NEGOTIABLE (ROUND-17 §W rule 6): the bootstrap resamples PARTICIPANTS.
Resampling observations would treat repeated measures as independent
participants and understate the interval. ``UncertaintyResult`` refuses any
other resampling unit, and ``tests/unit/test_bootstrap.py`` asserts it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..constants import BOOTSTRAP_B, CI_ALPHA, SEED, DataStatus
from ..schemas import UncertaintyResult

__all__ = ["bootstrap_participants", "bootstrap_from_frame"]

MIN_PARTICIPANTS_FOR_CI = 10


def bootstrap_participants(log_ratios, *, n_resamples: int | None = None,
                           seed: int = SEED, alpha: float = CI_ALPHA,
                           data_status: DataStatus = DataStatus.SYNTHETIC,
                           method: str = "participant-cluster percentile bootstrap"
                           ) -> UncertaintyResult:
    """Percentile CI on exp(mean(log rho*_p)), resampling participants."""
    B = BOOTSTRAP_B if n_resamples is None else int(n_resamples)
    v = np.asarray([x for x in np.asarray(log_ratios, dtype=float)
                    if np.isfinite(x)], dtype=float)
    P = int(len(v))
    if P < MIN_PARTICIPANTS_FOR_CI:
        return UncertaintyResult(
            method=method, n_participants=P, n_resamples=0,
            point=float(np.exp(v.mean())) if P else float("nan"),
            ci_low=float("nan"), ci_high=float("nan"), seed=seed,
            data_status=data_status)
    rng = np.random.default_rng(seed)
    # resample the PARTICIPANT index, not the observation index
    idx = rng.integers(0, P, size=(B, P))
    means = v[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return UncertaintyResult(
        method=method, n_participants=P, n_resamples=B,
        point=float(np.exp(v.mean())), ci_low=float(np.exp(lo)),
        ci_high=float(np.exp(hi)), seed=seed, data_status=data_status)


def bootstrap_from_frame(df: pd.DataFrame, sensor: str, K: int, *,
                         n_resamples: int | None = None, seed: int = SEED,
                         data_status: DataStatus = DataStatus.SYNTHETIC
                         ) -> UncertaintyResult:
    """Convenience: compute the per-person log ratios, then bootstrap them."""
    from ..estimators.slope_ratio import person_log_ratio
    logs = []
    for pid, g in df.groupby("pid", sort=True):
        v, _why, _f = person_log_ratio(g, sensor, K, pid=str(pid),
                                       data_status=data_status)
        logs.append(v)
    return bootstrap_participants(logs, n_resamples=n_resamples, seed=seed,
                                  data_status=data_status)
