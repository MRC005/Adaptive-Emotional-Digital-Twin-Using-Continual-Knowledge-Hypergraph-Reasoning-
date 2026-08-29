"""THE FAILURE ANALYSIS BASELINE: the linear anchor estimator.

** OUR RESEARCH CONTRIBUTION (as a documented NEGATIVE result). **

This is the natural affine approach -- our OWN previous method, killed in Round
14. It models a BOUNDED ORDINAL response with an AFFINE equation and then tries
to remove the resulting bias analytically. The boundary is not noise to be
corrected; it is the measurement model.

Its documented failure is a scientific asset and is asserted by a test:
under the TRUE NULL on a 5-point Likert scale it reports a spurious ~10.7%
apparent scale compression (bias -0.107, ROUND-17 §Q). ``tests`` assert that
this known failure is REPRODUCED, so that a future "fix" which silently breaks
the baseline fails loudly rather than making our method look better.

Five remedies were tried and killed (Round 14). They are not re-implemented.

PORTED VERBATIM in its numerical logic from ``ordinal.py::method_linear``.
"""
from __future__ import annotations

import numpy as np

__all__ = ["linear_anchor_ratio", "linear_anchor_from_cohort"]


def linear_anchor_ratio(people, K: int, n_anchor_bins: int = 5
                        ) -> tuple[float, float]:
    """Pooled, person-standardised, moment-corrected affine anchor estimator.

    ``people`` is a sequence of (R1, R2, s1, s2) tuples. Returns
    (rho_hat, convergence_flag). Anchors are tercile-style bins of the sensed
    level with EPOCH-1 edges, so no epoch-2 information defines the bins.
    """
    DR, DS, S1, WV, PID = [], [], [], [], []
    for i, rec in enumerate(people):
        R1, R2, s1, s2 = rec[0], rec[1], rec[2], rec[3]
        edges = np.quantile(s1, np.linspace(0, 1, n_anchor_bins + 1)[1:-1])
        b1 = np.searchsorted(edges, s1)
        b2 = np.searchsorted(edges, s2)
        dr, ds, sb, wv = [], [], [], []
        for k in range(n_anchor_bins):
            m1 = b1 == k
            m2 = b2 == k
            if m1.sum() < 3 or m2.sum() < 3:
                continue
            dr.append(R2[m2].mean() - R1[m1].mean())
            ds.append(s2[m2].mean() - s1[m1].mean())
            sb.append(s1[m1].mean())
            wv.append(0.5 * (s1[m1].var(ddof=1) / m1.sum()
                             + s2[m2].var(ddof=1) / m2.sum()))
        if len(dr) < 3:
            continue
        sr = R1.std(ddof=1)
        sq = s1.std(ddof=1)
        if sr < 1e-9 or sq < 1e-9:
            continue
        DR.append(np.array(dr) / sr)
        DS.append(np.array(ds) / sq)
        S1.append(np.array(sb) / sq)
        WV.append(np.array(wv) / sq ** 2)
        PID.append(np.full(len(dr), i))
    if len(DR) < 5:
        return float("nan"), 0.0
    dr = np.concatenate(DR)
    ds = np.concatenate(DS)
    s1a = np.concatenate(S1)
    wv = np.concatenate(WV)
    pid = np.concatenate(PID)

    def demean(v):
        o = np.empty_like(v)
        for p in np.unique(pid):
            k = pid == p
            o[k] = v[k] - v[k].mean()
        return o

    y = demean(dr)
    X = np.column_stack([demean(s1a), demean(ds)])
    dfc = max(len(y) - len(np.unique(pid)), 1)
    e = float(np.mean(wv))
    A = X.T @ X / dfc - np.array([[e, -e], [-e, 2 * e]])
    if np.min(np.linalg.eigvalsh(A)) <= 1e-9:
        return float("nan"), 0.0
    g = np.linalg.solve(A, X.T @ y / dfc)
    if abs(g[1]) < 1e-9 or abs(1 - g[0] / g[1]) < 1e-9:
        return float("nan"), 0.0
    return float(1.0 / (1.0 - g[0] / g[1])), 1.0


def linear_anchor_from_cohort(cohort, K: int) -> tuple[float, float]:
    """Adapter from ``SimulatedPerson`` records to the ported signature."""
    return linear_anchor_ratio([(p.R1, p.R2, p.s1, p.s2) for p in cohort], K)
