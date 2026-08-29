"""THE FAILURE ANALYSIS OF THE NATURAL AFFINE APPROACH.

** OUR RESEARCH CONTRIBUTION -- a documented NEGATIVE result (ROUND-17 §E.3). **

PORTED VERBATIM in its numerical logic from ``likert_fix.py`` (Round 14, "O6").

WHAT THIS ESTABLISHES. The obvious way to attack the problem is an affine
difference-in-differences over recurring contexts: for each anchor, regress the
change in mean self-report on the epoch-1 level and the change in the sensed
level, correct the moments for sampling noise, and read rho off the
coefficients. Run on CONTINUOUS responses it is nearly unbiased (-0.008 under
the null). Run on a 5-point LIKERT response with one fixed scale per person it
fabricates a **-0.107 null bias** -- roughly 10% apparent scale compression
when nothing whatsoever has changed. On 7-point responses it is identical
(-0.107), so the problem is not resolution: it is that the estimator's LEVEL
coefficient absorbs threshold saturation.

WHY THIS IS A CONTRIBUTION AND NOT AN EMBARRASSMENT. It is the reason the
ordinal slope-ratio construction exists. The boundary is not noise to be
corrected analytically; it IS the measurement model. Five remedies were tried
and killed.

SELF-CORRECTION (Round 14, recorded). An EARLIER round reported -0.19 and froze
the project around it. That number came from a harness bug: each anchor's
responses were separately re-centred and re-scaled by that anchor's own 15
observations, erasing the between-anchor level variation the estimator reads.
A person has ONE response scale, not one per context. Both code paths are
retained below -- ``per-anchor`` reproduces the withdrawn -0.19 artefact and
``per-person`` the real -0.107 -- so the correction stays verifiable rather
than merely asserted.

THE TESTS ASSERT THE FAILURE. ``tests/regression/test_known_failures.py``
requires the -0.107 to be reproduced, so a future "improvement" that silently
breaks the baseline fails loudly instead of flattering our method.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..constants import SEED
from ..schemas import Serialisable

__all__ = ["affine_did_null_bias", "run_affine_did", "AffineDiDResult",
           "DISCRETISATION_MODES"]

DISCRETISATION_MODES = ("continuous", "per-person", "per-anchor")


@dataclass(frozen=True)
class AffineDiDResult(Serialisable):
    mode: str
    n_categories: int | None
    true_rho: float
    rho_hat: float
    bias: float
    boundary_rate_pct: float
    n_replications: int
    note: str = ""


def _demean(x: np.ndarray, pid: np.ndarray) -> np.ndarray:
    o = np.empty_like(x, dtype=float)
    for q in np.unique(pid):
        k = pid == q
        o[k] = x[k] - x[k].mean()
    return o


def _fit_pooled(dr, ds, s1, wv, pid, m):
    y = _demean(dr, pid)
    X = np.column_stack([_demean(s1, pid), _demean(ds, pid)])
    dfc = max(len(y) - len(np.unique(pid)), 1)
    e = float(np.mean(wv)) / m
    A = X.T @ X / dfc - np.array([[e, -e], [-e, 2 * e]])
    if np.min(np.linalg.eigvalsh(A)) <= 1e-9:
        return np.nan
    g = np.linalg.solve(A, X.T @ y / dfc)
    if abs(g[1]) < 1e-9 or abs(1 - g[0] / g[1]) < 1e-9:
        return np.nan
    return 1.0 / (1.0 - g[0] / g[1])


def _cohort(rng, P, C, m, sd, rho_bar, K, mode):
    DR, DS, S1, WV, PID, EDGE = [], [], [], [], [], []
    for pid in range(P):
        lam = rng.uniform(0.35, 0.90)
        smu = rng.uniform(0.6, 1.4)
        a2 = rng.normal(rho_bar, 0.10)
        off = rng.normal(0.4, 0.3)
        ms = rng.normal(-0.5, 0.4)
        mu = rng.normal(0, smu, C)
        shift = ms + rng.normal(0, sd / lam, C)

        R0 = np.empty((C, m)); R1 = np.empty((C, m))
        Q0 = np.empty((C, m)); Q1 = np.empty((C, m))
        for c in range(C):
            th0 = mu[c] + rng.normal(0, 0.25, m)
            th1 = mu[c] + shift[c] + rng.normal(0, 0.25, m)
            R0[c] = th0 + rng.normal(0, 1.0, m)
            R1[c] = a2 * th1 + 1.30 + rng.normal(0, 1.0, m)
            Q0[c] = lam * th0 + off + rng.normal(0, 1.0, m)
            Q1[c] = lam * th1 + off + rng.normal(0, 1.0, m)

        if mode == "per-anchor":
            # THE WITHDRAWN CODE PATH. Retained so the self-correction stays
            # verifiable: it re-scales each anchor by its OWN 15 observations,
            # erasing the between-anchor level variation the estimator reads.
            for c in range(C):
                g = ((K - 1) / 4.0) / max(np.std(R0[c]), 1e-9)
                ctr = (K + 1) / 2.0
                R0[c] = np.clip(np.round(ctr + g * R0[c]), 1, K)
                R1[c] = np.clip(np.round(ctr + g * R1[c]), 1, K)
        elif mode == "per-person":
            # CORRECT: one fixed response scale per person, from epoch-1 data.
            g = ((K - 1) / 4.0) / max(np.std(R0), 1e-9)
            ctr = (K + 1) / 2.0
            R0 = np.clip(np.round(ctr + g * R0), 1, K)
            R1 = np.clip(np.round(ctr + g * R1), 1, K)
        # mode == "continuous": no discretisation, the reference

        dr = R1.mean(1) - R0.mean(1)
        ds = Q1.mean(1) - Q0.mean(1)
        s1 = Q0.mean(1)
        wv = 0.5 * (Q0.var(1, ddof=1) + Q1.var(1, ddof=1))
        sr = float(R0.std(ddof=1))
        sq = float(Q0.std(ddof=1))
        if sr < 1e-8 or sq < 1e-8:
            continue
        if K is not None and mode != "continuous":
            EDGE.append(float(np.mean((R0 == 1) | (R0 == K))))
        DR.append(dr / sr); DS.append(ds / sq); S1.append(s1 / sq)
        WV.append(wv / sq ** 2); PID.append(np.full(C, pid))
    if not DR:
        return None
    return (np.concatenate(DR), np.concatenate(DS), np.concatenate(S1),
            np.concatenate(WV), np.concatenate(PID),
            100 * float(np.mean(EDGE)) if EDGE else 0.0)


def run_affine_did(mode: str, K: int | None, rho: float, *, P: int = 60,
                   C: int = 14, m: int = 15, nrep: int = 250, seed: int = 0
                   ) -> AffineDiDResult:
    """One cell of the failure table. ``mode`` in DISCRETISATION_MODES."""
    if mode not in DISCRETISATION_MODES:
        raise ValueError(f"unknown mode {mode!r}; known: {DISCRETISATION_MODES}")
    rng = np.random.default_rng(SEED + seed)
    R, E = [], []
    for _ in range(nrep):
        d = _cohort(rng, P, C, m, 0.6, rho, K, mode)
        if d is None:
            continue
        v = _fit_pooled(d[0], d[1], d[2], d[3], d[4], m)
        if np.isfinite(v) and abs(v) < 5:
            R.append(v)
            E.append(d[5])
    rho_hat = float(np.median(R)) if R else float("nan")
    note = {
        "continuous": "Reference: no discretisation. Nearly unbiased.",
        "per-person": ("THE REAL FAILURE. One fixed scale per person, which is "
                       "what a Likert item is. The level coefficient absorbs "
                       "threshold saturation and fabricates apparent scale "
                       "compression under the true null."),
        "per-anchor": ("THE WITHDRAWN ARTEFACT. Re-scaling per anchor erases "
                       "the between-anchor level variation the estimator "
                       "reads; the -0.19 reported in an earlier round measured "
                       "the harness, not the instrument."),
    }[mode]
    return AffineDiDResult(mode=mode, n_categories=K, true_rho=rho,
                           rho_hat=rho_hat, bias=rho_hat - rho,
                           boundary_rate_pct=float(np.mean(E)) if E else 0.0,
                           n_replications=len(R), note=note)


def affine_did_null_bias(K: int = 5, *, nrep: int = 250, seed: int = 3 * 37
                         ) -> float:
    """The headline number: the affine estimator's NULL bias on a K-point scale.

    Reproduces -0.107 for K = 5 and K = 7 at the frozen settings.
    """
    return run_affine_did("per-person", K, 1.00, nrep=nrep, seed=seed).bias


def failure_table(nrep: int = 250) -> list[AffineDiDResult]:
    """The full three-mode table used on the 'we were wrong' slide."""
    out, s = [], 0
    for mode, K in (("continuous", None), ("per-anchor", 5),
                    ("per-person", 5), ("per-anchor", 7), ("per-person", 7)):
        for rho in (1.00, 0.85):
            s += 1
            out.append(run_affine_did(mode, K, rho, nrep=nrep, seed=s * 37))
    return out
