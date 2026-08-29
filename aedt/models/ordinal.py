"""MODULE 9 -- ORDINAL SELF-REPORT MODEL.

Purpose  Model an ordinal self-report as what it is: a latent continuous
         feeling cut into K bands by K-1 person-specific thresholds.
Input    x (within-epoch standardised sensor covariate), y (1..K), K.
Output   OrdinalFit: slope beta, cutpoints c_k, convergence status.
Algorithm Maximum likelihood on
             P(R <= k | x) = Phi(c_k - beta * x)
         with the cutpoints parameterised as c_0 and log-increments so that
         ordering c_1 < c_2 < ... < c_{K-1} holds by construction.
Status   STANDARD MODELING COMPONENT USED IN OUR CONTRIBUTION
         (McCullagh 1980). ``statsmodels`` is deliberately not used -- see
         ROUND-17 §S.13.

PORTED VERBATIM in its numerical logic from the validated research code
(``ordinal.py::ordinal_probit`` and ``final_audit.py::ordinal_probit``,
Rounds 14-16). Only the packaging changed: the function now returns a typed
``OrdinalFit`` carrying the cutpoints, which the two-curve visualisation needs.

Sign convention: beta is NOT constrained to be positive. Conversation minutes
fall as stress rises, so the true slope is negative for the primary feature.
Requiring beta > 0 cut the fixture sample from 48 to 2 (self-correction 26).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

from ..constants import MIN_ABS_BETA, DataStatus
from ..schemas import OrdinalFit

__all__ = ["ordinal_probit", "ordinal_probit_fit", "predict_cumulative",
           "cutpoints_from_params", "negative_log_likelihood"]


def cutpoints_from_params(par: np.ndarray, K: int) -> np.ndarray:
    """Unpack (c_0, log-increments) into the ordered cutpoint vector."""
    c0 = par[0]
    if K > 2:
        inc = np.exp(par[1:K - 1])
        return np.concatenate([[c0], c0 + np.cumsum(inc)])
    return np.array([c0])


def negative_log_likelihood(par: np.ndarray, x: np.ndarray, y: np.ndarray,
                            K: int) -> float:
    """Negative log-likelihood of the ordered probit. Verbatim from Round 14."""
    cuts = cutpoints_from_params(par, K)
    beta = par[-1]
    lo = np.where(y > 1, cuts[np.clip(y - 2, 0, K - 2)], -np.inf) - beta * x
    hi = np.where(y < K, cuts[np.clip(y - 1, 0, K - 2)], np.inf) - beta * x
    p = norm.cdf(hi) - norm.cdf(lo)
    return -float(np.sum(np.log(np.clip(p, 1e-12, None))))


# alias kept so the ported provenance is obvious in a diff
_nll = negative_log_likelihood


def ordinal_probit_fit(x, y, K: int, *, pid: str = "", epoch: int = -1,
                       min_abs_beta: float = MIN_ABS_BETA,
                       standardiser: tuple[float, float] | None = None,
                       data_status: DataStatus = DataStatus.SYNTHETIC
                       ) -> OrdinalFit:
    """Fit the ordinal probit by ML and return a typed, auditable result.

    A non-convergent or degenerate fit returns ``converged=False`` with a
    stated reason and ``beta = nan``; it never returns a silent default.
    """
    y = np.asarray(y, dtype=int)
    x = np.asarray(x, dtype=float)
    mean, sd = (standardiser if standardiser is not None
                else (float("nan"), float("nan")))
    n = int(len(y))

    def fail(reason: str) -> OrdinalFit:
        return OrdinalFit(pid=pid, epoch=epoch, beta=float("nan"),
                          cutpoints=(), n=n, n_categories=K, converged=False,
                          reason=reason, standardiser_mean=mean,
                          standardiser_sd=sd, data_status=data_status)

    if n == 0:
        return fail("no observations")
    if len(np.unique(y)) < 2:
        return fail("fewer than 2 response categories used")
    if np.std(x) < 1e-9:
        return fail("no variation in the sensor covariate")

    q = np.clip(np.cumsum(np.bincount(y, minlength=K + 1)[1:K]) / len(y), .01, .99)
    c = norm.ppf(q)
    start = np.concatenate([[c[0]], np.log(np.clip(np.diff(c), 1e-3, None)), [0.5]])
    try:
        r = minimize(_nll, start, args=(x, y, K), method="BFGS",
                     options={"maxiter": 400, "gtol": 1e-5})
    except Exception as exc:                      # pragma: no cover - defensive
        return fail(f"optimiser raised {type(exc).__name__}: {exc}")

    beta = float(r.x[-1])
    # status 2 = "desired error not necessarily achieved due to precision loss",
    # which BFGS reports at a good optimum on flat likelihoods. Accepted in the
    # validated research code and kept identical here.
    converged = bool(r.success or r.status == 2)
    if not converged:
        return fail(f"optimiser did not converge (status {r.status})")
    if abs(beta) < min_abs_beta:
        return fail(f"|beta| = {abs(beta):.4f} below the floor {min_abs_beta}")
    return OrdinalFit(pid=pid, epoch=epoch, beta=beta,
                      cutpoints=tuple(float(v) for v in
                                      cutpoints_from_params(r.x, K)),
                      n=n, n_categories=K, converged=True,
                      loglik=float(-r.fun), standardiser_mean=mean,
                      standardiser_sd=sd, data_status=data_status)


def ordinal_probit(x, y, K: int, min_abs_beta: float = MIN_ABS_BETA) -> float:
    """Slope-only interface, identical in behaviour to the research code.

    Returns nan if the fit is degenerate, non-convergent, or the slope is not
    well determined. Kept because the ported estimator and the regression tests
    are written against this signature.
    """
    fit = ordinal_probit_fit(x, y, K, min_abs_beta=min_abs_beta)
    return fit.beta if fit.converged else float("nan")


def predict_cumulative(fit: OrdinalFit, x_grid: np.ndarray) -> np.ndarray:
    """P(R <= k | x) for every k = 1..K-1 over a grid of x.

    Returns an array of shape (len(x_grid), K-1). This is what the two-curve
    epoch-1 vs epoch-2 figure plots.
    """
    if not fit.converged or not fit.cutpoints:
        raise ValueError("Cannot predict from a non-convergent fit.")
    x_grid = np.asarray(x_grid, dtype=float)
    cuts = np.asarray(fit.cutpoints, dtype=float)
    return norm.cdf(cuts[None, :] - fit.beta * x_grid[:, None])


def predict_expected_category(fit: OrdinalFit, x_grid: np.ndarray) -> np.ndarray:
    """E[R | x] under the fit -- a single readable curve per epoch."""
    cum = predict_cumulative(fit, x_grid)
    K = fit.n_categories
    # P(R = k) = P(R<=k) - P(R<=k-1); expected value = sum_k k * P(R=k)
    full = np.hstack([np.zeros((len(cum), 1)), cum, np.ones((len(cum), 1))])
    probs = np.diff(full, axis=1)
    return probs @ np.arange(1, K + 1, dtype=float)
