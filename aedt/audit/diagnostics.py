"""ASSUMPTION DIAGNOSTICS, including [9b] -- the first number read on real data.

[9b] SENSOR-REPORT ASSOCIATION STRENGTH. The single best predictor of whether a
study can detect anything at all. A weak association makes every per-person
slope noisy and the pooled CI uselessly wide REGARDLESS of how well calibrated
the estimator is: the synthetic fixture produced a CI of [0.768, 1.450] with a
perfectly calibrated estimator, purely because the association was weak.

CALIBRATION AND USEFULNESS ARE DIFFERENT PROPERTIES, and only the association
strength predicts the second. Read [9b] BEFORE the placebo and long before the
primary estimate (ROUND-17 §R step 2).

The other diagnostics map one-to-one onto assumptions A1-A5:
  A1  linear vs spline sensed-level term (curvature check)
  A2  untestable by construction -- it IS the null
  A3  Var(s) epoch ratio
  A4  lag-1/2/7 autocorrelation of the sensor within epoch
  A5  category usage and standardised sensor SD
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..constants import WEAK_ASSOCIATION_BETA, DataStatus
from ..models.ordinal import ordinal_probit_fit
from ..schemas import Serialisable

__all__ = ["acf", "association_strength", "AssociationStrength",
           "assumption_diagnostics", "curvature_check"]


def acf(v, lag: int) -> float:
    """Lag-``lag`` autocorrelation. Verbatim from ``final_audit.py::acf``."""
    v = np.asarray(v, dtype=float)
    if len(v) <= lag + 2 or np.std(v) < 1e-12:
        return float("nan")
    a, b = v[:-lag], v[lag:]
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


@dataclass(frozen=True)
class AssociationStrength(Serialisable):
    """Diagnostic [9b]. Read this first."""

    n_determined: int
    n_screened: int
    median_abs_beta: float
    iqr_low: float
    iqr_high: float
    share_below_0_10: float
    weak: bool
    recommendation: str
    data_status: DataStatus = DataStatus.SYNTHETIC


def association_strength(df: pd.DataFrame, sensor: str, K: int, *,
                         epoch: int = 0,
                         threshold: float = WEAK_ASSOCIATION_BETA,
                         data_status: DataStatus = DataStatus.SYNTHETIC
                         ) -> AssociationStrength:
    """Distribution of |beta| in epoch 1 across participants."""
    from ..estimators.slope_ratio import standardise_within_epoch

    betas = []
    n_screened = 0
    for pid, g in df.groupby("pid", sort=True):
        n_screened += 1
        ge = g[g["epoch"] == epoch].sort_values("ts")
        s = ge[sensor].to_numpy(dtype=float)
        if len(ge) < 5 or np.std(s) < 1e-12:
            continue
        x, m, sd = standardise_within_epoch(s)
        fit = ordinal_probit_fit(x, ge["report"].to_numpy(int), K, pid=str(pid),
                                 epoch=epoch, standardiser=(m, sd),
                                 data_status=data_status)
        if fit.converged:
            betas.append(abs(fit.beta))
    b = np.asarray(betas, dtype=float)
    if len(b) == 0:
        return AssociationStrength(
            0, n_screened, float("nan"), float("nan"), float("nan"),
            float("nan"), True,
            "No determined slopes -- the sensor carries no usable signal. "
            "Do not interpret any downstream estimate.", data_status)
    med = float(np.median(b))
    weak = med < threshold
    return AssociationStrength(
        n_determined=int(len(b)), n_screened=n_screened, median_abs_beta=med,
        iqr_low=float(np.percentile(b, 25)), iqr_high=float(np.percentile(b, 75)),
        share_below_0_10=float(np.mean(b < 0.10)), weak=weak,
        recommendation=(
            f"WEAK ASSOCIATION (median |beta| = {med:.3f} < {threshold}). "
            "Per-person slopes will be noisy and the pooled CI wide. Switch to "
            "the pre-specified PC1 fallback covariate BEFORE interpreting "
            "anything." if weak else
            f"Association adequate (median |beta| = {med:.3f}). Proceed to the "
            "placebo."),
        data_status=data_status)


def curvature_check(df: pd.DataFrame, sensor: str, K: int) -> pd.DataFrame:
    """A1 diagnostic: does a quadratic term in x improve the epoch-1 fit?

    Reported as the share of participants for whom adding x^2 raises the
    log-likelihood by more than the 0.95 chi-square(1) critical value. This is
    a descriptive flag: residual curvature beyond the threshold structure
    biases the null by up to -0.068 (ROUND-14 §7), which is the number that
    goes in the limitations.
    """
    from ..estimators.slope_ratio import standardise_within_epoch
    from scipy.optimize import minimize
    from scipy.stats import norm

    rows = []
    for pid, g in df.groupby("pid", sort=True):
        g0 = g[g["epoch"] == 0].sort_values("ts")
        s = g0[sensor].to_numpy(dtype=float)
        y = g0["report"].to_numpy(dtype=int)
        if len(y) < 30 or np.std(s) < 1e-12 or len(np.unique(y)) < 2:
            continue
        x, _m, _sd = standardise_within_epoch(s)
        lin = ordinal_probit_fit(x, y, K, pid=str(pid), epoch=0)
        if not lin.converged:
            continue

        def nll2(par):
            c0, *rest = par
            inc = np.exp(np.asarray(rest[:K - 2])) if K > 2 else np.array([])
            cuts = (np.concatenate([[c0], c0 + np.cumsum(inc)])
                    if K > 2 else np.array([c0]))
            b1, b2 = rest[K - 2], rest[K - 1]
            eta = b1 * x + b2 * x ** 2
            lo = np.where(y > 1, cuts[np.clip(y - 2, 0, K - 2)], -np.inf) - eta
            hi = np.where(y < K, cuts[np.clip(y - 1, 0, K - 2)], np.inf) - eta
            return -float(np.sum(np.log(np.clip(
                norm.cdf(hi) - norm.cdf(lo), 1e-12, None))))

        start = list(lin.cutpoints[:1]) + \
            list(np.log(np.clip(np.diff(lin.cutpoints), 1e-3, None))) + \
            [lin.beta, 0.0]
        try:
            r = minimize(nll2, start, method="BFGS",
                         options={"maxiter": 400, "gtol": 1e-5})
        except Exception:
            continue
        lr = 2.0 * (lin.loglik - (-float(r.fun)) * -1.0) if False else \
            2.0 * ((-float(r.fun)) - lin.loglik)
        rows.append({"pid": str(pid), "loglik_linear": lin.loglik,
                     "loglik_quadratic": -float(r.fun),
                     "lr_stat": lr, "curvature_flagged": bool(lr > 3.841)})
    return pd.DataFrame(rows)


def assumption_diagnostics(df: pd.DataFrame, sensor: str, K: int, *,
                           data_status: DataStatus = DataStatus.SYNTHETIC
                           ) -> pd.DataFrame:
    """Per-participant table of the observable proxies for A3, A4 and A5."""
    rows = []
    for pid, g in df.groupby("pid", sort=True):
        rec = {"pid": str(pid)}
        for e in (0, 1):
            ge = g[g["epoch"] == e].sort_values("ts")
            s = ge[sensor].to_numpy(dtype=float)
            R = ge["report"].to_numpy(dtype=int)
            rec[f"n_epoch{e}"] = len(R)
            rec[f"var_epoch{e}"] = float(np.var(s, ddof=1)) if len(s) > 2 else np.nan
            rec[f"categories_epoch{e}"] = int(len(np.unique(R))) if len(R) else 0
            rec[f"floor_epoch{e}"] = float(np.mean(R == 1)) if len(R) else np.nan
            rec[f"ceiling_epoch{e}"] = float(np.mean(R == K)) if len(R) else np.nan
            for L in (1, 2, 7):
                rec[f"ar{L}_epoch{e}"] = acf(s, L)
        v0, v1 = rec.get("var_epoch0", np.nan), rec.get("var_epoch1", np.nan)
        rec["var_ratio_A3"] = v1 / v0 if (np.isfinite(v0) and v0 > 0) else np.nan
        rec["data_status"] = data_status.value
        rows.append(rec)
    return pd.DataFrame(rows)
