"""ABLATION 1 -- CONTEXT REPRESENTATION: WITH vs WITHOUT the hypergraph.

The only experiment that connects the architecture to the title, and the one
whose result is reported WHICHEVER WAY IT FALLS (ROUND-17 §M, §AC).

Three arms, all estimating the SAME estimand rho*:

  continuous          the frozen primary: within-epoch standardised sensed
                      level, ordinal probit slope ratio, every observation used
  feature_vector      the non-graph control: epoch-1 prototypes, compensatory
                      distance, context-effect spread ratio
  nary_hyperedge      the hypergraph-native form: exact conjunctive contexts,
                      context-effect spread ratio

JUDGEMENT CRITERIA, FIXED IN ADVANCE (ROUND-17 §M):
  effect retention at matched calibration, CI width, placebo rejection rate,
  convergence rate.

They are NOT judged on which produces the largest effect.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..constants import MIN_REPORTS_PER_EPOCH, SEED, DataStatus
from ..contexts.vector import vector_context
from ..estimators.slope_ratio import estimate_rho_star
from ..estimators.spread_ratio import hyperedge_spread_ratio
from ..inference.placebo import placebo_split_half
from .structure import nary_context

log = logging.getLogger(__name__)

__all__ = ["run_context_ablation", "ABLATION_CRITERIA"]

ABLATION_CRITERIA = ("effect_retention", "ci_width", "placebo_rejects",
                     "convergence_rate", "null_calibrated")


def null_calibration_check(*, n_participants: int, n_per_epoch: int,
                           ctx_cols: list[str], sensor: str, K: int,
                           seed: int, n_resamples: int | None,
                           n_bins: int) -> dict[str, bool | None]:
    """Does each arm hold its size on a cohort where the TRUTH is rho = 1?

    This is the "matched calibration" half of the frozen judgement criteria
    (ROUND-17 §M: *effect retention AT MATCHED CALIBRATION*). Comparing CI
    widths across arms is meaningless until each arm is known to hold its size,
    because a narrow interval around a wrong answer is worse than a wide one
    around a right answer.

    WHY THE PLACEBO IS NOT ENOUGH. The split-half placebo builds its two
    pseudo-epochs from within epoch 1, so they contain no genuine change in the
    construct. A null cohort DOES contain one (the generator applies a real
    between-epoch shift in theta). An arm that confounds genuine change with
    scale change therefore passes the placebo and still fails here -- which is
    exactly what the n-ary spread-ratio form does.

    Only available where a simulator exists, i.e. on synthetic runs. Returns
    None per arm when it cannot be judged; None never disqualifies.
    """
    from ..simulate.generator import cohort_to_long_frame, simulate_cohort

    out: dict[str, bool | None] = {"continuous": None, "feature_vector": None,
                                   "nary_hyperedge": None}
    try:
        df = cohort_to_long_frame(simulate_cohort(
            1.00, n_participants=n_participants, n_per_epoch=n_per_epoch,
            seed=seed + 5077))
    except Exception as exc:                      # pragma: no cover - defensive
        log.warning("null-calibration cohort could not be built: %s", exc)
        return out

    cont = estimate_rho_star(df, sensor, K, seed=seed, n_resamples=n_resamples)
    if cont.uncertainty is not None:
        out["continuous"] = not cont.uncertainty.excludes_null

    ctx = [c for c in ctx_cols if c in df.columns] or [sensor]
    vec = pd.concat([vector_context(g, ctx, seed=seed)
                     for _p, g in df.groupby("pid", sort=True)],
                    ignore_index=True)
    v = hyperedge_spread_ratio(vec.dropna(subset=["context_vector"]),
                               "context_vector", seed=seed,
                               n_resamples=n_resamples)
    if v.uncertainty is not None:
        out["feature_vector"] = not v.uncertainty.excludes_null

    nry, _g = nary_context(df, ctx, n_bins=n_bins)
    n = hyperedge_spread_ratio(nry.dropna(subset=["context_hyperedge"]),
                               "context_hyperedge", seed=seed,
                               n_resamples=n_resamples)
    if n.uncertainty is not None:
        out["nary_hyperedge"] = not n.uncertainty.excludes_null

    log.info("null calibration (truth rho = 1): %s", out)
    return out


def _categorical_placebo(df: pd.DataFrame, edge_col: str, *, seed: int,
                         n_resamples: int | None,
                         data_status: DataStatus) -> bool | None:
    """Contiguous epoch-1 split-half placebo for a CATEGORICAL context arm.

    Same design as the continuous placebo: no response shift can exist between
    two contiguous halves of epoch 1, so the arm must not reject. An arm that
    cannot be judged this way returns None rather than a flattering default.
    """
    parts = []
    for pid, g in df.groupby("pid", sort=True):
        g0 = g[g["epoch"] == 0].sort_values("ts")
        if len(g0) < 2 * MIN_REPORTS_PER_EPOCH:
            continue
        h = len(g0) // 2
        a, b = g0.iloc[:h].copy(), g0.iloc[h:].copy()
        a["epoch"] = 0
        b["epoch"] = 1
        parts.append(pd.concat([a, b]))
    if len(parts) < 10:
        return None
    res = hyperedge_spread_ratio(pd.concat(parts, ignore_index=True), edge_col,
                                 seed=seed, n_resamples=n_resamples,
                                 data_status=data_status)
    return bool(res.uncertainty.excludes_null) if res.uncertainty else None


def _row(name: str, res, true_rho: float | None,
         placebo_rejects: bool | None = None,
         null_calibrated: bool | None = None) -> dict:
    unc = res.uncertainty
    lo = unc.ci_low if unc else float("nan")
    hi = unc.ci_high if unc else float("nan")
    effect = 1.0 - res.rho_star
    retention = (effect / (1.0 - true_rho)
                 if true_rho is not None and true_rho < 1.0 else float("nan"))
    return {
        "representation": name,
        "uses_hypergraph": name == "nary_hyperedge",
        "rho_star": res.rho_star,
        "ci_low": lo,
        "ci_high": hi,
        "ci_width": hi - lo,
        "effect_retention": retention,
        "convergence_rate": (res.n_participants_used /
                             max(res.n_participants_screened, 1)),
        "n_used": res.n_participants_used,
        "n_screened": res.n_participants_screened,
        "excludes_null": unc.excludes_null if unc else False,
        # A representation that fires on a split-half of epoch 1 is detecting
        # something other than recalibration, whatever its CI width.
        "placebo_rejects": placebo_rejects,
        # Holds its size on a cohort where the truth is rho = 1. A narrow
        # interval around a wrong answer is worse than a wide correct one.
        "null_calibrated": null_calibrated,
        "diagnostic_status": res.diagnostic_status,
        "data_status": res.data_status.value,
    }


def run_context_ablation(df: pd.DataFrame, sensor: str, K: int, *,
                         ctx_cols: list[str] | None = None,
                         true_rho: float | None = None, seed: int = SEED,
                         n_resamples: int | None = 399, n_bins: int = 3,
                         check_null_calibration: bool = True,
                         data_status: DataStatus = DataStatus.SYNTHETIC
                         ) -> pd.DataFrame:
    """Run all three arms on the same eligible participants and compare."""
    ctx = ctx_cols or [sensor]
    rows = []
    nullcal: dict[str, bool | None] = {"continuous": None,
                                       "feature_vector": None,
                                       "nary_hyperedge": None}
    if check_null_calibration:
        nullcal = null_calibration_check(
            n_participants=int(df["pid"].nunique()),
            n_per_epoch=int(max(60, np.median(
                [int((g["epoch"] == 0).sum()) for _p, g in df.groupby("pid")]))),
            ctx_cols=ctx, sensor=sensor, K=K, seed=seed,
            n_resamples=n_resamples, n_bins=n_bins)

    cont = estimate_rho_star(df, sensor, K, seed=seed, n_resamples=n_resamples,
                             context_representation="continuous",
                             data_status=data_status)
    cont_placebo = placebo_split_half(df, sensor, K, seed=seed,
                                      n_resamples=n_resamples,
                                      data_status=data_status)
    rows.append(_row("continuous", cont, true_rho,
                     cont_placebo.rejected if cont_placebo.runnable else None,
                     nullcal["continuous"]))

    vec = pd.concat([vector_context(g, ctx, seed=seed)
                     for _p, g in df.groupby("pid", sort=True)],
                    ignore_index=True)
    vres = hyperedge_spread_ratio(vec.dropna(subset=["context_vector"]),
                                  "context_vector", seed=seed,
                                  n_resamples=n_resamples,
                                  representation="feature_vector",
                                  data_status=data_status)
    rows.append(_row("feature_vector", vres, true_rho,
                     _categorical_placebo(vec.dropna(subset=["context_vector"]),
                                          "context_vector", seed=seed,
                                          n_resamples=n_resamples,
                                          data_status=data_status),
                     nullcal["feature_vector"]))

    nry, graphs = nary_context(df, ctx, n_bins=n_bins)
    nres = hyperedge_spread_ratio(nry.dropna(subset=["context_hyperedge"]),
                                  "context_hyperedge", seed=seed,
                                  n_resamples=n_resamples,
                                  representation="nary_hyperedge",
                                  data_status=data_status)
    row = _row("nary_hyperedge", nres, true_rho,
               _categorical_placebo(nry.dropna(subset=["context_hyperedge"]),
                                    "context_hyperedge", seed=seed,
                                    n_resamples=n_resamples,
                                    data_status=data_status),
               nullcal["nary_hyperedge"])
    if graphs:
        row["mean_hyperedges_per_participant"] = float(
            np.mean([g.n_edges for g in graphs.values()]))
        row["mean_occupancy_overlap"] = float(np.nanmean(
            [g.occupancy_overlap() for g in graphs.values()]))
    rows.append(row)

    out = pd.DataFrame(rows)
    log.info("context ablation complete:\n%s", out.to_string(index=False))
    return out


def ablation_verdict(table: pd.DataFrame) -> str:
    """A one-sentence, honest reading of the ablation. Used on the slide."""
    if table.empty:
        return "Ablation not run."
    ok = table[np.isfinite(table["rho_star"])]
    if ok.empty:
        return ("No context representation produced a usable estimate on this "
                "data. Reported as a negative result.")
    # DISQUALIFICATION COMES BEFORE CI WIDTH. A narrow interval around a wrong
    # answer is worse than a wide interval around a right one, so an arm is
    # ranked on precision only once it is known to be usable. A None value
    # means "not judged" and disqualifies nothing.
    reasons: dict[str, list[str]] = {}

    def disqualify(mask, why: str) -> None:
        for name in ok.loc[mask, "representation"]:
            reasons.setdefault(name, []).append(why)

    if "placebo_rejects" in ok.columns:
        disqualify(ok["placebo_rejects"] == True,           # noqa: E712
                   "its placebo fired where no recalibration can exist")
    if "null_calibrated" in ok.columns:
        disqualify(ok["null_calibrated"] == False,          # noqa: E712
                   "it does not hold its size when the truth is rho = 1")
    if "effect_retention" in ok.columns:
        disqualify((ok["effect_retention"] < 0)
                   & np.isfinite(ok["effect_retention"]),
                   "it reports the effect in the WRONG DIRECTION")

    passed = ok[~ok["representation"].isin(reasons)]
    prefix = ""
    if reasons:
        prefix = ("DISQUALIFIED: "
                  + "; ".join(f"{k} ({', '.join(v)})"
                              for k, v in reasons.items()) + ". ")
    if passed.empty:
        return prefix + ("No context representation is usable on this data. "
                         "Reported as a negative result.")
    ok = passed
    best = ok.loc[ok["ci_width"].idxmin()]
    hyper = ok[ok["representation"] == "nary_hyperedge"]
    if hyper.empty and "nary_hyperedge" in reasons:
        return prefix + (
            "The continuous covariate is retained as the frozen primary. The "
            "hypergraph is the twin's contextual knowledge representation and "
            "its trust signal, NOT the methodological contribution -- and we "
            "report that rather than fudging it.")
    if best["representation"] == "nary_hyperedge":
        return prefix + ("The n-ary hyperedge representation gave the narrowest interval "
                f"(width {best['ci_width']:.3f}); the hypergraph earns its "
                "place in the architecture on this data.")
    if hyper.empty or not np.isfinite(hyper.iloc[0]["rho_star"]):
        return prefix + ("The hyperedge representation did not yield a usable estimate "
                "at this observation density, while the continuous covariate "
                "did. We report this: the hypergraph is the twin's knowledge "
                "representation, not the methodological contribution.")
    return prefix + (f"The {best['representation']} representation gave the narrowest "
            f"interval (width {best['ci_width']:.3f}) against "
            f"{hyper.iloc[0]['ci_width']:.3f} for the hyperedge form. The "
            "continuous covariate is retained as the frozen primary, and we "
            "report that the hypergraph did not improve the estimator.")
