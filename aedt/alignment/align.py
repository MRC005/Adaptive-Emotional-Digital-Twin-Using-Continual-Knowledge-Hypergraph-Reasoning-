"""MODULE 3 -- TEMPORAL ALIGNMENT.

Purpose  Attach sensor observations to self-report occasions.
Input    Timestamped sensor samples and timestamped reports.
Output   Aligned participant-time records.
Algorithm Configurable CAUSAL alignment window [ts - lookback, ts - lag].
Status   STANDARD.

CRITICAL: no future information may be used. Every sensor sample contributing
to the covariate at report time ``ts`` must satisfy ``sample_ts <= ts - lag``.
``assert_no_leakage`` enforces this and is exercised by
``tests/unit/test_leakage.py`` with a deliberately planted future sample.

A same-day aggregate is permitted only when the ``lag`` is zero AND the window
end is clamped to the report timestamp, which is what
``align_sensor_to_reports`` does; it never uses the whole calendar day when
the report arrived at midday.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..errors import DecisionRequired

log = logging.getLogger(__name__)

__all__ = ["CausalWindow", "align_sensor_to_reports", "assert_no_leakage"]


@dataclass(frozen=True)
class CausalWindow:
    """The causal window used to build a covariate for a report at time ts.

    ``lookback_hours``  how far back the window extends
    ``lag_hours``       an exclusion gap immediately before the report; use a
                        positive lag when a sensor stream is only available
                        after a processing delay
    ``min_samples``     below this, the covariate is missing, never imputed
    """

    lookback_hours: float = 24.0
    lag_hours: float = 0.0
    min_samples: int = 1
    aggregation: str = "sum"

    def bounds(self, ts: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
        end = ts - pd.Timedelta(hours=self.lag_hours)
        start = end - pd.Timedelta(hours=self.lookback_hours)
        return start, end


_AGG = {"sum": np.sum, "mean": np.mean, "median": np.median,
        "max": np.max, "min": np.min, "std": lambda v: np.std(v, ddof=1),
        "count": len}


def align_sensor_to_reports(reports: pd.DataFrame, samples: pd.DataFrame,
                            value_col: str, window: CausalWindow, *,
                            out_col: str | None = None) -> pd.DataFrame:
    """Aggregate sensor samples into a causal window ending at each report.

    ``reports`` needs columns (pid, ts); ``samples`` needs (pid, ts, value_col).
    Returns ``reports`` with the aggregated column added, plus
    ``<col>_n_samples`` and ``<col>_window_start`` / ``_window_end`` so that the
    leakage assertion has something to check.
    """
    for name, frame, cols in (("reports", reports, ("pid", "ts")),
                              ("samples", samples, ("pid", "ts", value_col))):
        missing = [c for c in cols if c not in frame.columns]
        if missing:
            raise DecisionRequired(
                f"Alignment input '{name}' lacks columns {missing}; "
                f"present: {list(frame.columns)}")
    if window.lookback_hours <= 0:
        raise DecisionRequired("Causal window lookback must be positive.")

    col = out_col or value_col
    agg = _AGG.get(window.aggregation)
    if agg is None:
        raise DecisionRequired(
            f"Unknown aggregation {window.aggregation!r}; "
            f"known: {sorted(_AGG)}")

    out = reports.sort_values(["pid", "ts"]).reset_index(drop=True).copy()
    vals = np.full(len(out), np.nan)
    counts = np.zeros(len(out), dtype=int)
    wstart = np.empty(len(out), dtype="datetime64[ns]")
    wend = np.empty(len(out), dtype="datetime64[ns]")

    by_pid = {str(p): g.sort_values("ts") for p, g in samples.groupby("pid")}
    for i, row in enumerate(out.itertuples(index=False)):
        start, end = window.bounds(row.ts)
        wstart[i], wend[i] = np.datetime64(start), np.datetime64(end)
        g = by_pid.get(str(row.pid))
        if g is None or g.empty:
            continue
        t = g["ts"].to_numpy()
        # strictly causal: (start, end], and end <= ts by construction
        m = (t > np.datetime64(start)) & (t <= np.datetime64(end))
        if m.sum() < window.min_samples:
            continue
        v = g.loc[m, value_col].to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        if len(v) < window.min_samples:
            continue
        counts[i] = len(v)
        vals[i] = float(agg(v))

    out[col] = vals
    out[f"{col}_n_samples"] = counts
    out[f"{col}_window_start"] = wstart
    out[f"{col}_window_end"] = wend
    n_missing = int(np.isnan(vals).sum())
    if n_missing:
        log.info("alignment: %d/%d reports have no sensor coverage in the "
                 "causal window (left missing, never imputed)", n_missing,
                 len(out))
    assert_no_leakage(out, col)
    return out


def assert_no_leakage(df: pd.DataFrame, col: str) -> None:
    """Raise if any aligned window ends after the report it feeds.

    This is a hard assertion, not a warning. A pipeline that leaks the future
    produces a result that cannot be interpreted, so it must not run.
    """
    end_col = f"{col}_window_end"
    if end_col not in df.columns:
        raise DecisionRequired(
            f"Cannot verify causality for {col!r}: no {end_col} column. "
            "Alignment metadata must be retained.")
    end = pd.to_datetime(df[end_col])
    bad = df[end > df["ts"]]
    if len(bad):
        raise DecisionRequired(
            f"FUTURE LEAKAGE: {len(bad)} aligned windows for {col!r} end after "
            f"the report they feed. First offender: pid={bad.iloc[0]['pid']} "
            f"report_ts={bad.iloc[0]['ts']} window_end={bad.iloc[0][end_col]}.")
