"""THE PREDICTABILITY CEILING — how much of day-to-day affect is forecastable.

The primary experiment answered "does personalisation beat persistence?" (no).
This module answers the follow-up: *how much signal was ever available?* Every
statistic quoted on the website under "the ceiling" is defined and computed
here, so that no number reaches a reader without a function behind it.

WHY THIS MODULE EXISTS
Until now these twelve values were literals typed into
``scripts/export_findings.py``. They may well have been correct; nothing could
show it. Definitions matter more than usual here, because several of these
quantities have more than one defensible definition and the reported figure
depends on which was used. Each function therefore states its definition, and
``scripts/run_ceiling_analysis.py`` reports the primary AND the sensitivity
variant side by side rather than choosing one silently.

DECLARED CHOICES (fixed here, before the numbers are seen again)
  * A "consecutive pair" is a report and the participant's NEXT report.
  * PRIMARY restricts pairs to a gap of 1-7 days, matching the prediction task
    that the pre-registration defines. ALL-PAIRS is reported alongside it as a
    sensitivity, because a reader may reasonably mean either.
  * A participant enters the per-person analysis with at least
    ``MIN_OBS_PER_PERSON`` reports, and each half of the early/late split needs
    at least ``MIN_PAIRS_PER_HALF`` usable pairs.
  * A participant whose stress never varies has no defined autocorrelation and
    is EXCLUDED, never imputed as 0. The exclusion count is reported.

Nothing here reads, alters or recomputes the primary experiment. It is a
description of the data, not a re-analysis of the result.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

__all__ = [
    "MIN_OBS_PER_PERSON", "MIN_PAIRS_PER_HALF", "MAX_GAP_DAYS",
    "NEAR_UNPREDICTABLE_BELOW", "WELL_PREDICTABLE_ABOVE",
    "consecutive_pairs", "person_autocorrelation", "pooled_autocorrelation",
    "icc_one_way", "per_person_autocorrelations", "early_late_stability",
    "PersonR", "CeilingStats", "ceiling_statistics",
]

#: A participant needs this many reports before a per-person autocorrelation is
#: estimated at all. Below it the estimate is dominated by sampling noise.
MIN_OBS_PER_PERSON = 30

#: Each half of the early/late split needs this many usable pairs.
MIN_PAIRS_PER_HALF = 15

#: The prediction task's horizon, from the pre-registration.
MAX_GAP_DAYS = 7

NEAR_UNPREDICTABLE_BELOW = 0.15
WELL_PREDICTABLE_ABOVE = 0.50


# --------------------------------------------------------------------- pairs
def consecutive_pairs(values, days=None, max_gap_days: int | None = MAX_GAP_DAYS):
    """Return ``(x_t, x_next)`` for one participant's ordered series.

    ``values`` must already be in time order. When ``days`` is supplied and
    ``max_gap_days`` is not None, a pair is kept only if the next report falls
    within the horizon — the same restriction the prediction task applies, so
    the ceiling describes the task that was actually run.
    """
    v = np.asarray(values, dtype=float)
    if len(v) < 2:
        return np.empty(0), np.empty(0)
    x, y = v[:-1], v[1:]
    keep = np.isfinite(x) & np.isfinite(y)
    if days is not None and max_gap_days is not None:
        d = np.asarray(days, dtype=float)
        gap = d[1:] - d[:-1]
        keep &= (gap > 0) & (gap <= max_gap_days)
    return x[keep], y[keep]


def _pearson(x, y) -> float:
    """Pearson r, or NaN when either side has no variance (never 0.0)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def person_autocorrelation(values, days=None,
                           max_gap_days: int | None = MAX_GAP_DAYS) -> float:
    """Lag-1 autocorrelation over one participant's consecutive report pairs."""
    x, y = consecutive_pairs(values, days, max_gap_days)
    return _pearson(x, y)


def pooled_autocorrelation(series_by_person, max_gap_days: int | None = MAX_GAP_DAYS
                           ) -> float:
    """WITHIN-person lag-1 autocorrelation, pooled across participants.

    Each participant's pairs are centred on that participant's own mean before
    pooling, so between-person differences in level cannot inflate the result.
    This is the quantity reported as "within-person autocorrelation"; it is
    deliberately not the same as correlating the raw pooled pairs, which would
    also absorb between-person variance.

    ``series_by_person`` is an iterable of ``(values, days)`` tuples.
    """
    xs, ys = [], []
    for values, days in series_by_person:
        x, y = consecutive_pairs(values, days, max_gap_days)
        if len(x) < 2:
            continue
        m = float(np.mean(np.concatenate([x, y])))
        xs.append(x - m)
        ys.append(y - m)
    if not xs:
        return float("nan")
    return _pearson(np.concatenate(xs), np.concatenate(ys))


# ----------------------------------------------------------------------- ICC
def icc_one_way(groups) -> float:
    """ICC(1): the share of total variance that lies BETWEEN participants.

    One-way random-effects ANOVA estimator with unequal group sizes. Returns a
    value in [0, 1]; a negative variance estimate is reported as 0.0, which is
    the conventional truncation and is stated rather than hidden.

    ``groups`` is an iterable of per-participant value arrays.
    """
    gs = [np.asarray(g, dtype=float) for g in groups]
    gs = [g[np.isfinite(g)] for g in gs]
    gs = [g for g in gs if len(g) >= 2]
    k = len(gs)
    if k < 2:
        return float("nan")
    n_i = np.array([len(g) for g in gs], dtype=float)
    N = n_i.sum()
    means = np.array([g.mean() for g in gs])
    grand = float(np.concatenate(gs).mean())

    ss_between = float((n_i * (means - grand) ** 2).sum())
    ss_within = float(sum(((g - g.mean()) ** 2).sum() for g in gs))
    ms_between = ss_between / (k - 1)
    ms_within = ss_within / (N - k)
    # n_0: the "effective" group size for unequal n
    n0 = (N - (n_i ** 2).sum() / N) / (k - 1)
    var_between = (ms_between - ms_within) / n0
    if not np.isfinite(var_between) or var_between <= 0:
        return 0.0
    return float(var_between / (var_between + ms_within))


# ------------------------------------------------------------ per-person set
@dataclass
class PersonR:
    participant_id: str
    n_obs: int
    n_pairs: int
    r: float
    r_early: float = float("nan")
    r_late: float = float("nan")


def per_person_autocorrelations(frame: pd.DataFrame, *, value_col: str,
                                day_col: str, id_col: str = "participant_id",
                                min_obs: int = MIN_OBS_PER_PERSON,
                                max_gap_days: int | None = MAX_GAP_DAYS
                                ) -> tuple[list[PersonR], dict]:
    """One autocorrelation per eligible participant, plus an exclusion ledger.

    The frame must be sorted within participant by ``day_col``. Participants
    below ``min_obs``, or with no variance, are excluded and counted — never
    imputed.
    """
    rows: list[PersonR] = []
    excluded = {"too_few_observations": 0, "no_variance_or_too_few_pairs": 0}
    for pid, g in frame.groupby(id_col, sort=True):
        g = g.sort_values(day_col)
        v = g[value_col].to_numpy(dtype=float)
        d = g[day_col].to_numpy(dtype=float)
        if len(v) < min_obs:
            excluded["too_few_observations"] += 1
            continue
        x, y = consecutive_pairs(v, d, max_gap_days)
        r = _pearson(x, y)
        if not np.isfinite(r):
            excluded["no_variance_or_too_few_pairs"] += 1
            continue
        rec = PersonR(participant_id=str(pid), n_obs=int(len(v)),
                      n_pairs=int(len(x)), r=float(r))
        half = len(v) // 2
        xe, ye = consecutive_pairs(v[:half], d[:half], max_gap_days)
        xl, yl = consecutive_pairs(v[half:], d[half:], max_gap_days)
        if len(xe) >= MIN_PAIRS_PER_HALF and len(xl) >= MIN_PAIRS_PER_HALF:
            rec.r_early = _pearson(xe, ye)
            rec.r_late = _pearson(xl, yl)
        rows.append(rec)
    return rows, excluded


def early_late_stability(rows: list[PersonR]) -> tuple[float, int]:
    """Correlation across participants between early-half r and late-half r.

    This is the claim that predictability is a stable property of the person
    rather than of the moment. Returns ``(r, n_participants_used)``.
    """
    e = np.array([p.r_early for p in rows], dtype=float)
    l = np.array([p.r_late for p in rows], dtype=float)
    keep = np.isfinite(e) & np.isfinite(l)
    if keep.sum() < 3:
        return float("nan"), int(keep.sum())
    return _pearson(e[keep], l[keep]), int(keep.sum())


# ------------------------------------------------------------------ assembly
@dataclass
class CeilingStats:
    """Every number the website quotes under "the ceiling", with its inputs."""
    within_person_autocorrelation: float
    variance_explained: float
    icc_between_person: float
    per_person_r_median: float
    per_person_r_iqr: list
    per_person_r_range: list
    frac_near_unpredictable: float
    frac_well_predictable: float
    early_late_r: float
    n_participants_analysed: int
    n_participants_early_late: int
    excluded: dict
    definition: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "within_person_autocorrelation": self.within_person_autocorrelation,
            "variance_explained": self.variance_explained,
            "icc_between_person": self.icc_between_person,
            "per_person_r_median": self.per_person_r_median,
            "per_person_r_iqr": self.per_person_r_iqr,
            "per_person_r_range": self.per_person_r_range,
            "frac_near_unpredictable": self.frac_near_unpredictable,
            "frac_well_predictable": self.frac_well_predictable,
            "early_late_r": self.early_late_r,
            "n_participants_analysed": self.n_participants_analysed,
            "n_participants_early_late": self.n_participants_early_late,
            "excluded": self.excluded,
            "definition": self.definition,
        }


def ceiling_statistics(frame: pd.DataFrame, *, value_col: str, day_col: str,
                       id_col: str = "participant_id",
                       min_obs: int = MIN_OBS_PER_PERSON,
                       max_gap_days: int | None = MAX_GAP_DAYS) -> CeilingStats:
    """Compute the full ceiling block from a long (person, day, value) frame."""
    rows, excluded = per_person_autocorrelations(
        frame, value_col=value_col, day_col=day_col, id_col=id_col,
        min_obs=min_obs, max_gap_days=max_gap_days)

    eligible_ids = {p.participant_id for p in rows}
    series, groups = [], []
    for pid, g in frame.groupby(id_col, sort=True):
        if str(pid) not in eligible_ids:
            continue
        g = g.sort_values(day_col)
        series.append((g[value_col].to_numpy(dtype=float),
                       g[day_col].to_numpy(dtype=float)))
        groups.append(g[value_col].to_numpy(dtype=float))

    pooled = pooled_autocorrelation(series, max_gap_days)
    icc = icc_one_way(groups)
    rs = np.array([p.r for p in rows], dtype=float)
    el_r, el_n = early_late_stability(rows)

    return CeilingStats(
        within_person_autocorrelation=float(pooled),
        variance_explained=float(pooled ** 2) if np.isfinite(pooled) else float("nan"),
        icc_between_person=float(icc),
        per_person_r_median=float(np.median(rs)) if len(rs) else float("nan"),
        per_person_r_iqr=[float(np.percentile(rs, 25)),
                          float(np.percentile(rs, 75))] if len(rs) else [],
        per_person_r_range=[float(rs.min()), float(rs.max())] if len(rs) else [],
        frac_near_unpredictable=float((rs < NEAR_UNPREDICTABLE_BELOW).mean())
                                if len(rs) else float("nan"),
        frac_well_predictable=float((rs > WELL_PREDICTABLE_ABOVE).mean())
                              if len(rs) else float("nan"),
        early_late_r=float(el_r),
        n_participants_analysed=len(rows),
        n_participants_early_late=el_n,
        excluded=excluded,
        definition={
            "pair_rule": ("report to NEXT report"
                          + (f", gap 1-{max_gap_days} days"
                             if max_gap_days else ", any gap")),
            "min_obs_per_person": min_obs,
            "min_pairs_per_half": MIN_PAIRS_PER_HALF,
            "near_unpredictable_below": NEAR_UNPREDICTABLE_BELOW,
            "well_predictable_above": WELL_PREDICTABLE_ABOVE,
            "pooled_autocorrelation": "person-mean-centred pairs, pooled",
            "icc": "one-way random effects ANOVA, unequal group sizes",
        },
    )
