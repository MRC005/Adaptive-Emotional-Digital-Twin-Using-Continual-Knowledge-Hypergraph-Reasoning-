"""TEMPORAL EPOCHS: halves of each participant's OWN enrolment span.

FROZEN RULE (ROUND-17 §W): "Epochs = halves of each participant's own span."

Why not a calendar split: participants enrol and drop out at different times,
so a shared calendar midpoint would give some people a 2-week epoch 1 and a
10-week epoch 2. The estimand is a within-person ratio; the split must be
within-person too.

LEAKAGE. The midpoint uses only that participant's own first and last
timestamps. No cross-participant statistic, and no epoch-2 statistic, enters
the definition of epoch 1.
"""
from __future__ import annotations

import logging

import pandas as pd

from ..errors import DecisionRequired
from ..schemas import EpochDefinition

log = logging.getLogger(__name__)

__all__ = ["assign_epochs", "epoch_definitions", "EPOCH_RULE"]

EPOCH_RULE = "halves of each participant's own enrolment span"


def assign_epochs(df: pd.DataFrame, *, rule: str = "own_span_halves"
                  ) -> pd.DataFrame:
    """Add an 'epoch' column (0 or 1) per participant.

    ``rule`` is exposed only so that Ablation 5 (epoch definition) can be run
    as a pre-specified sensitivity analysis. The frozen default is
    ``own_span_halves``.
    """
    if rule not in ("own_span_halves", "calendar_median", "observation_halves"):
        raise DecisionRequired(
            f"Unknown epoch rule {rule!r}. The frozen rule is "
            "'own_span_halves'.")
    out = []
    for pid, g in df.groupby("pid", sort=True):
        g = g.sort_values("ts").copy()
        if rule == "own_span_halves":
            lo, hi = g["ts"].min(), g["ts"].max()
            mid = lo + (hi - lo) / 2
            g["epoch"] = (g["ts"] > mid).astype(int)
        elif rule == "observation_halves":
            h = len(g) // 2
            g["epoch"] = ([0] * h) + ([1] * (len(g) - h))
        else:  # calendar_median, over the whole cohort
            mid = df["ts"].min() + (df["ts"].max() - df["ts"].min()) / 2
            g["epoch"] = (g["ts"] > mid).astype(int)
        out.append(g)
    res = pd.concat(out, ignore_index=True)
    res.attrs.update(df.attrs)
    res.attrs["epoch_rule"] = rule
    return res


def epoch_definitions(df: pd.DataFrame) -> list[EpochDefinition]:
    """Typed record of how each participant's timeline was cut."""
    defs = []
    for pid, g in df.groupby("pid", sort=True):
        g = g.sort_values("ts")
        lo, hi = g["ts"].min(), g["ts"].max()
        defs.append(EpochDefinition(
            pid=str(pid), rule=df.attrs.get("epoch_rule", "own_span_halves"),
            start=lo, midpoint=lo + (hi - lo) / 2, end=hi,
            n_epoch0=int((g["epoch"] == 0).sum()),
            n_epoch1=int((g["epoch"] == 1).sum())))
    return defs
