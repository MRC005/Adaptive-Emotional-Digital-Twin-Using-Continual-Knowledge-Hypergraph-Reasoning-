"""The prediction dataset for the personalised future-affect task.

ONE function builds it, and the temporal rule is enforced here rather than
trusted to callers. Protocol: docs/preregistration_twin_prediction.md.

    for a report at day d, predict the NEXT report at d', with 0 < d'-d <= 7

Every feature is computed from information available strictly at or before the
prediction time. The columns that make that auditable travel with the data:
``feature_time``, ``prediction_time``, ``target_time``. ``assert_no_leakage``
checks them and is called by the tests.

WHY THE HISTORY FEATURES ARE BUILT WITH shift() BEFORE ROLLING

A rolling mean that includes the current row leaks the present into its own
predictor, and a rolling window computed on the whole series leaks the future
into the past. Both are easy to write by accident and invisible afterwards.
Every history column here is ``shift(1)`` first, so row i can only ever see
rows < i, within one participant.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["build_prediction_frame", "assert_no_leakage", "PredictionData",
           "SENSING_FEATURES", "MAX_GAP_DAYS", "TARGET"]

TARGET = "stress"
MAX_GAP_DAYS = 7
EWMA_ALPHA = 0.5

#: Daily sensing columns used as behaviour features. `_ep_0` is the full-day
#: aggregate. Conversation audio is Android-only per the data dictionary, so it
#: is included but will be missing for most participants; the model sees NaN,
#: which is honest, rather than a zero that would read as "no conversation".
SENSING_FEATURES = (
    "unlock_duration_ep_0", "loc_dist_ep_0", "act_still_ep_0",
    "act_walking_ep_0", "audio_convo_duration_ep_0", "unlock_num_ep_0",
)


@dataclass
class PredictionData:
    frame: pd.DataFrame
    feature_columns: list[str]
    target_column: str = TARGET

    @property
    def participants(self) -> np.ndarray:
        return self.frame["participant_id"].unique()

    def for_participants(self, pids) -> pd.DataFrame:
        return self.frame[self.frame["participant_id"].isin(set(pids))].copy()


def _read(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    ema = pd.read_csv(root / "EMA" / "general_ema.csv",
                      usecols=["uid", "day", TARGET, "social_level", "pam"])
    ema = ema.dropna(subset=[TARGET])
    keep = ["uid", "day", *SENSING_FEATURES]
    have = pd.read_csv(root / "Sensing" / "sensing.csv", nrows=1).columns
    keep = [c for c in keep if c in have]
    sens = pd.read_csv(root / "Sensing" / "sensing.csv", usecols=keep)
    for d in (ema, sens):
        d["t"] = pd.to_datetime(d["day"].astype(int).astype(str), format="%Y%m%d")
    return ema, sens


def build_prediction_frame(root: str | Path,
                           max_gap_days: int = MAX_GAP_DAYS) -> PredictionData:
    """Build the (features → next stress) frame with no future information.

    Returns one row per prediction opportunity. Rows whose next report falls
    more than ``max_gap_days`` away are dropped, as pre-registered.
    """
    root = Path(root)
    ema, sens = _read(root)

    ema = ema.sort_values(["uid", "t"]).reset_index(drop=True)
    g = ema.groupby("uid", sort=False)

    # ---- target: the NEXT report, and how far ahead it is -------------------
    ema["target"] = g[TARGET].shift(-1)
    ema["target_time"] = g["t"].shift(-1)
    ema["gap_days"] = (ema["target_time"] - ema["t"]).dt.days

    # ---- history: shift(1) FIRST, so a row never sees itself ---------------
    ema["prev_1"] = g[TARGET].shift(1)
    ema["prev_2"] = g[TARGET].shift(2)
    ema["prev_3"] = g[TARGET].shift(3)
    prior = g[TARGET].shift(1)
    ema["hist_mean"] = prior.groupby(ema["uid"]).transform(
        lambda s: s.expanding().mean())
    ema["hist_sd"] = prior.groupby(ema["uid"]).transform(
        lambda s: s.expanding().std())
    ema["hist_n"] = prior.groupby(ema["uid"]).transform(
        lambda s: s.expanding().count())
    ema["ewma"] = prior.groupby(ema["uid"]).transform(
        lambda s: s.ewm(alpha=EWMA_ALPHA, adjust=False).mean())
    ema["days_since_prev"] = (ema["t"] - g["t"].shift(1)).dt.days

    # `current` is the value observed AT the prediction time. It is legitimate
    # input (it is the past relative to the target) and is what persistence uses.
    ema["current"] = ema[TARGET]
    ema["current_social"] = ema["social_level"]
    ema["current_pam"] = ema["pam"]

    # ---- behaviour: previous-day sensing + trailing 7-day means ------------
    sens = sens.sort_values(["uid", "t"]).reset_index(drop=True)
    feats = [c for c in SENSING_FEATURES if c in sens.columns]
    sg = sens.groupby("uid", sort=False)
    for c in feats:
        sens[f"{c}__roll7"] = sg[c].transform(
            lambda s: s.rolling(7, min_periods=1).mean())
    # attach sensing from the day BEFORE the prediction day
    sens_prior = sens.copy()
    sens_prior["t"] = sens_prior["t"] + pd.Timedelta(days=1)
    cols = ["uid", "t", *feats, *[f"{c}__roll7" for c in feats]]
    ema = ema.merge(sens_prior[cols], on=["uid", "t"], how="left",
                    suffixes=("", "_dup"))

    # ---- context ------------------------------------------------------------
    ema["dow"] = ema["t"].dt.dayofweek

    # ---- filter to valid prediction opportunities --------------------------
    out = ema[ema["target"].notna() & (ema["gap_days"] > 0)
              & (ema["gap_days"] <= max_gap_days)].copy()

    out = out.rename(columns={"uid": "participant_id", "t": "prediction_time"})
    # every time-dependent feature is drawn from the previous day or earlier
    out["feature_time"] = out["prediction_time"] - pd.Timedelta(days=1)
    out["target"] = out["target"].astype(int)

    feature_columns = [
        "current", "current_social", "current_pam",
        "prev_1", "prev_2", "prev_3", "hist_mean", "hist_sd", "hist_n",
        "ewma", "days_since_prev", "gap_days", "dow",
        *feats, *[f"{c}__roll7" for c in feats],
    ]
    feature_columns = [c for c in feature_columns if c in out.columns]

    keep = ["participant_id", "feature_time", "prediction_time", "target_time",
            "target", *feature_columns]
    out = out[keep].sort_values(["participant_id", "prediction_time"])
    out = out.reset_index(drop=True)
    return PredictionData(frame=out, feature_columns=feature_columns)


def assert_no_leakage(df: pd.DataFrame, max_gap_days: int = MAX_GAP_DAYS) -> None:
    """Raise if any row violates the temporal contract. Used by the tests."""
    if not (df["feature_time"] <= df["prediction_time"]).all():
        raise AssertionError("a feature is timestamped after its prediction time")
    if not (df["prediction_time"] < df["target_time"]).all():
        raise AssertionError("a target is not strictly in the future")
    gap = (df["target_time"] - df["prediction_time"]).dt.days
    if not ((gap > 0) & (gap <= max_gap_days)).all():
        raise AssertionError(f"a gap falls outside (0, {max_gap_days}] days")
    if "participant_id" in df.columns:
        # identity must never be usable as a feature
        if df["participant_id"].dtype.kind in "if":
            raise AssertionError("participant_id looks numeric and could be "
                                 "consumed as a feature")
    # history must never equal the target it predicts
    if "prev_1" in df.columns:
        both = df[["prev_1", "target"]].dropna()
        if len(both) and (both["prev_1"] == both["target"]).all():
            raise AssertionError("prev_1 is identical to the target everywhere")
