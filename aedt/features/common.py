"""Concrete feature extractors, all behind the common interface.

The PRIMARY covariate for StudentLife is daily conversation minutes and for
PMData resting heart rate, both frozen in ROUND-17 §W. The remaining
extractors exist so that (a) the context/hypergraph layer has several features
to form conjunctions over, and (b) the pre-specified PC1 fallback covariate can
be constructed if the primary association turns out to be too weak on real data
(diagnostic [9b]).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import FeatureExtractor, FeatureSpec, register

__all__ = ["ConversationMinutes", "RestingHeartRate", "ActivityLevel",
           "LocationEntropy", "UnlockCount", "PhysiologicalWindowStats",
           "PC1Fallback"]


class ConversationMinutes(FeatureExtractor):
    """Daily minutes of detected conversation, from (start, end) episode pairs.

    Expected NEGATIVE association with stress: stressed students talk less.
    That expectation is documentation only -- see self-correction 26.
    """

    spec = FeatureSpec(
        name="conversation_minutes", unit="minutes/day",
        description="Summed duration of detected conversation episodes per day.",
        expected_sign_vs_stress="negative", primary_eligible=True,
        modality="audio/conversation")

    def extract(self, raw: pd.DataFrame, *, start_col: str = "start_timestamp",
                end_col: str = "end_timestamp") -> pd.DataFrame:
        s = pd.to_numeric(raw[start_col], errors="coerce")
        e = pd.to_numeric(raw[end_col], errors="coerce")
        ok = s.notna() & e.notna() & (e > s)
        if not ok.any():
            return pd.DataFrame(columns=["pid", "ts", self.spec.name])
        day = pd.to_datetime(s[ok], unit="s").dt.normalize()
        mins = (e[ok] - s[ok]) / 60.0
        pid = raw.loc[ok, "pid"] if "pid" in raw.columns else "unknown"
        out = (pd.DataFrame({"pid": pid, "ts": day.values, "m": mins.values})
               .groupby(["pid", "ts"], as_index=False)["m"].sum())
        return out.rename(columns={"m": self.spec.name})


class RestingHeartRate(FeatureExtractor):
    """Daily resting heart rate. Expected POSITIVE association with stress."""

    spec = FeatureSpec(
        name="resting_hr", unit="bpm",
        description="Device-reported daily resting heart rate.",
        expected_sign_vs_stress="positive", primary_eligible=True,
        modality="physiological")

    def extract(self, raw: pd.DataFrame, *, value_col: str = "value"
                ) -> pd.DataFrame:
        out = raw.copy()
        out["ts"] = pd.to_datetime(out["ts"], errors="coerce").dt.normalize()
        out[self.spec.name] = pd.to_numeric(out[value_col], errors="coerce")
        return out[["pid", "ts", self.spec.name]].dropna()


class ActivityLevel(FeatureExtractor):
    """Fraction of sampled epochs classified as non-stationary."""

    spec = FeatureSpec(
        name="activity_level", unit="fraction",
        description="Share of activity-inference samples that are non-stationary.",
        expected_sign_vs_stress="unknown", modality="accelerometer")

    def extract(self, raw: pd.DataFrame, *, value_col: str = "activity"
                ) -> pd.DataFrame:
        out = raw.copy()
        out["ts"] = pd.to_datetime(out["ts"], errors="coerce").dt.normalize()
        v = pd.to_numeric(out[value_col], errors="coerce")
        out[self.spec.name] = (v > 0).astype(float)
        return (out.dropna(subset=["ts"])
                .groupby(["pid", "ts"], as_index=False)[self.spec.name].mean())


class LocationEntropy(FeatureExtractor):
    """Shannon entropy of time spent across visited location clusters."""

    spec = FeatureSpec(
        name="location_entropy", unit="nats",
        description="Entropy of the day's distribution over location clusters.",
        expected_sign_vs_stress="unknown", modality="gps")

    def extract(self, raw: pd.DataFrame, *, cluster_col: str = "cluster"
                ) -> pd.DataFrame:
        out = raw.copy()
        out["ts"] = pd.to_datetime(out["ts"], errors="coerce").dt.normalize()
        rows = []
        for (pid, day), g in out.dropna(subset=["ts"]).groupby(["pid", "ts"]):
            p = g[cluster_col].value_counts(normalize=True).to_numpy()
            p = p[p > 0]
            rows.append((pid, day, float(-(p * np.log(p)).sum())))
        return pd.DataFrame(rows, columns=["pid", "ts", self.spec.name])


class UnlockCount(FeatureExtractor):
    """Number of phone unlock events per day."""

    spec = FeatureSpec(
        name="unlock_count", unit="events/day",
        description="Count of phone-lock release events per day.",
        expected_sign_vs_stress="unknown", modality="phone usage")

    def extract(self, raw: pd.DataFrame) -> pd.DataFrame:
        out = raw.copy()
        out["ts"] = pd.to_datetime(out["ts"], errors="coerce").dt.normalize()
        g = (out.dropna(subset=["ts"]).groupby(["pid", "ts"])
             .size().rename(self.spec.name).reset_index())
        return g


class PhysiologicalWindowStats(FeatureExtractor):
    """Mean / SD / range of a high-rate physiological channel over a window.

    Used by the BENCHMARK datasets (WESAD, SWELL-KW, AffectiveROAD), where the
    scientific role is to validate feature extraction and robustness -- NOT to
    validate the longitudinal estimand.
    """

    spec = FeatureSpec(
        name="physio_window_mean", unit="channel units",
        description="Windowed summary statistics of a physiological channel.",
        expected_sign_vs_stress="unknown", modality="physiological")

    def extract(self, raw: pd.DataFrame, *, value_col: str = "value",
                window: str = "60s") -> pd.DataFrame:
        out = raw.copy()
        out["ts"] = pd.to_datetime(out["ts"], errors="coerce")
        out = out.dropna(subset=["ts"]).set_index("ts")
        rows = []
        for pid, g in out.groupby("pid"):
            r = g[value_col].resample(window)
            agg = pd.DataFrame({
                "physio_window_mean": r.mean(),
                "physio_window_sd": r.std(ddof=1),
                "physio_window_range": r.max() - r.min()}).dropna()
            agg["pid"] = pid
            rows.append(agg.reset_index())
        if not rows:
            return pd.DataFrame(columns=["pid", "ts", self.spec.name])
        return pd.concat(rows, ignore_index=True)


class PC1Fallback(FeatureExtractor):
    """The PRE-SPECIFIED fallback covariate (ROUND-17 §R step 2).

    First principal component of the standardised feature set, with loadings
    fitted on EPOCH 1 ONLY and applied unchanged to epoch 2. Fitting on pooled
    data would let epoch-2 structure define the covariate and would partly
    absorb the drift under test.

    Use only when the primary feature's median |beta| is below
    ``WEAK_ASSOCIATION_BETA``, and record the switch in the run log.
    """

    spec = FeatureSpec(
        name="pc1_fallback", unit="standardised units",
        description=("Epoch-1-fitted first principal component of "
                     "{conversation, activity, location entropy, unlocks}."),
        expected_sign_vs_stress="unknown", primary_eligible=True,
        modality="composite")

    def extract(self, raw: pd.DataFrame, *, feature_cols: list[str] | None = None
                ) -> pd.DataFrame:
        cols = feature_cols or [c for c in raw.columns
                                if c not in ("pid", "ts", "day", "report",
                                             "epoch")]
        rows = []
        for pid, g in raw.groupby("pid", sort=True):
            g = g.sort_values("ts")
            X = g[cols].to_numpy(dtype=float)
            e1 = (g["epoch"] == 0).to_numpy() if "epoch" in g else np.ones(len(g), bool)
            if e1.sum() < 5 or not np.isfinite(X).all():
                continue
            mu = X[e1].mean(axis=0)
            sd = X[e1].std(axis=0, ddof=1)
            sd[sd < 1e-12] = 1.0
            Z = (X - mu) / sd                     # epoch-1 standardiser
            _u, _s, vt = np.linalg.svd(Z[e1] - Z[e1].mean(axis=0),
                                       full_matrices=False)
            pc1 = Z @ vt[0]                       # epoch-1 loadings applied to all
            rows.append(pd.DataFrame({"pid": pid, "ts": g["ts"].to_numpy(),
                                      self.spec.name: pc1}))
        if not rows:
            return pd.DataFrame(columns=["pid", "ts", self.spec.name])
        return pd.concat(rows, ignore_index=True)


for _e in (ConversationMinutes(), RestingHeartRate(), ActivityLevel(),
           LocationEntropy(), UnlockCount(), PhysiologicalWindowStats(),
           PC1Fallback()):
    register(_e)
