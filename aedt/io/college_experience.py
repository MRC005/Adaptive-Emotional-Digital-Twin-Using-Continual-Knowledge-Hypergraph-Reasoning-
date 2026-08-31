"""MODULE 1 -- College Experience Study adapter (REAL, PRIMARY LONGITUDINAL).

Purpose  Load the Dartmouth College Experience Study into the canonical LongFrame.
Input    data/raw/college-experience/{EMA/general_ema.csv, Sensing/sensing.csv}
Output   LongFrame(pid, ts, report, raw_response, <sensor>) + DatasetAudit.
Status   STANDARD.

WHY THIS DATASET CLEARS A BAR THE OTHERS DID NOT

Three earlier archives were audited and none produced enough repeated
observations per participant to support a within-person epoch ratio. This one
does, and it does so under the ORIGINAL pre-specified screen. No threshold was
moved to admit it:

    participants with >= 60 aligned observations in BOTH halves of their own
    span, by window rule:  halves-of-own-span 121, count-half 148, first/last
    40% 133.  The screen requires >= 10.

TWO PROPERTIES THAT MATTER MORE THAN THE COUNT

1. THE SCALE DIRECTION IS DOCUMENTED. The published data dictionary states
   `stress`: "1: Not at All; 2: A Little Bit; 3: Somewhat; 4: Very Much;
   5: Extremely". Direction is therefore CONFIRMED ASCENDING from the codebook,
   not assumed from the data. PMData failed on exactly this point, and no
   amount of data can settle it -- only a codebook can.

2. THE JOIN IS EXACT, NOT INFERRED. EMA and sensing are both keyed by
   (uid, day) on the same calendar day, so aligning a report to a sensing
   feature involves no tolerance window and no interpolation. Nothing is
   imputed and no row is created that the source did not contain.

SENSOR CHOICE. `audio_convo_duration_ep_0` is conversation duration over the
full day in seconds, converted here to minutes. It is the same construct the
project has used throughout, which keeps this dataset comparable with the
simulated controls rather than introducing a new feature at the same time as a
new dataset.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ..constants import DataStatus, DatasetRole
from ..errors import DecisionRequired
from ..schemas import DatasetAudit
from .base import DatasetAdapter, LoadResult, register_adapter

log = logging.getLogger(__name__)

__all__ = ["CollegeExperienceAdapter", "CE_SENSORS", "CE_REPORTS"]

#: Sensing features offered for analysis. Each is a full-day (`_ep_0`) daily
#: aggregate present for ~100% of sensing rows. `scale` converts to the unit
#: named in `label`; the estimator standardises within each window regardless,
#: so the unit affects readability only, never the estimate.
CE_SENSORS: dict[str, dict] = {
    "conversation_minutes": {
        "column": "audio_convo_duration_ep_0", "scale": 1 / 60.0,
        "label": "Conversation minutes per day", "platforms": "android",
        "detail": "Total time the microphone classified as conversation, full day.",
    },
    "conversation_episodes": {
        "column": "audio_convo_num_ep_0", "scale": 1.0,
        "label": "Conversation episodes per day", "platforms": "android",
        "detail": "Number of distinct conversation episodes detected, full day.",
    },
    "unlock_minutes": {
        "column": "unlock_duration_ep_0", "scale": 1 / 60.0,
        "label": "Phone unlock minutes per day", "platforms": "all",
        "detail": "Total time the phone was unlocked, full day.",
    },
    "travel_distance": {
        "column": "loc_dist_ep_0", "scale": 1.0,
        "label": "Distance travelled per day", "platforms": "all",
        "detail": "Total distance covered from location traces, full day.",
    },
}

#: Ordinal self-reports. Direction is taken from the published data dictionary,
#: never inferred. `k` is the number of categories the probit is fitted with.
CE_REPORTS: dict[str, dict] = {
    "stress": {
        "column": "stress", "k": 5, "ascending": True,
        "label": "Stress right now (1-5)",
        "codebook": "1: Not at All; 2: A Little Bit; 3: Somewhat; "
                    "4: Very Much; 5: Extremely",
        "question": "Are you feeling stressed now?",
    },
    "social_level": {
        "column": "social_level", "k": 5, "ascending": True,
        "label": "Time spent with others (1-5)",
        "codebook": "1: Almost always alone; ... 5: Almost always with others",
        "question": "Have you spent most of your time alone or with others today?",
    },
}

_EMA_REL = Path("EMA/general_ema.csv")
_SENSING_REL = Path("Sensing/sensing.csv")
_DICT_REL = Path("EMA/Data Dictionary (general EMA).csv")


@register_adapter
class CollegeExperienceAdapter(DatasetAdapter):
    """Dartmouth College Experience Study (Nepal et al., IMWUT 2024)."""

    name = "college_experience"
    role = DatasetRole.PRIMARY_LONGITUDINAL
    primary_sensor = "conversation_minutes"
    report_variable = "general EMA `stress` (1-5, ascending, per data dictionary)"
    can_support_longitudinal_estimand = True
    acquisition_instructions = (
        "College Experience Study. Kaggle: "
        "https://www.kaggle.com/datasets/subigyanepal/college-experience-dataset\n"
        "Place the extracted archive at data/raw/college-experience/ so that "
        "EMA/general_ema.csv and Sensing/sensing.csv exist.\n"
        "Cite: Nepal et al. 2024, Proc. ACM IMWUT 8(1) art. 38. "
        "https://doi.org/10.1145/3643501"
    )

    def __init__(self, report: str = "stress", sensor: str = "conversation_minutes"):
        if report not in CE_REPORTS:
            raise KeyError(f"unknown report {report!r}; known: {sorted(CE_REPORTS)}")
        if sensor not in CE_SENSORS:
            raise KeyError(f"unknown sensor {sensor!r}; known: {sorted(CE_SENSORS)}")
        self.report = report
        self.sensor = sensor
        self.primary_sensor = sensor

    # ----------------------------------------------------------- discovery
    def locate(self, root: Path) -> dict[str, list[Path]]:
        if root is None or not Path(root).is_dir():
            return {}
        root = Path(root)
        found: dict[str, list[Path]] = {}
        for key, rel in (("ema", _EMA_REL), ("sensing", _SENSING_REL),
                         ("dictionary", _DICT_REL)):
            p = root / rel
            if p.is_file():
                found[key] = [p]
        return found

    # -------------------------------------------------------------- parsing
    def _read_pair(self, root: Path) -> pd.DataFrame:
        """Join EMA to sensing on (uid, day). Reads only the columns needed."""
        rep = CE_REPORTS[self.report]
        sen = CE_SENSORS[self.sensor]

        ema = pd.read_csv(root / _EMA_REL, usecols=["uid", "day", rep["column"]])
        try:
            sensing = pd.read_csv(root / _SENSING_REL,
                                  usecols=["uid", "day", "is_ios", sen["column"]])
        except ValueError as exc:                      # column genuinely absent
            raise DecisionRequired(
                f"College Experience: sensing column {sen['column']!r} is not in "
                f"{root / _SENSING_REL}. Do not substitute another feature "
                f"silently -- confirm the column name against the data "
                f"dictionary.\n{exc}") from exc

        # The data dictionary marks each sensing feature Android-only, iOS-only or
        # All. On a platform where the feature is not collected the archive still
        # stores a row, and it stores 0 -- so an unfiltered read would silently
        # treat "never measured" as "no conversation happened". Conversation audio
        # is documented Android only, and is 87.8% zero on iOS against 13.1% on
        # Android. Dropping those rows removes absent instrumentation, not data.
        if sen["platforms"] == "android":
            sensing = sensing[sensing["is_ios"] == 0]
        elif sen["platforms"] == "ios":
            sensing = sensing[sensing["is_ios"] == 1]
        sensing = sensing.drop(columns=["is_ios"])

        m = ema.merge(sensing, on=["uid", "day"], how="inner")
        m = m.dropna(subset=[rep["column"], sen["column"]])
        if m.empty:
            raise DecisionRequired(
                "College Experience: no (uid, day) carries both "
                f"{rep['column']!r} and {sen['column']!r}.")

        out = pd.DataFrame({
            "pid": m["uid"].astype(str),
            "ts": pd.to_datetime(m["day"].astype(int).astype(str), format="%Y%m%d"),
            "raw_response": m[rep["column"]].astype(float).round().astype(int),
            self.sensor: m[sen["column"]].astype(float) * sen["scale"],
        })

        lo, hi = 1, rep["k"]
        bad = ~out["raw_response"].between(lo, hi)
        if bad.any():
            offending = sorted(out.loc[bad, "raw_response"].unique())[:10]
            raise DecisionRequired(
                f"College Experience: {int(bad.sum())} {self.report!r} values "
                f"outside the documented range {lo}-{hi}: {offending}. The data "
                f"dictionary does not describe these codes.")
        return out.sort_values(["pid", "ts"]).reset_index(drop=True)

    # ---------------------------------------------------------------- load
    def load(self, root: str | Path | None) -> LoadResult:
        p = self.require_files(root)
        rep = CE_REPORTS[self.report]
        df = self._read_pair(p)

        # Direction comes from the published codebook, so severity is the stored
        # value when ascending and is reflected when it is not. Never guessed.
        df["report"] = (df["raw_response"] if rep["ascending"]
                        else (rep["k"] + 1) - df["raw_response"])

        prov = {
            "dataset": self.name,
            "source_files": [str(p / _EMA_REL), str(p / _SENSING_REL)],
            "citation": "Nepal et al. 2024, Proc. ACM IMWUT 8(1) art. 38, "
                        "doi:10.1145/3643501",
            "report_variable": rep["column"],
            "report_question": rep["question"],
            "report_codebook": rep["codebook"],
            "severity_direction_confirmed": True,
            "severity_note": (
                "Direction CONFIRMED ASCENDING from the published data "
                "dictionary, not inferred from the data."),
            "sensor_variable": CE_SENSORS[self.sensor]["column"],
            "sensor_units": CE_SENSORS[self.sensor]["label"],
            "alignment": "exact inner join on (uid, day); no tolerance window, "
                         "no interpolation, no imputation",
            "n_participants": int(df["pid"].nunique()),
            "n_observations": int(len(df)),
        }
        df.attrs["data_status"] = DataStatus.REAL.value
        df.attrs["n_categories"] = rep["k"]
        df.attrs["severity_direction_confirmed"] = True
        return LoadResult(frame=df, n_categories=rep["k"], sensor=self.sensor,
                          data_status=DataStatus.REAL, provenance=prov,
                          audit=self.audit(root))

    # --------------------------------------------------------------- audit
    def audit(self, root: str | Path | None) -> DatasetAudit:
        p = self.resolve_root(root)
        if p is None or not p.is_dir():
            return self.unavailable_audit(root, "directory not present")
        found = self.locate(p)
        if "ema" not in found or "sensing" not in found:
            missing = [n for n in ("ema", "sensing") if n not in found]
            return self.unavailable_audit(
                root, f"directory present but missing: {', '.join(missing)}")

        rep = CE_REPORTS[self.report]
        try:
            df = self._read_pair(p)
        except DecisionRequired as exc:
            return self.unavailable_audit(root, f"files present but unusable: {exc}")

        per = df.groupby("pid").size()
        span_days = float((df["ts"].max() - df["ts"].min()).days)
        per_span = df.groupby("pid")["ts"].agg(lambda s: (s.max() - s.min()).days)

        # Per-window counts under the pre-specified rule (halves of own span).
        both60 = 0
        for _, g in df.groupby("pid"):
            t = g["ts"].to_numpy()
            if len(t) < 2:
                continue
            mid = t.min() + (t.max() - t.min()) / 2
            if (t <= mid).sum() >= 60 and (t > mid).sum() >= 60:
                both60 += 1

        return DatasetAudit(
            dataset_name=self.name, role=self.role,
            data_status=DataStatus.REAL,
            source_status="present and parsed",
            local_files_available=True, root_path=str(p),
            files_found=tuple(str(f) for fs in found.values() for f in fs),
            participant_count=int(df["pid"].nunique()),
            observation_count=int(len(df)),
            sensor_modalities=tuple(CE_SENSORS),
            self_report_variables=tuple(CE_REPORTS),
            self_report_scale=rep["codebook"],
            timestamps_present=True, timestamp_format="YYYYMMDD (calendar day)",
            timezone="not stated in the release; days are local calendar days",
            longitudinal_span_days=span_days,
            observations_per_participant={str(k): int(v) for k, v in per.items()},
            median_observations_per_participant=float(per.median()),
            missingness={"rows dropped for a missing report or sensor value":
                         float("nan")},
            eligible_for_primary_analysis=both60 >= 10,
            eligible_for_benchmark_analysis=True,
            acquisition_instructions=self.acquisition_instructions,
            exclusion_reasons=(),
            notes=(
                f"Participants with >= 60 aligned observations in BOTH halves "
                f"of their own span: {both60} (the screen requires >= 10).",
                f"Median aligned observations per participant: {per.median():.0f}; "
                f"median per-participant span {per_span.median():.0f} days.",
                "Scale direction CONFIRMED ASCENDING from the published data "
                "dictionary. This is the property PMData could not establish.",
                "EMA and sensing join exactly on (uid, day); nothing is "
                "interpolated and nothing is imputed.",
            ),
        )
