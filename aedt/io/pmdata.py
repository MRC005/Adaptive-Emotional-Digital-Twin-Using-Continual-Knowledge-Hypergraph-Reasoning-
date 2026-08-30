"""PMDATA ADAPTER -- CONDITIONAL SECONDARY target.

** WRITTEN AGAINST THE ACTUAL RELEASED FILES, WHICH HAVE BEEN OPENED AND
   INSPECTED. ** Not a declared expectation.

Source   Thambawita, Hicks, Borgli, Stensland, Jha, Svensen, Pettersen,
         Johansen, Johansen, Pettersen, Nordvang, Pettersen, Vonstad,
         Nygard, Riegler & Halvorsen (2020), "PMData: A Sports Logging Dataset",
         ACM MMSys. https://datasets.simula.no/pmdata/
Licence  CC BY 4.0 (per the publisher's dataset page; the released archive
         itself carries NO licence file -- see the audit notes).

VERIFIED STRUCTURE (read from the archive):

    participant-overview.xlsx            demographics only (age, height, sex,
                                         max HR, stride) -- NO questionnaire
                                         definition, NO scale documentation
    p01..p16/pmsys/wellness.csv          effective_time_frame (ISO-8601 with Z),
                                         fatigue, mood, readiness,
                                         sleep_duration_h, sleep_quality,
                                         soreness, soreness_area, stress
    p01..p16/pmsys/srpe.csv              session RPE (not used here)
    p01..p16/fitbit/resting_heart_rate.json
                                         dateTime (NAIVE local datetime),
                                         value = {date, value (bpm), error}
    p01..p16/fitbit/heart_rate.json      intraday HR (~1.6 GB, not used)
    p01..p16/food-images/                photographs (not used)

MEASURED FACTS (all 16 participants, this release):
    16 participants, ~105-150 day spans
    1747 wellness rows; stress values {0:4, 1:39, 2:389, 3:1016, 4:283, 5:16}
    resting_heart_rate.json present for 14 of 16 (p12 and p13 have none)

TWO THINGS THIS ADAPTER REFUSES TO GUESS

1. THE SCALE DIRECTION. PMSys `stress` is a bare integer. The released archive
   contains no README, no codebook and no questionnaire definition, so nothing
   in the data states whether 5 means "most stressed" or "least stressed".
   Unlike StudentLife (label text) and RELAX (answer-label anchors), there is
   no text to key the remap on. ``direction_confirmed`` is therefore False by
   default and the PRIMARY analysis is refused; the audit still runs and
   reports the structure.

2. `stress == 0`, which is outside the 1..5 range every other value occupies.
   Two of the four such rows have EVERY wellness item at 0 (a blank
   submission); the other two carry valid answers elsewhere. The first kind is
   unambiguous and is dropped as a blank submission; the second is genuinely
   ambiguous and is controlled by ``zero_stress_handling``.

Neither is imputed, and nothing is reversed on a hunch.
"""
from __future__ import annotations

import glob
import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

from ..constants import DataStatus, DatasetRole
from ..errors import DecisionRequired
from ..schemas import DatasetAudit
from .base import DatasetAdapter, LoadResult, register_adapter

log = logging.getLogger(__name__)

__all__ = ["PMDataAdapter", "WELLNESS_ITEMS", "PMSYS_STRESS_RANGE"]

ACQUISITION = """\
ACQUISITION -- PMData (Simula), Thambawita et al. 2020, MMSys.
  1. Download from https://datasets.simula.no/pmdata/ (open access).
  2. Either extract the archive so that --root contains p01 ... p16, or leave
     the zip in place and extract only what is needed (~0.3 MB of 1.4 GB):
         unzip -o pmdata.zip 'participant-overview.xlsx' \\
             'p*/pmsys/wellness.csv' 'p*/fitbit/resting_heart_rate.json' \\
             -d data/raw/pmdata
     The food-image and intraday-heart-rate files are NOT used by this analysis.
  3. Reading participant-overview.xlsx needs:  pip install -e ".[relax]"
     (openpyxl; optional -- the adapter works without it).
  4. BEFORE any primary analysis you must confirm the PMSys stress scale
     DIRECTION from the PMSys instrument documentation and set
     `pmdata.direction_confirmed` in configs/pmdata.yaml. The released archive
     does not state it."""

WELLNESS_ITEMS = ("fatigue", "mood", "readiness", "sleep_quality", "soreness",
                  "stress")
PMSYS_STRESS_RANGE = (1, 5)


class PMDataAdapter(DatasetAdapter):
    name = "pmdata"
    role = DatasetRole.CONDITIONAL_SECONDARY
    primary_sensor = "resting_hr"
    report_variable = "PMSys daily wellness `stress` (bare integer, 1-5)"
    can_support_longitudinal_estimand = True
    acquisition_instructions = ACQUISITION

    def __init__(self, *, direction_confirmed: bool = False,
                 ascending: bool = True,
                 zero_stress_handling: str = "treat_as_missing"):
        if zero_stress_handling not in ("halt", "treat_as_missing"):
            raise DecisionRequired(
                "pmdata.zero_stress_handling must be 'halt' or "
                f"'treat_as_missing', not {zero_stress_handling!r}")
        self.direction_confirmed = direction_confirmed
        self.ascending = ascending
        self.zero_stress_handling = zero_stress_handling

    @classmethod
    def from_config(cls, cfg) -> "PMDataAdapter":
        return cls(
            direction_confirmed=bool(cfg.get("pmdata.direction_confirmed",
                                             False)),
            ascending=bool(cfg.get("pmdata.severity_ascending", True)),
            zero_stress_handling=str(cfg.get("pmdata.zero_stress_handling",
                                             "treat_as_missing")))

    # ------------------------------------------------------------- locate
    def locate(self, root: Path) -> dict[str, list[Path]]:
        if root is None or not Path(root).exists():
            return {}
        r = str(root)
        pdirs = [Path(d) for d in sorted(glob.glob(os.path.join(r, "p[0-9]*")))
                 if os.path.isdir(d)]
        if not pdirs:                       # tolerate one nesting level
            for cand in sorted(glob.glob(os.path.join(r, "*"))):
                if os.path.isdir(cand):
                    inner = sorted(glob.glob(os.path.join(cand, "p[0-9]*")))
                    if inner:
                        pdirs = [Path(d) for d in inner if os.path.isdir(d)]
                        break
        return {
            "participant_dirs": pdirs,
            "wellness": [Path(f) for d in pdirs for f in
                         glob.glob(os.path.join(str(d), "pmsys", "wellness*.csv"))],
            "resting_hr": [Path(f) for d in pdirs for f in
                           glob.glob(os.path.join(str(d), "fitbit",
                                                  "resting_heart_rate.json"))],
            "overview": [Path(f) for f in
                         glob.glob(os.path.join(r, "**",
                                                "participant-overview.xlsx"),
                                   recursive=True)],
        }

    # -------------------------------------------------------------- helpers
    @staticmethod
    def _to_utc_day(s: pd.Series) -> pd.Series:
        """Normalise to a tz-naive UTC calendar day.

        PMData mixes clocks: wellness timestamps are ISO-8601 with a 'Z'
        (tz-aware UTC) while resting_heart_rate.json carries naive datetimes.
        Merging them directly raises; both sides are pinned to the same clock
        here so the join is well defined rather than accidentally working.
        """
        t = pd.to_datetime(s, errors="coerce", format="ISO8601", utc=True)
        return t.dt.tz_convert("UTC").dt.tz_localize(None).dt.normalize()

    def _read_wellness(self, path: Path, pid: str) -> tuple[pd.DataFrame, dict]:
        w = pd.read_csv(path)
        w.columns = [c.strip().lower() for c in w.columns]
        tc = next((c for c in w.columns if "time" in c or "date" in c), None)
        if tc is None or "stress" not in w.columns:
            raise DecisionRequired(
                f"PMData required variables unavailable -- {pid}: "
                f"{path} lacks a timestamp and/or 'stress' column. "
                f"Columns present: {list(w.columns)}")
        ts = pd.to_datetime(w[tc], errors="coerce", format="ISO8601", utc=True)
        raw = pd.to_numeric(w["stress"], errors="coerce")
        stats = {"n_rows": int(len(w)), "n_unparsable_ts": int(ts.isna().sum()),
                 "n_missing_stress": int(raw.isna().sum())}

        # ---- blank submissions: EVERY wellness item is zero ---------------
        present = [c for c in WELLNESS_ITEMS if c in w.columns]
        allzero = pd.Series(True, index=w.index)
        for c in present:
            allzero &= (pd.to_numeric(w[c], errors="coerce") == 0)
        stats["n_blank_submissions_dropped"] = int(allzero.sum())

        d = pd.DataFrame({"pid": pid, "ts": ts, "raw_response": raw,
                          "day": self._to_utc_day(w[tc])})
        d = d[~allzero.to_numpy()].dropna(subset=["ts", "raw_response"])

        # ---- remaining out-of-range values -------------------------------
        lo, hi = PMSYS_STRESS_RANGE
        bad = d[(d["raw_response"] < lo) | (d["raw_response"] > hi)]
        stats["n_out_of_range"] = int(len(bad))
        if len(bad):
            if self.zero_stress_handling == "halt":
                raise DecisionRequired(
                    f"PMData {pid}: {len(bad)} wellness rows have `stress` "
                    f"outside the documented {lo}..{hi} range "
                    f"(values {sorted(bad['raw_response'].unique().tolist())}) "
                    "while their other wellness items are valid, so they are "
                    "NOT blank submissions. The released archive does not say "
                    "whether these encode 'not answered' or a real response. "
                    "Do NOT guess: confirm against the PMSys instrument, then "
                    "set pmdata.zero_stress_handling in configs/pmdata.yaml.")
            d = d[(d["raw_response"] >= lo) & (d["raw_response"] <= hi)]
        d["raw_response"] = d["raw_response"].round().astype(int)
        return d, stats

    def _read_resting_hr(self, path: Path) -> pd.DataFrame:
        try:
            h = pd.read_json(path)
        except Exception as exc:
            raise DecisionRequired(
                f"PMData: cannot read {path} ({exc}). Confirm the release "
                "version; this adapter expects a JSON list of "
                "{dateTime, value:{date, value, error}} records.")
        if not len(h):
            return pd.DataFrame(columns=["day", "resting_hr"])
        dc = next((c for c in h.columns if "date" in str(c).lower()), None)
        vc = next((c for c in h.columns if c != dc), None)
        if dc is None or vc is None:
            raise DecisionRequired(
                f"PMData: resting HR columns unclear in {path}: "
                f"{list(h.columns)}")
        val = h[vc].apply(lambda v: v.get("value") if isinstance(v, dict) else v)
        out = pd.DataFrame({"day": self._to_utc_day(h[dc]),
                            "resting_hr": pd.to_numeric(val, errors="coerce")})
        return out.dropna()

    # --------------------------------------------------------------- load
    def load(self, root=None) -> LoadResult:
        p = self.require_files(root)
        found = self.locate(p)
        if not found.get("participant_dirs"):
            raise DecisionRequired(f"No participant folders p01..pNN under {p}")
        if not found.get("wellness"):
            raise DecisionRequired(
                "PMData required variables unavailable -- no "
                f"pmsys/wellness*.csv under {p}")
        if not found.get("resting_hr"):
            raise DecisionRequired(
                "PMData required variables unavailable -- no "
                f"fitbit/resting_heart_rate.json under {p}")

        prov: dict = {
            "dataset_page": "https://datasets.simula.no/pmdata/",
            "citation": ("Thambawita et al. (2020), PMData: A Sports Logging "
                         "Dataset, ACM MMSys"),
            "licence": "CC BY 4.0 (publisher page; NO licence file in the archive)",
            "severity_direction_confirmed": self.direction_confirmed,
            "zero_stress_handling": self.zero_stress_handling,
        }
        quality, frames = {}, []
        no_hr = []
        for d in found["participant_dirs"]:
            pid = d.name
            wf = sorted(glob.glob(os.path.join(str(d), "pmsys", "wellness*.csv")))
            hf = sorted(glob.glob(os.path.join(str(d), "fitbit",
                                              "resting_heart_rate.json")))
            if not wf:
                quality[pid] = {"skipped": "no pmsys/wellness*.csv"}
                continue
            w, st = self._read_wellness(Path(wf[0]), pid)
            if not hf:
                st["skipped"] = "no fitbit/resting_heart_rate.json"
                quality[pid] = st
                no_hr.append(pid)
                continue
            hr = self._read_resting_hr(Path(hf[0]))
            st["n_resting_hr_days"] = int(len(hr))
            m = w.merge(hr, on="day", how="inner")
            st["n_matched"] = int(len(m))
            quality[pid] = st
            if len(m):
                frames.append(m)

        if not frames:
            raise DecisionRequired(
                "No PMData participant has both a parsable wellness file and "
                "resting heart rate on matching days.")
        df = pd.concat(frames, ignore_index=True)
        prov["participant_quality"] = quality
        prov["participants_without_resting_hr"] = no_hr

        # ---- severity: NOT applied unless the direction is confirmed ------
        if self.direction_confirmed:
            df["report"] = (df["raw_response"] if self.ascending
                            else (PMSYS_STRESS_RANGE[1] + 1) - df["raw_response"])
            prov["severity_note"] = (
                "direction CONFIRMED by the analyst; severity "
                + ("ascends with" if self.ascending else "is REVERSED relative "
                   "to") + " the stored value")
        else:
            # Carry the raw value through so the AUDIT can characterise the
            # data, but record loudly that this is not a severity scale yet.
            df["report"] = df["raw_response"]
            prov["severity_note"] = (
                "DIRECTION NOT CONFIRMED. 'report' is the RAW stored value, "
                "not a verified severity scale. No primary result may be "
                "derived from it.")
        prov["n_matched_total"] = int(len(df))
        df = df.sort_values(["pid", "ts"]).reset_index(drop=True)
        df.attrs["data_status"] = DataStatus.REAL.value
        df.attrs["n_categories"] = PMSYS_STRESS_RANGE[1]
        df.attrs["severity_direction_confirmed"] = self.direction_confirmed
        return LoadResult(frame=df, n_categories=PMSYS_STRESS_RANGE[1],
                          sensor=self.primary_sensor,
                          data_status=DataStatus.REAL, provenance=prov)

    # -------------------------------------------------------------- audit
    def audit(self, root=None) -> DatasetAudit:
        p = self.resolve_root(root)
        if p is None:
            return self.unavailable_audit(
                root, "REAL DATA UNAVAILABLE - PMDATA AUDIT NOT RUN: "
                      f"no directory at {root!r}")
        found = self.locate(p)
        if not found.get("participant_dirs"):
            return self.unavailable_audit(
                root, "REAL DATA UNAVAILABLE - PMDATA AUDIT NOT RUN: no "
                      f"participant folders p01..pNN under {p}")
        missing = [k for k in ("wellness", "resting_hr") if not found.get(k)]
        if missing:
            return self.unavailable_audit(
                root, "DECISION REQUIRED: PMData required variables "
                      f"unavailable -- {missing} not found under {p}")

        res = self.load(root)
        df, prov = res.frame, res.provenance
        per = df.groupby("pid").size()
        span = (df["ts"].max() - df["ts"].min()).total_seconds() / 86400.0
        codes = sorted(int(v) for v in pd.unique(df["raw_response"]))
        enough = int((per >= 120).sum())
        q = prov.get("participant_quality", {})
        n_rows = sum(v.get("n_rows", 0) for v in q.values())
        n_blank = sum(v.get("n_blank_submissions_dropped", 0) for v in q.values())
        n_oor = sum(v.get("n_out_of_range", 0) for v in q.values())
        no_hr = prov.get("participants_without_resting_hr", [])

        reasons = []
        if enough < 10:
            reasons.append(
                f"Only {enough} of {len(per)} participants reach the 120 "
                "matched daily reports the frozen screen needs (60 per epoch); "
                f"the densest has {int(per.max())}. At least 10 eligible "
                "participants are required for a participant-cluster interval.")
        if no_hr:
            reasons.append(
                f"{len(no_hr)} participant(s) have no resting_heart_rate.json "
                f"and carry no primary sensor at all: {no_hr}.")
        if not self.direction_confirmed:
            reasons.append(
                "The PMSys stress scale DIRECTION is not documented in the "
                "released archive (no README, no codebook, no questionnaire "
                "definition) and has not been confirmed. Until it is, the "
                "stored integer is not a verified severity scale and NO "
                "primary result may be derived from it.")

        return DatasetAudit(
            dataset_name=self.name, role=self.role, data_status=DataStatus.REAL,
            source_status=(f"local files audited at {p}; "
                           "https://datasets.simula.no/pmdata/"),
            local_files_available=True, root_path=str(p),
            files_found=tuple(f"{k}: {len(v)}" for k, v in found.items()),
            participant_count=int(df["pid"].nunique()),
            observation_count=int(len(df)),
            sensor_modalities=("Fitbit resting heart rate (daily)",),
            self_report_variables=(self.report_variable,),
            self_report_scale=(
                f"PMSys `stress`, integers {PMSYS_STRESS_RANGE[0]}-"
                f"{PMSYS_STRESS_RANGE[1]}; DIRECTION "
                + ("confirmed by the analyst" if self.direction_confirmed
                   else "NOT DOCUMENTED AND NOT CONFIRMED")),
            stress_labels=(),
            raw_stored_codes=tuple(codes),
            code_to_label_mapping={
                c: f"{c} = stored integer; no label text exists in the release"
                for c in codes},
            code_to_severity_mapping=(
                {c: (c if self.ascending
                     else PMSYS_STRESS_RANGE[1] + 1 - c) for c in codes}
                if self.direction_confirmed else {}),
            timestamps_present=True,
            timestamp_format="ISO-8601 with Z (wellness); naive datetime (Fitbit)",
            timezone=("wellness is UTC; Fitbit resting HR is naive local. Both "
                      "are pinned to a UTC calendar day for the join"),
            longitudinal_span_days=float(span),
            observations_per_participant={str(k): int(v) for k, v in per.items()},
            median_observations_per_participant=float(per.median()),
            missingness={
                "blank_submissions_dropped":
                    float(n_blank / n_rows) if n_rows else float("nan"),
                "stress_out_of_documented_range":
                    float(n_oor / n_rows) if n_rows else float("nan"),
                "wellness_rows_without_same_day_resting_hr":
                    float(1.0 - len(df) / n_rows) if n_rows else float("nan")},
            participant_level_coverage={
                str(k): float(v / max(per.max(), 1)) for k, v in per.items()},
            sensor_report_alignment=("daily wellness report joined to the same "
                                     "UTC calendar day's resting heart rate"),
            conversation_context_available=False,
            eligible_for_primary_analysis=bool(enough >= 10
                                               and self.direction_confirmed),
            eligible_for_benchmark_analysis=True,
            exclusion_reasons=tuple(reasons),
            acquisition_instructions=self.acquisition_instructions,
            notes=(
                "The released archive contains NO README, NO codebook and NO "
                "questionnaire definition. participant-overview.xlsx holds "
                "demographics only.",
                f"{n_blank} wellness row(s) had EVERY item at 0 and were "
                "dropped as blank submissions; this is unambiguous cleaning, "
                "not imputation.",
                f"{n_oor} row(s) had `stress` outside 1-5 with other items "
                f"valid; handling = {self.zero_stress_handling!r}.",
                "Fitbit resting heart rate is an ALGORITHMIC daily estimate, "
                "not a raw measurement; its variance can shift between epochs "
                "for device reasons rather than physiological ones, which "
                "assumption A3 is sensitive to."))


register_adapter(PMDataAdapter())
