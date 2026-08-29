"""PMDATA ADAPTER -- CONDITIONAL SECONDARY target.

PORTED from the validated ``final_audit.py::load_pmdata`` (Round 16).

Conditional because: 16 participants over ~5 months with ONE wellness report
per day gives roughly 75 reports per epoch, close to the frozen minimum of 60.
Whether it clears the screen is an empirical question the audit answers; it is
not assumed either way here.

Verified by ``audit()``: PMSys stress scale, the exact stress variable, raw
codes and labels, timestamps, participant IDs, resting-HR availability,
longitudinal structure, missingness, participant count, longitudinal span.

If a required variable is unavailable the audit says so and ``load`` raises
``DECISION REQUIRED: PMData required variables unavailable``.
"""
from __future__ import annotations

import glob
import logging
import os
from pathlib import Path

import pandas as pd

from ..constants import DataStatus, DatasetRole
from ..errors import DecisionRequired
from ..schemas import DatasetAudit
from .base import DatasetAdapter, LoadResult, register_adapter

log = logging.getLogger(__name__)

__all__ = ["PMDataAdapter"]

ACQUISITION = """\
ACQUISITION -- PMData (Simula), Thambawita et al. 2020, CC BY 4.0.
  1. Download from https://datasets.simula.no/pmdata/ (open access).
  2. Extract and pass the directory that contains the participant folders
     p01 ... p16 as --root, e.g.
         python scripts/audit_dataset.py --dataset pmdata --root ~/data/pmdata
  3. Expected layout:
         <root>/p01/pmsys/wellness.csv        (has a `stress` column)
         <root>/p01/fitbit/resting_heart_rate.json
  4. PMData is CONDITIONAL: one wellness report per day over ~5 months gives
     roughly 75 reports per epoch against a frozen minimum of 60. The audit
     reports whether it actually clears that bar."""


class PMDataAdapter(DatasetAdapter):
    name = "pmdata"
    role = DatasetRole.CONDITIONAL_SECONDARY
    primary_sensor = "resting_hr"
    report_variable = "PMSys daily wellness `stress`"
    can_support_longitudinal_estimand = True
    acquisition_instructions = ACQUISITION

    def locate(self, root: Path) -> dict[str, list[Path]]:
        if root is None or not Path(root).exists():
            return {}
        r = str(root)
        pdirs = [Path(d) for d in sorted(glob.glob(os.path.join(r, "p*")))
                 if os.path.isdir(d)]
        return {
            "participant_dirs": pdirs,
            "wellness": [Path(f) for d in pdirs for f in
                         glob.glob(os.path.join(str(d), "pmsys", "wellness*.csv"))],
            "resting_hr": [Path(f) for d in pdirs for f in
                           glob.glob(os.path.join(str(d), "fitbit",
                                                  "resting_heart_rate.json"))],
        }

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
        missing = []
        if not found.get("wellness"):
            missing.append("pmsys/wellness*.csv (the `stress` variable)")
        if not found.get("resting_hr"):
            missing.append("fitbit/resting_heart_rate.json")
        if missing:
            return self.unavailable_audit(
                root, "DECISION REQUIRED: PMData required variables "
                      f"unavailable -- {', '.join(missing)} not found under {p}")

        res = self.load(root)
        df = res.frame
        per = df.groupby("pid").size()
        span = (df["ts"].max() - df["ts"].min()).total_seconds() / 86400.0
        codes = sorted(int(v) for v in pd.unique(df["report"]))
        enough = int((per >= 120).sum())
        return DatasetAudit(
            dataset_name=self.name, role=self.role, data_status=DataStatus.REAL,
            source_status=f"local files audited at {p}",
            local_files_available=True, root_path=str(p),
            files_found=tuple(f"{k}: {len(v)}" for k, v in found.items()),
            participant_count=int(df["pid"].nunique()),
            observation_count=int(len(df)),
            sensor_modalities=("Fitbit resting heart rate",),
            self_report_variables=(self.report_variable,),
            self_report_scale=res.provenance.get("report_scale"),
            stress_labels=(),
            raw_stored_codes=tuple(codes),
            code_to_label_mapping={c: f"PMSys stress = {c} (numeric, no text "
                                      "labels in the source)" for c in codes},
            code_to_severity_mapping={c: c for c in codes},
            timestamps_present=True, timestamp_format="ISO date",
            timezone="not recorded in the source files",
            longitudinal_span_days=float(span),
            observations_per_participant={str(k): int(v) for k, v in per.items()},
            median_observations_per_participant=float(per.median()),
            missingness={"matched_report_days": 1.0 - (
                len(df) / max(res.provenance.get("n_wellness_rows", 1), 1))},
            participant_level_coverage={
                str(k): float(v / max(per.max(), 1)) for k, v in per.items()},
            sensor_report_alignment="daily wellness joined to same-day resting HR",
            conversation_context_available=False,
            eligible_for_primary_analysis=bool(enough >= 10),
            eligible_for_benchmark_analysis=False,
            exclusion_reasons=() if enough >= 10 else (
                f"only {enough} participants have >= 120 matched daily reports "
                "(60 per epoch). PMData is CONDITIONAL for exactly this reason.",),
            acquisition_instructions=self.acquisition_instructions,
            notes=("PMSys `stress` is stored as a bare integer with no label "
                   "text, so the label-text remap cannot be applied. The audit "
                   "records the observed code range verbatim; confirm the "
                   "scale direction against the PMData documentation before "
                   "any primary analysis.",))

    def load(self, root=None) -> LoadResult:
        p = self.require_files(root)
        found = self.locate(p)
        prov: dict = {}
        if not found["participant_dirs"]:
            raise DecisionRequired(
                f"No participant folders p01..pNN under {p}")
        frames, n_wellness = [], 0
        for d in found["participant_dirs"]:
            pid = d.name
            well = sorted(glob.glob(os.path.join(str(d), "pmsys", "wellness*.csv")))
            if not well:
                log.info("skip %s: no pmsys/wellness*.csv", pid)
                continue
            w = pd.read_csv(well[0])
            w.columns = [c.strip().lower() for c in w.columns]
            tc = next((c for c in w.columns if "date" in c or "time" in c), None)
            rc = next((c for c in w.columns if c == "stress"), None)
            if tc is None or rc is None:
                raise DecisionRequired(
                    "PMData required variables unavailable -- "
                    f"{pid}: cannot find date/stress columns. "
                    f"Found: {list(w.columns)}")
            w = w[[tc, rc]].dropna()
            w.columns = ["ts", "report"]
            w["pid"] = pid
            w["ts"] = pd.to_datetime(w["ts"], errors="coerce")
            w = w.dropna(subset=["ts"])
            w["day"] = w["ts"].dt.normalize()
            w["report"] = pd.to_numeric(w["report"], errors="coerce")
            w = w.dropna(subset=["report"])
            w["report"] = w["report"].round().astype(int)
            n_wellness += len(w)

            hrf = sorted(glob.glob(os.path.join(str(d), "fitbit",
                                                "resting_heart_rate.json")))
            if not hrf:
                log.info("skip %s: no fitbit/resting_heart_rate.json", pid)
                continue
            try:
                h = pd.read_json(hrf[0])
            except Exception as e:
                log.info("skip %s: cannot read resting HR (%s)", pid, e)
                continue
            dc = next((c for c in h.columns if "date" in str(c).lower()), None)
            vc = next((c for c in h.columns if c != dc), None)
            if dc is None or vc is None:
                raise DecisionRequired(
                    f"{pid}: resting HR columns unclear: {list(h.columns)}")
            hh = h[[dc, vc]].copy()
            hh.columns = ["day", self.primary_sensor]
            hh["day"] = pd.to_datetime(hh["day"], errors="coerce").dt.normalize()
            hh[self.primary_sensor] = hh[self.primary_sensor].apply(
                lambda v: v.get("value") if isinstance(v, dict) else v)
            hh[self.primary_sensor] = pd.to_numeric(hh[self.primary_sensor],
                                                    errors="coerce")
            frames.append(w.merge(hh.dropna(), on="day", how="inner"))
        if not frames:
            raise DecisionRequired(
                "PMData required variables unavailable -- no participant has "
                "both pmsys/wellness*.csv and fitbit/resting_heart_rate.json.")
        df = pd.concat(frames, ignore_index=True)
        lo, hi = int(df["report"].min()), int(df["report"].max())
        if lo < 1 or hi > 10:
            raise DecisionRequired(
                f"PMSys stress out of the expected 1-5 / 1-10 range: [{lo},{hi}]. "
                "Confirm the instrument before proceeding.")
        prov["report_range"] = (lo, hi)
        prov["report_scale"] = f"{hi}-point numeric, observed range [{lo}, {hi}]"
        prov["n_wellness_rows"] = n_wellness
        df.attrs["data_status"] = DataStatus.REAL.value
        df.attrs["n_categories"] = hi
        return LoadResult(frame=df, n_categories=hi, sensor=self.primary_sensor,
                          data_status=DataStatus.REAL, provenance=prov)


register_adapter(PMDataAdapter())
