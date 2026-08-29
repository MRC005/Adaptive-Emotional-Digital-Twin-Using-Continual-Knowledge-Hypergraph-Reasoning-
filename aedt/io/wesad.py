"""WESAD ADAPTER -- PHYSIOLOGICAL BENCHMARK ONLY.

** WESAD CANNOT SUPPORT THE PRIMARY LONGITUDINAL ESTIMAND, AND THIS ADAPTER
   REFUSES TO PRETEND OTHERWISE. **

WESAD is a single laboratory session per subject with protocol-defined
condition blocks (baseline / stress / amusement / meditation). It has:
  - subject IDs                                       yes
  - protocol structure                                yes
  - condition labels                                  yes
  - rich sensor modalities (chest + wrist)            yes
  - sample timing                                     yes
  - REPEATED SELF-REPORT ACROSS SEPARATED TIME EPOCHS no

rho* is a ratio of the sensor->report slope in epoch 2 to epoch 1, where the
epochs are halves of a MULTI-WEEK enrolment span and the thresholds are assumed
fixed across them (A2). A single session contains no such span. Splitting one
session into "epochs" would produce a number, and that number would not be an
estimate of longitudinal reporting-scale change.

Therefore ``can_support_longitudinal_estimand = False``, ``audit()`` reports
``eligible_for_primary_analysis = False`` with the reason, and
``assert_benchmark_only`` raises if any caller tries to route WESAD into the
primary path. This constraint is a property of the DATA and is not overridable
by configuration.

What WESAD IS good for, and what it is used for here: validating that the
feature-extraction interface and the ordinal model behave sensibly on real
physiological signals with real condition labels -- an ENGINEERING BENCHMARK.
"""
from __future__ import annotations

import glob
import logging
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from ..constants import DataStatus, DatasetRole
from ..errors import DecisionRequired, ScientificError
from ..schemas import DatasetAudit
from .base import DatasetAdapter, LoadResult, register_adapter

log = logging.getLogger(__name__)

__all__ = ["WesadAdapter", "assert_benchmark_only", "WESAD_CONDITION_LABELS"]

# The protocol labels as documented by the WESAD release. Codes 0, 5, 6, 7 are
# transient/undefined and are NOT analysed.
WESAD_CONDITION_LABELS = {
    0: "not defined / transient",
    1: "baseline",
    2: "stress",
    3: "amusement",
    4: "meditation",
    5: "transient",
    6: "transient",
    7: "transient",
}
ANALYSED_CONDITIONS = (1, 2, 3, 4)

ACQUISITION = """\
ACQUISITION -- WESAD (Schmidt et al. 2018).
  1. Download WESAD.zip from the UniPassau public release page.
  2. Extract and pass the directory containing S2 ... S17 as --root, e.g.
         python scripts/audit_dataset.py --dataset wesad --root ~/data/WESAD
  3. Expected layout:
         <root>/S2/S2.pkl   (dict with 'signal', 'label', 'subject')
  4. WESAD IS A BENCHMARK ONLY. It cannot validate rho*, and the system will
     refuse to run the primary analysis on it. Use it to check feature
     extraction and model behaviour on real physiological signals."""

BENCHMARK_ONLY_REASON = (
    "WESAD is a single-session laboratory protocol. It has no repeated "
    "self-report across separated longitudinal epochs, so it cannot identify "
    "rho*, which is defined as a ratio of within-person sensor->report slopes "
    "across halves of a multi-week enrolment span under fixed thresholds (A2). "
    "Presenting a WESAD number as longitudinal validation of rho* would be "
    "false. It is eligible for BENCHMARK analysis only.")


def assert_benchmark_only(dataset_name: str) -> None:
    """Hard stop if a benchmark dataset is routed into the primary analysis."""
    raise ScientificError(
        f"{dataset_name.upper()} CANNOT SUPPORT THE PRIMARY LONGITUDINAL "
        f"ANALYSIS.\n{BENCHMARK_ONLY_REASON}")


class WesadAdapter(DatasetAdapter):
    name = "wesad"
    role = DatasetRole.BENCHMARK_PHYSIOLOGICAL
    primary_sensor = "chest_ecg_window_mean"
    report_variable = "protocol condition label (NOT a self-report time series)"
    can_support_longitudinal_estimand = False
    acquisition_instructions = ACQUISITION

    def locate(self, root: Path) -> dict[str, list[Path]]:
        if root is None or not Path(root).exists():
            return {}
        r = str(root)
        return {"subject_pickles": [Path(f) for f in sorted(glob.glob(
            os.path.join(r, "**", "S*.pkl"), recursive=True))]}

    def audit(self, root=None) -> DatasetAudit:
        p = self.resolve_root(root)
        if p is None:
            return self.unavailable_audit(
                root, "REAL DATA UNAVAILABLE - WESAD AUDIT NOT RUN: "
                      f"no directory at {root!r}")
        found = self.locate(p)
        if not found.get("subject_pickles"):
            return self.unavailable_audit(
                root, "REAL DATA UNAVAILABLE - WESAD AUDIT NOT RUN: no S*.pkl "
                      f"subject files under {p}")

        res = self.load(root)
        df = res.frame
        per = df.groupby("pid").size()
        labels = sorted(int(v) for v in pd.unique(df["report"]))
        span = (df["ts"].max() - df["ts"].min()).total_seconds() / 86400.0
        return DatasetAudit(
            dataset_name=self.name, role=self.role, data_status=DataStatus.REAL,
            source_status=f"local files audited at {p}",
            local_files_available=True, root_path=str(p),
            files_found=tuple(f"{k}: {len(v)}" for k, v in found.items()),
            participant_count=int(df["pid"].nunique()),
            observation_count=int(len(df)),
            sensor_modalities=tuple(res.provenance.get("modalities", ())),
            self_report_variables=(self.report_variable,),
            self_report_scale=("protocol condition, ordered here as "
                               "baseline < amusement < meditation < stress "
                               "ONLY for benchmark modelling; this is NOT a "
                               "self-reported severity scale"),
            stress_labels=tuple(WESAD_CONDITION_LABELS[c]
                                for c in ANALYSED_CONDITIONS),
            raw_stored_codes=tuple(ANALYSED_CONDITIONS),
            code_to_label_mapping={c: WESAD_CONDITION_LABELS[c]
                                   for c in ANALYSED_CONDITIONS},
            code_to_severity_mapping=res.provenance.get("condition_order", {}),
            timestamps_present=True,
            timestamp_format="derived from the 700 Hz chest sample index",
            timezone="none -- session-relative clock",
            longitudinal_span_days=float(span),
            observations_per_participant={str(k): int(v) for k, v in per.items()},
            median_observations_per_participant=float(per.median()),
            missingness={},
            participant_level_coverage={
                str(k): float(v / max(per.max(), 1)) for k, v in per.items()},
            sensor_report_alignment="60 s windows labelled by protocol condition",
            conversation_context_available=False,
            eligible_for_primary_analysis=False,
            eligible_for_benchmark_analysis=True,
            exclusion_reasons=(BENCHMARK_ONLY_REASON,),
            acquisition_instructions=self.acquisition_instructions,
            notes=("BENCHMARK ONLY. Any attempt to route this dataset into the "
                   "primary rho* analysis raises ScientificError.",
                   f"Condition codes present: {labels}"))

    def load(self, root=None) -> LoadResult:
        p = self.require_files(root)
        found = self.locate(p)
        rows = []
        modalities: set[str] = set()
        # Ordered ONLY so a benchmark ordinal fit is defined. This ordering is
        # a modelling convenience for the benchmark and is NOT a claim that the
        # protocol conditions form a self-reported severity scale.
        order = {1: 1, 3: 2, 4: 3, 2: 4}
        for fp in found["subject_pickles"]:
            try:
                with open(fp, "rb") as f:
                    d = pickle.load(f, encoding="latin1")
            except Exception as e:
                raise DecisionRequired(
                    f"WESAD: cannot unpickle {fp} ({e}). Confirm the release "
                    "version; this adapter expects the documented dict with "
                    "'signal', 'label' and 'subject' keys.")
            if not {"signal", "label"} <= set(d):
                raise DecisionRequired(
                    f"WESAD: {fp} lacks 'signal'/'label' keys. "
                    f"Keys present: {list(d)}")
            sig = d["signal"]
            chest = sig.get("chest", {}) if isinstance(sig, dict) else {}
            modalities.update(f"chest/{k}" for k in chest)
            if isinstance(sig, dict) and "wrist" in sig:
                modalities.update(f"wrist/{k}" for k in sig["wrist"])
            if "ECG" not in chest:
                raise DecisionRequired(
                    f"WESAD: {fp} has no chest/ECG channel. Channels present: "
                    f"{list(chest)}")
            ecg = np.asarray(chest["ECG"], dtype=float).ravel()
            lab = np.asarray(d["label"]).ravel()
            n = min(len(ecg), len(lab))
            ecg, lab = ecg[:n], lab[:n]
            fs = 700                      # documented chest sampling rate
            win = 60 * fs
            pid = str(d.get("subject", fp.stem))
            for i in range(0, n - win + 1, win):
                block = lab[i:i + win]
                vals, counts = np.unique(block, return_counts=True)
                dom = int(vals[counts.argmax()])
                if dom not in ANALYSED_CONDITIONS:
                    continue
                if counts.max() / len(block) < 0.9:   # mixed window, skip
                    continue
                seg = ecg[i:i + win]
                rows.append((pid, i / fs, order[dom], float(np.mean(seg)),
                             float(np.std(seg, ddof=1)),
                             float(np.ptp(seg)), dom))
        if not rows:
            raise DecisionRequired(
                "WESAD: no clean 60 s condition windows were extracted. "
                "Confirm the label encoding in this release.")
        df = pd.DataFrame(rows, columns=[
            "pid", "seconds", "report", self.primary_sensor,
            "chest_ecg_window_sd", "chest_ecg_window_range", "condition_code"])
        df["ts"] = pd.Timestamp("2018-01-01") + pd.to_timedelta(df["seconds"],
                                                                unit="s")
        df["day"] = df["ts"].dt.normalize()
        df.attrs["data_status"] = DataStatus.REAL.value
        df.attrs["n_categories"] = 4
        df.attrs["benchmark_only"] = True
        return LoadResult(
            frame=df, n_categories=4, sensor=self.primary_sensor,
            data_status=DataStatus.REAL,
            provenance={"modalities": sorted(modalities),
                        "condition_order": order,
                        "window_seconds": 60, "sampling_rate_hz": 700,
                        "benchmark_only": True,
                        "benchmark_only_reason": BENCHMARK_ONLY_REASON})


register_adapter(WesadAdapter())
