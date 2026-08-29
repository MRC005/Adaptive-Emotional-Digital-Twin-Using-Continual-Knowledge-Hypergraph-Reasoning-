"""RELAX ADAPTER -- LONGITUDINAL ALTERNATIVE / ADDITIONAL VALIDATION TARGET.

RELAX is a longitudinal wearable stress dataset with repeated subjective stress
reports. Its role here is a SECOND longitudinal target: if StudentLife's
sensor-report association turns out to be too weak (diagnostic [9b]), an
alternative longitudinal cohort matters more than a physiological benchmark,
because only a longitudinal cohort can support the estimand at all.

Verified by ``audit()``:
  repeated subjective stress observations; participant identifiers; timestamps;
  wearable physiology; longitudinal span; repeated measures sufficient for
  epoch analysis; alignment between physiology and reports.

STATUS. The file layout below is a DECLARED EXPECTATION, not a verified fact:
no RELAX file has been opened by this project. The adapter is written to fail
loudly and specifically when the layout differs, so the first real run produces
a precise DECISION REQUIRED rather than a wrong number. Column names are
configurable in ``configs/relax.yaml`` so that adapting to the actual release is
a config change, not a code change.
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

__all__ = ["RelaxAdapter"]

ACQUISITION = """\
ACQUISITION -- RELAX longitudinal stress dataset.
  1. Obtain the release archive from its published repository record and note
     the DOI and licence in docs/dataset_audit.md.
  2. Extract it and pass the directory containing the per-participant report
     and physiology tables as --root.
  3. The adapter looks for, in order of preference:
         <root>/**/*report*.csv    (participant id, timestamp, stress rating)
         <root>/**/*physio*.csv    (participant id, timestamp, signal columns)
     Column names are configurable in configs/relax.yaml.
  4. IMPORTANT: no RELAX file has been opened by this project. The expected
     layout is a declared assumption. If the audit halts with DECISION
     REQUIRED, record the ACTUAL column names in configs/relax.yaml -- do not
     edit the adapter to guess."""


class RelaxAdapter(DatasetAdapter):
    name = "relax"
    role = DatasetRole.LONGITUDINAL_ALTERNATIVE
    primary_sensor = "physio_window_mean"
    report_variable = "repeated subjective stress rating"
    can_support_longitudinal_estimand = True
    acquisition_instructions = ACQUISITION

    def __init__(self, *, report_glob: str = "*report*.csv",
                 physio_glob: str = "*physio*.csv",
                 pid_col: str = "participant_id", ts_col: str = "timestamp",
                 report_col: str = "stress", signal_col: str = "hr",
                 n_categories: int | None = None):
        self.report_glob = report_glob
        self.physio_glob = physio_glob
        self.pid_col = pid_col
        self.ts_col = ts_col
        self.report_col = report_col
        self.signal_col = signal_col
        self.n_categories = n_categories

    @classmethod
    def from_config(cls, cfg) -> "RelaxAdapter":
        return cls(
            report_glob=str(cfg.get("adapter.report_glob", "*report*.csv")),
            physio_glob=str(cfg.get("adapter.physio_glob", "*physio*.csv")),
            pid_col=str(cfg.get("adapter.pid_col", "participant_id")),
            ts_col=str(cfg.get("adapter.ts_col", "timestamp")),
            report_col=str(cfg.get("adapter.report_col", "stress")),
            signal_col=str(cfg.get("adapter.signal_col", "hr")),
            n_categories=cfg.get("dataset.n_categories", None))

    def locate(self, root: Path) -> dict[str, list[Path]]:
        if root is None or not Path(root).exists():
            return {}
        r = str(root)
        return {
            "reports": [Path(f) for f in glob.glob(
                os.path.join(r, "**", self.report_glob), recursive=True)],
            "physiology": [Path(f) for f in glob.glob(
                os.path.join(r, "**", self.physio_glob), recursive=True)],
        }

    # ------------------------------------------------------------ helpers
    def _pick(self, cols: list[str], wanted: str, alternatives: tuple[str, ...],
              path: Path) -> str:
        low = {c.lower().strip(): c for c in cols}
        if wanted.lower() in low:
            return low[wanted.lower()]
        for a in alternatives:
            if a in low:
                return low[a]
            for k, orig in low.items():
                if a in k:
                    return orig
        raise DecisionRequired(
            f"RELAX: cannot identify the {wanted!r} column in {path}. "
            f"Columns present: {cols}. Record the actual name in "
            "configs/relax.yaml rather than guessing.")

    def audit(self, root=None) -> DatasetAudit:
        p = self.resolve_root(root)
        if p is None:
            return self.unavailable_audit(
                root, "REAL DATA UNAVAILABLE - RELAX AUDIT NOT RUN: "
                      f"no directory at {root!r}")
        found = self.locate(p)
        if not found.get("reports"):
            return self.unavailable_audit(
                root, "REAL DATA UNAVAILABLE - RELAX AUDIT NOT RUN: no files "
                      f"matching {self.report_glob!r} under {p}")
        if not found.get("physiology"):
            return self.unavailable_audit(
                root, "REAL DATA UNAVAILABLE - RELAX AUDIT NOT RUN: no files "
                      f"matching {self.physio_glob!r} under {p}")
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
            sensor_modalities=("wearable physiology",),
            self_report_variables=(self.report_variable,),
            self_report_scale=res.provenance.get("report_scale"),
            raw_stored_codes=tuple(codes),
            code_to_severity_mapping={c: c for c in codes},
            timestamps_present=True, timestamp_format="parsed datetime",
            timezone="not recorded in the source files",
            longitudinal_span_days=float(span),
            observations_per_participant={str(k): int(v) for k, v in per.items()},
            median_observations_per_participant=float(per.median()),
            missingness={"unmatched_reports": res.provenance.get(
                "unmatched_report_fraction", float("nan"))},
            participant_level_coverage={
                str(k): float(v / max(per.max(), 1)) for k, v in per.items()},
            sensor_report_alignment="physiology aggregated into a causal window "
                                    "ending at each report",
            conversation_context_available=False,
            eligible_for_primary_analysis=bool(enough >= 10),
            eligible_for_benchmark_analysis=True,
            exclusion_reasons=() if enough >= 10 else (
                f"only {enough} participants have >= 120 aligned reports "
                "(60 per epoch).",),
            acquisition_instructions=self.acquisition_instructions,
            notes=("The scale DIRECTION of the RELAX stress rating must be "
                   "confirmed against its documentation before any primary "
                   "analysis. This adapter does not infer it.",))

    def load(self, root=None) -> LoadResult:
        from ..alignment.align import CausalWindow, align_sensor_to_reports

        p = self.require_files(root)
        found = self.locate(p)
        prov: dict = {}

        reps = []
        for fp in sorted(found["reports"]):
            d = pd.read_csv(fp)
            cols = list(d.columns)
            pc = self._pick(cols, self.pid_col, ("participant", "subject", "pid",
                                                 "id"), fp)
            tc = self._pick(cols, self.ts_col, ("timestamp", "time", "date"), fp)
            rc = self._pick(cols, self.report_col, ("stress", "rating",
                                                    "self_report"), fp)
            t = d[[pc, tc, rc]].copy()
            t.columns = ["pid", "ts", "report"]
            reps.append(t)
        rep = pd.concat(reps, ignore_index=True)
        rep["pid"] = rep["pid"].astype(str)
        rep["ts"] = pd.to_datetime(rep["ts"], errors="coerce")
        rep["report"] = pd.to_numeric(rep["report"], errors="coerce")
        rep = rep.dropna(subset=["ts", "report"])
        if rep.empty:
            raise DecisionRequired(
                f"RELAX: no parsable (timestamp, stress) rows under {p}.")
        rep["report"] = rep["report"].round().astype(int)
        lo, hi = int(rep["report"].min()), int(rep["report"].max())
        if lo < 1:
            raise DecisionRequired(
                f"RELAX stress ratings start at {lo}; the ordinal model needs "
                "1-based severity codes. Record the true coding in "
                "configs/relax.yaml.")
        K = self.n_categories or hi
        prov["report_scale"] = f"{K}-point, observed range [{lo}, {hi}]"

        phys = []
        for fp in sorted(found["physiology"]):
            d = pd.read_csv(fp)
            cols = list(d.columns)
            pc = self._pick(cols, self.pid_col, ("participant", "subject",
                                                 "pid", "id"), fp)
            tc = self._pick(cols, self.ts_col, ("timestamp", "time", "date"), fp)
            sc = self._pick(cols, self.signal_col, ("hr", "heart", "eda", "bvp",
                                                    "value"), fp)
            t = d[[pc, tc, sc]].copy()
            t.columns = ["pid", "ts", "signal"]
            phys.append(t)
        ph = pd.concat(phys, ignore_index=True)
        ph["pid"] = ph["pid"].astype(str)
        ph["ts"] = pd.to_datetime(ph["ts"], errors="coerce")
        ph["signal"] = pd.to_numeric(ph["signal"], errors="coerce")
        ph = ph.dropna(subset=["ts", "signal"])

        aligned = align_sensor_to_reports(
            rep, ph, "signal", CausalWindow(lookback_hours=2.0,
                                            aggregation="mean", min_samples=1),
            out_col=self.primary_sensor)
        n_before = len(aligned)
        df = aligned.dropna(subset=[self.primary_sensor]).copy()
        prov["unmatched_report_fraction"] = float(
            1.0 - len(df) / max(n_before, 1))
        if df.empty:
            raise DecisionRequired(
                "RELAX: no report has physiology in its causal window. The "
                "timestamps of the two tables are not on a common clock.")
        df["day"] = df["ts"].dt.normalize()
        df.attrs["data_status"] = DataStatus.REAL.value
        df.attrs["n_categories"] = K
        return LoadResult(frame=df, n_categories=K, sensor=self.primary_sensor,
                          data_status=DataStatus.REAL, provenance=prov)


register_adapter(RelaxAdapter())
