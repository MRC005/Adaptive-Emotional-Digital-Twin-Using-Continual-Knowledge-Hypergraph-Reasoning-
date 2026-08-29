"""RELAX ADAPTER -- LONGITUDINAL ALTERNATIVE / ADDITIONAL VALIDATION TARGET.

** THIS ADAPTER IS WRITTEN AGAINST THE ACTUAL RELEASED FILES, WHICH HAVE BEEN
   OPENED AND INSPECTED. ** It is not a declared expectation.

Source   Halmich, Jung, Schmoigl-Tonis, Schranz, Kremser, Kunas & Laireiter
         (2026), "A six-week longitudinal dataset of wearable and self-reported
         stress measurements in working adults", Scientific Data.
Data     Zenodo record 20701999, DOI 10.5281/zenodo.20701999, CC-BY-4.0, open.
Device   Polar Verity Sense (SDK v5.1.0, firmware v2.1.0).

VERIFIED STRUCTURE (read from the archive's own central directory and files):

    RELAXDataset/questionnaire_responses.xlsx     sheets: users, interv,
                                                  mfb, ifb, afb, profile1..7
    RELAXDataset/metadata/questionnaires.xlsx     item + answer-label definitions
    RELAXDataset/metadata/README.md               data dictionary
    RELAXDataset/data/<pid>/ibi_data.parquet      ibi_ppi (ms, int32),
                                                  ibi_blocker (bool),
                                                  ibi_errorEstimate (ms),
                                                  timestamp (UTC, tz-aware)
    RELAXDataset/data/<pid>/acc_data.parquet      52 Hz triaxial accelerometer
                                                  (NOT required by this adapter)

31 participants (ids 12..63), four study phases spanning 2024-02-25 to
2024-04-28 UTC.

THE SELF-REPORT DIRECTION IS READ FROM THE ANSWER-LABEL TEXT, NEVER FROM THE
STORED INTEGER. RELAX Likert items are anchored in both directions -- some
ascend in stress severity and some descend -- so mapping by position would
silently invert the scale for half of them. ``ITEM_SPECS`` records the exact
anchor pair for every supported item and the loader HALTS with
``DecisionRequired`` if the anchors in the file differ from those recorded here.

HONEST LIMITATION ON THE ITEM ITSELF. RELAX has no single-item "stress" scale
of the kind the frozen specification names for StudentLife. The densest
repeated ordinal item is ``ifb-2`` ("I feel: excited <-> calm"), which measures
momentary tension/arousal and is a stress PROXY, not a stress scale. That
choice is recorded in ``docs/decision_required.md`` rather than buried here.
"""
from __future__ import annotations

import glob
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..constants import DataStatus, DatasetRole
from ..errors import DecisionRequired
from ..schemas import DatasetAudit
from .base import DatasetAdapter, LoadResult, register_adapter

log = logging.getLogger(__name__)

__all__ = ["RelaxAdapter", "ITEM_SPECS", "RelaxItemSpec"]

ACQUISITION = """\
ACQUISITION -- RELAX (Halmich et al. 2026, Sci Data), CC-BY-4.0.
  1. Record: https://doi.org/10.5281/zenodo.20701999  (open access)
     The published archive is a single 16.5 GB zip, RELAXDataset.zip, of which
     ~15.9 GB is accelerometer data this analysis does NOT use.
  2. Fetch only what is needed (~0.5 GB) with the helper in this repository:
         python scripts/fetch_relax.py --root data/raw/relax
     It uses HTTP range requests to pull, from inside the remote zip:
         questionnaire_responses.xlsx      (0.45 MB, the self-reports)
         metadata/questionnaires.xlsx      (0.08 MB, the item definitions)
         metadata/README.md                (the data dictionary)
         data/<pid>/ibi_data.parquet       (~500 MB total, 31 participants)
  3. Or download the full zip manually and extract it so that --root contains
     `questionnaire_responses.xlsx`, `metadata/` and `data/`.
  4. Reading .xlsx needs the optional extra:  pip install -e ".[relax]"
"""


@dataclass(frozen=True)
class RelaxItemSpec:
    """One repeated ordinal item, with its VERIFIED anchor labels.

    ``ascending`` is True when a LARGER stored value means MORE stress. It is
    derived from, and checked against, the anchor text in the released
    questionnaire definition -- never assumed from the item's position.
    """

    sheet: str
    question_id: str
    column: str
    n_categories: int
    anchors_en: tuple[str, str]
    ascending: bool
    text_en: str
    note: str = ""

    def severity(self, raw: pd.Series) -> pd.Series:
        """Map the stored value to 1..K severity, larger = more stress."""
        return raw if self.ascending else (self.n_categories + 1) - raw


# Every entry below was read from RELAXDataset/metadata/questionnaires.xlsx.
ITEM_SPECS: dict[str, RelaxItemSpec] = {
    # ---- in-the-moment feedback (densest repeated item) -------------------
    "ifb-2": RelaxItemSpec(
        sheet="ifb", question_id="ifb-2", column="answers/ifb-2-1",
        n_categories=7, anchors_en=("excited", "calm"), ascending=False,
        text_en="I feel:",
        note="Momentary tension/arousal. Anchor 7 = 'calm' = LEAST stressed, "
             "so severity is REVERSED. A stress PROXY, not a stress scale."),
    "ifb-7": RelaxItemSpec(
        sheet="ifb", question_id="ifb-7", column="answers/ifb-7-1",
        n_categories=7, anchors_en=("low", "high"), ascending=True,
        text_en="My mental effort is:",
        note="Momentary mental load. Anchor 7 = 'high' effort, so severity "
             "ascends. Measures demand rather than stress response."),
    # ---- morning feedback --------------------------------------------------
    "mfb-3": RelaxItemSpec(
        sheet="mfb", question_id="mfb-3", column="answers/mfb-3-1",
        n_categories=7, anchors_en=("no stress at all", "a lot of stress"),
        ascending=True, text_en="I expect for today:",
        note="The only explicitly stress-worded item, but it is ANTICIPATORY "
             "(expected stress), not experienced stress."),
    # ---- evening feedback (PSS-like) --------------------------------------
    "afb-9": RelaxItemSpec(
        sheet="afb", question_id="afb-9", column="answers/afb-9-1",
        n_categories=7, anchors_en=("strongly agree", "strongly disagree"),
        ascending=False, text_en="Overall I felt overwhelmed today:",
        note="Anchor 1 = 'strongly agree' = MOST overwhelmed, so severity is "
             "REVERSED. Once per day only."),
    "afb-11": RelaxItemSpec(
        sheet="afb", question_id="afb-11", column="answers/afb-11-1",
        n_categories=7, anchors_en=("strongly agree", "strongly disagree"),
        ascending=False, text_en="Today I couldn't relax:",
        note="As afb-9. Once per day only."),
}

# Physiologically implausible pulse-to-pulse intervals are DROPPED, never
# repaired. 300 ms .. 2000 ms corresponds to 30 .. 200 bpm.
PPI_MIN_MS, PPI_MAX_MS = 300, 2000


class RelaxAdapter(DatasetAdapter):
    name = "relax"
    role = DatasetRole.LONGITUDINAL_ALTERNATIVE
    primary_sensor = "heart_rate_bpm"
    report_variable = "RELAX repeated 7-point Likert self-report"
    can_support_longitudinal_estimand = True
    acquisition_instructions = ACQUISITION

    def __init__(self, *, item: str = "ifb-2", lookback_hours: float = 2.0,
                 lag_hours: float = 0.0, min_ibi_samples: int = 30,
                 drop_blocked: bool = True):
        if item not in ITEM_SPECS:
            raise DecisionRequired(
                f"Unknown RELAX item {item!r}. Supported items, with their "
                f"verified anchors: "
                + "; ".join(f"{k} ({v.anchors_en[0]}->{v.anchors_en[1]})"
                            for k, v in ITEM_SPECS.items()))
        self.item = item
        self.lookback_hours = lookback_hours
        self.lag_hours = lag_hours
        self.min_ibi_samples = min_ibi_samples
        self.drop_blocked = drop_blocked

    @property
    def spec(self) -> RelaxItemSpec:
        return ITEM_SPECS[self.item]

    @classmethod
    def from_config(cls, cfg) -> "RelaxAdapter":
        return cls(
            item=str(cfg.get("relax.item", "ifb-2")),
            lookback_hours=float(cfg.get("alignment.lookback_hours", 2.0)),
            lag_hours=float(cfg.get("alignment.lag_hours", 0.0)),
            min_ibi_samples=int(cfg.get("relax.min_ibi_samples", 30)),
            drop_blocked=bool(cfg.get("relax.drop_blocked_ibi", True)))

    # ------------------------------------------------------------- locate
    def locate(self, root: Path) -> dict[str, list[Path]]:
        if root is None or not Path(root).exists():
            return {}
        r = str(root)
        return {
            "responses": [Path(p) for p in
                          glob.glob(os.path.join(r, "questionnaire_responses.xlsx"))
                          + glob.glob(os.path.join(r, "**",
                                                   "questionnaire_responses.xlsx"),
                                      recursive=True)],
            "definitions": [Path(p) for p in glob.glob(
                os.path.join(r, "**", "questionnaires.xlsx"), recursive=True)],
            "ibi": [Path(p) for p in sorted(glob.glob(
                os.path.join(r, "**", "ibi_data.parquet"), recursive=True))],
        }

    # ------------------------------------------------- label verification
    def _verify_anchors(self, defs_path: Path) -> dict:
        """Read the released item definition and CHECK it against ITEM_SPECS.

        A mismatch means the release differs from what this adapter was written
        against, so the direction of the scale can no longer be trusted.
        """
        sp = self.spec
        try:
            d = pd.read_excel(defs_path, sheet_name=sp.sheet)
        except ImportError as exc:
            raise DecisionRequired(
                f"Reading {defs_path} needs the optional 'openpyxl' extra: "
                f'pip install -e ".[relax]"  ({exc})')
        rows = d[d["question_id"].astype(str) == sp.question_id]
        if rows.empty:
            raise DecisionRequired(
                f"RELAX item {sp.question_id!r} is absent from sheet "
                f"{sp.sheet!r} of {defs_path}. Items present: "
                f"{sorted(d['question_id'].astype(str).unique())}")
        row = rows.iloc[0]
        atype = str(row.get("answer_type", "")).strip()
        if atype != f"likert-{sp.n_categories}":
            raise DecisionRequired(
                f"RELAX item {sp.question_id} has answer_type {atype!r}, not "
                f"'likert-{sp.n_categories}'. The ordinal model's category "
                "count would be wrong.")
        raw_labels = str(row.get("answer_labels_en", ""))
        try:
            import ast
            labels = tuple(str(x).strip().lower()
                           for x in ast.literal_eval(raw_labels))
        except Exception:
            raise DecisionRequired(
                f"Cannot parse answer_labels_en for {sp.question_id}: "
                f"{raw_labels!r}")
        if labels != tuple(a.lower() for a in sp.anchors_en):
            raise DecisionRequired(
                f"Dataset stress labels differ from expected mapping. RELAX "
                f"item {sp.question_id} is anchored {labels} in this release, "
                f"but this adapter was written against {sp.anchors_en}. The "
                "DIRECTION of the severity scale therefore cannot be trusted. "
                "Do NOT guess: confirm the anchors and update "
                "aedt.io.relax.ITEM_SPECS.")
        return {
            "item": sp.question_id, "sheet": sp.sheet,
            "question_text_en": str(row.get("answer_text_en", "")),
            "anchors_en": list(labels),
            "answer_type": atype,
            "severity_direction": ("ascending (larger stored value = more "
                                   "stress)" if sp.ascending else
                                   "DESCENDING (larger stored value = LESS "
                                   "stress; severity is reversed)"),
            "note": sp.note,
        }

    # ------------------------------------------------------------- reports
    def _load_reports(self, resp_path: Path) -> tuple[pd.DataFrame, dict]:
        sp = self.spec
        try:
            d = pd.read_excel(resp_path, sheet_name=sp.sheet)
        except ImportError as exc:
            raise DecisionRequired(
                f"Reading {resp_path} needs the optional 'openpyxl' extra: "
                f'pip install -e ".[relax]"  ({exc})')
        for col in ("user", "manual_date", sp.column):
            if col not in d.columns:
                raise DecisionRequired(
                    f"RELAX sheet {sp.sheet!r} lacks column {col!r}. "
                    f"Columns present: {list(d.columns)}")

        # ---- timestamps: manual_date is epoch MILLISECONDS, and the release
        # also carries readable_date. Cross-check them rather than trusting one.
        ts = pd.to_datetime(d["manual_date"], unit="ms", utc=True)
        prov: dict = {"timestamp_source": "manual_date (unix epoch ms) -> UTC"}
        if "readable_date" in d.columns:
            rd = pd.to_datetime(d["readable_date"], errors="coerce")
            delta = (ts.dt.tz_localize(None) - rd).abs().dropna()
            worst = float(delta.max().total_seconds()) if len(delta) else 0.0
            prov["timestamp_crosscheck_max_delta_s"] = worst
            if worst > 1.0:
                raise DecisionRequired(
                    f"RELAX timestamp fields disagree by up to {worst:.1f}s: "
                    "manual_date (epoch ms) and readable_date do not describe "
                    "the same instant. The timezone of readable_date is "
                    "therefore unknown and epoch ordering cannot be trusted.")

        raw = pd.to_numeric(d[sp.column], errors="coerce")
        out = pd.DataFrame({"pid": d["user"].astype(str),
                            "ts": ts, "raw_response": raw})
        n_before = len(out)
        out = out.dropna(subset=["raw_response"])
        prov["n_rows_in_sheet"] = n_before
        prov["n_missing_response_dropped"] = int(n_before - len(out))
        if out.empty:
            raise DecisionRequired(
                f"No parsable responses for RELAX item {sp.question_id}.")

        lo, hi = int(out["raw_response"].min()), int(out["raw_response"].max())
        if lo < 1 or hi > sp.n_categories:
            raise DecisionRequired(
                f"RELAX item {sp.question_id} has stored values in [{lo},{hi}], "
                f"outside the declared 1..{sp.n_categories} Likert range.")
        out["raw_response"] = out["raw_response"].round().astype(int)
        out["report"] = sp.severity(out["raw_response"]).astype(int)
        prov.update(observed_raw_range=(lo, hi),
                    severity_reversed=not sp.ascending)
        return out.sort_values(["pid", "ts"]).reset_index(drop=True), prov

    # ------------------------------------------------------------- sensor
    def _load_ibi(self, path: Path, pid: str) -> tuple[pd.DataFrame, dict]:
        d = pd.read_parquet(path)
        for col in ("ibi_ppi", "timestamp"):
            if col not in d.columns:
                raise DecisionRequired(
                    f"RELAX IBI file {path} lacks column {col!r}. "
                    f"Columns present: {list(d.columns)}")
        n0 = len(d)
        stats = {"n_raw": n0}
        if self.drop_blocked and "ibi_blocker" in d.columns:
            d = d[~d["ibi_blocker"].astype(bool)]
            stats["n_dropped_device_flagged"] = int(n0 - len(d))
        n1 = len(d)
        ppi = pd.to_numeric(d["ibi_ppi"], errors="coerce")
        d = d[(ppi >= PPI_MIN_MS) & (ppi <= PPI_MAX_MS)]
        stats["n_dropped_implausible_ppi"] = int(n1 - len(d))
        stats["n_kept"] = int(len(d))
        if d.empty:
            return pd.DataFrame(columns=["pid", "ts", "hr"]), stats
        ts = pd.to_datetime(d["timestamp"], utc=True)
        return (pd.DataFrame({"pid": pid, "ts": ts.to_numpy(),
                              "hr": 60000.0 / d["ibi_ppi"].to_numpy(float)}),
                stats)

    # --------------------------------------------------------------- load
    def load(self, root=None) -> LoadResult:
        from ..alignment.align import CausalWindow, align_sensor_to_reports

        p = self.require_files(root)
        found = self.locate(p)
        if not found.get("responses"):
            raise DecisionRequired(
                f"questionnaire_responses.xlsx not found under {p}.")
        if not found.get("definitions"):
            raise DecisionRequired(
                f"metadata/questionnaires.xlsx not found under {p}. The "
                "severity DIRECTION cannot be verified without it, and this "
                "adapter will not guess it.")
        if not found.get("ibi"):
            raise DecisionRequired(
                f"No data/<pid>/ibi_data.parquet files under {p}.")

        prov: dict = {"dataset_doi": "10.5281/zenodo.20701999",
                      "licence": "CC-BY-4.0",
                      "device": "Polar Verity Sense"}
        prov["item_definition"] = self._verify_anchors(found["definitions"][0])
        reports, rprov = self._load_reports(found["responses"][0])
        prov.update(rprov)

        frames, sensor_stats = [], {}
        for f in found["ibi"]:
            pid = f.parent.name
            g, st = self._load_ibi(f, pid)
            sensor_stats[pid] = st
            if len(g):
                frames.append(g)
        if not frames:
            raise DecisionRequired("No usable IBI samples in any RELAX file.")
        ibi = pd.concat(frames, ignore_index=True)
        prov["sensor_quality"] = sensor_stats
        prov["n_ibi_samples_kept"] = int(len(ibi))

        reports = reports[reports["pid"].isin(set(ibi["pid"]))]
        if reports.empty:
            raise DecisionRequired(
                "No participant has BOTH self-reports and IBI data. The "
                "participant identifiers in questionnaire_responses.xlsx do "
                "not match the data/<pid>/ directory names.")

        aligned = align_sensor_to_reports(
            reports, ibi, "hr",
            CausalWindow(lookback_hours=self.lookback_hours,
                         lag_hours=self.lag_hours, aggregation="mean",
                         min_samples=self.min_ibi_samples),
            out_col=self.primary_sensor)
        n_before = len(aligned)
        df = aligned.dropna(subset=[self.primary_sensor]).copy()
        prov["n_reports_before_alignment"] = int(n_before)
        prov["n_reports_with_sensor_coverage"] = int(len(df))
        prov["unmatched_report_fraction"] = float(1.0 - len(df) / max(n_before, 1))
        if df.empty:
            raise DecisionRequired(
                "No RELAX report has at least "
                f"{self.min_ibi_samples} valid IBI samples in its "
                f"{self.lookback_hours}h causal window. Reports and physiology "
                "do not overlap in time.")
        df["day"] = df["ts"].dt.tz_convert("UTC").dt.normalize()
        df.attrs["data_status"] = DataStatus.REAL.value
        df.attrs["n_categories"] = self.spec.n_categories
        return LoadResult(frame=df, n_categories=self.spec.n_categories,
                          sensor=self.primary_sensor,
                          data_status=DataStatus.REAL, provenance=prov)

    # -------------------------------------------------------------- audit
    def audit(self, root=None) -> DatasetAudit:
        p = self.resolve_root(root)
        if p is None:
            return self.unavailable_audit(
                root, "REAL DATA UNAVAILABLE - RELAX AUDIT NOT RUN: "
                      f"no directory at {root!r}")
        found = self.locate(p)
        missing = [k for k in ("responses", "definitions", "ibi")
                   if not found.get(k)]
        if missing:
            return self.unavailable_audit(
                root, "REAL DATA UNAVAILABLE - RELAX AUDIT NOT RUN: missing "
                      f"{missing} under {p}")

        res = self.load(root)
        df, sp, prov = res.frame, self.spec, res.provenance
        per = df.groupby("pid").size()
        span = (df["ts"].max() - df["ts"].min()).total_seconds() / 86400.0
        codes = sorted(int(v) for v in pd.unique(df["raw_response"]))
        enough = int((per >= 120).sum())
        sq = prov.get("sensor_quality", {})
        kept = sum(v.get("n_kept", 0) for v in sq.values())
        rawn = sum(v.get("n_raw", 0) for v in sq.values())

        return DatasetAudit(
            dataset_name=self.name, role=self.role, data_status=DataStatus.REAL,
            source_status=(f"local files audited at {p}; Zenodo "
                           "10.5281/zenodo.20701999, CC-BY-4.0"),
            local_files_available=True, root_path=str(p),
            files_found=tuple(f"{k}: {len(v)}" for k, v in found.items()),
            participant_count=int(df["pid"].nunique()),
            observation_count=int(len(df)),
            sensor_modalities=("Polar Verity Sense interbeat intervals (IBI) "
                               "-> heart rate",),
            self_report_variables=(f"{sp.question_id}: {sp.text_en}",),
            self_report_scale=(f"{sp.n_categories}-point Likert anchored "
                               f"{sp.anchors_en[0]!r} .. {sp.anchors_en[1]!r}; "
                               + ("severity ascends with the stored value"
                                  if sp.ascending else
                                  "severity is REVERSED relative to the "
                                  "stored value")),
            stress_labels=tuple(sp.anchors_en),
            raw_stored_codes=tuple(codes),
            code_to_label_mapping={c: (f"{c} = toward {sp.anchors_en[0]!r}"
                                       if c <= sp.n_categories // 2
                                       else f"{c} = toward {sp.anchors_en[1]!r}")
                                   for c in codes},
            code_to_severity_mapping={c: int(sp.severity(pd.Series([c])).iloc[0])
                                      for c in codes},
            timestamps_present=True,
            timestamp_format=prov.get("timestamp_source"),
            timezone="UTC (tz-aware; cross-checked against readable_date)",
            longitudinal_span_days=float(span),
            observations_per_participant={str(k): int(v) for k, v in per.items()},
            median_observations_per_participant=float(per.median()),
            missingness={
                "reports_without_sensor_coverage":
                    prov.get("unmatched_report_fraction", float("nan")),
                "ibi_samples_discarded":
                    float(1.0 - kept / rawn) if rawn else float("nan")},
            participant_level_coverage={
                str(k): float(v / max(per.max(), 1)) for k, v in per.items()},
            sensor_report_alignment=(
                f"mean heart rate over a strictly causal {self.lookback_hours}h "
                f"window ending at each report (lag {self.lag_hours}h, "
                f"minimum {self.min_ibi_samples} valid IBI samples)"),
            conversation_context_available=False,
            eligible_for_primary_analysis=bool(enough >= 10),
            eligible_for_benchmark_analysis=True,
            exclusion_reasons=() if enough >= 10 else (
                f"Only {enough} of {len(per)} participants reach the 120 "
                "aligned reports the frozen screen needs (60 per epoch); the "
                f"densest participant has {int(per.max())}. RELAX cannot "
                "support the frozen PRIMARY endpoint.",),
            acquisition_instructions=self.acquisition_instructions,
            notes=(
                f"Item {sp.question_id} selected: {sp.note}",
                "Severity direction was VERIFIED against the released anchor "
                "text, not inferred from the stored integer.",
                "IBI samples flagged by the device (ibi_blocker) and "
                f"physiologically implausible intervals (<{PPI_MIN_MS} or "
                f">{PPI_MAX_MS} ms) are DROPPED, never repaired or imputed.",
                "Accelerometer data (~15.9 GB of the archive) is not used by "
                "this analysis and is not downloaded."))


register_adapter(RelaxAdapter())
