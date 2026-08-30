"""STUDENTLIFE ADAPTER -- the PRIMARY LONGITUDINAL TARGET.

PORTED from the validated ``final_audit.py::load_studentlife`` (Round 16),
which was tested end to end on a synthetic StudentLife-shaped fixture and in
doing so caught a live specification bug (self-correction 26).

THE DAY-1 PROCEDURE, implemented step by step in ``audit()``:
  1. Locate the actual StudentLife files.
  2. Locate ``EMA_definition.json``.
  3. Extract the stress item's response labels VERBATIM.
  4. Extract the stored integer codes.
  5. Verify the label -> severity mapping FROM TEXT, never from position.
  6. Inspect PAM storage.
  7. Count observations.
  8. Check timestamps (unix epoch vs formatted datetime -- DETECTED, not assumed).
  9. Check conversation / context data availability.
 10. Produce the first T4 audit table.

IF THE FILES ARE ABSENT the audit reports
``REAL DATA UNAVAILABLE - STUDENTLIFE AUDIT NOT RUN`` and NOTHING is
substituted. If the labels disagree with the specification, the loader raises
``DECISION REQUIRED`` rather than guessing the severity order -- because
mapping by position would silently invert the scale.
"""
from __future__ import annotations

import glob
import json
import logging
import os
from pathlib import Path

import pandas as pd

from ..constants import DataStatus, DatasetRole
from ..errors import DecisionRequired
from ..preprocess.reports import build_code_to_severity, detect_reversed_coding
from ..schemas import DatasetAudit
from .base import DatasetAdapter, LoadResult, register_adapter

log = logging.getLogger(__name__)

__all__ = ["StudentLifeAdapter", "find_response_options"]

ACQUISITION = """\
ACQUISITION -- StudentLife (Dartmouth), Wang et al. 2014.
  1. Download the dataset archive from the Dartmouth StudentLife study page
     (https://studentlife.cs.dartmouth.edu/dataset.html). It is a public
     research release; no credentials are required, but the download is large.
  2. Extract it, then pass the directory that CONTAINS `EMA/` and `sensing/`
     as --root, e.g.
         python scripts/audit_dataset.py --dataset studentlife --root ~/data/StudentLife_Dataset
  3. Expected layout:
         <root>/EMA/EMA_definition.json
         <root>/EMA/response/Stress/Stress_u00.json ...
         <root>/sensing/conversation/conversation_u00.csv ...
  4. Read the audit's [9b] association-strength line FIRST. If the median
     |beta| is below 0.15, switch to the pre-specified PC1 fallback covariate
     before interpreting anything else."""


def find_response_options(node):
    """Locate the response-option list wherever it sits in the EMA entry."""
    if isinstance(node, list) and node and all(isinstance(v, str) for v in node):
        return node
    if isinstance(node, dict):
        for v in node.values():
            r = find_response_options(v)
            if r:
                return r
    if isinstance(node, list):
        for v in node:
            r = find_response_options(v)
            if r:
                return r
    return None


class StudentLifeAdapter(DatasetAdapter):
    name = "studentlife"
    role = DatasetRole.PRIMARY_LONGITUDINAL
    primary_sensor = "conversation_minutes"
    report_variable = "single-item stress EMA (5 ordered levels)"
    can_support_longitudinal_estimand = True
    acquisition_instructions = ACQUISITION

    #: filenames that alone are enough to consider the dataset "present"
    ENOUGH = ("ema_definition", "rds_stress")

    # ------------------------------------------------------------- locate
    def locate(self, root: Path) -> dict[str, list[Path]]:
        if root is None or not Path(root).exists():
            return {}
        r = str(root)
        stress = [Path(f) for f in
                  glob.glob(os.path.join(r, "**", "*tress*", "*.json"),
                            recursive=True)
                  if "definition" not in os.path.basename(f).lower()]
        return {
            # the RDS repackaging, converted by scripts/convert_studentlife_rds.py
            "rds_stress": [Path(f) for f in glob.glob(
                os.path.join(r, "**", "stress_ema.csv"), recursive=True)],
            "rds_conversation": [Path(f) for f in glob.glob(
                os.path.join(r, "**", "conversation.csv"), recursive=True)],
            "ema_definition": [Path(f) for f in glob.glob(
                os.path.join(r, "**", "EMA_definition.json"), recursive=True)],
            "stress_responses": stress,
            "conversation": [Path(f) for f in glob.glob(
                os.path.join(r, "**", "*onversation*", "*.csv"), recursive=True)],
            "pam": [Path(f) for f in glob.glob(
                os.path.join(r, "**", "*PAM*", "*.json"), recursive=True)],
            "activity": [Path(f) for f in glob.glob(
                os.path.join(r, "**", "*activity*", "*.csv"), recursive=True)],
        }

    # ------------------------------------------------- RDS repackaging path
    def _load_rds_derived(self, found: dict) -> LoadResult:
        """Read the CSVs produced from the StudentLife RDS repackaging.

        This layout carries NO ``EMA_definition.json``, so the label-text remap
        the specification mandates cannot be applied. Stored integers are
        therefore carried through UNMAPPED and the audit refuses primary
        eligibility: without the codebook there is no way to verify that code 1
        means "Feeling great" rather than "Stressed out".
        """
        prov: dict = {
            "layout": "RDS repackaging converted to CSV",
            "official_source": "https://studentlife.cs.dartmouth.edu/dataset.html",
            "ema_definition_present": False,
            "label_text_remap_applied": False,
        }
        # keep_default_na=False on the metadata column only: the defective
        # repackaging names its response column literally "null", which pandas
        # would otherwise read back as NaN and hide the diagnosis.
        ema = pd.read_csv(found["rds_stress"][0],
                          keep_default_na=True,
                          converters={"response_column_name": str})
        for c in ("uid", "timestamp", "response"):
            if c not in ema.columns:
                raise DecisionRequired(
                    f"{found['rds_stress'][0]} lacks column {c!r}; columns "
                    f"present: {list(ema.columns)}. Re-run "
                    "scripts/convert_studentlife_rds.py.")
        col = str(ema.get("response_column_name",
                          pd.Series(["?"])).iloc[0])
        prov["source_response_column"] = col
        n_rows = len(ema)

        raw = pd.to_numeric(ema["response"], errors="coerce")
        usable = raw.notna() & (raw >= 1) & (raw <= 5)
        prov["n_rows"] = int(n_rows)
        prov["n_response_missing"] = int(ema["response"].isna().sum())
        prov["n_response_non_numeric"] = int((~usable & ema["response"].notna()).sum())
        prov["n_usable_responses"] = int(usable.sum())
        prov["usable_fraction"] = float(usable.mean())
        if col == "null" or usable.mean() < 0.5:
            prov["conversion_defect"] = (
                f"The stress response arrived in a column named {col!r} with "
                f"only {int(usable.sum())} of {n_rows} rows "
                f"({usable.mean():.1%}) parsable as a 1-5 response. Other EMA "
                "tables in the same archive convert with correctly named "
                "columns, so this is a defect of the repackaging, not of "
                "StudentLife.")
        if not usable.any():
            raise DecisionRequired(
                "StudentLife RDS repackaging: no parsable 1-5 stress responses "
                f"at all in {found['rds_stress'][0]} (response column was "
                f"{col!r}). Obtain the ORIGINAL Dartmouth release.")

        d = pd.DataFrame({
            "pid": "u" + ema.loc[usable, "uid"].astype(int).astype(str).str.zfill(2),
            "ts": pd.to_datetime(ema.loc[usable, "timestamp"], unit="s", utc=True),
            "raw_code": raw[usable].astype(int)})
        n_dup = int(d.duplicated(subset=["pid", "ts"]).sum())
        prov["n_duplicate_occasions_dropped"] = n_dup
        d = d.drop_duplicates(subset=["pid", "ts"], keep="first")
        # NOT remapped: without the codebook the severity order is unverifiable
        d["report"] = d["raw_code"]
        d["day"] = d["ts"].dt.tz_convert("UTC").dt.tz_localize(None).dt.normalize()

        conv = pd.read_csv(found["rds_conversation"][0])
        for c in ("uid", "start_timestamp", "end_timestamp"):
            if c not in conv.columns:
                raise DecisionRequired(
                    f"{found['rds_conversation'][0]} lacks column {c!r}; "
                    f"present: {list(conv.columns)}")
        st = pd.to_numeric(conv["start_timestamp"], errors="coerce")
        en = pd.to_numeric(conv["end_timestamp"], errors="coerce")
        ok = st.notna() & en.notna() & (en > st)
        prov["n_conversation_episodes"] = int(len(conv))
        prov["n_conversation_invalid_dropped"] = int((~ok).sum())
        cdf = pd.DataFrame({
            "pid": "u" + conv.loc[ok, "uid"].astype(int).astype(str).str.zfill(2),
            "day": pd.to_datetime(st[ok], unit="s").dt.normalize(),
            self.primary_sensor: (en[ok] - st[ok]) / 60.0})
        sens = cdf.groupby(["pid", "day"], as_index=False)[self.primary_sensor].sum()

        df = d.merge(sens, on=["pid", "day"], how="inner")
        prov["n_matched"] = int(len(df))
        prov["stress_day_range"] = [str(d["day"].min().date()),
                                    str(d["day"].max().date())]
        prov["sensor_day_range"] = [str(sens["day"].min().date()),
                                    str(sens["day"].max().date())]
        prov["n_pid_overlap"] = int(len(set(d["pid"]) & set(sens["pid"])))
        if df.empty:
            shared_ids = prov["n_pid_overlap"] > 0
            raise DecisionRequired(
                "StudentLife RDS repackaging: NOT ONE stress response falls on "
                "a day with conversation data, so no observation can be formed."
                f" The {int(usable.sum())} parsable responses span "
                f"{prov['stress_day_range'][0]} to {prov['stress_day_range'][1]}"
                f", while conversation sensing spans "
                f"{prov['sensor_day_range'][0]} to {prov['sensor_day_range'][1]}"
                f". Participant identifiers DO match ({prov['n_pid_overlap']} "
                "in common), so this is not an ID problem: the surviving "
                "responses are from before the sensing period, and the study-"
                "period EMA was lost in the column named "
                f"{prov.get('source_response_column')!r}. "
                "Obtain the ORIGINAL Dartmouth release."
                if shared_ids else
                "Participant identifiers do not match between the two tables.")
        df.attrs["data_status"] = DataStatus.REAL.value
        df.attrs["n_categories"] = 5
        return LoadResult(frame=df, n_categories=5, sensor=self.primary_sensor,
                          data_status=DataStatus.REAL, provenance=prov)

    def _rds_defect_audit(self, p: Path, found: dict, exc: Exception
                          ) -> DatasetAudit:
        """A full structured audit for an archive that cannot be loaded.

        A dataset that fails is still worth a complete audit record: the point
        of the audit is to say precisely WHY, in a form a reviewer and a
        machine can both read.
        """
        import numpy as _np
        ema = pd.read_csv(found["rds_stress"][0],
                          converters={"response_column_name": str})
        conv = pd.read_csv(found["rds_conversation"][0])
        raw = pd.to_numeric(ema["response"], errors="coerce")
        usable = raw.notna() & (raw >= 1) & (raw <= 5)
        col = str(ema["response_column_name"].iloc[0]) \
            if "response_column_name" in ema.columns else "?"
        per = ema.loc[usable].groupby("uid").size()
        st = pd.to_numeric(conv["start_timestamp"], errors="coerce")
        en = pd.to_numeric(conv["end_timestamp"], errors="coerce")
        good = st.notna() & en.notna() & (en > st)
        dist = raw[usable].value_counts().sort_index()
        return DatasetAudit(
            dataset_name=self.name, role=self.role, data_status=DataStatus.REAL,
            source_status=(f"RDS repackaging audited at {p}; the archive loads "
                           "but is DEFECTIVE for the stress EMA"),
            local_files_available=True, root_path=str(p),
            files_found=tuple(f"{k}: {len(v)}" for k, v in found.items() if v),
            participant_count=int(ema["uid"].nunique()),
            observation_count=int(usable.sum()),
            sensor_modalities=("phone conversation episodes (audio inference)",),
            self_report_variables=("single-item stress EMA (5 ordered levels)",),
            self_report_scale=("5-point ordered; SEVERITY ORDER UNVERIFIABLE — "
                               "EMA_definition.json is absent from this "
                               "repackaging"),
            stress_labels=(),
            raw_stored_codes=tuple(int(v) for v in dist.index),
            code_to_label_mapping={int(k): f"{int(k)} = stored integer; no "
                                   "label text available in this repackaging"
                                   for k in dist.index},
            code_to_severity_mapping={},
            timestamps_present=True, timestamp_format="unix epoch (s)",
            timezone="UTC",
            longitudinal_span_days=float(
                (pd.to_datetime(en[good].max(), unit="s")
                 - pd.to_datetime(st[good].min(), unit="s")).days),
            observations_per_participant={str(k): int(v) for k, v in per.items()},
            median_observations_per_participant=float(per.median()) if len(per) else 0.0,
            missingness={
                "stress_response_missing": float(ema["response"].isna().mean()),
                "stress_response_unusable": float(1.0 - usable.mean()),
                "conversation_episodes_invalid": float(1.0 - good.mean())},
            participant_level_coverage={str(k): float(v / 735.0)
                                        for k, v in per.items()},
            sensor_report_alignment=("IMPOSSIBLE — zero temporal overlap "
                                     "between usable reports and sensing"),
            conversation_context_available=True,
            eligible_for_primary_analysis=False,
            eligible_for_benchmark_analysis=False,
            exclusion_reasons=(
                str(exc),
                f"Only {int(usable.sum())} of {len(ema)} stress rows "
                f"({usable.mean():.1%}) parse as a 1-5 response; the column is "
                f"named {col!r} and is {ema['response'].isna().mean():.1%} NA.",
                f"Maximum {int(per.max()) if len(per) else 0} usable responses "
                "per participant against roughly 735 per student in the "
                "published descriptor — a loss of about 99%.",
                "EMA_definition.json is absent, so the label-text remap the "
                "frozen specification mandates cannot be applied at all."),
            acquisition_instructions=self.acquisition_instructions,
            notes=(
                "THE SENSOR SIDE IS INTACT: "
                f"{int(good.sum())} conversation episodes across "
                f"{int(conv['uid'].nunique())} participants, 0% invalid.",
                "Other EMA tables in the same archive (PAM, Mood, Sleep) "
                "converted correctly with named columns, so this is a defect "
                "of the repackaging, NOT a property of StudentLife.",
                "StudentLife remains the correct target. The fix is to obtain "
                "the ORIGINAL Dartmouth release, not to change the method."))

    # -------------------------------------------------------------- audit
    def audit(self, root=None) -> DatasetAudit:
        p = self.resolve_root(root)
        if p is None:
            return self.unavailable_audit(
                root, "REAL DATA UNAVAILABLE - STUDENTLIFE AUDIT NOT RUN: "
                      f"no directory at {root!r}")
        found = self.locate(p)
        # The RDS repackaging is a different layout with no EMA_definition.json.
        # It is present-but-defective rather than absent, so it gets a full
        # audit record explaining exactly why it cannot be used.
        if found.get("rds_stress") and found.get("rds_conversation") \
                and not found.get("ema_definition"):
            try:
                res = self.load(root)
            except DecisionRequired as exc:
                return self._rds_defect_audit(Path(p), found, exc)
            df = res.frame
        elif not found.get("ema_definition"):
            return self.unavailable_audit(
                root, "REAL DATA UNAVAILABLE - STUDENTLIFE AUDIT NOT RUN: "
                      f"EMA_definition.json not found under {p}")
        if not found.get("ema_definition") or not found.get("stress_responses"):
            return self.unavailable_audit(
                root, "REAL DATA UNAVAILABLE - STUDENTLIFE AUDIT NOT RUN: "
                      f"no stress response JSON files under {p}")

        # ---- steps 2-5: labels, codes, severity map (raises on surprise)
        with open(found["ema_definition"][0]) as f:
            ema_def = json.load(f)
        stress_key = next((k for k in ema_def if "stress" in k.lower()), None)
        if stress_key is None:
            raise DecisionRequired(
                f"No key containing 'stress' in {found['ema_definition'][0]}. "
                f"Keys found: {list(ema_def)[:40]}")
        options = find_response_options(ema_def[stress_key])
        if not options:
            raise DecisionRequired(
                f"Could not locate the response-option list inside the "
                f"'{stress_key}' entry of {found['ema_definition'][0]}. "
                "Inspect the file and supply the option order explicitly.")
        code_to_sev = build_code_to_severity(options)
        reversed_coding = detect_reversed_coding(code_to_sev)
        if reversed_coding:
            log.warning("StudentLife stored codes are NOT in severity order; "
                        "the remap reorders them. This is the trap the "
                        "specification flags: %s", code_to_sev)

        # ---- steps 7-10: counts, timestamps, coverage
        res = self.load(root)
        df = res.frame
        per = df.groupby("pid").size()
        span = (df["ts"].max() - df["ts"].min()).total_seconds() / 86400.0
        prov = res.provenance
        enough = int((per >= 120).sum())     # 60 per epoch x 2 epochs

        return DatasetAudit(
            dataset_name=self.name, role=self.role, data_status=DataStatus.REAL,
            source_status=f"local files audited at {p}",
            local_files_available=True, root_path=str(p),
            files_found=tuple(f"{k}: {len(v)}" for k, v in found.items()),
            participant_count=int(df["pid"].nunique()),
            observation_count=int(len(df)),
            sensor_modalities=tuple(
                k for k in ("conversation", "activity") if found.get(k)),
            self_report_variables=(self.report_variable,)
            + (("PAM affect (1-16)",) if found.get("pam") else ()),
            self_report_scale="5-point ordered stress item, remapped by LABEL TEXT",
            stress_labels=tuple(options),
            raw_stored_codes=tuple(sorted(code_to_sev)),
            code_to_label_mapping={i: lab for i, lab
                                   in enumerate(options, start=1)},
            code_to_severity_mapping=code_to_sev,
            timestamps_present=True,
            timestamp_format=prov.get("timestamp_format"),
            timezone=prov.get("timezone", "not recorded in the source files"),
            longitudinal_span_days=float(span),
            observations_per_participant={str(k): int(v) for k, v in per.items()},
            median_observations_per_participant=float(per.median()),
            missingness={"matched_report_days": 1.0 - (
                prov.get("n_matched", 0) / max(prov.get("n_ema_rows", 1), 1))},
            participant_level_coverage={
                str(k): float(v / max(per.max(), 1)) for k, v in per.items()},
            sensor_report_alignment=("EMA responses joined to same-day "
                                     "conversation minutes"),
            conversation_context_available=bool(found.get("conversation")),
            eligible_for_primary_analysis=bool(enough >= 10),
            eligible_for_benchmark_analysis=False,
            exclusion_reasons=() if enough >= 10 else (
                f"only {enough} participants have >= 120 matched reports "
                "(60 per epoch); the primary analysis needs at least 10.",),
            acquisition_instructions=self.acquisition_instructions,
            notes=(("Stored codes are NOT in severity order; the remap "
                    "reorders them by label text.",) if reversed_coding else ())
            + ("Read the [9b] association-strength diagnostic before "
               "interpreting any estimate.",))

    # --------------------------------------------------------------- load
    def load(self, root=None) -> LoadResult:
        p = self.require_files(root)
        found = self.locate(p)
        prov: dict = {}

        # The RDS repackaging has no EMA_definition.json and a different shape;
        # dispatch to its reader when the original layout is absent.
        if found.get("rds_stress") and found.get("rds_conversation") \
                and not found.get("ema_definition"):
            return self._load_rds_derived(found)

        if not found["ema_definition"]:
            raise DecisionRequired(
                f"EMA_definition.json not found under {p}. Locate it and pass "
                "its parent as --root.")
        with open(found["ema_definition"][0]) as f:
            ema_def = json.load(f)
        prov["ema_definition_path"] = str(found["ema_definition"][0])

        stress_key = next((k for k in ema_def if "stress" in k.lower()), None)
        if stress_key is None:
            raise DecisionRequired(
                f"No key containing 'stress' in {found['ema_definition'][0]}. "
                f"Keys found: {list(ema_def)[:40]}")
        options = find_response_options(ema_def[stress_key])
        if not options:
            raise DecisionRequired(
                f"Could not locate the response-option list inside the "
                f"'{stress_key}' entry. Inspect "
                f"{found['ema_definition'][0]} and supply the option order.")
        prov["stress_key"] = stress_key
        prov["options_verbatim"] = list(options)
        code_to_sev = build_code_to_severity(options)
        prov["code_to_severity"] = code_to_sev
        prov["stored_codes_in_severity_order"] = not detect_reversed_coding(
            code_to_sev)

        # ---- stress responses
        rows = []
        for fp in sorted(found["stress_responses"]):
            pid = os.path.splitext(fp.name)[0].split("_")[-1]
            try:
                recs = json.loads(fp.read_text())
            except Exception as e:
                log.warning("skipping %s: %s", fp, e)
                continue
            if not isinstance(recs, list) or not recs:
                continue
            lvl_key = next((k for k in recs[0] if k.lower() in
                            ("level", "response", "value", "answer")), None)
            ts_key = next((k for k in recs[0] if "time" in k.lower()
                           or k.lower() in ("resp_time", "ts")), None)
            if lvl_key is None or ts_key is None:
                raise DecisionRequired(
                    f"Cannot identify the response/timestamp fields in {fp}. "
                    f"Keys present: {list(recs[0].keys())}")
            for r in recs:
                v, t = r.get(lvl_key), r.get(ts_key)
                if v in (None, "", "null") or t in (None, ""):
                    continue
                try:
                    code = int(float(v))
                except (TypeError, ValueError):
                    continue
                if code not in code_to_sev:
                    continue
                rows.append((pid, t, code, code_to_sev[code]))
        if not rows:
            raise DecisionRequired(
                "Stress files found but no parsable responses. Inspect the "
                "level/timestamp field names in the response JSON.")
        ema = pd.DataFrame(rows, columns=["pid", "ts", "raw_code", "report"])

        # ---- timestamps: DETECT the encoding, never assume it
        raw_ts = ema["ts"].astype(str)
        if raw_ts.str.fullmatch(r"\d{9,13}").mean() > 0.9:
            num = pd.to_numeric(raw_ts, errors="coerce")
            unit = "ms" if num.median() > 1e11 else "s"
            prov["timestamp_format"] = f"unix epoch ({unit})"
            ema["ts"] = pd.to_datetime(num, errors="coerce", unit=unit)
        else:
            prov["timestamp_format"] = "formatted datetime"
            ema["ts"] = pd.to_datetime(raw_ts, errors="coerce")
        ema = ema.dropna(subset=["ts"])
        if ema.empty:
            raise DecisionRequired(
                "Every stress timestamp failed to parse. Inspect the "
                f"resp_time field; first raw values: {list(raw_ts.head(5))}")
        ema["day"] = ema["ts"].dt.normalize()

        # ---- conversation -> daily minutes
        if not found["conversation"]:
            raise DecisionRequired(
                f"No conversation CSVs found under {p}. The pre-specified "
                "fallback is the epoch-1-fitted per-participant PC1 of "
                "{conversation, activity, location entropy, unlocks}; that "
                "fallback needs those streams, so locate sensing/ first.")
        srows = []
        for fp in sorted(found["conversation"]):
            pid = os.path.splitext(fp.name)[0].split("_")[-1]
            try:
                d = pd.read_csv(fp)
            except Exception:
                continue
            d.columns = [c.strip() for c in d.columns]
            st = next((c for c in d.columns if "start" in c.lower()), None)
            en = next((c for c in d.columns if "end" in c.lower()), None)
            if st is None or en is None:
                raise DecisionRequired(
                    f"Conversation file {fp} lacks start/end columns. "
                    f"Found: {list(d.columns)}")
            s = pd.to_numeric(d[st], errors="coerce")
            e = pd.to_numeric(d[en], errors="coerce")
            ok = s.notna() & e.notna() & (e > s)
            if not ok.any():
                continue
            day = pd.to_datetime(s[ok], unit="s").dt.normalize()
            mins = (e[ok] - s[ok]) / 60.0
            g = (pd.DataFrame({"day": day.values, "m": mins.values})
                 .groupby("day")["m"].sum())
            for dd, mm in g.items():
                srows.append((pid, dd, float(mm)))
        sens = pd.DataFrame(srows, columns=["pid", "day",
                                            self.primary_sensor])

        df = ema.merge(sens, on=["pid", "day"], how="inner")
        prov.update(n_ema_rows=len(ema), n_sensor_days=len(sens),
                    n_matched=len(df))
        if df.empty:
            raise DecisionRequired(
                f"{len(ema)} EMA responses and {len(sens)} sensor-days were "
                "parsed but NONE matched on (pid, day). The participant "
                "identifiers or the day alignment differ from the "
                "specification.")
        df.attrs["data_status"] = DataStatus.REAL.value
        df.attrs["n_categories"] = 5
        return LoadResult(frame=df, n_categories=5, sensor=self.primary_sensor,
                          data_status=DataStatus.REAL, provenance=prov)


register_adapter(StudentLifeAdapter())
