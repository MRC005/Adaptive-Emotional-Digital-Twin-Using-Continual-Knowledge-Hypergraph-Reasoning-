"""TABLE GENERATION. Every table carries a visible data-status column.

The tables the report and the deck need:
  dataset audit - participant characteristics - eligibility - primary estimator
  - placebo - uncertainty - sensitivity - baseline comparison - hypergraph
  ablation - status board - title alignment - standard vs contribution.

``write_table`` writes CSV and Markdown side by side and REFUSES to write a
table that carries no ``data_status`` column.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..constants import DataStatus
from ..schemas import (BiasEnvelopeResult, DatasetAudit, EligibilityResult,
                       EstimatorResult, PlaceboResult, UncertaintyResult)

def _to_markdown(df: pd.DataFrame, *, max_cell: int = 320) -> str:
    """Render a Markdown table WITHOUT pulling in ``tabulate``.

    The frozen software list (ROUND-17 §S.13) is numpy, scipy, pandas,
    matplotlib, pyyaml and pytest. ``DataFrame.to_markdown`` needs ``tabulate``,
    which is not on that list, so the 30 lines below buy us one fewer
    dependency in the environment specification.
    """
    def cell(v) -> str:
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return ""
        if isinstance(v, float):
            t = f"{v:.6g}"
        else:
            t = str(v)
        t = t.replace("|", r"\|").replace("\n", " ")
        return t if len(t) <= max_cell else t[:max_cell - 3] + "..."

    cols = [str(c) for c in df.columns]
    rows = [[cell(v) for v in rec] for rec in df.itertuples(index=False)]
    widths = [max(len(c), *(len(r[i]) for r in rows)) if rows else len(c)
              for i, c in enumerate(cols)]
    out = ["| " + " | ".join(c.ljust(w) for c, w in zip(cols, widths)) + " |",
           "| " + " | ".join("-" * w for w in widths) + " |"]
    out += ["| " + " | ".join(v.ljust(w) for v, w in zip(r, widths)) + " |"
            for r in rows]
    return "\n".join(out)


__all__ = ["write_table", "dataset_audit_table", "participant_table",
           "eligibility_summary", "estimator_table", "placebo_table",
           "uncertainty_table", "sensitivity_table", "status_board",
           "title_alignment_table", "contribution_table"]


def write_table(df: pd.DataFrame, path: str | Path, *,
                status: DataStatus | None = None, title: str = "") -> Path:
    """Write CSV + Markdown. Refuses an unstamped table."""
    if "data_status" not in df.columns:
        if status is None:
            raise ValueError(
                "Refusing to write a table without a data_status column. "
                "Every table must state REAL / SYNTHETIC / PLANNED.")
        df = df.copy()
        df["data_status"] = status.value
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p.with_suffix(".csv"), index=False)
    stamps = sorted(set(str(v) for v in df["data_status"]))
    header = (f"# {title}\n\n" if title else "")
    header += f"**DATA STATUS: {' / '.join(stamps)}**\n\n"
    p.with_suffix(".md").write_text(header + _to_markdown(df) + "\n")
    return p.with_suffix(".csv")


# ------------------------------------------------------------------ audits
def dataset_audit_table(audits: list[DatasetAudit]) -> pd.DataFrame:
    """The T4 audit table -- one row per dataset, every mandated field."""
    rows = []
    for a in audits:
        rows.append({
            "dataset": a.dataset_name,
            "role": a.role.value,
            "data_status": a.data_status.value,
            "source_status": a.source_status,
            "local_files_available": a.local_files_available,
            "participants": a.participant_count,
            "observations": a.observation_count,
            "sensor_modalities": "; ".join(a.sensor_modalities) or None,
            "self_report_variables": "; ".join(a.self_report_variables) or None,
            "self_report_scale": a.self_report_scale,
            "stress_labels": "; ".join(a.stress_labels) or None,
            "raw_stored_codes": ", ".join(map(str, a.raw_stored_codes)) or None,
            "code_to_severity": (str(dict(a.code_to_severity_mapping))
                                 if a.code_to_severity_mapping else None),
            "timestamps_present": a.timestamps_present,
            "timestamp_format": a.timestamp_format,
            "timezone": a.timezone,
            "longitudinal_span_days": a.longitudinal_span_days,
            "median_obs_per_participant": a.median_observations_per_participant,
            "missingness": (str(dict(a.missingness)) if a.missingness else None),
            "sensor_report_alignment": a.sensor_report_alignment,
            "conversation_context_available": a.conversation_context_available,
            "eligible_for_primary": a.eligible_for_primary_analysis,
            "eligible_for_benchmark": a.eligible_for_benchmark_analysis,
            "exclusion_reasons": " | ".join(a.exclusion_reasons) or None,
        })
    return pd.DataFrame(rows)


def participant_table(df: pd.DataFrame, sensor: str, K: int,
                      status: DataStatus) -> pd.DataFrame:
    """Participant characteristics: coverage, span, category use, boundary."""
    rows = []
    for pid, g in df.groupby("pid", sort=True):
        R = g["report"].to_numpy(int)
        rows.append({
            "pid": str(pid),
            "n_observations": len(g),
            "n_epoch0": int((g["epoch"] == 0).sum()) if "epoch" in g else None,
            "n_epoch1": int((g["epoch"] == 1).sum()) if "epoch" in g else None,
            "span_days": float((g["ts"].max() - g["ts"].min()).total_seconds()
                               / 86400.0),
            "categories_used": int(len(np.unique(R))),
            "floor_rate": float(np.mean(R == 1)),
            "ceiling_rate": float(np.mean(R == K)),
            "sensor_mean": float(g[sensor].mean()),
            "sensor_sd": float(g[sensor].std(ddof=1)) if len(g) > 1 else np.nan,
            "data_status": status.value,
        })
    return pd.DataFrame(rows)


def eligibility_summary(results: list[EligibilityResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append({
            "pid": r.pid, "eligible": r.eligible,
            "n_epoch0": r.n_epoch0, "n_epoch1": r.n_epoch1,
            "categories_epoch0": r.categories_epoch0,
            "categories_epoch1": r.categories_epoch1,
            "var_ratio": r.var_ratio,
            "floor_epoch0": r.floor_rate_epoch0,
            "floor_epoch1": r.floor_rate_epoch1,
            "ar1_epoch0": r.ar1_epoch0,
            "exclusion_reasons": "; ".join(r.reasons),
            "data_status": r.data_status.value,
        })
    return pd.DataFrame(rows)


# -------------------------------------------------------------- estimates
def estimator_table(result: EstimatorResult) -> pd.DataFrame:
    u = result.uncertainty
    return pd.DataFrame([{
        "estimand": "rho_star (IDENTIFIED)",
        "context_representation": result.context_representation,
        "rho_star": result.rho_star,
        "median_rho_star": result.median_rho_star,
        "log_rho_star": result.log_rho_star,
        "ci_low": u.ci_low if u else np.nan,
        "ci_high": u.ci_high if u else np.nan,
        "ci_excludes_1": u.excludes_null if u else None,
        "lower_bound_on_recalibration_1_minus_rho_star":
            result.lower_bound_on_recalibration,
        "n_participants_used": result.n_participants_used,
        "n_participants_screened": result.n_participants_screened,
        "n_excluded": len(result.exclusions),
        "uncertainty_method": u.method if u else None,
        "resampling_unit": u.resampling_unit if u else None,
        "n_resamples": u.n_resamples if u else None,
        "diagnostic_status": result.diagnostic_status,
        "eligibility_status": result.eligibility_status,
        "additive_component": "NOT IDENTIFIED - never estimated",
        "rho_itself": "NOT point-identified; 1 - rho* is a LOWER BOUND",
        "data_status": result.data_status.value,
    }])


def exclusions_table(result: EstimatorResult) -> pd.DataFrame:
    if not result.exclusions:
        return pd.DataFrame([{"pid": "-", "reason": "no exclusions",
                              "data_status": result.data_status.value}])
    return pd.DataFrame([{"pid": k, "reason": v,
                          "data_status": result.data_status.value}
                         for k, v in sorted(result.exclusions.items())])


def placebo_table(p: PlaceboResult) -> pd.DataFrame:
    return pd.DataFrame([{
        "test": "contiguous epoch-1 split-half placebo",
        "runs_before_primary": True,
        "gates_primary": p.gates_primary,
        "n_participants": p.n_participants,
        "rho_star": p.rho_star, "ci_low": p.ci_low, "ci_high": p.ci_high,
        "rejected": p.rejected, "verdict": p.verdict,
        "data_status": p.data_status.value,
    }])


def uncertainty_table(u: UncertaintyResult) -> pd.DataFrame:
    return pd.DataFrame([{
        "method": u.method, "resampling_unit": u.resampling_unit,
        "n_participants": u.n_participants, "n_resamples": u.n_resamples,
        "point": u.point, "ci_low": u.ci_low, "ci_high": u.ci_high,
        "excludes_null": u.excludes_null, "seed": u.seed,
        "data_status": u.data_status.value,
    }])


def sensitivity_table(env: BiasEnvelopeResult) -> pd.DataFrame:
    rows = [{"scenario": k, "rho_star_under_null": v,
             "inside_envelope": bool(env.envelope_low <= v <= env.envelope_high)
             if np.isfinite(v) else None,
             "data_status": env.data_status.value}
            for k, v in env.rho_star_by_scenario.items()]
    rows.append({"scenario": "ENVELOPE [5th, 95th percentile]",
                 "rho_star_under_null": np.nan, "inside_envelope": None,
                 "data_status": env.data_status.value})
    df = pd.DataFrame(rows)
    df.attrs["envelope"] = (env.envelope_low, env.envelope_high)
    return df


# --------------------------------------------------- review-2 support tables
def title_alignment_table() -> pd.DataFrame:
    """| Title Term | Actual Module | Status | Role |

    Slide 9b. This pre-empts the single most dangerous question in the room.
    Where a title term is only WEAKLY represented, this table says so.
    """
    rows = [
        ("Adaptive",
         "aedt/twin/update.py - close_epoch(); aedt/knowledge/store.py",
         "IMPLEMENTED",
         "The twin re-estimates its per-person parameters as each epoch closes "
         "and adapts its own trust state (ACCEPTED / FLAGGED-UNTRUSTWORTHY). "
         "HONEST LIMIT: 'adaptive' here means scheduled re-estimation, not an "
         "online learning algorithm."),
        ("Emotional",
         "aedt/preprocess/reports.py; aedt/models/ordinal.py",
         "IMPLEMENTED (narrow)",
         "We model the REPORTING of self-reported stress, not affect itself. "
         "HONEST LIMIT: no emotion recognition is performed or claimed. The "
         "construct is the self-report, which is the object under study."),
        ("Digital Twin",
         "aedt/twin/state.py - TwinState, PersonalDigitalTwin",
         "IMPLEMENTED",
         "A persistent, per-person, individually parameterised model of that "
         "person's MEASURING INSTRUMENT. Persistence is load-bearing: the "
         "estimand is a ratio across epochs. HONEST LIMIT: the term is "
         "descriptive; no novelty is claimed for it."),
        ("Continual Knowledge",
         "aedt/knowledge/store.py - ContinualKnowledgeStore",
         "IMPLEMENTED",
         "Append-only, provenance-carrying, temporally inspectable per-person "
         "store; exactly four things update. HONEST LIMIT: NO continual-"
         "learning algorithm is imported (no replay, no EWC) - a deliberate "
         "scope decision."),
        ("Hypergraph",
         "aedt/hypergraph/structure.py; aedt/hypergraph/ablation.py",
         "IMPLEMENTED as knowledge representation + ablation",
         "Conjunctive feature-value hyperedges as the contextual knowledge "
         "representation, and an ablation arm against continuous and "
         "feature-vector contexts. HONEST LIMIT - STATE THIS FIRST: the "
         "hypergraph is NOT the identification mechanism for rho*. If the "
         "continuous covariate wins the ablation, we report that."),
        ("Reasoning",
         "aedt/twin/update.py - close_epoch() hyperedge-overlap trust rule",
         "WEAKLY REPRESENTED",
         "The twin reasons over higher-order context in one specific, real "
         "way: it reads epoch-to-epoch hyperedge occupancy overlap and flags "
         "an update as untrustworthy when the epochs describe different "
         "situations. HONEST LIMIT: this is a rule-based trust decision, not "
         "a general reasoning engine. This is the weakest title term and we "
         "say so before being asked."),
    ]
    return pd.DataFrame(
        [{"title_term": a, "actual_module": b, "status": c, "role": d,
          "data_status": DataStatus.SYNTHETIC.value} for a, b, c, d in rows])


def contribution_table() -> pd.DataFrame:
    """| Component | STANDARD | OUR CONTRIBUTION | Evidence |  (ROUND-17 §J)"""
    rows = [
        ("Data ingestion, cleaning, temporal alignment", True, False,
         "aedt/io/, aedt/preprocess/, aedt/alignment/ - standard parsing and "
         "causal windowing"),
        ("Sensor feature extraction", True, False,
         "aedt/features/ - conversation minutes, resting HR, activity, entropy"),
        ("Self-report label remap", True, False,
         "aedt/preprocess/reports.py - engineering integration, but "
         "SAFETY-CRITICAL: mapping by position would invert the scale"),
        ("Context formation - continuous", True, False,
         "aedt/contexts/continuous.py"),
        ("Context formation - hypergraph / n-ary", True, False,
         "aedt/hypergraph/structure.py - the REPRESENTATION is standard; its "
         "integration here is engineering"),
        ("Personal Digital Twin (persistent state)", False, False,
         "aedt/twin/state.py - ENGINEERING INTEGRATION, not claimed novel"),
        ("Continual knowledge (rolling re-estimation)", False, False,
         "aedt/knowledge/store.py - ENGINEERING INTEGRATION, not claimed novel"),
        ("Ordinal probit model", True, False,
         "aedt/models/ordinal.py - McCullagh 1980"),
        ("Slope-ratio estimator and its calibration argument", False, True,
         "aedt/estimators/slope_ratio.py - exactly 1 under the null, "
         "attenuated otherwise; validated across 13 misspecification scenarios"),
        ("Identifiability analysis", False, True,
         "docs/frozen_scientific_specification.md T1-T4: rho* identified, "
         "additive component provably NOT identified"),
        ("Failure analysis of the affine approach", False, True,
         "aedt/estimators/affine_did.py - reproduces the -0.107 null bias; "
         "asserted by tests/regression/test_known_failures.py"),
        ("Eligibility screen", False, False,
         "aedt/audit/eligibility.py - ENGINEERING INTEGRATION"),
        ("Placebo validation design", False, True,
         "aedt/inference/placebo.py - the DESIGN is ours; the technique is "
         "standard. Validated: does not fire even with a real 30% shift"),
        ("Bias envelope", False, True,
         "aedt/audit/envelope.py - rho* range under the enumerated assumption "
         "violations, under the true null"),
        ("Bootstrap inference", True, False,
         "aedt/inference/bootstrap.py - standard, clustered on participants"),
        ("Visualisation", False, False,
         "aedt/viz/ - ENGINEERING INTEGRATION"),
    ]
    return pd.DataFrame([{
        "component": c,
        "STANDARD": "YES" if std else ("-" if contrib else "engineering integration"),
        "OUR_CONTRIBUTION": "YES" if contrib else "no",
        "evidence": ev,
        "data_status": DataStatus.SYNTHETIC.value,
    } for c, std, contrib, ev in rows])


def status_board(rows: list[dict] | None = None) -> pd.DataFrame:
    """| Item | Completed | In Progress | Planned |

    COMPLETED means implemented AND verified, or an existing historical result
    WITH evidence. Code that merely exists is not completed.
    """
    default = [
        # (item, state, evidence)
        ("Repository structure and packaging", "COMPLETED",
         "aedt/ package, pyproject.toml, configs/, scripts/, tests/"),
        ("Configuration system", "COMPLETED",
         "aedt/config.py + 6 YAML configs; deviations from frozen detected"),
        ("Canonical schemas", "COMPLETED",
         "aedt/schemas.py - 15 typed dataclasses with validation"),
        ("Synthetic dataset ingestion", "COMPLETED",
         "aedt/io/synthetic.py, aedt/simulate/ - frozen generative model"),
        ("Preprocessing + missingness ledger", "COMPLETED",
         "aedt/preprocess/clean.py; outcome never imputed"),
        ("Temporal alignment (causal)", "COMPLETED",
         "aedt/alignment/align.py; leakage assertion + test"),
        ("Feature extraction interface", "COMPLETED",
         "aedt/features/ - 7 extractors behind one interface"),
        ("Context representation (3 forms)", "COMPLETED",
         "aedt/contexts/ + aedt/hypergraph/structure.py"),
        ("TwinState + persistence", "COMPLETED",
         "aedt/twin/state.py; JSON round-trip test"),
        ("Continual knowledge store", "COMPLETED",
         "aedt/knowledge/store.py; append-only, causality-checked"),
        ("Hypergraph / context layer", "COMPLETED",
         "aedt/hypergraph/ - built, visualised, ablated"),
        ("Ordinal probit model", "COMPLETED",
         "aedt/models/ordinal.py - ported + known-answer test"),
        ("Slope-ratio estimator (rho*)", "COMPLETED",
         "aedt/estimators/slope_ratio.py - ported + known-answer tests"),
        ("Eligibility screen", "COMPLETED",
         "aedt/audit/eligibility.py - frozen thresholds"),
        ("Placebo framework", "COMPLETED",
         "aedt/inference/placebo.py - gates the primary; exit code 5"),
        ("Uncertainty (participant bootstrap)", "COMPLETED",
         "aedt/inference/bootstrap.py - resampling unit asserted"),
        ("Bias envelope", "COMPLETED",
         "aedt/audit/envelope.py - 9 enumerated scenarios"),
        ("Dataset audit framework", "COMPLETED",
         "aedt/io/base.py + per-adapter audit(); every mandated field"),
        ("StudentLife adapter", "COMPLETED (fixture-tested only)",
         "aedt/io/studentlife.py - passes on a StudentLife-shaped fixture; "
         "NEVER RUN ON THE REAL ARCHIVE"),
        ("PMData adapter", "COMPLETED (run on REAL files)",
         "aedt/io/pmdata.py - real archive opened and audited "
         "(Thambawita et al. 2020); two real bugs fixed that only the real "
         "files exposed (mixed tz join, stress==0 sentinel)"),
        ("RELAX adapter", "COMPLETED (run on REAL files)",
         "aedt/io/relax.py - real archive acquired and audited "
         "(Zenodo 10.5281/zenodo.20701999, CC-BY-4.0); schema, anchors and "
         "timestamps verified against the release"),
        ("WESAD adapter", "COMPLETED (benchmark only)",
         "aedt/io/wesad.py - refuses the primary analysis by construction; "
         "NEVER RUN ON THE REAL ARCHIVE"),
        ("Epoch 1 vs Epoch 2 visualisation", "COMPLETED",
         "aedt/viz/curves.py - the two-curve plot"),
        ("Figure / table generation pipeline", "COMPLETED",
         "aedt/viz/, aedt/reporting/tables.py - every artefact stamped"),
        ("End-to-end synthetic demo", "COMPLETED",
         "scripts/run_demo.py --dataset synthetic - 9 stages"),
        ("Baselines", "COMPLETED",
         "aedt/baselines/ - 6 pre-specified baselines"),
        ("Hypergraph ablation (Ablation 1)", "COMPLETED (synthetic)",
         "aedt/hypergraph/ablation.py - run on synthetic data; result "
         "reported whichever way it falls"),
        ("Test suite", "COMPLETED",
         "tests/unit, integration, synthetic, regression"),
        ("Controlled synthetic simulations", "COMPLETED",
         "Historical: 13 misspecification scenarios (Rounds 14-15)"),
        ("Realistic synthetic simulation gate", "COMPLETED",
         "Historical: G1/G2/G3, power and placebo at realistic density"),
        ("REAL-DATA ACQUISITION (RELAX)", "COMPLETED",
         "31 participants acquired from Zenodo 10.5281/zenodo.20701999 "
         "(CC-BY-4.0); 453 MB of a 16.5 GB archive via range requests; "
         "provenance with per-file SHA-256 recorded"),
        ("REAL-DATA AUDIT (T4 on real RELAX files)", "COMPLETED",
         "Strict audit run on the real files: schema, answer-label anchors, "
         "timestamp cross-check, participant IDs, missingness"),
        ("REAL-DATA primary rho* estimate", "PLANNED",
         "BLOCKED and correctly so on BOTH audited datasets. RELAX: 0/31 "
         "eligible (density). PMData: 0/14 eligible (density + A3 + "
         "undocumented scale direction). Pipeline exits 3 on each. NO "
         "estimate exists from any real dataset"),
        ("REAL-DATA ACQUISITION (PMData)", "COMPLETED",
         "1.4 GB archive present locally; 0.32 MB extracted (wellness + "
         "resting HR + overview). 16 participants, ~105-150 day spans"),
        ("REAL-DATA AUDIT (T4 on real PMData files)", "COMPLETED",
         "Strict audit run: 0 of 14 eligible. Fails on report density, on "
         "assumption A3 (Var(s) ratios to 13.6), and on an undocumented "
         "scale direction"),
        ("REAL-DATA ACQUISITION (StudentLife / WESAD)", "PLANNED",
         "No file opened. StudentLife is currently unreachable from this "
         "environment (connection timeout on every URL tried)"),
        ("Cross-dataset pooling", "PLANNED",
         "Requires two audited real cohorts"),
        ("Ablations 2-7", "PLANNED",
         "Only Ablation 1 (context representation) is implemented and run"),
        ("Manuscript", "IN PROGRESS",
         "Theory and simulation sections are supported; the empirical section "
         "has no evidence yet"),
        ("Patent", "PLANNED (position RED)",
         "No concrete novel mechanism found. No drafting undertaken."),
    ]
    src = rows if rows is not None else [
        {"item": i, "state": s, "evidence": e} for i, s, e in default]
    out = []
    for r in src:
        state = r["state"]
        out.append({
            "item": r["item"],
            "completed": "YES" if state.startswith("COMPLETED") else "",
            "in_progress": "YES" if state.startswith("IN PROGRESS") else "",
            "planned": "YES" if state.startswith("PLANNED") else "",
            "qualifier": (state.split("(", 1)[1].rstrip(")")
                          if "(" in state else ""),
            "evidence": r["evidence"],
            "data_status": (DataStatus.PLANNED.value
                            if state.startswith("PLANNED")
                            else DataStatus.SYNTHETIC.value),
        })
    return pd.DataFrame(out)
