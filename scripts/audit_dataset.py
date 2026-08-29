#!/usr/bin/env python3
"""STRICT DATASET AUDIT. Always safe to run; never guesses; never substitutes.

    python scripts/audit_dataset.py --dataset studentlife --root /path/to/archive
    python scripts/audit_dataset.py --all              # audit every dataset
    python scripts/audit_dataset.py --fixture studentlife --out /tmp/sl_fixture

If the files are absent this prints, for example:

    REAL DATA UNAVAILABLE - STUDENTLIFE AUDIT NOT RUN

together with the acquisition instructions, and exits 6. It never falls back to
synthetic data.

Exit codes: 0 audited - 2 DECISION REQUIRED - 6 REAL DATA UNAVAILABLE
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aedt.constants import DataStatus
from aedt.errors import DecisionRequired, ScientificError
from aedt.io import ADAPTERS, get_adapter
from aedt.io.fixtures import (make_pmdata_fixture, make_relax_fixture,
                              make_studentlife_fixture)
from aedt.logging_setup import setup_logging
from aedt.reporting.tables import dataset_audit_table, write_table

W = 78
FIXTURES = {"studentlife": make_studentlife_fixture,
            "pmdata": make_pmdata_fixture, "relax": make_relax_fixture}


def show(a) -> None:
    print("=" * W)
    print(f"DATASET AUDIT   {a.dataset_name.upper()}")
    print("=" * W)
    fields = [
        ("dataset name", a.dataset_name),
        ("scientific role", a.role.value),
        ("DATA STATUS", a.data_status.value),
        ("source status", a.source_status),
        ("local files available", a.local_files_available),
        ("root path", a.root_path),
        ("files found", "; ".join(a.files_found) or None),
        ("participant count", a.participant_count),
        ("observation count", a.observation_count),
        ("sensor modalities", ", ".join(a.sensor_modalities) or None),
        ("self-report variables", ", ".join(a.self_report_variables) or None),
        ("self-report scale", a.self_report_scale),
        ("stress labels", ", ".join(a.stress_labels) or None),
        ("raw stored codes", ", ".join(map(str, a.raw_stored_codes)) or None),
        ("code -> label", dict(a.code_to_label_mapping) or None),
        ("code -> severity", dict(a.code_to_severity_mapping) or None),
        ("timestamps present", a.timestamps_present),
        ("timestamp format", a.timestamp_format),
        ("timezone", a.timezone),
        ("longitudinal span (days)", a.longitudinal_span_days),
        ("median obs / participant", a.median_observations_per_participant),
        ("missingness", dict(a.missingness) or None),
        ("sensor/report alignment", a.sensor_report_alignment),
        ("conversation / context", a.conversation_context_available),
        ("ELIGIBLE for PRIMARY", a.eligible_for_primary_analysis),
        ("ELIGIBLE for BENCHMARK", a.eligible_for_benchmark_analysis),
    ]
    for k, v in fields:
        print(f"  {k:<26} {v}")
    if a.observations_per_participant:
        n = len(a.observations_per_participant)
        head = list(a.observations_per_participant.items())[:8]
        print(f"  {'obs per participant':<26} "
              f"{dict(head)}{' ...' if n > 8 else ''}  ({n} participants)")
    if a.exclusion_reasons:
        print(f"\n  EXCLUSION REASONS")
        for r in a.exclusion_reasons:
            print(f"    - {r}")
    if a.notes:
        print(f"\n  NOTES")
        for nte in a.notes:
            print(f"    - {nte}")
    if not a.local_files_available and a.acquisition_instructions:
        print(f"\n  HOW TO OBTAIN THIS DATASET")
        for line in a.acquisition_instructions.splitlines():
            print(f"    {line}")
    print("=" * W)
    print()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=sorted(ADAPTERS))
    ap.add_argument("--root")
    ap.add_argument("--all", action="store_true",
                    help="audit every registered dataset")
    ap.add_argument("--fixture", choices=sorted(FIXTURES),
                    help="write a SYNTHETIC dataset-shaped fixture and audit it")
    ap.add_argument("--out", default=None, help="where to write the fixture")
    ap.add_argument("--tables", default="tables",
                    help="directory for the T4 audit table")
    a = ap.parse_args(argv)
    setup_logging("INFO")

    if a.fixture:
        out = Path(a.out or f"data/synthetic/{a.fixture}_fixture")
        FIXTURES[a.fixture](out)
        print(f"SYNTHETIC FIXTURE written to {out}. NOT REAL DATA.\n")
        a.dataset, a.root = a.fixture, str(out)

    if not (a.dataset or a.all):
        ap.error("pass --dataset, --all or --fixture")

    names = sorted(ADAPTERS) if a.all else [a.dataset]
    audits, worst = [], 0
    for name in names:
        adapter = get_adapter(name)
        try:
            au = adapter.audit(a.root if not a.all or name == a.dataset else None)
        except DecisionRequired as exc:
            print("!" * W)
            print(f"  {exc}")
            print("!" * W)
            worst = max(worst, exc.exit_code)
            continue
        except ScientificError as exc:
            print("!" * W)
            print(f"  {type(exc).__name__}: {exc}")
            print("!" * W)
            worst = max(worst, exc.exit_code)
            continue
        show(au)
        audits.append(au)
        if not au.local_files_available and au.data_status is DataStatus.PLANNED:
            worst = max(worst, 6)

    if audits:
        p = write_table(dataset_audit_table(audits),
                        Path(a.tables) / "t01_dataset_audit",
                        title="Dataset audit (T4) - every mandated field")
        print(f"T4 audit table written to {p} and {p.with_suffix('.md')}")
    return worst


if __name__ == "__main__":
    raise SystemExit(main())
