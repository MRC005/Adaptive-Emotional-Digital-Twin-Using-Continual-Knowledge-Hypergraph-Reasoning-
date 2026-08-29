#!/usr/bin/env python3
"""Regenerate EVERY Review-2 figure and table into figures/ and tables/.

    python scripts/generate_review2_outputs.py

Writes the presentation-ready artefacts to the repository's top-level
``figures/`` and ``tables/`` directories (versioned run folders live under
``results/``). Everything is stamped SYNTHETIC, because no real dataset file
has been opened by this project.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aedt.config import load_config
from aedt.constants import DataStatus
from aedt.demo_artefacts import (representative_participant,
                                 write_demo_artefacts)
from aedt.io import ADAPTERS, get_adapter
from aedt.logging_setup import setup_logging
from aedt.pipeline import run_pipeline
from aedt.reporting.tables import dataset_audit_table, write_table


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="synthetic", choices=sorted(ADAPTERS))
    ap.add_argument("--root", default=None)
    ap.add_argument("--bootstrap", type=int, default=999)
    ap.add_argument("--figures", default="figures")
    ap.add_argument("--tables", default="tables")
    a = ap.parse_args(argv)
    setup_logging("INFO")

    cfg = load_config("simulation" if a.dataset == "synthetic" else a.dataset)
    res = run_pipeline(a.dataset, root=a.root, config=cfg,
                       n_resamples=a.bootstrap, halt_on_placebo_failure=False)
    if res.frame is None:
        print("\n".join(res.blocking_reasons))
        return 6

    staging = Path(".review2_staging")
    if staging.exists():
        shutil.rmtree(staging)
    (staging / "figures").mkdir(parents=True)
    (staging / "tables").mkdir(parents=True)
    pid = representative_participant(res)
    write_demo_artefacts(
        res, staging, pid=pid, n_resamples=a.bootstrap, cfg=cfg,
        selection_rule=(f"Participant {pid} selected by a rule fixed in "
                        "advance: the participant whose own rho* is CLOSEST "
                        "to the cohort rho* - the representative case, never "
                        "the most extreme."))

    figs, tabs = Path(a.figures), Path(a.tables)
    figs.mkdir(parents=True, exist_ok=True)
    tabs.mkdir(parents=True, exist_ok=True)
    n = 0
    for src, dst in ((staging / "figures", figs), (staging / "tables", tabs)):
        for f in sorted(src.iterdir()):
            shutil.copy2(f, dst / f.name)
            n += 1

    # the audit table for EVERY dataset, including the ones with no files
    audits = [get_adapter(name).audit(a.root if name == a.dataset else None)
              for name in sorted(ADAPTERS)]
    write_table(dataset_audit_table(audits), tabs / "t00_all_dataset_audits",
                title="Dataset audit across every registered dataset")

    shutil.rmtree(staging)
    print(f"\n{n + 2} artefacts written to {figs}/ and {tabs}/   "
          f"DATA STATUS: {res.data_status.value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
