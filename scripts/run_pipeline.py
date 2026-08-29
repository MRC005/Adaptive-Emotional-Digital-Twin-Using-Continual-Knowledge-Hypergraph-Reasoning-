#!/usr/bin/env python3
"""Run the full analysis pipeline and write every artefact.

    python scripts/run_pipeline.py --dataset synthetic
    python scripts/run_pipeline.py --dataset studentlife --root /path/to/archive

Unlike ``run_demo.py`` this prints no narrative -- it is the batch entry point.
The frozen primary uses 2000 bootstrap resamples, which is the default here.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aedt.config import load_config
from aedt.constants import BOOTSTRAP_B
from aedt.demo_artefacts import (representative_participant,
                                 write_demo_artefacts)
from aedt.errors import ScientificError
from aedt.io import ADAPTERS
from aedt.logging_setup import log_run_header, setup_logging
from aedt.pipeline import run_pipeline
from aedt.reporting.metadata import (make_run_metadata, new_run_dir,
                                     write_metadata)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="synthetic", choices=sorted(ADAPTERS))
    ap.add_argument("--root", default=None)
    ap.add_argument("--out", default="results")
    ap.add_argument("--bootstrap", type=int, default=BOOTSTRAP_B)
    ap.add_argument("--no-artefacts", action="store_true")
    ap.add_argument("--strict-real", action="store_true",
                    help="fail hard (exit 6) if the real archive is missing")
    ap.add_argument("--log-level", default="INFO")
    a = ap.parse_args(argv)

    t0 = time.time()
    cfg = load_config("simulation" if a.dataset == "synthetic" else a.dataset)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log = setup_logging(a.log_level, log_file=out / "_last_run.jsonl")
    log_run_header(log, dataset=a.dataset, root=a.root, seed=cfg.seed,
                   config=", ".join(cfg.sources), bootstrap=a.bootstrap)

    try:
        res = run_pipeline(a.dataset, root=a.root, config=cfg,
                           n_resamples=a.bootstrap,
                           strict_real=a.strict_real)
    except ScientificError as exc:
        log.error("%s: %s", type(exc).__name__, exc)
        print(f"\n{'!' * 78}\n  {exc}\n{'!' * 78}\n")
        return exc.exit_code

    if res.frame is None:
        print(f"\n{'!' * 78}")
        for r in res.blocking_reasons:
            print(f"  {r}")
        print(f"{'!' * 78}\n")
        return 6

    run_dir = new_run_dir(out, a.dataset, res.data_status)
    if not a.no_artefacts:
        pid = representative_participant(res)
        write_demo_artefacts(
            res, run_dir, pid=pid, n_resamples=a.bootstrap, cfg=cfg,
            selection_rule=(f"Participant {pid} selected by a rule fixed in "
                            "advance: the participant whose own rho* is "
                            "CLOSEST to the cohort rho*."))
    (run_dir / "pipeline_result.json").write_text(
        json.dumps(res.to_dict(), indent=2, default=str))
    meta = make_run_metadata(dataset=a.dataset, data_status=res.data_status,
                            config=cfg, started=t0, output_dir=run_dir)
    write_metadata(meta, run_dir, cfg)
    log.info("artefacts written to %s", run_dir)
    print(f"\nOutput: {run_dir}   DATA STATUS: {res.data_status.value}   "
          f"validated: {res.validated}")
    if res.blocking_reasons:
        for r in res.blocking_reasons:
            print(f"  BLOCKING: {r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
