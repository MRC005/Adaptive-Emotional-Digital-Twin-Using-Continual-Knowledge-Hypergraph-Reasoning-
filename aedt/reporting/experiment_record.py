"""One record per experiment run, so a number can always be traced back.

WHY THIS EXISTS
The first study's K=80 twin figure cannot be explained today. The archive is
digest-verified, the code is one unmodified commit, and every environmental
hypothesis has been eliminated -- but none of that could be checked from the
run's own output, because the run recorded nothing about itself. The machinery
to record it already existed in ``aedt/reporting/metadata.py`` and was never
wired in.

Every experiment written from now on calls ``write_experiment_record()`` and
writes ``run_metadata.json`` beside its results, carrying:

    git commit · Python version · versions of every package that can move a
    number · thread-pool configuration · the dataset identifier and its
    recorded SHA-256 digests · the full experiment configuration · the seed

It touches nothing in the frozen experiment. It is additive, and its only job
is to make the next discrepancy answerable from the artefact itself.
"""
from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

from ..constants import DataStatus
from .metadata import (git_commit, make_run_metadata,  # noqa: F401
                       package_versions, thread_environment)

__all__ = ["dataset_provenance", "build_experiment_record",
           "write_experiment_record", "RECORD_FILENAME"]

RECORD_FILENAME = "run_metadata.json"


def dataset_provenance(data_root: str | Path) -> dict:
    """The dataset identifier and the digests recorded for it.

    Reads ``PROVENANCE.json`` from the dataset directory rather than restating
    anything: the citation, DOI, source and per-file SHA-256 prefixes travel
    with the result. A missing record is reported as missing, never invented.
    """
    root = Path(data_root)
    p = root / "PROVENANCE.json"
    if not p.exists():
        return {"root": str(root), "status": "NO PROVENANCE RECORD",
                "digests": None}
    rec = json.loads(p.read_text(encoding="utf-8"))
    return {
        "root": str(root),
        "dataset": rec.get("dataset"),
        "doi": rec.get("doi"),
        "official_source": rec.get("official_source"),
        "file_count": rec.get("file_count"),
        "total_bytes": rec.get("total_bytes"),
        "recorded_utc": rec.get("recorded_utc"),
        "digests": rec.get("key_file_digests"),
    }


def build_experiment_record(*, experiment: str, dataset: str, seed: int,
                            config: dict, started: float,
                            data_root: str | Path | None = None,
                            data_status: DataStatus = DataStatus.REAL,
                            output_dir: str | Path | None = None) -> dict:
    """Assemble the record. Pure -- takes no side effects, writes nothing."""
    meta = make_run_metadata(dataset=dataset, data_status=data_status,
                             config=None, started=started,
                             output_dir=output_dir, seed=seed)
    return {
        "_what_this_is": (
            "Reproducibility record for one experiment run. If a number in the "
            "result beside this file is ever questioned, everything needed to "
            "reconstruct the run that produced it is here."),
        "experiment": experiment,
        "run": json.loads(meta.to_json()) if hasattr(meta, "to_json")
               else meta.__dict__,
        "git_commit": git_commit(),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "package_versions": package_versions(),
        "thread_environment": thread_environment(),
        "dataset_provenance": (dataset_provenance(data_root)
                               if data_root is not None else None),
        "seed": seed,
        "config": config,
        "elapsed_seconds": round(time.time() - started, 3),
    }


def write_experiment_record(out_dir: str | Path, **kwargs) -> Path:
    """Write ``run_metadata.json`` into ``out_dir`` and return its path."""
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    record = build_experiment_record(**kwargs)
    p = d / RECORD_FILENAME
    p.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    return p
