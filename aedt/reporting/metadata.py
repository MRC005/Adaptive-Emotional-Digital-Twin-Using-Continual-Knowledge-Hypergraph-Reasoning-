"""REPRODUCIBILITY: run metadata, versioned output folders, provenance.

Every result records: data source, REAL or SYNTHETIC, configuration, random
seed, software version, and execution time. Output folders are timestamped so
a re-run never silently overwrites an earlier result.
"""
from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from .. import __version__
from ..config import Config
from ..constants import DataStatus
from ..schemas import RunMetadata

__all__ = ["make_run_metadata", "new_run_dir", "write_metadata",
           "package_versions", "git_commit"]


def package_versions() -> dict[str, str]:
    out = {}
    for mod in ("numpy", "scipy", "pandas", "matplotlib", "yaml", "pytest"):
        try:
            m = __import__(mod)
            out[mod] = getattr(m, "__version__", "unknown")
        except Exception:
            out[mod] = "not installed"
    return out


def git_commit() -> str | None:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                           text=True, timeout=5,
                           cwd=Path(__file__).resolve().parents[2])
        return r.stdout.strip() or None if r.returncode == 0 else None
    except Exception:
        return None


def new_run_dir(base: str | Path, dataset: str, status: DataStatus) -> Path:
    """A timestamped, status-stamped output folder. Never overwrites."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    p = Path(base) / f"{stamp}_{dataset}_{status.value}"
    p.mkdir(parents=True, exist_ok=True)
    for sub in ("figures", "tables", "twins"):
        (p / sub).mkdir(exist_ok=True)
    return p


def make_run_metadata(*, dataset: str, data_status: DataStatus,
                      config: Config | None, started: float,
                      output_dir: str | Path | None = None,
                      command: str | None = None) -> RunMetadata:
    now = time.time()
    return RunMetadata(
        run_id=datetime.now().strftime("%Y%m%d-%H%M%S") + f"-{dataset}",
        started_at=datetime.fromtimestamp(started, timezone.utc).isoformat(
            timespec="seconds"),
        finished_at=datetime.fromtimestamp(now, timezone.utc).isoformat(
            timespec="seconds"),
        dataset=dataset, data_status=data_status,
        config_path=", ".join(config.sources) if config else None,
        config_digest=config.digest if config else "none",
        seed=config.seed if config else 0,
        software_version=__version__,
        python_version=platform.python_version(),
        package_versions=package_versions(),
        git_commit=git_commit(),
        command=command or " ".join(sys.argv),
        elapsed_seconds=round(now - started, 3),
        output_dir=str(output_dir) if output_dir else None)


def write_metadata(meta: RunMetadata, out_dir: str | Path,
                   config: Config | None = None) -> Path:
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    (d / "run_metadata.json").write_text(meta.to_json())
    if config is not None:
        (d / "resolved_config.json").write_text(config.to_json())
        dev = config.deviations_from_frozen()
        (d / "config_deviations_from_frozen.json").write_text(
            json.dumps({k: {"frozen": f, "used": u} for k, (f, u) in dev.items()},
                       indent=2))
    return d / "run_metadata.json"
