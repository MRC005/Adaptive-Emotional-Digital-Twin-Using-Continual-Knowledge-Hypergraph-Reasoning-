"""Structured logging. One configuration, used by every entry point.

Every pipeline run logs, at INFO: dataset, data status, participant count,
observations, exclusions, epoch definition, model convergence, estimator
result, placebo status and output location. Every exclusion is logged with its
pid and reason. Fitted models are logged at DEBUG.

Fatal scientific errors are never logged and swallowed; they propagate to a
distinct non-zero exit code (see ``aedt/errors.py``).
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

__all__ = ["setup_logging", "JsonLineFormatter", "log_run_header"]


class JsonLineFormatter(logging.Formatter):
    """One JSON object per line, for the machine-readable run log."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "time": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        for k, v in getattr(record, "extra_fields", {}).items():
            payload[k] = v
        return json.dumps(payload, default=str)


def setup_logging(level: str = "INFO", *, log_file: str | Path | None = None,
                  quiet: bool = False) -> logging.Logger:
    """Console (human) + optional JSON-lines file (machine)."""
    root = logging.getLogger("aedt")
    root.setLevel(logging.DEBUG)
    root.handlers.clear()
    root.propagate = False

    if not quiet:
        con = logging.StreamHandler(sys.stdout)
        con.setLevel(getattr(logging, level.upper(), logging.INFO))
        con.setFormatter(logging.Formatter("%(levelname)-7s %(name)-28s %(message)s"))
        root.addHandler(con)

    if log_file is not None:
        p = Path(log_file)
        p.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(p)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(JsonLineFormatter())
        root.addHandler(fh)
    return root


def log_run_header(log: logging.Logger, **fields) -> None:
    log.info("=" * 78)
    for k, v in fields.items():
        log.info("%-24s %s", k, v)
    log.info("=" * 78)
