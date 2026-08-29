"""Configuration system. Every scientific decision lives in a config file.

Layering: ``configs/base.yaml`` is loaded first, then the dataset config is
merged over it. Command-line overrides are applied last and are RECORDED in the
run metadata, so a result computed with a non-default threshold is visibly
non-default.

The frozen constants in ``aedt/constants.py`` are the DEFAULTS. A config may
restate them (which is how a pre-specified sensitivity analysis is declared),
and ``Config.deviations_from_frozen`` reports every value that differs -- that
report is written into every result folder.
"""
from __future__ import annotations

import copy
import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .constants import SEED
from .errors import DecisionRequired

log = logging.getLogger(__name__)

__all__ = ["Config", "load_config", "CONFIG_DIR", "deep_merge"]

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


@dataclass
class Config:
    """A resolved configuration, with provenance."""

    data: dict[str, Any] = field(default_factory=dict)
    sources: list[str] = field(default_factory=list)
    overrides: dict[str, Any] = field(default_factory=dict)

    # --------------------------------------------------------------- access
    def get(self, path: str, default: Any = None) -> Any:
        node: Any = self.data
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def require(self, path: str) -> Any:
        v = self.get(path, None)
        if v is None:
            raise DecisionRequired(
                f"Configuration key {path!r} is unset and has no safe default. "
                f"Set it in one of: {self.sources}")
        return v

    def __getitem__(self, path: str) -> Any:
        return self.require(path)

    # ------------------------------------------------------------ metadata
    @property
    def seed(self) -> int:
        return int(self.get("run.seed", SEED))

    @property
    def digest(self) -> str:
        return hashlib.sha256(
            json.dumps(self.data, sort_keys=True, default=str).encode()
        ).hexdigest()[:16]

    def deviations_from_frozen(self) -> dict[str, tuple[Any, Any]]:
        """Every eligibility/inference value that differs from the frozen one."""
        from . import constants as C
        out: dict[str, tuple[Any, Any]] = {}
        checks = {
            "eligibility.min_reports_per_epoch": C.MIN_REPORTS_PER_EPOCH,
            "eligibility.min_categories_used": C.MIN_CATEGORIES_USED,
            "eligibility.min_sensor_sd": C.MIN_SENSOR_SD,
            "eligibility.var_ratio_lo": C.VAR_RATIO_LO,
            "eligibility.var_ratio_hi": C.VAR_RATIO_HI,
            "eligibility.min_abs_beta": C.MIN_ABS_BETA,
            "inference.bootstrap_resamples": C.BOOTSTRAP_B,
            "run.seed": C.SEED,
        }
        for key, frozen in checks.items():
            v = self.get(key, None)
            if v is not None and v != frozen:
                out[key] = (frozen, v)
        if out:
            log.warning("CONFIG DEVIATES FROM THE FROZEN SPECIFICATION: %s. "
                        "This must be a PRE-SPECIFIED sensitivity analysis.",
                        out)
        return out

    def eligibility_thresholds(self) -> dict[str, Any]:
        """The screen's thresholds, as this run will actually apply them."""
        from . import constants as C
        return {
            "MIN_REPORTS_PER_EPOCH": self.get(
                "eligibility.min_reports_per_epoch", C.MIN_REPORTS_PER_EPOCH),
            "MIN_CATEGORIES_USED": self.get(
                "eligibility.min_categories_used", C.MIN_CATEGORIES_USED),
            "MIN_SENSOR_SD": self.get("eligibility.min_sensor_sd",
                                      C.MIN_SENSOR_SD),
            "VAR_RATIO_LO": self.get("eligibility.var_ratio_lo", C.VAR_RATIO_LO),
            "VAR_RATIO_HI": self.get("eligibility.var_ratio_hi", C.VAR_RATIO_HI),
            "MIN_ABS_BETA": self.get("eligibility.min_abs_beta", C.MIN_ABS_BETA),
        }

    def to_json(self) -> str:
        return json.dumps({"data": self.data, "sources": self.sources,
                           "overrides": self.overrides, "digest": self.digest},
                          indent=2, default=str)


def _read(path: Path) -> dict:
    if not path.exists():
        raise DecisionRequired(f"Configuration file not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_config(dataset: str | None = None, *, path: str | Path | None = None,
                overrides: dict | None = None,
                config_dir: str | Path | None = None) -> Config:
    """Load base.yaml, merge the dataset config, then apply overrides."""
    cdir = Path(config_dir) if config_dir else CONFIG_DIR
    sources: list[str] = []
    base_path = cdir / "base.yaml"
    data = _read(base_path)
    sources.append(str(base_path))

    if path is not None:
        p = Path(path)
        data = deep_merge(data, _read(p))
        sources.append(str(p))
    elif dataset is not None:
        p = cdir / f"{dataset}.yaml"
        if not p.exists():
            raise DecisionRequired(
                f"No configuration for dataset {dataset!r} at {p}. Known "
                f"configs: {sorted(f.stem for f in cdir.glob('*.yaml'))}")
        data = deep_merge(data, _read(p))
        sources.append(str(p))

    ov = overrides or {}
    for dotted, value in ov.items():
        node = data
        parts = dotted.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    cfg = Config(data=data, sources=sources, overrides=dict(ov))
    cfg.deviations_from_frozen()
    return cfg
