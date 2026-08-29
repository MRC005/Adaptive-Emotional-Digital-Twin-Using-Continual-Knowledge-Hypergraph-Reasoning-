"""Configuration: layering, digests, and detection of frozen-spec deviations."""
from __future__ import annotations

import pytest

from aedt import constants as C
from aedt.config import Config, deep_merge, load_config
from aedt.errors import DecisionRequired


@pytest.mark.parametrize("name", ["simulation", "studentlife", "pmdata",
                                  "relax", "wesad"])
def test_every_dataset_config_loads_and_matches_the_frozen_spec(name):
    cfg = load_config(name)
    assert cfg.seed == C.SEED
    assert cfg.get("project.estimand") == "rho_star"
    assert cfg.deviations_from_frozen() == {}, (
        "a shipped config deviates from the frozen specification")


def test_eligibility_thresholds_come_through_unchanged():
    cfg = load_config("simulation")
    assert cfg.eligibility_thresholds() == {
        "MIN_REPORTS_PER_EPOCH": 60, "MIN_CATEGORIES_USED": 2,
        "MIN_SENSOR_SD": 0.10, "VAR_RATIO_LO": 0.25, "VAR_RATIO_HI": 4.0,
        "MIN_ABS_BETA": 0.02}


def test_a_deviation_is_detected_and_reported():
    cfg = load_config("simulation",
                      overrides={"eligibility.min_reports_per_epoch": 30})
    dev = cfg.deviations_from_frozen()
    assert "eligibility.min_reports_per_epoch" in dev
    assert dev["eligibility.min_reports_per_epoch"] == (60, 30)
    assert cfg.eligibility_thresholds()["MIN_REPORTS_PER_EPOCH"] == 30


def test_dataset_config_overrides_base():
    assert load_config("wesad").get("dataset.sensor") == "chest_ecg_window_mean"
    assert load_config("pmdata").get("dataset.sensor") == "resting_hr"


def test_wesad_config_cannot_enable_the_primary_analysis():
    """A property of the DATA, not a preference."""
    cfg = load_config("wesad")
    assert cfg.get("analysis.primary_rho_star") is False
    assert cfg.get("analysis.benchmark_only") is True


def test_missing_key_raises_rather_than_defaulting():
    cfg = load_config("simulation")
    with pytest.raises(DecisionRequired, match="has no safe default"):
        cfg.require("nonexistent.key")


def test_unknown_dataset_config_is_refused():
    with pytest.raises(DecisionRequired, match="No configuration for dataset"):
        load_config("not_a_dataset")


def test_digest_changes_with_content():
    a = load_config("simulation")
    b = load_config("simulation", overrides={"run.seed": 1})
    assert a.digest != b.digest
    assert load_config("simulation").digest == a.digest


def test_deep_merge_is_recursive_and_non_mutating():
    base = {"a": {"x": 1, "y": 2}, "b": 3}
    out = deep_merge(base, {"a": {"y": 9}, "c": 4})
    assert out == {"a": {"x": 1, "y": 9}, "b": 3, "c": 4}
    assert base == {"a": {"x": 1, "y": 2}, "b": 3}
