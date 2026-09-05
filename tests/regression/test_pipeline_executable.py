"""Can the analysis pipeline actually RUN? A smoke test on a fake archive.

READ THIS BEFORE TRUSTING ANYTHING HERE.

The archive these scripts consume is 2.76 GB and is not redistributable, so it
is absent from any fresh checkout. That leaves a real question unanswered: is
the code still executable at all, or has it rotted? This file answers exactly
that question and NOTHING else.

The fixture is INVENTED. It has the shape of the College Experience archive --
the same file layout, the same column names -- and none of its content. No
number produced here is a finding, may be quoted, or may reach the website. A
green run means "the pipeline executes end to end"; it does not mean any
reported result has been reproduced. Reproducing the reported results requires
the real archive, and there is no substitute for it.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]

#: Sensing columns the modelling frame consumes, plus filler so the screen has
#: more than one column to choose between.
SENSING_COLS = ("unlock_duration_ep_0", "loc_dist_ep_0", "act_still_ep_0",
                "act_walking_ep_0", "audio_convo_duration_ep_0",
                "unlock_num_ep_0", "filler_a_ep_0", "filler_b_ep_0")


def _load(name: str, relpath: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relpath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def fake_archive(tmp_path) -> Path:
    """An archive-SHAPED fixture. Invented content; never a stand-in for data."""
    rng = np.random.default_rng(1234)
    root = tmp_path / "college-experience"
    (root / "EMA").mkdir(parents=True)
    (root / "Sensing").mkdir(parents=True)

    ema_rows, sens_rows = [], []
    start = pd.Timestamp("2020-01-01")
    for p in range(40):
        uid = f"u{p:03d}"
        rho = float(rng.uniform(0.0, 0.6))
        latent = 0.0
        # 80 reports every other day: dense enough to clear MIN_OBS_PER_PERSON
        for i in range(80):
            latent = rho * latent + rng.normal(0, 1)
            day = start + pd.Timedelta(days=2 * i)
            stress = int(np.clip(round(3 + latent), 1, 5))
            ema_rows.append({"uid": uid, "day": int(day.strftime("%Y%m%d")),
                             "stress": stress,
                             "social_level": int(rng.integers(1, 6)),
                             "pam": int(rng.integers(1, 17))})
        for i in range(170):
            day = start + pd.Timedelta(days=i)
            row = {"uid": uid, "day": int(day.strftime("%Y%m%d")),
                   "is_ios": int(p % 2)}
            for c in SENSING_COLS:
                row[c] = float(rng.normal(100, 25))
            sens_rows.append(row)

    pd.DataFrame(ema_rows).to_csv(root / "EMA" / "general_ema.csv", index=False)
    pd.DataFrame(sens_rows).to_csv(root / "Sensing" / "sensing.csv", index=False)
    return root


# ------------------------------------------------- the pre-registered frame
def test_the_original_prediction_frame_still_builds_and_is_leakage_free(
        fake_archive):
    """The v1 data path is executable and its temporal contract still holds.

    This touches nothing in the primary experiment; it only proves that the
    function it depends on runs and that ``assert_no_leakage`` accepts what it
    produces.
    """
    from aedt.twin.prediction_data import (assert_no_leakage,
                                           build_prediction_frame)
    data = build_prediction_frame(fake_archive)
    assert len(data.frame) > 0
    assert data.frame["participant_id"].nunique() == 40
    assert_no_leakage(data.frame)
    for col in ("current", "prev_1", "hist_mean", "ewma", "hist_n"):
        assert col in data.feature_columns


def test_the_leakage_check_still_rejects_a_frame_that_violates_the_contract(
        fake_archive):
    """A check that never fires proves nothing, so make it fire."""
    from aedt.twin.prediction_data import (assert_no_leakage,
                                           build_prediction_frame)
    frame = build_prediction_frame(fake_archive).frame.copy()
    frame["target_time"] = frame["prediction_time"]        # target no longer future
    with pytest.raises(AssertionError):
        assert_no_leakage(frame)


def test_the_original_experiment_still_runs_end_to_end(fake_archive, tmp_path,
                                                       monkeypatch):
    """The pre-registered experiment EXECUTES. It is not re-run or re-scored.

    This answers "has the v1 pipeline rotted?" and nothing else. The archive it
    runs on here is invented, so every number it produces is meaningless and
    none is asserted against the recorded result. Reproducing the recorded
    result requires the real archive; there is no substitute.

    The script itself is untouched -- it is imported, and only its DATA and OUT
    paths are redirected.
    """
    mod = _load("run_twin_experiment", "scripts/run_twin_experiment.py")
    out = tmp_path / "results" / "twin"
    monkeypatch.setattr(mod, "DATA", fake_archive)
    monkeypatch.setattr(mod, "OUT", out)
    monkeypatch.setattr(mod, "K_GRID", (0, 20))       # two points is enough
    monkeypatch.setattr(mod.sys, "argv", ["run_twin_experiment.py", "--quick"])

    assert mod.main() == 0
    payload = json.loads((out / "twin_experiment.json").read_text(
        encoding="utf-8"))

    assert payload["primary_metric"] == "macro_f1", (
        "the pre-registered primary metric has changed")
    assert payload["seed"] == 20260828, "the pre-registered seed has changed"
    assert payload["n_train"] + payload["n_val"] + payload["n_test"] == 40

    for result in payload["results"]:
        models = result["models"]
        for name in ("B0_majority", "B1_persistence", "B2_global",
                     "B3_static_prior", "B4_calibrated", "T_twin"):
            assert name in models, f"baseline {name} disappeared from the run"
        assert "twin_vs_B1_persistence" in result, (
            "persistence is no longer the comparison the experiment reports")
    assert (out / "twin_results_table.md").exists()


# ----------------------------------------------------------- ceiling driver
def test_ceiling_analysis_runs_end_to_end_and_writes_a_complete_artefact(
        fake_archive, tmp_path, monkeypatch):
    mod = _load("run_ceiling_analysis", "scripts/run_ceiling_analysis.py")
    out = tmp_path / "results" / "twin"
    monkeypatch.setattr(mod, "DATA", fake_archive)
    monkeypatch.setattr(mod, "OUT", out)
    monkeypatch.setattr(mod.sys, "argv", ["run_ceiling_analysis.py"])

    assert mod.main() == 0
    payload = json.loads((out / "ceiling.json").read_text(encoding="utf-8"))

    for key in ("_generated_by", "_generated_utc", "_inputs", "_declared",
                "cohort", "ceiling", "ceiling_all_pairs", "behaviour_screen"):
        assert key in payload, f"{key} missing from ceiling.json"

    c = payload["ceiling"]
    assert c["n_participants_analysed"] == 40
    assert -1.0 <= c["within_person_autocorrelation"] <= 1.0
    assert c["variance_explained"] == pytest.approx(
        c["within_person_autocorrelation"] ** 2)
    assert 0.0 <= c["icc_between_person"] <= 1.0
    assert 0.0 <= c["strongest_behaviour_r"] <= 1.0

    # every sensing column is screened -- the count is measured, not asserted
    assert payload["behaviour_screen"]["n_sensing_columns_screened"] == \
        len(SENSING_COLS)

    # the cohort descriptors are read off the fixture, not remembered
    assert payload["cohort"]["participants"] == 40
    assert payload["cohort"]["reports"] == 40 * 80
    assert payload["cohort"]["prediction_pairs"] > 0

    # both definitions are present, so a reader can see the choice
    assert payload["ceiling_all_pairs"]["n_participants_analysed"] == 40


def test_ceiling_analysis_refuses_when_the_archive_is_absent(tmp_path,
                                                             monkeypatch):
    mod = _load("run_ceiling_analysis", "scripts/run_ceiling_analysis.py")
    monkeypatch.setattr(mod, "DATA", tmp_path / "nothing-here")
    monkeypatch.setattr(mod, "OUT", tmp_path / "out")
    monkeypatch.setattr(mod.sys, "argv", ["run_ceiling_analysis.py"])
    assert mod.main() == 6, "missing data must exit 6, never approximate"
    assert not (tmp_path / "out").exists()


# ---------------------------------------------------------------- exporter
def test_exporter_writes_only_from_artefacts_and_matches_them(tmp_path,
                                                              monkeypatch):
    """Feed the exporter synthetic artefacts; the output must mirror them."""
    mod = _load("export_findings", "scripts/export_findings.py")
    fake_root = tmp_path / "repo"
    (fake_root / "results" / "twin").mkdir(parents=True)
    (fake_root / "frontend" / "src" / "data").mkdir(parents=True)
    monkeypatch.setattr(mod, "ROOT", fake_root)

    def model(v):
        return {"macro_f1_mean": v, "macro_f1_ci": [v - 0.01, v + 0.01],
                "accuracy": v + 0.2, "mae": 0.6, "qwk": 0.3, "n_participants": 31}

    cmp_block = {"mean_diff": -0.047, "median_diff": -0.042, "ci_low": -0.075,
                 "ci_high": -0.020, "improved": 9, "harmed": 22, "tied": 0,
                 "n_participants": 31, "frac_improved": 0.29}
    results = [{"K": k, "models": {"T_twin": model(0.28 + k / 10000),
                                   "B1_persistence": model(0.33)},
                "twin_vs_B1_persistence": cmp_block,
                "twin_vs_B4_calibrated": cmp_block} for k in (0, 20, 80)]
    (fake_root / "results/twin/twin_experiment.json").write_text(json.dumps(
        {"results": results, "n_train": 130, "n_val": 43, "n_test": 44}),
        encoding="utf-8")
    (fake_root / "results/twin/twin_ablation.json").write_text(json.dumps(
        {"arms": {"A6_no_behaviour": {"macro_f1": 0.285}}}), encoding="utf-8")

    ceiling = {k: 0.5 for k in mod.CEILING_KEYS}
    ceiling.update({"per_person_r_iqr": [0.2, 0.5], "per_person_r_range": [-0.2, 0.7],
                    "n_participants_analysed": 194, "definition": {"x": 1}})
    (fake_root / "results/twin/ceiling.json").write_text(json.dumps(
        {"_generated_utc": "2026-01-01T00:00:00+00:00", "ceiling": ceiling,
         "cohort": {"participants": 218, "years": 4.8, "reports": 35348,
                    "prediction_pairs": 25966}}), encoding="utf-8")

    assert mod.main() == 0
    out = json.loads((fake_root / "frontend/src/data/findings.json").read_text(
        encoding="utf-8"))
    assert out["ceiling"]["n_participants_analysed"] == 194
    assert out["cohort"]["participants"] == 218
    assert out["cohort"]["val"] == 43
    assert out["_provenance"]["ceiling"] == "results/twin/ceiling.json :: ceiling"
    assert "input_digests" in out["_provenance"]
    assert "_source" not in out


def test_exporter_refuses_when_any_artefact_is_missing(tmp_path, monkeypatch):
    mod = _load("export_findings", "scripts/export_findings.py")
    fake_root = tmp_path / "repo"
    (fake_root / "results" / "twin").mkdir(parents=True)
    monkeypatch.setattr(mod, "ROOT", fake_root)
    assert mod.main() == 6
    assert not (fake_root / "frontend").exists(), (
        "a refusal must not create or touch the frontend data file")
