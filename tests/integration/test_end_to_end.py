"""END-TO-END PIPELINE TEST, and the experiment EXECUTION ORDER it enforces."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from aedt.config import load_config
from aedt.constants import DataStatus
from aedt.errors import PlaceboFailed
from aedt.pipeline import run_pipeline

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def run():
    cfg = load_config("simulation",
                      overrides={"simulation.n_participants": 30,
                                 "simulation.n_per_epoch": 200})
    return run_pipeline("synthetic", config=cfg, n_resamples=399,
                        twin_pids=["p00", "p01"])


def test_pipeline_completes_and_produces_every_stage(run):
    assert run.frame is not None and len(run.frame) > 0
    assert run.ledger is not None and run.ledger.n_output > 0
    assert run.epochs and len(run.epochs) == 30
    assert run.category_usage is not None and len(run.category_usage) == 60
    assert run.association is not None
    assert run.eligibility and len(run.eligibility) == 30
    assert run.placebo is not None
    assert run.primary is not None
    assert run.twins and set(run.twins) == {"p00", "p01"}
    assert run.hypergraphs


def test_everything_carries_the_same_synthetic_stamp(run):
    assert run.data_status is DataStatus.SYNTHETIC
    assert run.primary.data_status is DataStatus.SYNTHETIC
    assert run.placebo.data_status is DataStatus.SYNTHETIC
    assert run.association.data_status is DataStatus.SYNTHETIC
    assert all(e.data_status is DataStatus.SYNTHETIC for e in run.eligibility)
    assert all(t.state.data_status is DataStatus.SYNTHETIC
               for t in run.twins.values())
    assert not run.validated, "SYNTHETIC output must never be called validated"


def test_the_primary_estimand_is_rho_star_and_nothing_else(run):
    assert run.primary.estimand == "rho_star"
    assert run.primary.additive_component is None
    assert run.primary.additive_component_status == "NOT IDENTIFIED"
    assert 0.5 < run.primary.rho_star < 1.5


def test_uncertainty_is_clustered_on_participants(run):
    u = run.primary.uncertainty
    assert u.resampling_unit == "participant"
    assert u.n_participants == run.primary.n_participants_used
    assert u.ci_low < u.point < u.ci_high


def test_hypergraph_is_built_with_genuine_conjunctions(run):
    hg = run.hypergraphs["p00"]
    assert hg.n_edges > 1
    assert hg.mean_arity() > 1.0, (
        "a one-factor 'hyperedge' is just a bin; the ablation would be "
        "meaningless")
    assert hg.n_vertices >= hg.mean_arity()


def test_twins_persist_and_reload(run, tmp_path):
    from aedt.twin.state import PersonalDigitalTwin
    t = run.twins["p00"]
    p = t.save(tmp_path / "p00.json")
    back = PersonalDigitalTwin.load(p)
    assert back.state.history[-1]["verdict"] in ("ACCEPTED",
                                                 "FLAGGED-UNTRUSTWORTHY")
    assert back.state.n_observations_seen == t.state.n_observations_seen


def test_result_serialises_to_json(run):
    d = run.to_dict()
    json.loads(json.dumps(d, default=str))
    assert d["data_status"] == "SYNTHETIC"
    assert d["validated"] is False
    assert "primary" in d and "placebo" in d


def test_a_failing_placebo_blocks_the_primary(monkeypatch):
    """The gate must actually gate: when the placebo rejects, no primary
    estimate is produced at all."""
    import aedt.pipeline as pl
    from aedt.schemas import PlaceboResult

    def fired(*a, **k):
        return PlaceboResult(n_participants=30, rho_star=0.80, ci_low=0.70,
                             ci_high=0.90, rejected=True,
                             verdict="REJECTS (injected)",
                             data_status=DataStatus.SYNTHETIC)

    monkeypatch.setattr(pl, "placebo_split_half", fired)
    cfg = load_config("simulation",
                      overrides={"simulation.n_participants": 12,
                                 "simulation.n_per_epoch": 150})
    with pytest.raises(PlaceboFailed):
        run_pipeline("synthetic", config=cfg, n_resamples=99)

    res = run_pipeline("synthetic", config=cfg, n_resamples=99,
                       halt_on_placebo_failure=False)
    assert res.primary is None, "the primary ran despite a failed placebo"
    assert any("PLACEBO FAILED" in r for r in res.blocking_reasons)
    assert not res.validated


def test_a_weak_association_is_recorded_as_blocking(monkeypatch):
    """[9b] is read before the placebo and its failure is carried forward."""
    import aedt.pipeline as pl
    from aedt.audit.diagnostics import AssociationStrength

    def weak(*a, **k):
        return AssociationStrength(10, 12, 0.03, 0.01, 0.05, 0.9, True,
                                   "WEAK", DataStatus.SYNTHETIC)

    monkeypatch.setattr(pl, "association_strength", weak)
    cfg = load_config("simulation",
                      overrides={"simulation.n_participants": 12,
                                 "simulation.n_per_epoch": 150})
    res = run_pipeline("synthetic", config=cfg, n_resamples=99,
                       halt_on_placebo_failure=False, build_twins=False)
    assert any("WEAK SENSOR-REPORT ASSOCIATION" in r
               for r in res.blocking_reasons)


def test_demo_script_runs_end_to_end_and_writes_artefacts(tmp_path):
    r = subprocess.run(
        [sys.executable, "scripts/run_demo.py", "--dataset", "synthetic",
         "--participant", "p07", "--bootstrap", "199", "--quiet",
         "--out", str(tmp_path)],
        capture_output=True, text=True, cwd=ROOT, timeout=1200)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    for stage in ("STAGE 1", "STAGE 2", "STAGE 3", "STAGE 4", "STAGE 5",
                  "STAGE 6", "STAGE 7", "STAGE 8", "STAGE 9"):
        assert stage in r.stdout, f"{stage} missing from the demo output"
    assert "MONEY SHOT" in r.stdout
    assert "rho ITSELF IS NOT POINT-IDENTIFIED" in r.stdout
    assert "NOT IDENTIFIED and is NOT ESTIMATED" in r.stdout
    assert "EVERY FIGURE AND TABLE ABOVE IS STAMPED SYNTHETIC" in r.stdout

    runs = sorted(tmp_path.glob("*_synthetic_SYNTHETIC"))
    assert runs, "no timestamped, status-stamped run directory was created"
    out = runs[-1]
    figs = list((out / "figures").glob("*.png"))
    tabs = list((out / "tables").glob("*.csv"))
    assert len(figs) >= 8, [f.name for f in figs]
    assert len(tabs) >= 10, [t.name for t in tabs]
    assert (out / "figures" / "fig01_two_curve_epoch1_vs_epoch2.png").exists()
    assert (out / "run_metadata.json").exists()
    assert (out / "resolved_config.json").exists()
    meta = json.loads((out / "run_metadata.json").read_text())
    assert meta["seed"] == 20260828
    assert meta["data_status"] == "SYNTHETIC"
    for t in tabs:
        assert "data_status" in t.read_text().splitlines()[0], t.name
