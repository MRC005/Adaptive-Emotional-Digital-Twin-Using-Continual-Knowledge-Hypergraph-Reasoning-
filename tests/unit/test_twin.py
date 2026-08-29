"""TwinState, persistence, and the continual knowledge store."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from aedt.audit.eligibility import screen_participant
from aedt.constants import DataStatus
from aedt.errors import DecisionRequired
from aedt.knowledge.store import KNOWLEDGE_KINDS, ContinualKnowledgeStore
from aedt.twin.state import PersonalDigitalTwin, new_twin
from aedt.twin.update import (MIN_HYPEREDGE_OVERLAP, TwinVerdict, close_epoch,
                              observe)


def _feed(twin, g, sensor="conversation_minutes", K=5):
    for _i, row in g.sort_values("ts").iterrows():
        observe(twin, row, sensor=sensor, n_categories=K)
    return twin


def test_twin_round_trips_through_json(tmp_path, small_frame):
    g = small_frame[small_frame["pid"] == "p00"]
    twin = _feed(new_twin("p00", "synthetic", DataStatus.SYNTHETIC), g)
    elig = screen_participant(g, "conversation_minutes", 5, pid="p00")
    close_epoch(twin, g, sensor="conversation_minutes", n_categories=5,
                eligibility=elig)

    p = twin.save(tmp_path / "p00.json")
    back = PersonalDigitalTwin.load(p)
    assert back.state.pid == twin.state.pid
    assert back.state.n_observations_seen == twin.state.n_observations_seen
    assert back.state.current_time == twin.state.current_time
    assert back.state.category_usage == twin.state.category_usage
    assert len(back.state.history) == len(twin.state.history)
    assert len(back.knowledge.nodes) == len(twin.knowledge.nodes)
    assert back.state.data_status is DataStatus.SYNTHETIC


def test_history_is_append_only(small_frame):
    g = small_frame[small_frame["pid"] == "p00"]
    twin = _feed(new_twin("p00", "synthetic", DataStatus.SYNTHETIC), g)
    elig = screen_participant(g, "conversation_minutes", 5, pid="p00")
    close_epoch(twin, g, sensor="conversation_minutes", n_categories=5,
                eligibility=elig)
    first = json.dumps(twin.state.history[0], default=str)
    close_epoch(twin, g, sensor="conversation_minutes", n_categories=5,
                eligibility=elig)
    assert len(twin.state.history) == 2
    assert json.dumps(twin.state.history[0], default=str) == first


def test_twin_gains_a_history_row_with_a_verdict(small_frame):
    g = small_frame[small_frame["pid"] == "p00"]
    twin = _feed(new_twin("p00", "synthetic", DataStatus.SYNTHETIC), g)
    elig = screen_participant(g, "conversation_minutes", 5, pid="p00")
    out = close_epoch(twin, g, sensor="conversation_minutes", n_categories=5,
                      eligibility=elig)
    assert out.verdict in (TwinVerdict.ACCEPTED,
                           TwinVerdict.FLAGGED_UNTRUSTWORTHY)
    assert twin.state.history[-1]["verdict"] == out.verdict.value
    assert "boundary_rate" in twin.state.audit_flags


def test_ineligible_participant_is_flagged_and_calibration_unchanged(small_frame):
    g = small_frame[small_frame["pid"] == "p00"].copy()
    twin = _feed(new_twin("p00", "synthetic", DataStatus.SYNTHETIC), g)
    bad = screen_participant(g.head(20), "conversation_minutes", 5, pid="p00")
    assert not bad.eligible
    out = close_epoch(twin, g, sensor="conversation_minutes", n_categories=5,
                      eligibility=bad)
    assert out.verdict is TwinVerdict.FLAGGED_UNTRUSTWORTHY
    assert not out.calibration_state_changed
    assert out.reasons


def test_failed_placebo_flags_the_twin(small_frame):
    from aedt.schemas import PlaceboResult
    g = small_frame[small_frame["pid"] == "p00"]
    twin = _feed(new_twin("p00", "synthetic", DataStatus.SYNTHETIC), g)
    elig = screen_participant(g, "conversation_minutes", 5, pid="p00")
    fired = PlaceboResult(n_participants=40, rho_star=0.8, ci_low=0.7,
                          ci_high=0.9, rejected=True, verdict="REJECTS")
    out = close_epoch(twin, g, sensor="conversation_minutes", n_categories=5,
                      eligibility=elig, placebo=fired)
    assert out.verdict is TwinVerdict.FLAGGED_UNTRUSTWORTHY
    assert any("placebo gate" in r for r in out.reasons)


def test_twin_reasons_over_hyperedge_overlap(small_frame, monkeypatch):
    """The one place the twin actually uses the higher-order structure: a low
    epoch-to-epoch context overlap flags the update as untrustworthy."""
    import aedt.hypergraph.structure as hgmod

    class FakeHG:
        n_edges = 4

        def occupancy_overlap(self):
            return 0.05                    # far below MIN_HYPEREDGE_OVERLAP

        def summary(self):
            return {"pid": "p00", "n_hyperedges": 4, "occupancy_overlap": 0.05}

    # close_epoch imports build_hypergraph lazily from this module, so
    # patching it here is what the running code will actually resolve.
    monkeypatch.setattr(hgmod, "build_hypergraph", lambda *a, **k: FakeHG())
    g = small_frame[small_frame["pid"] == "p00"]
    twin = _feed(new_twin("p00", "synthetic", DataStatus.SYNTHETIC), g)
    elig = screen_participant(g, "conversation_minutes", 5, pid="p00")
    out = close_epoch(twin, g, sensor="conversation_minutes", n_categories=5,
                      eligibility=elig)
    assert out.verdict is TwinVerdict.FLAGGED_UNTRUSTWORTHY
    assert any("occupancy overlap" in r for r in out.reasons)
    assert MIN_HYPEREDGE_OVERLAP == 0.20


def test_observe_refuses_an_out_of_range_report():
    twin = new_twin("a", "synthetic", DataStatus.SYNTHETIC)
    row = pd.Series({"ts": pd.Timestamp("2026-01-01"), "report": 9, "epoch": 0,
                     "conversation_minutes": 30.0})
    with pytest.raises(DecisionRequired, match="outside 1..5"):
        observe(twin, row, sensor="conversation_minutes", n_categories=5)


def test_observe_refuses_a_missing_report():
    twin = new_twin("a", "synthetic", DataStatus.SYNTHETIC)
    row = pd.Series({"ts": pd.Timestamp("2026-01-01"), "report": np.nan,
                     "epoch": 0, "conversation_minutes": 30.0})
    with pytest.raises(DecisionRequired, match="never imputed"):
        observe(twin, row, sensor="conversation_minutes", n_categories=5)


# ------------------------------------------------------ knowledge store
def test_exactly_four_knowledge_kinds_update():
    assert set(KNOWLEDGE_KINDS) == {
        "personalised_parameters", "state_history", "context_relationship",
        "uncertainty_audit"}


def test_unknown_knowledge_kind_is_refused():
    s = ContinualKnowledgeStore(pid="a")
    with pytest.raises(DecisionRequired, match="Exactly four things update"):
        s.append("replay_buffer", {}, valid_from=pd.Timestamp("2026-01-01"),
                 provenance="t")


def test_supersession_preserves_the_superseded_node():
    s = ContinualKnowledgeStore(pid="a")
    old = s.append("personalised_parameters", {"beta": 1.0},
                   valid_from=pd.Timestamp("2026-01-01"), provenance="fit1")
    new = s.append("personalised_parameters", {"beta": 2.0},
                   valid_from=pd.Timestamp("2026-02-01"), provenance="fit2",
                   supersedes=old.node_id)
    kept = next(n for n in s.nodes if n.node_id == old.node_id)
    assert kept.payload == {"beta": 1.0}, "history was overwritten"
    assert kept.superseded_by == new.node_id
    assert s.latest("personalised_parameters").payload == {"beta": 2.0}
    assert any(r.kind == "supersedes" for r in s.relations)


def test_temporal_inspection_as_of():
    s = ContinualKnowledgeStore(pid="a")
    s.append("state_history", {"n": 1}, valid_from=pd.Timestamp("2026-01-01"),
             provenance="t")
    s.append("state_history", {"n": 2}, valid_from=pd.Timestamp("2026-03-01"),
             provenance="t")
    assert len(s.as_of(pd.Timestamp("2026-02-01"))) == 1
    assert len(s.as_of(pd.Timestamp("2026-04-01"))) == 2


def test_knowledge_store_round_trips(tmp_path):
    s = ContinualKnowledgeStore(pid="a")
    s.append("uncertainty_audit", {"rho_star": 0.9},
             valid_from=pd.Timestamp("2026-01-01"), provenance="t")
    p = s.save(tmp_path / "k.json")
    back = ContinualKnowledgeStore.load(p)
    assert len(back.nodes) == 1
    assert back.nodes[0].payload == {"rho_star": 0.9}
    assert back.current_time == s.current_time
