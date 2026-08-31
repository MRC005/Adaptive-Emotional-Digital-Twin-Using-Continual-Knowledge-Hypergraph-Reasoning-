"""Layer 1 research components: HGNN and EWC.

These test that the components are what they claim to be. The headline numbers
come from ``scripts/run_hgnn_experiment.py`` and ``scripts/run_ewc_experiment.py``
with their own protocols; what matters here is that the machinery is real:

  * the label is genuinely removed from the model's input (otherwise every
    reported metric is meaningless);
  * the hypergraph convolution actually uses the incidence structure, so it
    cannot be a renamed MLP;
  * the Fisher is non-zero and the EWC penalty grows as parameters drift away
    from the anchor, so the regulariser is not a no-op.

They skip without torch rather than passing vacuously.
"""
from __future__ import annotations

import numpy as np
import pytest

from aedt.models.hgnn import (build_tensors, mask_emotion_vertices,
                              torch_available)
from aedt.simulate.event_cohort import RULE, build_cohort_hypergraph

torch_only = pytest.mark.skipif(not torch_available(),
                                reason="torch not installed")


@pytest.fixture(scope="module")
def cohort():
    return build_cohort_hypergraph(n_people=12, n_events=10, seed=20260828)


# --------------------------------------------------------------- leakage
def test_emotion_vertices_are_removed_from_the_input(cohort):
    """If the label stays in the incidence matrix, every metric is a lie."""
    graph, _ = cohort
    vk = graph.vertex_keys
    assert any(k.startswith("Emotion=") for k in vk)      # they exist...
    H, keep = mask_emotion_vertices(graph.incidence(), vk)
    kept = [k for k, m in zip(vk, keep) if m]
    assert not any(k.startswith("Emotion=") for k in kept)  # ...and are gone
    assert H.shape[0] == len(kept)


def test_build_tensors_never_exposes_the_label(cohort):
    graph, _ = cohort
    X, H, y, meta = build_tensors(graph)
    assert not any(k.startswith("Emotion=") for k in meta["vertex_keys"])
    assert X.shape[0] == H.shape[0] == len(meta["vertex_keys"])
    assert len(y) == H.shape[1] == len(graph.edges)
    assert set(np.unique(y)).issubset(set(range(len(meta["classes"]))))


def test_simulator_rule_is_conjunctive():
    """The generator must need combinations, or the experiment tests nothing."""
    for a, b, emotion in RULE:
        assert a[0] != b[0], "a rule that depends on one field is not conjunctive"
    fields = {a[0] for a, _, _ in RULE} | {b[0] for _, b, _ in RULE}
    assert len(fields) >= 3


# ------------------------------------------------------------------ HGNN
@torch_only
def test_hypergraph_conv_actually_uses_the_incidence(cohort):
    """Change H, and the output must change. A renamed MLP would not."""
    import torch
    from aedt.models.hgnn import HGNN

    graph, _ = cohort
    X, H, y, meta = build_tensors(graph)
    torch.manual_seed(0)
    model = HGNN(X.shape[1], 8, len(meta["classes"]), H.shape[1])
    model.eval()
    Xt = torch.tensor(X, dtype=torch.float32)
    Ht = torch.tensor(H, dtype=torch.float32)
    with torch.no_grad():
        a = model(Xt, Ht)
        shuffled = Ht[torch.randperm(Ht.shape[0])]
        b = model(Xt, shuffled)
    assert not torch.allclose(a, b), "output ignored the hypergraph structure"


@torch_only
def test_all_three_models_emit_one_row_per_edge(cohort):
    import torch
    from aedt.models.hgnn import GCNBaseline, HGNN, MLPBaseline

    graph, _ = cohort
    X, H, y, meta = build_tensors(graph)
    Xt = torch.tensor(X, dtype=torch.float32)
    Ht = torch.tensor(H, dtype=torch.float32)
    n_c = len(meta["classes"])
    for model in (HGNN(X.shape[1], 8, n_c, H.shape[1]),
                  GCNBaseline(X.shape[1], 8, n_c),
                  MLPBaseline(X.shape[1], 8, n_c)):
        model.eval()
        with torch.no_grad():
            out = model(Xt, Ht)
        assert out.shape == (H.shape[1], n_c)


@torch_only
def test_hgnn_and_gcn_are_different_operators(cohort):
    """They must not collapse to the same computation, or the control is void."""
    import torch
    from aedt.models.hgnn import GCNBaseline, HGNN

    graph, _ = cohort
    X, H, y, meta = build_tensors(graph)
    Xt = torch.tensor(X, dtype=torch.float32)
    Ht = torch.tensor(H, dtype=torch.float32)
    torch.manual_seed(0)
    h = HGNN(X.shape[1], 8, len(meta["classes"]), H.shape[1])
    torch.manual_seed(0)
    g = GCNBaseline(X.shape[1], 8, len(meta["classes"]))
    h.eval(); g.eval()
    with torch.no_grad():
        assert not torch.allclose(h(Xt, Ht), g(Xt, Ht))


# ------------------------------------------------------------------- EWC
@torch_only
def test_fisher_is_non_zero_and_penalty_grows_with_drift(cohort):
    import torch
    import torch.nn.functional as F
    from aedt.continual.ewc import EWC
    from aedt.models.hgnn import MLPBaseline

    graph, _ = cohort
    X, H, y, meta = build_tensors(graph)
    Xt = torch.tensor(X, dtype=torch.float32)
    Ht = torch.tensor(H, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.long)

    torch.manual_seed(0)
    model = MLPBaseline(X.shape[1], 16, len(meta["classes"]))
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(20):
        opt.zero_grad(); F.cross_entropy(model(Xt, Ht), yt).backward(); opt.step()

    ewc = EWC(model, lam=1000.0)
    ewc.consolidate("t0", [((Xt, Ht), yt) for _ in range(5)], n_samples=5)

    summary = ewc.fisher_summary()["t0"]
    assert summary["all_zero"] is False, "a zero Fisher means EWC does nothing"
    assert summary["mean_abs_fisher"] > 0

    # at the anchor the penalty is zero; moving away makes it grow
    assert float(ewc.penalty().detach()) == pytest.approx(0.0, abs=1e-9)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.1)
    moved = float(ewc.penalty().detach())
    assert moved > 0
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.1)
    assert float(ewc.penalty().detach()) > moved


@torch_only
def test_fisher_can_be_restricted_to_a_subset_of_rows(cohort):
    """The index argument is what keeps the Fisher on the task's own data."""
    import torch
    from aedt.continual.ewc import EWC
    from aedt.models.hgnn import MLPBaseline

    graph, _ = cohort
    X, H, y, meta = build_tensors(graph)
    Xt = torch.tensor(X, dtype=torch.float32)
    Ht = torch.tensor(H, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.long)
    torch.manual_seed(0)
    model = MLPBaseline(X.shape[1], 16, len(meta["classes"]))

    idx = torch.arange(0, len(y) // 2)
    ewc = EWC(model, lam=1.0)
    ewc.consolidate("subset", [((Xt, Ht), yt, idx) for _ in range(3)], n_samples=3)
    assert ewc.fisher_summary()["subset"]["all_zero"] is False


def test_forgetting_metrics_match_their_definitions():
    from aedt.continual.ewc import forgetting_metrics

    # task 0 learned to 1.0 then decayed to 0.5; task 1 learned to 0.8
    m = forgetting_metrics([[1.0, 0.0], [0.5, 0.8]])
    assert m["average_accuracy"] == pytest.approx(0.65)
    assert m["forgetting"] == pytest.approx(0.5)          # 1.0 -> 0.5
    assert m["backward_transfer"] == pytest.approx(-0.5)  # negative = forgot
    assert m["final_per_task"] == [0.5, 0.8]

    # no forgetting at all
    m2 = forgetting_metrics([[1.0, 0.0], [1.0, 0.9]])
    assert m2["forgetting"] == pytest.approx(0.0)
