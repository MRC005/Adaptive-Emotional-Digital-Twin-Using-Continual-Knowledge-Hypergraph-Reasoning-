"""LAYER 1 / MODULE 5 -- Hypergraph Neural Network.

Purpose  Learn representations over the event hypergraph and predict the
         emotion of a new contextual event.
Input    Node features X, incidence H.
Output   Per-edge emotion logits.
Status   RESEARCH COMPONENT, trained offline. Not in the browser.

THE LAYER

This is the Feng et al. (2019) hypergraph convolution, implemented directly:

    X' = sigma( Dv^-1/2 H W De^-1 H^T Dv^-1/2 X Theta )

with Dv the vertex-degree diagonal, De the edge-degree diagonal, W the
(learned, diagonal) edge weights and Theta the learned linear map. The two
H multiplications are the whole point: information travels vertex -> edge ->
vertex, so a vertex is updated by everything that co-occurred with it *in the
same episode*, not by a pairwise neighbour list. That is what makes it a
hypergraph network rather than a GCN over a clique expansion.

WHAT WOULD MAKE THIS FAKE, AND IS NOT DONE HERE

  - cosine similarity between context vectors, called "neural reasoning";
  - a rule table consulted at inference and described as a learned model;
  - a GCN on a pairwise projection, called an HGNN.

The clique-expansion GCN IS implemented, deliberately, as ``GCNBaseline`` in
this file: it is the honest control that tells you whether the higher-order
structure earns its place. If the HGNN does not beat it, the evaluation says
so, and ``scripts/run_hgnn_experiment.py`` prints both either way.

EDGE-LEVEL PREDICTION. The task is "what emotion does this episode carry", so
predictions are made per hyperedge. Edge representations are the mean of the
member vertices' representations, with the emotion vertex REMOVED from the
input features -- otherwise the label is in the input and the metric is
meaningless. ``mask_emotion_vertices`` enforces that, and a test asserts it.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)

__all__ = ["HGNN", "GCNBaseline", "MLPBaseline", "build_tensors",
           "mask_emotion_vertices", "torch_available"]


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def mask_emotion_vertices(H: np.ndarray, vertex_keys: list[str]
                          ) -> tuple[np.ndarray, np.ndarray]:
    """Remove Emotion vertices from the structure used as INPUT.

    The emotion is the prediction target. If its vertex stays in the incidence
    matrix, every edge carries its own answer and any model scores ~1.0. This
    returns (H_without_emotion_rows, boolean mask of kept rows).
    """
    keep = np.array([not k.startswith("Emotion=") for k in vertex_keys], dtype=bool)
    return H[keep, :], keep


def build_tensors(graph, label_field: str = "emotion"):
    """Turn an ``EventHypergraph`` into (X, H, y, meta) for training.

    Vertex features are one-hot over vertex IDENTITY plus a normalised degree.

    Identity rather than type: with type features every vertex of a type is
    indistinguishable, so every edge's mean-of-members is the same vector and
    all three models collapse to the majority class. (That was observed before
    this was fixed, and it is exactly the sort of silent failure that makes a
    model comparison meaningless.) Identity features are the standard choice
    here and give each model the same information; what differs between them is
    only how that information is propagated.
    """
    vk = graph.vertex_keys
    H_full = graph.incidence()
    H, keep = mask_emotion_vertices(H_full, vk)
    kept_keys = [k for k, m in zip(vk, keep) if m]

    n_v = len(kept_keys)
    deg = H.sum(axis=1)
    X = np.zeros((n_v, n_v + 1), dtype=np.float32)
    for i in range(n_v):
        X[i, i] = 1.0                                   # identity
        X[i, -1] = deg[i] / max(deg.max(), 1.0)         # normalised degree

    labels, y = [], []
    for e in graph.edges:
        emo = next((v.split("=", 1)[1] for v in e.vertices
                    if v.startswith("Emotion=")), None)
        labels.append(emo)
    classes = sorted({l for l in labels if l is not None})
    cindex = {c: i for i, c in enumerate(classes)}
    y = np.array([cindex.get(l, -1) for l in labels], dtype=np.int64)

    meta = {"vertex_keys": kept_keys, "classes": classes,
            "n_vertices": len(kept_keys), "n_edges": H.shape[1],
            "edge_ids": [e.edge_id for e in graph.edges]}
    return X, H, y, meta


# --------------------------------------------------------------------------
# torch models. Imported lazily so the package still imports without torch.
# --------------------------------------------------------------------------
def _torch_modules():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class HGNNConv(nn.Module):
        """One Feng et al. hypergraph convolution with learnable edge weights."""

        def __init__(self, d_in: int, d_out: int, n_edges: int):
            super().__init__()
            self.theta = nn.Linear(d_in, d_out)
            # A vertex must keep its own identity alongside the propagated
            # signal. The entity universe here is small and categorical, so
            # every vertex co-occurs with nearly every other and two rounds of
            # pure propagation drive all representations to a constant -- which
            # is what happened before this term was added: HGNN and GCN both
            # collapsed to the majority class while the structure-free MLP
            # learned. The same term is given to the GCN control, so the only
            # difference that remains between them is the propagation operator.
            self.theta_self = nn.Linear(d_in, d_out)
            # diagonal W, learned, initialised at 1
            self.edge_weight = nn.Parameter(torch.ones(n_edges))

        def forward(self, X, H):
            W = torch.diag(self.edge_weight)
            Dv = H.mul(self.edge_weight).sum(dim=1).clamp(min=1e-6)
            De = H.sum(dim=0).clamp(min=1e-6)
            Dv_inv_sqrt = torch.diag(Dv.pow(-0.5))
            De_inv = torch.diag(De.pow(-1.0))
            # Dv^-1/2 H W De^-1 H^T Dv^-1/2 X Theta
            A = Dv_inv_sqrt @ H @ W @ De_inv @ H.t() @ Dv_inv_sqrt
            return self.theta(A @ X) + self.theta_self(X)

    class HGNNNet(nn.Module):
        """Two hypergraph convolutions, then an edge-level classifier."""

        def __init__(self, d_in: int, d_hidden: int, n_classes: int,
                     n_edges: int, dropout: float = 0.3):
            super().__init__()
            self.c1 = HGNNConv(d_in, d_hidden, n_edges)
            self.c2 = HGNNConv(d_hidden, d_hidden, n_edges)
            self.head = nn.Linear(d_hidden, n_classes)
            self.dropout = dropout

        def forward(self, X, H):
            h = F.relu(self.c1(X, H))
            h = F.dropout(h, self.dropout, self.training)
            h = F.relu(self.c2(h, H))
            # edge representation = mean of its member vertices
            De = H.sum(dim=0).clamp(min=1e-6)
            edge_h = (H.t() @ h) / De.unsqueeze(1)
            return self.head(edge_h)

    class GCNNet(nn.Module):
        """Clique-expansion GCN: the pairwise control.

        The hypergraph is flattened to a graph by connecting every pair of
        vertices that share an edge. This keeps co-occurrence but destroys the
        record of WHICH episode produced it, which is exactly the information
        the hyperedge preserves.
        """

        def __init__(self, d_in: int, d_hidden: int, n_classes: int,
                     dropout: float = 0.3):
            super().__init__()
            self.l1 = nn.Linear(d_in, d_hidden)
            self.l1_self = nn.Linear(d_in, d_hidden)
            self.l2 = nn.Linear(d_hidden, d_hidden)
            self.l2_self = nn.Linear(d_hidden, d_hidden)
            self.head = nn.Linear(d_hidden, n_classes)
            self.dropout = dropout

        def forward(self, X, H):
            A = (H @ H.t() > 0).float()
            A.fill_diagonal_(1.0)
            d = A.sum(dim=1).clamp(min=1e-6).pow(-0.5)
            A = torch.diag(d) @ A @ torch.diag(d)
            h = F.relu(self.l1(A @ X) + self.l1_self(X))
            h = F.dropout(h, self.dropout, self.training)
            h = F.relu(self.l2(A @ h) + self.l2_self(h))
            De = H.sum(dim=0).clamp(min=1e-6)
            edge_h = (H.t() @ h) / De.unsqueeze(1)
            return self.head(edge_h)

    class MLPNet(nn.Module):
        """No structure at all: the floor. Edge features are a bag of vertices."""

        def __init__(self, d_in: int, d_hidden: int, n_classes: int,
                     dropout: float = 0.3):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(d_in, d_hidden), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(d_hidden, d_hidden), nn.ReLU(),
                nn.Linear(d_hidden, n_classes))

        def forward(self, X, H):
            De = H.sum(dim=0).clamp(min=1e-6)
            edge_x = (H.t() @ X) / De.unsqueeze(1)
            return self.net(edge_x)

    return HGNNNet, GCNNet, MLPNet


def HGNN(*a, **kw):
    return _torch_modules()[0](*a, **kw)


def GCNBaseline(*a, **kw):
    return _torch_modules()[1](*a, **kw)


def MLPBaseline(*a, **kw):
    return _torch_modules()[2](*a, **kw)
