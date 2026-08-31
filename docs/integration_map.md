# Review 1 + Review 2 integration map

Written **before** implementation, from an inspection of the repository as it
stood. It records what was already there, what was missing, and what was
therefore built. Anything that could not be built genuinely is marked
`NOT IMPLEMENTED` with the reason, here and in the README.

## What the inspection found

| Review 1 component | State before this pass | Evidence |
|---|---|---|
| Transformer emotion detection | **absent** | no NLP code anywhere in `aedt/` |
| Context extraction | **absent** | — |
| Structured emotional event | **absent** (Review 2 has `LongFrame`, a different thing) | `aedt/schemas.py` |
| Knowledge Hypergraph | **present, but over sensor features** | `aedt/hypergraph/structure.py` builds hyperedges from discretised sensor bins for the Review 2 ablation |
| HGNN | **absent** | no neural code; the "hypergraph-native estimator" in Ablation 1 is a statistical estimator, not a network |
| Continual learning / EWC | **absent, and explicitly disclaimed** | `aedt/knowledge/store.py`: *"NO CONTINUAL-LEARNING ALGORITHM IS IMPORTED. No replay buffer, no EWC, no rehearsal."* |
| Personal Digital Twin | **present, skeleton** | `aedt/twin/state.py`: append-only history, monotone clock, update log |
| Similar-episode retrieval | **absent** | — |
| Pattern forecast | **absent** | — |

Review 2 was found intact and passing: 285 tests, four audited datasets, the
College Experience analysis, guided controls, sandbox, CSV upload.

## Environment constraint that shaped the design

`torch` and `transformers` were **not installed**; `numpy`, `scipy`,
`sklearn`, `pandas` were. Network and pip were available, so both were
installed rather than simulating the missing pieces in numpy. This is the
reason the HGNN and EWC work is real rather than approximated.

The deployed frontend is static and client-side. A Transformer cannot run
there, so emotion detection runs in Python. The browser calls the backend when
one is configured, and otherwise falls back to a lexicon baseline that is
**labelled as a baseline in the interface**, never as the model.

## Where the two layers meet

They are not merged, because they answer different questions on different
data. They are connected at exactly one point, and the interface says so:

- **Layer 1** learns, for one person, which contexts have accompanied which
  reported emotions.
- **Layer 2** asks whether a behaviour-to-report relationship stays comparable
  across time in a *cohort*, using dense longitudinal data.

The connection is a **trust qualifier, not a data path**. Layer 2's finding
governs how far a Layer 1 history may be extrapolated: if a cohort's
measurement relationship drifts, a personal history spanning the same period
should not be read as if the scale meant one fixed thing throughout. Layer 2
never sees a personal chat, and Layer 1 never contributes to `rho*`.

**What is deliberately NOT claimed:** that a personal check-in history has
enough repeated observations to run the Review 2 estimator on one individual.
It does not, and the application says so where a user might expect otherwise.
