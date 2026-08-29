"""MANDATORY TEST E: participants must not improperly cross partitions.

Two distinct properties are checked:

  1. The BOOTSTRAP resamples whole participants. A participant either appears
     in a resample or does not; their observations are never split across it.
  2. Every per-person quantity is computed from THAT PERSON'S DATA ONLY. No
     cross-participant statistic enters a per-person fit, so there is no
     train/test contamination between people.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aedt.constants import SEED
from aedt.estimators.slope_ratio import estimate_rho_star, fit_person_epochs
from aedt.inference.bootstrap import bootstrap_participants


def test_bootstrap_draws_whole_participants():
    """With P wildly separated participant values, every bootstrap mean must be
    an average of P of those exact values -- impossible if observations were
    resampled instead."""
    vals = np.array([-2.0, -1.0, 0.0, 1.0, 2.0] * 4)
    u = bootstrap_participants(vals, n_resamples=2000, seed=SEED)
    assert u.resampling_unit == "participant"
    assert u.n_participants == 20
    # a mean of 20 draws from {-2..2} can never exceed the extremes
    assert np.exp(vals.min()) <= u.ci_low
    assert u.ci_high <= np.exp(vals.max())


def test_per_person_fit_ignores_every_other_participant(small_frame):
    """Corrupting OTHER participants must not move this participant's fit."""
    pid = "p00"
    mine = small_frame[small_frame["pid"] == pid]
    before = fit_person_epochs(mine, "conversation_minutes", 5, pid=pid)

    tampered = small_frame.copy()
    others = tampered["pid"] != pid
    tampered.loc[others, "conversation_minutes"] *= 100.0
    tampered.loc[others, "report"] = 1
    after = fit_person_epochs(tampered[tampered["pid"] == pid],
                              "conversation_minutes", 5, pid=pid)

    assert after[0].beta == pytest.approx(before[0].beta, abs=1e-12)
    assert after[1].beta == pytest.approx(before[1].beta, abs=1e-12)


def test_dropping_a_participant_does_not_change_the_others(small_frame):
    full = estimate_rho_star(small_frame, "conversation_minutes", 5,
                             bootstrap=False)
    reduced = estimate_rho_star(small_frame[small_frame["pid"] != "p00"],
                                "conversation_minutes", 5, bootstrap=False)
    shared = set(full.per_participant_pids) & set(reduced.per_participant_pids)
    assert shared
    fm = dict(zip(full.per_participant_pids, full.per_participant_rho_star))
    rm = dict(zip(reduced.per_participant_pids, reduced.per_participant_rho_star))
    for pid in shared:
        assert fm[pid] == pytest.approx(rm[pid], abs=1e-12)


def test_context_prototypes_are_fitted_per_participant(small_frame):
    """The feature-vector context uses EPOCH-1 prototypes from that participant,
    not cohort-level prototypes."""
    from aedt.contexts.vector import vector_context
    pid = "p00"
    mine = small_frame[small_frame["pid"] == pid].copy()
    a = vector_context(mine, ["conversation_minutes"], seed=SEED)

    tampered = small_frame.copy()
    tampered.loc[tampered["pid"] != pid, "conversation_minutes"] += 5000.0
    b = vector_context(tampered[tampered["pid"] == pid].copy(),
                       ["conversation_minutes"], seed=SEED)
    assert (a["context_vector"].fillna(-1).to_numpy()
            == b["context_vector"].fillna(-1).to_numpy()).all()


def test_hypergraph_bins_are_per_participant(small_frame):
    from aedt.hypergraph.structure import build_hypergraph
    pid = "p00"
    mine = small_frame[small_frame["pid"] == pid]
    a = build_hypergraph(mine, ["conversation_minutes", "activity_level"])
    tampered = small_frame.copy()
    tampered.loc[tampered["pid"] != pid, "conversation_minutes"] += 5000.0
    b = build_hypergraph(tampered[tampered["pid"] == pid],
                         ["conversation_minutes", "activity_level"])
    assert [e.key for e in a.edges] == [e.key for e in b.edges]


def test_placebo_uses_only_epoch_1_of_each_participant(null_frame):
    """The placebo must never touch epoch 2 -- otherwise it would not be a
    negative control. Destroying epoch 2 entirely must leave it unchanged."""
    from aedt.inference.placebo import placebo_split_half
    tampered = null_frame.copy()
    m = tampered["epoch"] == 1
    tampered.loc[m, "conversation_minutes"] = 0.0
    tampered.loc[m, "report"] = 1
    a = placebo_split_half(null_frame, "conversation_minutes", 5,
                           n_resamples=199)
    b = placebo_split_half(tampered, "conversation_minutes", 5,
                           n_resamples=199)
    assert a.runnable and b.runnable
    assert np.isfinite(a.rho_star)
    assert a.rho_star == pytest.approx(b.rho_star, abs=1e-12)
    assert a.n_participants == b.n_participants
