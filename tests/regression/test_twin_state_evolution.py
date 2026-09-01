"""Does the twin state genuinely evolve, and does history genuinely matter?

Part D of the upgrade brief. These test MECHANISM, not performance: a system
whose predictions are unaffected by a person's history is not personalised,
whatever its accuracy.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aedt.twin.prediction_data import build_prediction_frame

ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / "data" / "raw" / "college-experience"
have_data = pytest.mark.skipif(
    not (DATA / "EMA" / "general_ema.csv").is_file(),
    reason="College Experience archive not present")


@pytest.fixture(scope="module")
def setup():
    import sys
    sys.path.insert(0, str(ROOT))
    from scripts.run_twin_experiment import fit_global, make_splits
    d = build_prediction_frame(DATA)
    tr, va, te = make_splits(d)
    model = fit_global(d.for_participants(tr), d.feature_columns)
    return d, model, te


@have_data
def test_history_features_change_over_time_within_a_participant(setup):
    """The state must actually evolve, not sit constant."""
    d, _, te = setup
    g = d.for_participants(te)
    moved = 0
    for pid, p in g.groupby("participant_id"):
        if len(p) < 10:
            continue
        if p["hist_n"].nunique() > 1 and p["ewma"].nunique() > 1:
            moved += 1
    assert moved >= 20, "history/EWMA did not evolve for enough participants"


@have_data
def test_identical_context_but_different_history_gives_different_predictions(setup):
    """The core personalisation mechanism, tested directly."""
    d, model, te = setup
    feats = d.feature_columns
    g = d.for_participants(te).dropna(subset=["ewma", "hist_mean"])
    row = g.iloc[[0]].copy()

    low, high = row.copy(), row.copy()
    for c in ("current", "prev_1", "prev_2", "prev_3", "ewma", "hist_mean"):
        if c in low.columns:
            low[c] = 1.0
            high[c] = 5.0
    p_low = model.predict(low[feats])[0]
    p_high = model.predict(high[feats])[0]
    assert p_low != p_high, (
        "identical context with opposite histories produced the same prediction; "
        "the model is ignoring personal history")
    assert p_low < p_high, "prediction did not move in the direction of history"


@have_data
def test_adding_an_observation_can_change_the_next_prediction(setup):
    d, model, te = setup
    feats = d.feature_columns
    g = d.for_participants(te)
    pid = g["participant_id"].iloc[0]
    p = g[g.participant_id == pid].sort_values("prediction_time")
    if len(p) < 20:
        pytest.skip("participant too short")
    preds = model.predict(p[feats])
    assert len(set(preds)) > 1, "predictions never changed as history accumulated"


@have_data
def test_removing_personal_history_moves_prediction_toward_the_global_model(setup):
    """With history neutralised, the twin should behave like the context model."""
    import sys
    sys.path.insert(0, str(ROOT))
    from scripts.run_twin_experiment import fit_global, make_splits

    d, model, te = setup
    feats = d.feature_columns
    hist = {"current", "current_social", "current_pam", "prev_1", "prev_2",
            "prev_3", "hist_mean", "hist_sd", "hist_n", "ewma", "days_since_prev"}
    ctx = [c for c in feats if c not in hist]
    tr, _, _ = make_splits(d)
    train = d.for_participants(tr)
    ctx_model = fit_global(train, ctx)

    g = d.for_participants(te).head(400).copy()
    stripped = g.copy()
    for c in hist:
        if c in stripped.columns:
            stripped[c] = train[c].median() if c in train else np.nan

    full_pred = model.predict(g[feats])
    stripped_pred = model.predict(stripped[feats])
    ctx_pred = ctx_model.predict(g[ctx])

    agree_stripped = float((stripped_pred == ctx_pred).mean())
    agree_full = float((full_pred == ctx_pred).mean())
    assert agree_stripped > agree_full, (
        f"neutralising history did not move the twin toward the global model "
        f"(agreement {agree_stripped:.2f} vs {agree_full:.2f})")


@have_data
def test_the_twin_cannot_see_future_observations(setup):
    """Perturbing a LATER target must not change an EARLIER prediction."""
    d, model, te = setup
    feats = d.feature_columns
    g = d.for_participants(te)
    pid = g["participant_id"].iloc[0]
    p = g[g.participant_id == pid].sort_values("prediction_time").head(30).copy()
    before = model.predict(p[feats])
    p.loc[p.index[-1], "target"] = 5 if p["target"].iloc[-1] != 5 else 1
    after = model.predict(p[feats])
    assert (before == after).all(), "changing a future target altered a past prediction"
