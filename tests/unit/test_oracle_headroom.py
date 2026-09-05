"""The oracle is an upper bound. If it ever isn't, the number is wrong.

This guards the one figure the headroom read-out reports. The project's own
history is the reason: the ceiling statistics went unchecked for weeks because
nothing asserted anything about them.

The properties below are exact, not statistical, so they can be planted and
required back rather than eyeballed.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

spec = importlib.util.spec_from_file_location(
    "run_oracle_headroom", ROOT / "scripts" / "run_oracle_headroom.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def make(pairs):
    """{pid: {twin, persistence}} from a list of (twin_f1, persistence_f1)."""
    return {f"p{i}": {"n_warmup": 20, "n_eval": 30,
                      "twin": {"macro_f1": t},
                      "persistence": {"macro_f1": q}}
            for i, (t, q) in enumerate(pairs)}


def test_the_oracle_is_never_worse_than_either_strategy_it_chooses_between():
    rng = np.random.default_rng(7)
    pp = make([(float(rng.uniform(0, 1)), float(rng.uniform(0, 1)))
               for _ in range(60)])
    r = mod.headroom(pp, n_boot=200)
    assert r["R5_oracle"]["macro_f1"] >= r["R0_persistence"]["macro_f1"] - 1e-12
    assert r["R5_oracle"]["macro_f1"] >= r["R1_twin"]["macro_f1"] - 1e-12
    assert r["headroom"]["mean"] >= 0.0


def test_headroom_is_zero_when_the_twin_never_wins():
    pp = make([(0.2, 0.3), (0.1, 0.4), (0.25, 0.25)])
    r = mod.headroom(pp, n_boot=200)
    assert r["headroom"]["mean"] == pytest.approx(0.0, abs=1e-12), (
        "a twin that never wins leaves a perfect router nothing to gain")
    assert r["twin_wins"]["n"] == 0
    assert r["R5_oracle"]["macro_f1"] == pytest.approx(
        r["R0_persistence"]["macro_f1"])


def test_headroom_equals_the_planted_value():
    # advantages: +0.10, -0.20, +0.30, -0.05  ->  positive part mean = 0.10
    pp = make([(0.50, 0.40), (0.20, 0.40), (0.70, 0.40), (0.35, 0.40)])
    r = mod.headroom(pp, n_boot=200)
    assert r["headroom"]["mean"] == pytest.approx(0.40 / 4)
    assert r["twin_wins"]["n"] == 2
    assert r["twin_minus_persistence"]["range"] == [
        pytest.approx(-0.20), pytest.approx(0.30)]


def test_headroom_equals_the_oracle_minus_persistence_identity():
    """mean[max(0, t-q)] must equal mean[max(t,q)] - mean[q], exactly."""
    rng = np.random.default_rng(11)
    pp = make([(float(rng.uniform(0, 1)), float(rng.uniform(0, 1)))
               for _ in range(80)])
    r = mod.headroom(pp, n_boot=200)
    identity = r["R5_oracle"]["macro_f1"] - r["R0_persistence"]["macro_f1"]
    assert r["headroom"]["mean"] == pytest.approx(identity, abs=1e-12)


def test_concentration_reports_how_few_participants_carry_the_gain():
    # one participant carries everything
    pp = make([(0.9, 0.4)] + [(0.1, 0.4)] * 19)
    r = mod.headroom(pp, n_boot=200)
    assert r["gain_concentration"]["top_1_share"] == pytest.approx(1.0)


def test_concentration_is_none_rather_than_a_divide_by_zero_when_there_is_no_gain():
    pp = make([(0.1, 0.4), (0.2, 0.5)])
    r = mod.headroom(pp, n_boot=200)
    assert r["gain_concentration"]["top_1_share"] is None


def test_the_bootstrap_resamples_participants_not_observations():
    """The CI must widen when the cohort is small, as a participant CI does."""
    rng = np.random.default_rng(3)
    small = make([(float(rng.uniform(0, 1)), float(rng.uniform(0, 1)))
                  for _ in range(8)])
    large = make([(float(rng.uniform(0, 1)), float(rng.uniform(0, 1)))
                  for _ in range(200)])
    ws = np.diff(mod.headroom(small, n_boot=500)["headroom"]["ci"])[0]
    wl = np.diff(mod.headroom(large, n_boot=500)["headroom"]["ci"])[0]
    assert ws > wl, "the interval does not respond to cohort size"
