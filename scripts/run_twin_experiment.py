#!/usr/bin/env python3
"""The primary experiment: does personalisation predict future stress better?

Protocol frozen in docs/preregistration_twin_prediction.md BEFORE this file was
written. Nothing here may be adjusted in response to a result.

    python3 scripts/run_twin_experiment.py [--quick]

Writes results/twin/twin_experiment.json and a machine-generated table. No
number in the report is typed by hand.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from aedt.twin.prediction_data import (assert_no_leakage,               # noqa: E402
                                       build_prediction_frame)

SEED = 20260828
DATA = ROOT / "data" / "raw" / "college-experience"
OUT = ROOT / "results" / "twin"
K_GRID = (0, 5, 10, 20, 40, 80)          # pre-registered
CLASSES = np.array([1, 2, 3, 4, 5])


# ----------------------------------------------------------------- splits
def make_splits(data, seed: int = SEED, frac=(0.6, 0.2, 0.2)):
    """Participant-disjoint 60/20/20. Identity never becomes a feature."""
    pids = np.array(sorted(data.frame["participant_id"].unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(pids)
    n = len(pids)
    a, b = int(frac[0] * n), int((frac[0] + frac[1]) * n)
    return list(pids[:a]), list(pids[a:b]), list(pids[b:])


def split_warmup(g: pd.DataFrame, K: int):
    """First K observations are warm-up (never scored); the rest are evaluated."""
    g = g.sort_values("prediction_time")
    return g.iloc[:K], g.iloc[K:]


# ---------------------------------------------------------------- metrics
def score(y_true, y_pred) -> dict:
    from sklearn.metrics import (accuracy_score, cohen_kappa_score, f1_score,
                                 mean_absolute_error)
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "qwk": float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))
               if len(set(y_true)) > 1 and len(set(y_pred)) > 1 else float("nan"),
        "n": int(len(y_true)),
    }


def participant_bootstrap(per_person: dict, metric: str, n_boot=2000, seed=SEED):
    """Resample PARTICIPANTS, never observations. Returns (mean, lo, hi)."""
    pids = sorted(per_person)
    vals = np.array([per_person[p][metric] for p in pids], dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) < 2:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boots = [vals[rng.integers(0, len(vals), len(vals))].mean() for _ in range(n_boot)]
    return float(vals.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def paired_bootstrap(a: dict, b: dict, metric: str, n_boot=2000, seed=SEED):
    """Paired per-participant difference a-b, with a participant-clustered CI."""
    pids = sorted(set(a) & set(b))
    d = np.array([a[p][metric] - b[p][metric] for p in pids], dtype=float)
    d = d[np.isfinite(d)]
    if len(d) < 2:
        return {}
    rng = np.random.default_rng(seed)
    boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)]
    return {
        "mean_diff": float(d.mean()), "median_diff": float(np.median(d)),
        "ci_low": float(np.percentile(boots, 2.5)),
        "ci_high": float(np.percentile(boots, 97.5)),
        "improved": int((d > 0).sum()), "harmed": int((d < 0).sum()),
        "tied": int((d == 0).sum()), "n_participants": int(len(d)),
        "frac_improved": float((d > 0).mean()),
    }


# --------------------------------------------------------------- the models
def fit_global(train_df, features, quick=False):
    from sklearn.ensemble import HistGradientBoostingClassifier
    m = HistGradientBoostingClassifier(
        max_iter=120 if quick else 300, learning_rate=0.08,
        max_depth=6, l2_regularization=1.0, random_state=SEED)
    m.fit(train_df[features], train_df["target"])
    return m


CONTEXT_ONLY = None      # set at runtime: features with no personal history


def predict_baselines(model_ctx, model_full, warm, ev, features, ctx_features,
                      train_majority):
    """All baselines for one participant at one K. Returns {name: predictions}."""
    out = {}
    n = len(ev)

    # B0 population majority
    out["B0_majority"] = np.full(n, train_majority)

    # B1 persistence: the most recent observed value at prediction time
    out["B1_persistence"] = ev["current"].to_numpy()

    # B2 global model, context + behaviour only (no personal history)
    out["B2_global"] = model_ctx.predict(ev[ctx_features])

    # B3 global + STATIC personal prior from warm-up only, never updated
    if len(warm):
        prior = float(warm["target"].mean())
        shift = prior - float(model_ctx.predict(warm[ctx_features]).mean())
    else:
        shift = 0.0
    out["B3_static_prior"] = np.clip(
        np.round(out["B2_global"] + shift), 1, 5).astype(int)

    # B4 per-person calibrated global model: an ONLINE random-intercept
    # correction. The residual is updated after every observation, so this is a
    # genuinely personalised, genuinely adaptive baseline -- deliberately
    # strong, because beating a weak baseline would prove nothing.
    base = model_ctx.predict(ev[ctx_features]).astype(float)
    resid = 0.0
    seen = 0
    if len(warm):
        wb = model_ctx.predict(warm[ctx_features]).astype(float)
        r = (warm["target"].to_numpy() - wb)
        resid, seen = float(r.mean()), len(r)
    preds = []
    for i in range(n):
        preds.append(np.clip(round(base[i] + resid), 1, 5))
        # update AFTER predicting, using only the now-past observation
        seen += 1
        resid += (ev["target"].to_numpy()[i] - base[i] - resid) / seen
    out["B4_calibrated"] = np.array(preds, dtype=int)

    return out


def predict_twin(model_full, warm, ev, features, ctx_features):
    """The proposed twin: full dynamic state + online adaptation + shrinkage.

    State carried forward: the model's own history features (trajectory, EWMA,
    running mean/SD, evidence count) plus an online residual. The residual is
    SHRUNK by evidence count, so a participant with little history is pulled
    toward the global model and one with a long history is trusted more. That
    shrinkage is the only thing distinguishing this from B4, and the ablation
    reports whether it earns its place.
    """
    base = model_full.predict(ev[features]).astype(float)
    resid, seen = 0.0, 0
    if len(warm):
        wb = model_full.predict(warm[features]).astype(float)
        r = warm["target"].to_numpy() - wb
        resid, seen = float(r.mean()), len(r)
    preds = []
    y = ev["target"].to_numpy()
    for i in range(len(ev)):
        w = seen / (seen + 10.0)          # shrinkage: trust personal signal slowly
        preds.append(np.clip(round(base[i] + w * resid), 1, 5))
        seen += 1
        resid += (y[i] - base[i] - resid) / seen
    return np.array(preds, dtype=int)


# ------------------------------------------------------------------- driver
def run(K: int, data, splits, models, quick=False) -> dict:
    train_pids, val_pids, test_pids = splits
    model_ctx, model_full, ctx_features, train_majority = models
    features = data.feature_columns

    per_person: dict[str, dict[str, dict]] = {}
    rows_audit = []
    for pid, g in data.frame[data.frame.participant_id.isin(set(test_pids))].groupby(
            "participant_id"):
        warm, ev = split_warmup(g, K)
        if len(ev) < 5:                    # too little left to score
            continue
        preds = predict_baselines(model_ctx, model_full, warm, ev,
                                  features, ctx_features, train_majority)
        preds["T_twin"] = predict_twin(model_full, warm, ev, features, ctx_features)
        y = ev["target"].to_numpy()
        for name, p in preds.items():
            per_person.setdefault(name, {})[pid] = score(y, p)
        rows_audit.append({
            "participant_id": pid, "K": K, "n_warmup": int(len(warm)),
            "n_eval": int(len(ev)),
            "warmup_cutoff": (warm["prediction_time"].max().isoformat()
                              if len(warm) else None),
            "first_eval_time": ev["prediction_time"].min().isoformat(),
        })

    result = {"K": K, "n_test_participants": len(rows_audit), "models": {}}
    for name, pp in per_person.items():
        mean, lo, hi = participant_bootstrap(pp, "macro_f1")
        result["models"][name] = {
            "macro_f1_mean": mean, "macro_f1_ci": [lo, hi],
            "accuracy": float(np.mean([v["accuracy"] for v in pp.values()])),
            "mae": float(np.mean([v["mae"] for v in pp.values()])),
            "qwk": float(np.nanmean([v["qwk"] for v in pp.values()])),
            "n_participants": len(pp),
        }
    # paired comparisons against the two bars that matter
    for ref in ("B1_persistence", "B4_calibrated"):
        if ref in per_person and "T_twin" in per_person:
            result[f"twin_vs_{ref}"] = paired_bootstrap(
                per_person["T_twin"], per_person[ref], "macro_f1")
    result["_audit"] = rows_audit
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    print("Building the prediction frame...", flush=True)
    data = build_prediction_frame(DATA)
    assert_no_leakage(data.frame)
    print(f"  rows {len(data.frame):,} | participants "
          f"{data.frame.participant_id.nunique()} | features {len(data.feature_columns)}")

    train_pids, val_pids, test_pids = make_splits(data)
    assert not (set(train_pids) & set(test_pids))
    assert not (set(val_pids) & set(test_pids))
    print(f"  split: train {len(train_pids)} / val {len(val_pids)} / test {len(test_pids)}"
          f" | intersections 0")

    train_df = data.for_participants(train_pids)
    features = data.feature_columns
    # context-only = no personal-history columns
    hist = {"current", "current_social", "current_pam", "prev_1", "prev_2",
            "prev_3", "hist_mean", "hist_sd", "hist_n", "ewma", "days_since_prev"}
    ctx_features = [c for c in features if c not in hist]
    print(f"  full features {len(features)} | context-only {len(ctx_features)}")

    print("Fitting global models on TRAINING participants only...", flush=True)
    model_ctx = fit_global(train_df, ctx_features, args.quick)
    model_full = fit_global(train_df, features, args.quick)
    train_majority = int(train_df["target"].mode().iloc[0])

    models = (model_ctx, model_full, ctx_features, train_majority)
    results = []
    for K in K_GRID:
        r = run(K, data, (train_pids, val_pids, test_pids), models, args.quick)
        results.append(r)
        t = r["models"].get("T_twin", {})
        p = r["models"].get("B1_persistence", {})
        print(f"  K={K:3d}  n={r['n_test_participants']:3d}  "
              f"twin macro-F1 {t.get('macro_f1_mean', float('nan')):.4f}  "
              f"persistence {p.get('macro_f1_mean', float('nan')):.4f}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol": "docs/preregistration_twin_prediction.md",
        "seed": SEED, "K_grid": list(K_GRID),
        "primary_metric": "macro_f1",
        "n_train": len(train_pids), "n_val": len(val_pids), "n_test": len(test_pids),
        "results": results,
    }
    (OUT / "twin_experiment.json").write_text(json.dumps(payload, indent=2, default=str))

    # machine-generated table, no manual editing
    lines = ["| Model | K | Participant-disjoint | Strict future | macro-F1 | 95% CI | "
             "Median person diff | Improved | Harmed |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        for name, m in sorted(r["models"].items()):
            cmp = r.get("twin_vs_B1_persistence", {}) if name == "T_twin" else {}
            lines.append(
                f"| {name} | {r['K']} | yes | yes | {m['macro_f1_mean']:.4f} | "
                f"[{m['macro_f1_ci'][0]:.4f}, {m['macro_f1_ci'][1]:.4f}] | "
                f"{cmp.get('median_diff', float('nan')):.4f} | "
                f"{cmp.get('improved', '')} | {cmp.get('harmed', '')} |")
    (OUT / "twin_results_table.md").write_text("\n".join(lines))
    print(f"\nwritten: {OUT/'twin_experiment.json'} and twin_results_table.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
