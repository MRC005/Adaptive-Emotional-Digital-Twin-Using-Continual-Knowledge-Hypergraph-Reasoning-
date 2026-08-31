#!/usr/bin/env python3
"""Evaluate the emotion model on the GoEmotions held-out TEST split.

PROTOCOL
--------
data        GoEmotions "simplified" config, official `test` split (5,427
            examples). The checkpoint's author trained on the official `train`
            split, so `test` is genuinely held out from it and from us.
model       SamLowe/roberta-base-go_emotions (RoBERTa-base, 124.7M params,
            28-way multi-label sigmoid head). NOT trained by this project.
task        multi-label: an example may carry several gold labels.
threshold   a label is predicted when sigmoid(logit) >= THRESHOLD. Chosen ONCE
            on the `validation` split and then frozen for `test`, because
            tuning it on test would make the reported number meaningless.
baseline    the lexicon in aedt/emotion/detect.py, scored identically.
metrics     macro-F1, micro-F1, weighted-F1, and per-label F1. Accuracy is not
            the headline: GoEmotions is heavily imbalanced (27% neutral) and
            accuracy rewards predicting nothing.
seed        20260828 (no sampling is used, so this affects nothing; recorded
            for completeness).

    python3 scripts/eval_emotion_model.py [--limit N] [--threshold T]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SEED = 20260828
OUT = ROOT / "results" / "emotion"


def load_split(split: str, limit: int | None):
    from datasets import load_dataset
    ds = load_dataset("google-research-datasets/go_emotions", "simplified", split=split)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    return ds


def predict_scores(texts, batch=32):
    """Sigmoid scores for every label, in model-label order."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from aedt.emotion.detect import MODEL_NAME

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    mod = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    mod.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            enc = tok(list(texts[i:i + batch]), return_tensors="pt",
                      truncation=True, max_length=128, padding=True)
            out.append(torch.sigmoid(mod(**enc).logits).numpy())
            if (i // batch) % 20 == 0:
                print(f"    {i}/{len(texts)}", flush=True)
    return np.vstack(out), mod.config.id2label


def lexicon_scores(texts, id2label):
    """Score the lexicon baseline in the SAME label space, so it is comparable.

    The baseline speaks the check-in taxonomy, not GoEmotions, so its output is
    projected back: a check-in label lights up every GoEmotions label that maps
    onto it. This flatters the baseline if anything -- it gets credit for a
    whole group when it names the group -- which is the right direction for a
    baseline comparison.
    """
    from aedt.emotion.detect import _lexicon_predict, goemotions_to_checkin
    n_lab = len(id2label)
    back: dict[str, list[int]] = {}
    for i in range(n_lab):
        back.setdefault(goemotions_to_checkin(id2label[i]), []).append(i)
    S = np.zeros((len(texts), n_lab), dtype=float)
    for r, t in enumerate(texts):
        p = _lexicon_predict(t)
        for lab, sc in (p.distribution or ((p.label, p.score),)):
            for j in back.get(lab, []):
                S[r, j] = max(S[r, j], sc)
    return S


def prf(y_true, y_pred):
    from sklearn.metrics import (classification_report, f1_score,
                                 precision_score, recall_score)
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "subset_accuracy": float((y_true == y_pred).all(axis=1).mean()),
    }


def to_multihot(ds, n_lab):
    Y = np.zeros((len(ds), n_lab), dtype=int)
    for i, labs in enumerate(ds["labels"]):
        for l in labs:
            Y[i, l] = 1
    return Y


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--threshold", type=float, default=None,
                    help="skip validation tuning and force this threshold")
    args = ap.parse_args()

    print("Loading GoEmotions...", flush=True)
    val = load_split("validation", args.limit)
    test = load_split("test", args.limit)

    print(f"validation {len(val)} | test {len(test)}", flush=True)
    print("Scoring validation (threshold selection)...", flush=True)
    Sval, id2label = predict_scores(val["text"])
    n_lab = len(id2label)
    Yval = to_multihot(val, n_lab)

    if args.threshold is not None:
        thr, tuning = args.threshold, "forced on the command line"
    else:
        grid = np.arange(0.05, 0.61, 0.05)
        f1s = [prf(Yval, (Sval >= t).astype(int))["macro_f1"] for t in grid]
        thr = float(grid[int(np.argmax(f1s))])
        tuning = "argmax macro-F1 over 0.05..0.60 on the VALIDATION split"
        print("  threshold grid:", [f"{t:.2f}:{f:.3f}" for t, f in zip(grid, f1s)])
    print(f"  chosen threshold {thr:.2f} ({tuning})", flush=True)

    print("Scoring test...", flush=True)
    Stest, _ = predict_scores(test["text"])
    Ytest = to_multihot(test, n_lab)
    model_metrics = prf(Ytest, (Stest >= thr).astype(int))

    print("Scoring lexicon baseline on test...", flush=True)
    Slex = lexicon_scores(test["text"], id2label)
    base_metrics = prf(Ytest, (Slex >= 0.35).astype(int))

    from sklearn.metrics import f1_score
    per_label = f1_score(Ytest, (Stest >= thr).astype(int), average=None,
                         zero_division=0)
    per = sorted(((id2label[i], float(per_label[i]), int(Ytest[:, i].sum()))
                  for i in range(n_lab)), key=lambda r: -r[1])

    report = {
        "protocol": {
            "dataset": "google-research-datasets/go_emotions (simplified)",
            "split_evaluated": "test", "n_test": len(test),
            "n_validation": len(val),
            "threshold": thr, "threshold_selection": tuning,
            "model": "SamLowe/roberta-base-go_emotions",
            "trained_by_this_project": False,
            "baseline": "lexicon in aedt/emotion/detect.py, threshold 0.35",
            "seed": SEED,
        },
        "model": model_metrics,
        "lexicon_baseline": base_metrics,
        "per_label_f1": [{"label": l, "f1": round(f, 4), "support": s} for l, f, s in per],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "goemotions_eval.json").write_text(json.dumps(report, indent=2))

    print("\n================ RESULTS (GoEmotions test) ================")
    print(f"{'metric':18s} {'model':>10s} {'lexicon':>10s}")
    for k in ("macro_f1", "micro_f1", "weighted_f1", "macro_precision",
              "macro_recall", "subset_accuracy"):
        print(f"{k:18s} {model_metrics[k]:10.4f} {base_metrics[k]:10.4f}")
    print("\nbest 5 labels :", [(l, round(f, 3)) for l, f, _ in per[:5]])
    print("worst 5 labels:", [(l, round(f, 3), f"n={s}") for l, f, s in per[-5:]])
    print(f"\nwritten: {OUT / 'goemotions_eval.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
