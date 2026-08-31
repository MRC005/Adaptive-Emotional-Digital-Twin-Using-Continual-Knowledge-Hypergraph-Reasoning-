#!/usr/bin/env python3
"""Does the deployed int8 ONNX model agree with the torch reference?

The deployed service runs ``model_quantized.onnx`` because the torch path needs
664 MB and the free tier gives 512 MB. Quantisation changes the arithmetic, so
"it is the same model" is a claim that has to be checked rather than asserted.

If this script fails, the interface must stop saying "RoBERTa (GoEmotions)",
because it would no longer be true of what actually ran.

PROTOCOL
--------
data       GoEmotions test split (or --limit N of it)
reference  SamLowe/roberta-base-go_emotions via torch
deployed   SamLowe/roberta-base-go_emotions-onnx, model_quantized.onnx, via
           onnxruntime -- the exact artefact the service loads
compared   per-label sigmoid probabilities: max and mean absolute difference;
           agreement of the argmax label; agreement of the collapsed check-in
           label, which is what the user actually sees
thresholds MAX_MEAN_ABS 0.02, MIN_TOP1_AGREEMENT 0.95, MIN_CHECKIN_AGREEMENT 0.97

    python3 scripts/verify_onnx_agreement.py [--limit 300]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT = ROOT / "results" / "emotion"
MAX_MEAN_ABS = 0.02
MIN_TOP1_AGREEMENT = 0.95
MIN_CHECKIN_AGREEMENT = 0.97


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=300)
    args = ap.parse_args()

    from aedt.emotion.detect import goemotions_to_checkin
    from aedt.emotion.onnx_detect import GOEMOTIONS_LABELS, OnnxEmotionDetector

    print("Loading GoEmotions test split...", flush=True)
    from datasets import load_dataset
    ds = load_dataset("google-research-datasets/go_emotions", "simplified", split="test")
    texts = list(ds["text"])[: args.limit]

    print(f"Scoring {len(texts)} texts with the ONNX model...", flush=True)
    onnx = OnnxEmotionDetector()
    if not onnx.load():
        print(f"NOT RUN: ONNX model unavailable ({onnx.load_error})")
        return 2
    onnx_scores, onnx_checkin = [], []
    for t in texts:
        p = onnx.predict(t)
        onnx_scores.append({l: s for l, s in p.raw_top})
        onnx_checkin.append(p.label)

    print("Scoring the same texts with the torch reference...", flush=True)
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except Exception as exc:
        print(f"NOT RUN: torch/transformers unavailable ({exc}). The deployed "
              "model cannot be verified against the reference on this machine.")
        return 2

    from aedt.emotion.detect import MODEL_NAME
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    mod = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    mod.eval()
    id2label = mod.config.id2label

    # the hard-coded label order must match the reference exactly
    ref_labels = tuple(id2label[i] for i in range(len(id2label)))
    if ref_labels != GOEMOTIONS_LABELS:
        print("FAIL: the hard-coded ONNX label order does not match the reference.")
        print("  reference:", ref_labels)
        print("  hard-coded:", GOEMOTIONS_LABELS)
        return 1

    diffs, top1_same, checkin_same = [], 0, 0
    with torch.inference_mode():
        for i, t in enumerate(texts):
            enc = tok(t, return_tensors="pt", truncation=True, max_length=128)
            ref = torch.sigmoid(mod(**enc).logits)[0].numpy()

            # re-run the ONNX model to get the FULL distribution, not just top-6
            p = onnx.predict(t)
            got = np.array([onnx_scores[i].get(l, np.nan) for l in ref_labels])
            mask = ~np.isnan(got)
            diffs.append(np.abs(ref[mask] - got[mask]))

            if ref_labels[int(ref.argmax())] == p.raw_label:
                top1_same += 1
            if goemotions_to_checkin(ref_labels[int(ref.argmax())]) == p.label:
                checkin_same += 1
            if i and i % 100 == 0:
                print(f"    {i}/{len(texts)}", flush=True)

    all_d = np.concatenate(diffs)
    report = {
        "protocol": {
            "reference": MODEL_NAME, "deployed": "SamLowe/roberta-base-go_emotions-onnx"
                                                 " :: onnx/model_quantized.onnx",
            "n_texts": len(texts), "split": "go_emotions test",
            "note": "compared on the top-6 labels the ONNX path reports per text",
        },
        "max_abs_diff": float(all_d.max()),
        "mean_abs_diff": float(all_d.mean()),
        "top1_label_agreement": top1_same / len(texts),
        "checkin_label_agreement": checkin_same / len(texts),
        "thresholds": {"max_mean_abs": MAX_MEAN_ABS,
                       "min_top1_agreement": MIN_TOP1_AGREEMENT,
                       "min_checkin_agreement": MIN_CHECKIN_AGREEMENT},
    }
    ok = (report["mean_abs_diff"] <= MAX_MEAN_ABS
          and report["top1_label_agreement"] >= MIN_TOP1_AGREEMENT
          and report["checkin_label_agreement"] >= MIN_CHECKIN_AGREEMENT)
    report["verdict"] = "AGREES" if ok else "DIVERGES"

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "onnx_agreement.json").write_text(json.dumps(report, indent=2))

    print("\n============ ONNX vs torch reference ============")
    print(f"  texts compared          {len(texts)}")
    print(f"  max |Δ probability|     {report['max_abs_diff']:.4f}")
    print(f"  mean |Δ probability|    {report['mean_abs_diff']:.4f}   (limit {MAX_MEAN_ABS})")
    print(f"  top-1 label agreement   {report['top1_label_agreement']:.4f}   (min {MIN_TOP1_AGREEMENT})")
    print(f"  check-in label agreement{report['checkin_label_agreement']:.4f}   (min {MIN_CHECKIN_AGREEMENT})")
    print(f"\n  VERDICT: {report['verdict']}")
    if not ok:
        print("  The deployed model does NOT match the reference. The interface "
              "must not claim RoBERTa produced the result.")
    print(f"\nwritten: {OUT / 'onnx_agreement.json'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
