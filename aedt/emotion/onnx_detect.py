"""Deployable emotion detection: the SAME model, a runtime that fits.

WHY THIS FILE EXISTS -- a measurement, not a preference

The torch path in ``detect.py`` needs **664 MB** resident after one inference.
Render's free tier gives **512 MB**. It does not fit, and a service that OOMs
on its first request is worse than no service, because the browser then falls
back to a word list while the interface has already promised a Transformer.

Measured on this machine:

    torch + transformers                      664 MB     does not fit
    onnxruntime + transformers                403 MB on imports alone
    onnxruntime + tokenizers  (this file)     287 MB     fits

So the deployed path drops ``torch`` and ``transformers`` entirely and uses
``onnxruntime`` with the tokenizer from ``tokenizers``.

IT IS THE SAME MODEL. ``SamLowe/roberta-base-go_emotions-onnx`` is the same
author's ONNX export of the same fine-tune, and ``model_quantized.onnx`` is its
int8 quantisation (125 MB). Quantisation changes the arithmetic, so agreement
with the torch reference is not assumed -- it is measured by
``scripts/verify_onnx_agreement.py`` and reported in the README. If the two
ever diverge materially, the claim "RoBERTa (GoEmotions)" stops being true and
the script fails.

THE SESSION IS BUILT ONCE. ``get_detector()`` is cached, and the FastAPI
lifespan warms it at startup, so no request pays the load cost and the model is
never rebuilt per request.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .detect import goemotions_to_checkin

log = logging.getLogger(__name__)

__all__ = ["OnnxEmotionDetector", "get_detector", "ONNX_REPO", "ONNX_FILE",
           "GOEMOTIONS_LABELS"]

ONNX_REPO = "SamLowe/roberta-base-go_emotions-onnx"
ONNX_FILE = "onnx/model_quantized.onnx"
TOKENIZER_FILE = "onnx/tokenizer.json"
MAX_LEN = 128

#: The 28 GoEmotions labels in the model's own output order. Hard-coded so the
#: service does not need `transformers` just to read a config, and asserted
#: against the reference model by the agreement script.
GOEMOTIONS_LABELS = (
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization", "relief",
    "remorse", "sadness", "surprise", "neutral",
)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class OnnxPrediction:
    label: str
    score: float
    raw_label: str
    distribution: tuple[tuple[str, float], ...]
    raw_top: tuple[tuple[str, float], ...]
    backend: str = "transformer"
    model: str = ONNX_REPO

    @property
    def ambiguous(self) -> bool:
        d = self.distribution
        return len(d) > 1 and (d[0][1] - d[1][1]) < 0.15

    def to_dict(self) -> dict:
        return {"emotion": self.label, "confidence": round(float(self.score), 4),
                "raw_label": self.raw_label, "backend": self.backend,
                "model": self.model, "ambiguous": self.ambiguous,
                "runner_up": [self.distribution[1][0], round(self.distribution[1][1], 4)]
                             if len(self.distribution) > 1 else None,
                "distribution": [[l, round(float(s), 4)] for l, s in self.distribution],
                "raw_top": [[l, round(float(s), 4)] for l, s in self.raw_top]}


class OnnxEmotionDetector:
    """int8 ONNX RoBERTa. Loads once; every request reuses the session."""

    def __init__(self, *, repo: str = ONNX_REPO, filename: str = ONNX_FILE):
        self.repo, self.filename = repo, filename
        self._sess = None
        self._tok = None
        self._input_names: set[str] = set()
        self.load_error: str | None = None

    # ---------------------------------------------------------------- load
    def load(self) -> bool:
        """Build the session. Safe to call repeatedly; only the first does work."""
        if self._sess is not None:
            return True
        if self.load_error is not None:
            return False
        try:
            import onnxruntime as ort
            from huggingface_hub import hf_hub_download
            from tokenizers import Tokenizer

            model_path = hf_hub_download(self.repo, self.filename)
            tok_path = hf_hub_download(self.repo, TOKENIZER_FILE)

            tok = Tokenizer.from_file(tok_path)
            tok.enable_truncation(max_length=MAX_LEN)

            opts = ort.SessionOptions()
            # One thread: a free-tier container has a fraction of a core, and
            # extra threads cost memory without buying latency here.
            opts.intra_op_num_threads = 1
            opts.inter_op_num_threads = 1
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            sess = ort.InferenceSession(model_path, opts,
                                        providers=["CPUExecutionProvider"])
            self._tok, self._sess = tok, sess
            self._input_names = {i.name for i in sess.get_inputs()}
            log.info("ONNX emotion model ready (%s)", self.filename)
            return True
        except Exception as exc:                       # recorded, never hidden
            self.load_error = f"{type(exc).__name__}: {exc}"
            log.warning("ONNX emotion model unavailable: %s", self.load_error)
            return False

    @property
    def ready(self) -> bool:
        return self._sess is not None

    # ----------------------------------------------------------- inference
    def predict(self, text: str) -> OnnxPrediction | None:
        """Sigmoid scores over 28 labels, collapsed onto the check-in taxonomy.

        Returns None when the model is unavailable, so the caller must decide
        what to do rather than receiving a silent lexicon answer dressed up as
        a Transformer result.
        """
        if not self.load():
            return None
        enc = self._tok.encode(text or "")
        feed = {"input_ids": np.array([enc.ids], dtype=np.int64),
                "attention_mask": np.array([enc.attention_mask], dtype=np.int64)}
        feed = {k: v for k, v in feed.items() if k in self._input_names}
        logits = self._sess.run(None, feed)[0][0]

        # multi-label head: independent sigmoids, NOT a softmax over classes
        probs = _sigmoid(np.asarray(logits, dtype=np.float64))
        raw = sorted(zip(GOEMOTIONS_LABELS, probs), key=lambda kv: -kv[1])

        collapsed: dict[str, float] = {}
        for label, p in raw:
            k = goemotions_to_checkin(label)
            collapsed[k] = max(collapsed.get(k, 0.0), float(p))
        ranked = sorted(collapsed.items(), key=lambda kv: -kv[1])

        return OnnxPrediction(
            label=ranked[0][0], score=ranked[0][1], raw_label=raw[0][0],
            distribution=tuple(ranked[:6]),
            raw_top=tuple((l, float(p)) for l, p in raw[:6]))


@lru_cache(maxsize=1)
def get_detector() -> OnnxEmotionDetector:
    """The process-wide detector. Warmed by the FastAPI lifespan at startup."""
    return OnnxEmotionDetector()
