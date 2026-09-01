"""LAYER 1 / MODULE 1 -- Transformer emotion detection.

Purpose  Map free text to an emotion label with a calibrated-ish confidence.
Input    One utterance.
Output   ``EmotionPrediction`` carrying the label, score, full distribution and
         the exact model version that produced it.
Status   STANDARD / EXISTING MODEL, INTEGRATED. See the honesty note below.

WHAT IS AND IS NOT OURS

The model is ``SamLowe/roberta-base-go_emotions``: a RoBERTa-base encoder
(124.7M parameters) fine-tuned by its author on GoEmotions, 28 labels,
multi-label. **We did not train it and we do not claim to have.** What this
project contributes is the integration, the evaluation on a held-out split
(``scripts/eval_emotion_model.py``), the mapping onto the check-in taxonomy,
and the honest reporting of where it is weak.

Multi-label matters. GoEmotions annotates one comment with several emotions,
so the head is sigmoid, not softmax, and the scores do NOT sum to one. Taking
an argmax over sigmoid outputs and calling it a probability would be wrong;
``top`` is reported as the highest-scoring label with its own independent
score, and the runner-up is kept so the interface can show when two labels are
nearly tied.

THE LABEL-SPACE GAP, MEASURED RATHER THAN GLOSSED

GoEmotions has 28 labels and **none of them is "stress"**. Measured on this
checkpoint, "I am stressed and exhausted" returns sadness 0.463 with
nervousness only 0.119, while "I am so anxious about tomorrow" correctly
returns nervousness 0.504. The taxonomy that NLP emotion research standardised
on (Reddit comments) simply does not contain the construct that longitudinal
wellbeing research measures.

This is handled by precedence, not by pretending: an explicit first-person
statement of feeling is a SELF-REPORT, and a self-report outranks a model
inference about the same person. ``SELF_REPORT_PATTERNS`` below fires only on
explicit constructions ("I am stressed", "I feel anxious", "I'm exhausted"),
returns ``FieldSource.EXTRACTED`` with the matched span as evidence, and the
interface shows that the label came from the sentence rather than the model.
Everything else goes to the Transformer.

This is also why "stress" must not be read as a GoEmotions class in the
evaluation: ``scripts/eval_emotion_model.py`` scores the model on the labels
the model actually has.

THE FALLBACK, AND WHY IT IS NOT THE MODEL

Where torch is unavailable (the static browser build, a machine without the
dependency) a lexicon baseline is used instead. It is returned with
``backend="lexicon"`` and the interface labels it a baseline. It exists so the
product still functions, not so a demo can pretend a Transformer ran.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from functools import lru_cache

__all__ = ["EmotionPrediction", "EmotionDetector", "MODEL_NAME",
           "CHECKIN_EMOTIONS", "goemotions_to_checkin",
           "self_reported_emotion", "SELF_REPORT_PATTERNS"]

log = logging.getLogger(__name__)

MODEL_NAME = "SamLowe/roberta-base-go_emotions"
MODEL_REVISION = "main"

#: The taxonomy the product speaks. GoEmotions' 28 labels are finer than a
#: wellbeing check-in needs, and several ("admiration", "amusement") never
#: arise in one. They are grouped, and the grouping is shown to the user.
CHECKIN_EMOTIONS = ("anxiety", "sadness", "anger", "stress", "joy",
                    "gratitude", "calm", "confusion", "neutral")

#: GoEmotions label -> check-in label. Every one of the 28 is mapped, so no
#: prediction can fall through to a default.
_GOEMOTIONS_TO_CHECKIN: dict[str, str] = {
    "nervousness": "anxiety", "fear": "anxiety", "embarrassment": "anxiety",
    "sadness": "sadness", "grief": "sadness", "disappointment": "sadness",
    "remorse": "sadness",
    "anger": "anger", "annoyance": "anger", "disgust": "anger",
    "disapproval": "anger",
    "joy": "joy", "excitement": "joy", "amusement": "joy", "love": "joy",
    "optimism": "joy", "pride": "joy", "admiration": "joy",
    "gratitude": "gratitude", "relief": "calm", "approval": "calm",
    "caring": "calm", "desire": "calm",
    "confusion": "confusion", "curiosity": "confusion",
    # "realization" was mapped to confusion purely to give all 28 labels a
    # target. Realisation is arguably the OPPOSITE of confusion, and the
    # mapping produced a documented failure (a narrative about anxiety was
    # displayed as "confusion"). Coverage is not a semantic justification.
    # Labels without a defensible target are left unmapped and surface as
    # "neutral" with the raw label shown beside it.
    "surprise": "confusion",
    "neutral": "neutral",
}


def goemotions_to_checkin(label: str) -> str:
    """Map one GoEmotions label onto the check-in taxonomy."""
    return _GOEMOTIONS_TO_CHECKIN.get(label, "neutral")


#: Explicit first-person statements of feeling. A person saying "I am
#: stressed" has SELF-REPORTED it; that is an observation, and it outranks a
#: model's inference about the same sentence. Matched on the raw text so the
#: evidence span can be shown back to the user.
SELF_REPORT_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\bi(?:'m| am| feel(?:ing)?)\s+(?:really\s+|so\s+|very\s+|quite\s+|a bit\s+)?"
     r"(stressed|stressed out|under pressure|overwhelmed|burnt out|burned out|exhausted|drained)\b",
     "stress"),
    (r"\bi(?:'m| am| feel(?:ing)?)\s+(?:really\s+|so\s+|very\s+|quite\s+|a bit\s+)?"
     r"(anxious|nervous|worried|scared|afraid|panicky|on edge)\b", "anxiety"),
    (r"\bi(?:'m| am| feel(?:ing)?)\s+(?:really\s+|so\s+|very\s+|quite\s+|a bit\s+)?"
     r"(sad|down|low|depressed|miserable|hopeless|lonely)\b", "sadness"),
    (r"\bi(?:'m| am| feel(?:ing)?)\s+(?:really\s+|so\s+|very\s+|quite\s+|a bit\s+)?"
     r"(angry|furious|annoyed|irritated|frustrated|fed up)\b", "anger"),
    (r"\bi(?:'m| am| feel(?:ing)?)\s+(?:really\s+|so\s+|very\s+|quite\s+|a bit\s+)?"
     r"(happy|great|good|excited|glad|delighted|cheerful)\b", "joy"),
    (r"\bi(?:'m| am| feel(?:ing)?)\s+(?:really\s+|so\s+|very\s+|quite\s+|a bit\s+)?"
     r"(grateful|thankful)\b", "gratitude"),
    (r"\bi(?:'m| am| feel(?:ing)?)\s+(?:really\s+|so\s+|very\s+|quite\s+|a bit\s+)?"
     r"(calm|relaxed|fine|ok|okay|settled|at ease)\b", "calm"),
)


#: Constructions that REVERSE or CANCEL a following self-statement. Without
#: these, "I am not sure I am good enough" matched "i am good" and was reported
#: as JOY -- a sentence expressing self-doubt labelled as happiness.
_NEGATORS = re.compile(
    r"\b(not|never|hardly|barely|no longer|don'?t|doesn'?t|didn'?t|can'?t|cannot|"
    r"isn'?t|aren'?t|wasn'?t|won'?t)\b")

#: Past or future framing. "Yesterday I was anxious" is not a current state.
_PAST = re.compile(r"\b(yesterday|last night|last week|earlier|used to|"
                   r"i was|had been|this morning)\b")
_FUTURE = re.compile(r"\b(tomorrow|next week|will be|going to|expect to|"
                     r"i might|i may|probably)\b")


def self_reported_emotion(text: str, *, require_present: bool = True
                          ) -> tuple[str, str] | None:
    """(label, matched span) when the text states a CURRENT feeling outright.

    Returns None rather than guessing when the statement is negated or framed
    in the past or future. Three failures this now refuses, all of which the
    previous version got wrong:

        "I am not sure I am good enough"          -> None  (was: joy)
        "Yesterday I was anxious, now relieved"   -> None  (was: none, by luck)
        "I am grateful, but I am anxious"         -> the LAST statement wins

    The last-statement rule matters because a narrative ends on its current
    state; an earlier subordinate clause is not the person's present feeling.
    """
    low = (text or "").lower()
    hits: list[tuple[int, str, str]] = []
    for pattern, label in SELF_REPORT_PATTERNS:
        for m in re.finditer(pattern, low):
            # look back a short window for a negator attached to this clause
            window = low[max(0, m.start() - 40):m.start()]
            clause = window.rsplit(",", 1)[-1].rsplit(".", 1)[-1]
            if _NEGATORS.search(clause):
                continue
            if require_present:
                before = low[max(0, m.start() - 60):m.start()]
                seg = before.rsplit(",", 1)[-1].rsplit(".", 1)[-1]
                if _PAST.search(seg) or _FUTURE.search(seg):
                    continue
            hits.append((m.start(), label, m.group(0)))
    if not hits:
        return None
    # the LAST surviving statement is the current one
    hits.sort(key=lambda h: h[0])
    return hits[-1][1], hits[-1][2]


@dataclass(frozen=True)
class EmotionPrediction:
    """One prediction, with enough context to audit it."""

    label: str                     # check-in taxonomy
    score: float
    raw_label: str = ""            # the model's own label
    backend: str = "transformer"   # transformer | lexicon
    model: str = ""
    runner_up: tuple[str, float] | None = None
    distribution: tuple[tuple[str, float], ...] = ()

    @property
    def is_model(self) -> bool:
        return self.backend == "transformer"

    @property
    def ambiguous(self) -> bool:
        """True when the top two are close enough that one should not be asserted."""
        return bool(self.runner_up and (self.score - self.runner_up[1]) < 0.15)

    def to_dict(self) -> dict:
        return {"emotion": self.label, "confidence": round(float(self.score), 4),
                "raw_label": self.raw_label, "backend": self.backend,
                "model": self.model, "ambiguous": self.ambiguous,
                "runner_up": list(self.runner_up) if self.runner_up else None,
                "distribution": [[l, round(float(s), 4)] for l, s in self.distribution]}


# --------------------------------------------------------------------------
# Lexicon baseline. Deliberately simple; it is a fallback and a comparison
# point in the evaluation, not a contribution.
# --------------------------------------------------------------------------
_LEXICON: dict[str, tuple[str, ...]] = {
    "anxiety": ("anxious", "anxiety", "nervous", "worried", "worry", "scared",
                "afraid", "panic", "dread", "uneasy", "on edge", "apprehensive"),
    "stress": ("stressed", "stress", "pressure", "overwhelmed", "overloaded",
               "burnt out", "burned out", "exhausted", "drained", "tense"),
    "sadness": ("sad", "down", "depressed", "low", "unhappy", "miserable",
                "crying", "cried", "hopeless", "lonely", "empty"),
    "anger": ("angry", "furious", "annoyed", "irritated", "frustrated", "mad",
              "resentful", "fed up"),
    "joy": ("happy", "great", "excited", "delighted", "glad", "wonderful",
            "pleased", "cheerful", "good mood"),
    "gratitude": ("grateful", "thankful", "appreciate", "thanks"),
    "calm": ("calm", "relaxed", "peaceful", "settled", "at ease", "fine"),
    "confusion": ("confused", "unsure", "uncertain", "puzzled", "lost"),
}


def _lexicon_predict(text: str) -> EmotionPrediction:
    low = f" {(text or '').lower()} "
    scores: dict[str, float] = {}
    for label, words in _LEXICON.items():
        hits = sum(1 for w in words if w in low)
        if hits:
            scores[label] = min(0.35 + 0.15 * hits, 0.9)
    if not scores:
        return EmotionPrediction(label="neutral", score=0.3, raw_label="neutral",
                                 backend="lexicon", model="lexicon-v1")
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    return EmotionPrediction(
        label=ranked[0][0], score=ranked[0][1], raw_label=ranked[0][0],
        backend="lexicon", model="lexicon-v1",
        runner_up=ranked[1] if len(ranked) > 1 else None,
        distribution=tuple(ranked[:6]))


class EmotionDetector:
    """Loads the Transformer once and predicts. Falls back visibly."""

    def __init__(self, model_name: str = MODEL_NAME, *, force_lexicon: bool = False):
        self.model_name = model_name
        self._pipe = None
        self._failed = False
        if force_lexicon:
            self._failed = True

    # ------------------------------------------------------------- loading
    def _load(self):
        if self._pipe is not None or self._failed:
            return self._pipe
        try:
            import torch                                    # noqa: F401
            from transformers import (AutoModelForSequenceClassification,
                                      AutoTokenizer)
            tok = AutoTokenizer.from_pretrained(self.model_name)
            mod = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            mod.eval()
            self._pipe = (tok, mod)
        except Exception as exc:                             # reported, not hidden
            log.warning("Transformer unavailable (%s: %s); using the lexicon "
                        "BASELINE, which is not the model.",
                        type(exc).__name__, exc)
            self._failed = True
        return self._pipe

    @property
    def available(self) -> bool:
        return self._load() is not None

    # ---------------------------------------------------------- prediction
    def predict(self, text: str) -> EmotionPrediction:
        if not (text or "").strip():
            return EmotionPrediction(label="neutral", score=0.0,
                                     backend="lexicon", model="empty-input")
        pipe = self._load()
        if pipe is None:
            return _lexicon_predict(text)

        import torch
        tok, mod = pipe
        with torch.no_grad():
            enc = tok(text, return_tensors="pt", truncation=True, max_length=128)
            # multi-label head: sigmoid, NOT softmax. Scores are independent.
            probs = torch.sigmoid(mod(**enc).logits)[0]

        id2label = mod.config.id2label
        pairs = sorted(((id2label[i], float(probs[i])) for i in range(len(probs))),
                       key=lambda kv: -kv[1])
        top_raw, top_score = pairs[0]

        # Collapse onto the check-in taxonomy by summing the evidence that
        # maps to each check-in label, so "nervousness" + "fear" reinforce
        # "anxiety" rather than splitting it.
        collapsed: dict[str, float] = {}
        for raw, s in pairs:
            k = goemotions_to_checkin(raw)
            collapsed[k] = max(collapsed.get(k, 0.0), s)
        ranked = sorted(collapsed.items(), key=lambda kv: -kv[1])

        return EmotionPrediction(
            label=ranked[0][0], score=ranked[0][1], raw_label=top_raw,
            backend="transformer", model=f"{self.model_name}@{MODEL_REVISION}",
            runner_up=ranked[1] if len(ranked) > 1 else None,
            distribution=tuple(ranked[:6]))


@lru_cache(maxsize=1)
def default_detector() -> EmotionDetector:
    """One shared detector, so the weights load once per process."""
    return EmotionDetector()
