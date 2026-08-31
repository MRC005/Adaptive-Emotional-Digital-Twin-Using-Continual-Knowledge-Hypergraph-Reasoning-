"""LAYER 1 -- the personal emotional Digital Twin's perception modules.

text -> emotion (Transformer) + context (rules) -> structured event
"""
from .context import extract_context
from .detect import (CHECKIN_EMOTIONS, EmotionDetector, EmotionPrediction,
                     MODEL_NAME, default_detector, goemotions_to_checkin)
from .events import EmotionalEvent, FieldSource, Provenanced, new_event_id
from .pipeline import build_event

__all__ = ["extract_context", "EmotionDetector", "EmotionPrediction",
           "CHECKIN_EMOTIONS", "MODEL_NAME", "default_detector",
           "goemotions_to_checkin", "EmotionalEvent", "FieldSource",
           "Provenanced", "new_event_id", "build_event"]
