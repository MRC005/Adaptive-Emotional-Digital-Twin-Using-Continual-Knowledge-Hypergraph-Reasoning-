"""LAYER 1 -- continual learning of the personal emotional pattern model.

Model parameters, not stored events. See ``ewc.py`` for why that distinction
is enforced rather than assumed.
"""
from .ewc import EWC, forgetting_metrics

__all__ = ["EWC", "forgetting_metrics"]
