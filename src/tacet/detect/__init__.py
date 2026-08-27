"""Anomaly detectors, all sharing one interface.

``fit(train)`` then ``score(test)`` then ``alert(scored, budget=...)``, for every
method here -- so comparing a graph-convolutional detector against a robust
z-score is a loop, not a rewrite. Every detector also reports observability
trust alongside its score.
"""

from .base import BaseDetector, apply_budget, observability_trust
from .graph import FeatureGraphDetector
from .markov import MarkovDetector, mine_event_sequences
from .statistical import (
    CUSUMDetector,
    EWMADetector,
    MahalanobisDetector,
    RobustZScore,
)

#: Name -> class, for sweeps and configuration files.
REGISTRY = {
    "markov": MarkovDetector,
    "graph": FeatureGraphDetector,
    "robust_z": RobustZScore,
    "ewma": EWMADetector,
    "cusum": CUSUMDetector,
    "mahalanobis": MahalanobisDetector,
}

__all__ = [
    "BaseDetector",
    "apply_budget",
    "observability_trust",
    "MarkovDetector",
    "mine_event_sequences",
    "FeatureGraphDetector",
    "RobustZScore",
    "EWMADetector",
    "CUSUMDetector",
    "MahalanobisDetector",
    "REGISTRY",
]
