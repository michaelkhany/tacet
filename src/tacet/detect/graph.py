"""Graph-convolutional detection over a learned feature graph.

Features are treated as nodes in a graph whose edges are correlations learned
from healthy operation. Standardised deviations are then propagated across that
graph before scoring, so a window is judged by how anomalous its *neighbourhood*
of related signals is, not by each signal alone.

That distinction matters on real hardware. One sensor reading high is noise; one
sensor reading high while every metric it is normally coupled to stays flat is a
broken coupling, and it is exactly the situation a per-feature detector rates as
mild and an operator rates as urgent.

The propagation is a normalised-adjacency graph convolution -- the same operator
as a GCN layer with the weight matrix fixed to identity. That keeps the method
dependency-free (no PyTorch, no PyTorch Geometric, runs on a login node) and
keeps it interpretable, since every edge is a correlation you can inspect. The
interface matches the rest of the library, so swapping in a trained GCN later
changes nothing downstream.

Ported and generalised from the reference implementation used in the PDM /
EII Cloud evaluation study.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import schema
from .base import BaseDetector

__all__ = ["FeatureGraphDetector"]


class FeatureGraphDetector(BaseDetector):
    """Correlation-graph anomaly detector with relational explanations.

    Parameters
    ----------
    correlation_threshold:
        Minimum absolute correlation for an edge. Lower values build a denser
        graph that smooths harder and blurs localised faults.
    steps:
        Graph convolution steps. Each step widens the neighbourhood by one hop;
        beyond three the signal is usually smoothed into uniformity.
    method:
        Correlation used to build the graph. ``"spearman"`` resists the outliers
        that dominate raw telemetry.

    Examples
    --------
    >>> detector = FeatureGraphDetector(planes=["telemetry", "observability"])
    >>> scored = detector.fit(train).score(test)
    >>> detector.edges().head()
    >>> detector.explain(scored, top_k=5)
    """

    def __init__(
        self,
        correlation_threshold: float = 0.30,
        steps: int = 2,
        method: str = "spearman",
        **kwargs,
    ):
        super().__init__(**kwargs)
        if steps < 1:
            raise ValueError("steps must be at least 1")

        self.correlation_threshold = correlation_threshold
        self.steps = steps
        self.method = method

        self.correlation_: np.ndarray | None = None
        self.adjacency_: np.ndarray | None = None
        self.importance_: np.ndarray | None = None

    # -- detector contract --------------------------------------------------

    def _fit(self, values, frame):
        self.center_ = np.nanmean(values, axis=0)
        scale = np.nanstd(values, axis=0)
        self.scale_ = np.where(scale <= 1e-12, 1.0, scale)

        standardised = np.nan_to_num((values - self.center_) / self.scale_)

        self.correlation_ = self._correlation(standardised)
        self.adjacency_ = self._build_graph(self.correlation_)

        embeddings = self._propagate(np.abs(standardised))
        importance = np.abs(embeddings).mean(axis=0)
        self.importance_ = _unit_scale(importance)

        train_scores = embeddings.mean(axis=1)
        self.train_min_ = float(np.nanmin(train_scores))
        self.train_max_ = float(np.nanmax(train_scores))

    def _score(self, values, frame):
        standardised = np.nan_to_num((values - self.center_) / self.scale_)
        embeddings = self._propagate(np.abs(standardised))

        raw = embeddings.mean(axis=1)

        # Scaled against training extremes so runs stay comparable.
        span = max(self.train_max_ - self.train_min_, 1e-9)
        return np.clip((raw - self.train_min_) / span, 0.0, None)

    # -- graph construction -------------------------------------------------

    def _correlation(self, standardised: np.ndarray) -> np.ndarray:
        if standardised.shape[1] == 1:
            return np.ones((1, 1), dtype=float)

        frame = pd.DataFrame(standardised, columns=self.features_)
        # Constant features have zero variance, so their correlation is 0/0.
        # That is a NaN we replace with "no edge", not a condition to warn about.
        with np.errstate(invalid="ignore", divide="ignore"):
            correlation = frame.corr(method=self.method).to_numpy(dtype=float)

        return np.nan_to_num(correlation)

    def _build_graph(self, correlation: np.ndarray) -> np.ndarray:
        adjacency = (np.abs(correlation) >= self.correlation_threshold).astype(float)
        np.fill_diagonal(adjacency, 1.0)

        # Row-normalise: each node's update is a weighted average of its
        # neighbourhood, so hub features cannot dominate by degree alone.
        degree = adjacency.sum(axis=1, keepdims=True)
        return adjacency / np.where(degree <= 0, 1.0, degree)

    def _propagate(self, values: np.ndarray) -> np.ndarray:
        smoothed = values
        for _ in range(self.steps):
            smoothed = smoothed @ self.adjacency_
        return smoothed

    # -- inspection ---------------------------------------------------------

    def edges(self, min_weight: float = 0.0) -> pd.DataFrame:
        """The learned feature graph as an edge list, annotated with planes."""
        if self.adjacency_ is None:
            raise RuntimeError("fit the detector first")

        records = []
        for i, source in enumerate(self.features_):
            for j, target in enumerate(self.features_):
                if i >= j or self.adjacency_[i, j] <= min_weight:
                    continue

                source_plane = schema.plane_of(source)
                target_plane = schema.plane_of(target)

                records.append(
                    {
                        "source": source,
                        "target": target,
                        "correlation": float(self.correlation_[i, j]),
                        "weight": float(self.adjacency_[i, j]),
                        "source_plane": source_plane,
                        "target_plane": target_plane,
                        "cross_plane": source_plane != target_plane,
                    }
                )

        edges = pd.DataFrame(records)
        if edges.empty:
            return edges

        return edges.reindex(
            edges["correlation"].abs().sort_values(ascending=False).index
        ).reset_index(drop=True)

    def feature_importance(self) -> pd.DataFrame:
        """Per-feature contribution to the graph-smoothed score."""
        if self.importance_ is None:
            raise RuntimeError("fit the detector first")

        return (
            pd.DataFrame(
                {
                    "feature": self.features_,
                    "importance": self.importance_,
                    "plane": [schema.plane_of(f) for f in self.features_],
                    "degree": self.adjacency_.astype(bool).sum(axis=1) - 1,
                }
            )
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    def explain(self, scored: pd.DataFrame, top_k: int = 5) -> pd.DataFrame:
        """Which features drove each alerted window.

        Returns one row per (window, contributing feature), ranked by that
        feature's standardised deviation in that window. Only rows flagged by
        :meth:`~tacet.detect.base.BaseDetector.alert` are explained; call
        ``alert`` first.
        """
        if schema.ALERT not in scored.columns:
            raise KeyError(
                "no 'alert' column: call detector.alert(scored, budget=...) first"
            )

        flagged = scored[scored[schema.ALERT] == 1]
        if flagged.empty:
            return pd.DataFrame(
                columns=["row", "feature", "plane", "deviation", "rank"]
            )

        values = flagged[self.features_].to_numpy(dtype=float)
        deviations = np.abs(np.nan_to_num((values - self.center_) / self.scale_))

        order = np.argsort(-deviations, axis=1)[:, :top_k]

        records = []
        for position, (index, row) in enumerate(zip(flagged.index, order)):
            for rank, column in enumerate(row):
                feature = self.features_[column]
                records.append(
                    {
                        "row": index,
                        "feature": feature,
                        "plane": schema.plane_of(feature),
                        "deviation": float(deviations[position, column]),
                        "rank": rank + 1,
                    }
                )

        return pd.DataFrame(records)


def _unit_scale(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    low, high = np.nanmin(values), np.nanmax(values)
    return (values - low) / (high - low + 1e-9)
