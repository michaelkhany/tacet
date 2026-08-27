"""Classical statistical detectors.

These are the methods a reviewer will ask you to compare against, implemented
to the same interface as the graph and Markov detectors so the comparison is
one loop rather than one afternoon. They are deliberately simple and fast; on
well-behaved telemetry a robust z-score is a genuinely hard baseline to beat,
and a paper that omits it invites the question.

All of them inherit the observability machinery from
:class:`~tacet.detect.base.BaseDetector`, so even the plainest baseline reports
how much of each window it could actually see.
"""

from __future__ import annotations

import numpy as np

from .base import BaseDetector

__all__ = ["RobustZScore", "EWMADetector", "CUSUMDetector", "MahalanobisDetector"]


class RobustZScore(BaseDetector):
    """Median/MAD z-score, aggregated across features.

    Uses the median absolute deviation rather than the standard deviation, so a
    single catastrophic reading during training does not inflate the scale and
    mask everything after it. The scaling constant 1.4826 makes MAD a consistent
    estimator of sigma for normally distributed data.

    Parameters
    ----------
    aggregate:
        ``"max"`` scores a window by its single worst feature -- sensitive to
        localised faults, which is usually what you want on hardware.
        ``"mean"`` requires broad agreement and is quieter.
    """

    def __init__(self, aggregate: str = "max", **kwargs):
        super().__init__(**kwargs)
        if aggregate not in ("max", "mean"):
            raise ValueError("aggregate must be 'max' or 'mean'")
        self.aggregate = aggregate

    def _fit(self, values, frame):
        self.center_ = np.nanmedian(values, axis=0)
        deviation = np.nanmedian(np.abs(values - self.center_), axis=0)
        self.scale_ = 1.4826 * deviation
        # Constant features would divide by zero; give them a scale that makes
        # any deviation finite but unremarkable.
        self.scale_ = np.where(self.scale_ <= 1e-12, 1.0, self.scale_)

    def _score(self, values, frame):
        z = np.abs((values - self.center_) / self.scale_)
        return z.max(axis=1) if self.aggregate == "max" else z.mean(axis=1)


class EWMADetector(BaseDetector):
    """Exponentially weighted moving average control chart.

    Tracks a slow-moving expectation per feature and scores the standardised
    departure from it. Catches gradual drift that a fixed threshold never trips
    -- degradation rather than failure.

    Parameters
    ----------
    alpha:
        Smoothing factor in ``(0, 1]``. Smaller reacts more slowly and tolerates
        more noise; ``0.3`` suits window-level HPC telemetry.
    """

    def __init__(self, alpha: float = 0.3, **kwargs):
        super().__init__(**kwargs)
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = alpha

    def _fit(self, values, frame):
        self.center_ = np.nanmean(values, axis=0)
        scale = np.nanstd(values, axis=0)
        self.scale_ = np.where(scale <= 1e-12, 1.0, scale)

    def _score(self, values, frame):
        standardised = (values - self.center_) / self.scale_

        smoothed = np.empty_like(standardised)
        state = standardised[0] if len(standardised) else 0.0
        for i, row in enumerate(standardised):
            state = self.alpha * row + (1 - self.alpha) * state
            smoothed[i] = state

        return np.abs(smoothed).max(axis=1)


class CUSUMDetector(BaseDetector):
    """Two-sided cumulative sum change detector.

    Accumulates small standardised departures until they add up, which makes it
    the right tool for a shift too small to trip any per-sample threshold but
    too persistent to be noise. Classic for detecting the moment a component
    changes regime.

    Parameters
    ----------
    drift:
        Slack, in standard deviations, absorbed before the sum starts to build.
    reset_threshold:
        Score at which the accumulator resets, so one sustained event does not
        saturate every window after it.
    """

    def __init__(self, drift: float = 0.5, reset_threshold: float = 10.0, **kwargs):
        super().__init__(**kwargs)
        self.drift = drift
        self.reset_threshold = reset_threshold

    def _fit(self, values, frame):
        self.center_ = np.nanmean(values, axis=0)
        scale = np.nanstd(values, axis=0)
        self.scale_ = np.where(scale <= 1e-12, 1.0, scale)

    def _score(self, values, frame):
        standardised = (values - self.center_) / self.scale_

        high = np.zeros(standardised.shape[1])
        low = np.zeros(standardised.shape[1])
        out = np.zeros(len(standardised))

        for i, row in enumerate(standardised):
            high = np.clip(high + row - self.drift, 0.0, None)
            low = np.clip(low - row - self.drift, 0.0, None)

            combined = np.maximum(high, low)
            out[i] = combined.max()

            saturated = combined > self.reset_threshold
            high[saturated] = 0.0
            low[saturated] = 0.0

        return out


class MahalanobisDetector(BaseDetector):
    """Multivariate distance from the training distribution.

    Unlike per-feature methods, this one notices *combinations* that never occur
    together in healthy operation -- full clock with low power draw, high
    utilisation with cold silicon -- even when every individual reading is
    unremarkable. On HPC telemetry those contradictions are often the earliest
    honest sign of trouble.

    Parameters
    ----------
    shrinkage:
        Ledoit-Wolf style pull of the covariance toward its diagonal. Telemetry
        features are strongly collinear and wide matrices are routinely
        rank-deficient, so some shrinkage is effectively always required.
    """

    def __init__(self, shrinkage: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        if not 0 <= shrinkage <= 1:
            raise ValueError("shrinkage must be in [0, 1]")
        self.shrinkage = shrinkage

    def _fit(self, values, frame):
        self.center_ = np.nanmean(values, axis=0)
        covariance = np.cov(values, rowvar=False)
        covariance = np.atleast_2d(np.nan_to_num(covariance))

        target = np.diag(np.diag(covariance))
        blended = (1 - self.shrinkage) * covariance + self.shrinkage * target
        blended += 1e-8 * np.eye(blended.shape[0]) * max(np.trace(blended), 1.0)

        self.precision_ = np.linalg.pinv(blended)

    def _score(self, values, frame):
        centred = values - self.center_
        # Row-wise quadratic form without materialising an n x n matrix.
        distances = np.einsum("ij,jk,ik->i", centred, self.precision_, centred)
        return np.sqrt(np.clip(distances, 0.0, None))
