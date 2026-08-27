"""Detector contract, alert budgeting, and observability trust.

Every detector implements ``fit`` / ``score`` / ``alert``, so they are
interchangeable in an evaluation loop, and every detector emits an
**observability trust** column alongside its anomaly score.

That second output is the point. A conventional detector returns a single
number and leaves you to assume it means something. A window in which two thirds
of the scrapes never arrived produces a score computed from one third of the
evidence, and no amount of confidence in the model makes that score as good as
one from a fully observed window. Reporting trust separately lets you say
"quiet, and we could see clearly" apart from "quiet, but we were half blind" --
which are the same number and opposite conclusions.
"""

from __future__ import annotations

import abc

import numpy as np
import pandas as pd

from .. import schema

__all__ = ["BaseDetector", "apply_budget", "observability_trust"]


def apply_budget(
    frame: pd.DataFrame,
    budget: int | None = None,
    threshold: float | None = None,
    score_column: str = schema.SCORE,
) -> pd.DataFrame:
    """Flag windows for review, honouring an alert budget **exactly**.

    Parameters
    ----------
    budget:
        Number of windows an operator will actually look at. The top ``budget``
        by score are flagged -- no more, whatever the score distribution does.
    threshold:
        Alternative to ``budget``: flag everything at or above this score.

    Notes
    -----
    The obvious implementation, ``score >= sorted(scores)[-budget]``, silently
    overshoots whenever scores tie, and anomaly scores tie constantly (an
    entirely unobserved window has nothing to distinguish it from the next
    entirely unobserved window). A budget of 50 can become 8000 alerts, and
    because the extra alerts land on real positives, recall and F1 *improve* --
    the failure looks like success. Ranking with a deterministic tie-break makes
    the budget mean what it says.
    """
    result = frame.copy()
    rows = len(result)

    if rows == 0:
        result[schema.ALERT] = np.zeros(0, dtype=int)
        return result

    scores = pd.to_numeric(result[score_column], errors="coerce").fillna(-np.inf).to_numpy()

    if threshold is not None:
        result[schema.ALERT] = (scores >= threshold).astype(int)
        return result

    if budget is None:
        raise ValueError("pass either `budget` or `threshold`")

    cap = int(min(max(budget, 0), rows))
    # Ties broken by original position, so the count can never exceed `cap`.
    ranking = np.lexsort((np.arange(rows), -scores))

    flags = np.zeros(rows, dtype=int)
    flags[ranking[:cap]] = 1
    result[schema.ALERT] = flags

    return result


def observability_trust(frame: pd.DataFrame) -> np.ndarray:
    """How much of each window we could actually see, in ``[0, 1]``.

    Derived from the ``obs_`` plane when :func:`tacet.to_windows` built it:
    coverage, discounted by the longest silence inside the window. Frames with
    no observability plane trust everything, and say so by returning ones --
    which is the honest default, not a claim.
    """
    rows = len(frame)

    coverage_column = f"{schema.OBSERVABILITY}coverage"
    if coverage_column not in frame.columns:
        return np.ones(rows, dtype=float)

    coverage = pd.to_numeric(frame[coverage_column], errors="coerce").fillna(0.0).to_numpy()
    trust = np.clip(coverage, 0.0, 1.0)

    gap_column = f"{schema.OBSERVABILITY}max_gap_seconds"
    stale_column = f"{schema.OBSERVABILITY}stale_seconds"

    if gap_column in frame.columns:
        gaps = pd.to_numeric(frame[gap_column], errors="coerce").fillna(0.0).to_numpy()
        span = np.nanmax(gaps) if np.isfinite(gaps).any() else 0.0
        if span > 0:
            # A window can be 90% covered and still have been blind through the
            # only moment that mattered. Penalise the burst, not just the total.
            trust = trust * (1.0 - 0.5 * np.clip(gaps / span, 0.0, 1.0))

    if stale_column in frame.columns:
        stale = pd.to_numeric(frame[stale_column], errors="coerce").fillna(0.0).to_numpy()
        span = np.nanmax(stale) if np.isfinite(stale).any() else 0.0
        if span > 0:
            trust = trust * (1.0 - 0.25 * np.clip(stale / span, 0.0, 1.0))

    return np.clip(trust, 0.0, 1.0)


class BaseDetector(abc.ABC):
    """Common machinery for every ``tacet`` detector.

    Subclasses implement :meth:`_fit` and :meth:`_score`.

    Parameters
    ----------
    features:
        Feature columns to use. ``None`` uses every non-meta column.
    planes:
        Restrict features to these evidence planes, e.g.
        ``["telemetry"]`` to reproduce a conventional telemetry-only detector,
        or ``["telemetry", "observability", "eii"]`` for the full picture.
        Ignored when ``features`` is given.
    trust_weighting:
        How to combine the raw score with observability trust.

        ``"none"``    report the raw score (trust is still reported separately);
        ``"discount"`` scale the score by trust, so an anomaly seen through a
        half-blind window ranks below an equally strong one seen clearly;
        ``"boost"``   scale by ``2 - trust``, promoting windows we could *not*
        see. Counter-intuitive, and correct when silence is the failure mode you
        are hunting -- a node that stops reporting produces no anomaly at all in
        conventional detectors.
    """

    def __init__(
        self,
        features: list[str] | None = None,
        planes: list[str] | None = None,
        trust_weighting: str = "none",
    ):
        if trust_weighting not in ("none", "discount", "boost"):
            raise ValueError(
                f"trust_weighting must be 'none', 'discount' or 'boost', got {trust_weighting!r}"
            )

        self.features = features
        self.planes = planes
        self.trust_weighting = trust_weighting
        self.features_: list[str] = []
        self.fitted_ = False

    # -- subclass hooks -----------------------------------------------------

    @abc.abstractmethod
    def _fit(self, values: np.ndarray, frame: pd.DataFrame) -> None: ...

    @abc.abstractmethod
    def _score(self, values: np.ndarray, frame: pd.DataFrame) -> np.ndarray: ...

    # -- public API ---------------------------------------------------------

    def fit(self, frame: pd.DataFrame):
        self.features_ = self._resolve_features(frame)
        if not self.features_:
            raise ValueError("no feature columns selected; check `features`/`planes`")

        self._fit(self._matrix(frame), frame)
        self.fitted_ = True
        return self

    def score(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return ``frame`` with ``anomaly_score`` and ``observability_trust``."""
        if not self.fitted_:
            raise RuntimeError(f"{type(self).__name__} must be fitted before score()")

        missing = [c for c in self.features_ if c not in frame.columns]
        if missing:
            raise KeyError(f"columns missing at score time: {missing}")

        raw = np.asarray(self._score(self._matrix(frame), frame), dtype=float)
        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)

        trust = observability_trust(frame)

        if self.trust_weighting == "discount":
            adjusted = raw * trust
        elif self.trust_weighting == "boost":
            adjusted = raw * (2.0 - trust)
        else:
            adjusted = raw

        result = frame.copy()
        result[schema.SCORE] = adjusted
        result["raw_score"] = raw
        result[schema.TRUST] = trust

        return result

    def alert(self, scored: pd.DataFrame, budget: int | None = None, threshold=None):
        return apply_budget(scored, budget=budget, threshold=threshold)

    def fit_score(self, train: pd.DataFrame, test: pd.DataFrame | None = None):
        self.fit(train)
        return self.score(train if test is None else test)

    # -- helpers ------------------------------------------------------------

    def _resolve_features(self, frame: pd.DataFrame) -> list[str]:
        if self.features is not None:
            missing = [c for c in self.features if c not in frame.columns]
            if missing:
                raise KeyError(f"features not in frame: {missing}")
            return list(self.features)

        candidates = schema.feature_columns(frame, planes=self.planes)
        numeric = frame[candidates].select_dtypes(include="number").columns

        return [c for c in candidates if c in set(numeric)]

    def _matrix(self, frame: pd.DataFrame) -> np.ndarray:
        values = (
            frame[self.features_]
            .apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .to_numpy(dtype=float)
        )
        # NaN -> 0 only *after* the EII layer has already scored the absence.
        return np.nan_to_num(values, nan=0.0)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        state = "fitted" if self.fitted_ else "unfitted"
        return f"{type(self).__name__}({state}, features={len(self.features_)})"
