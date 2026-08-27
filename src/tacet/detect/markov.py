"""Markov-chain detection over discretised system states.

The system is reduced to a small alphabet of states by quantising a
standardised aggregate of its features, and a first-order transition matrix is
learned from healthy operation. A window then scores on two things: how
*surprising* the move into its state was, and how far the state sits from
normal.

Both terms are needed. Pure transition surprise is blind to level -- a system
that settles into its worst state and stays there stops being surprising after
one step, which is exactly backwards. Pure level is blind to dynamics. The blend
is controlled by ``transition_weight``.

Ported and generalised from the reference implementation used in the PDM /
EII Cloud evaluation study.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from .. import schema
from .base import BaseDetector

__all__ = ["MarkovDetector", "mine_event_sequences"]


class MarkovDetector(BaseDetector):
    """First-order Markov anomaly detector with transition mining.

    Parameters
    ----------
    n_states:
        Size of the state alphabet. Quantile-binned, so states are equally
        populated in training regardless of the score distribution's shape.
    transition_weight:
        Weight on the transition-surprise term; the remainder goes to state
        deviation. ``0.30`` was near-optimal by sweep on HPC and disk telemetry.
    smoothing:
        Laplace count added to every transition, so an unseen move is
        surprising rather than infinitely surprising.

    Notes
    -----
    Features are standardised with **training** statistics before aggregation.
    Skipping that step lets whichever feature happens to have the largest
    magnitude dictate the state, which collapses single-plane baselines to a
    constant score and makes the detector look no better than random.

    Examples
    --------
    >>> detector = MarkovDetector(planes=["telemetry", "observability", "eii"])
    >>> scored = detector.fit(train).score(test)
    >>> detector.transition_matrix()
    """

    def __init__(
        self,
        n_states: int = 5,
        transition_weight: float = 0.30,
        smoothing: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if n_states < 2:
            raise ValueError("n_states must be at least 2")

        self.n_states = n_states
        self.transition_weight = float(np.clip(transition_weight, 0.0, 1.0))
        self.smoothing = smoothing

        self.quantiles_: np.ndarray | None = None
        self.transitions_: np.ndarray | None = None

    # -- detector contract --------------------------------------------------

    def _fit(self, values, frame):
        self.center_ = np.nanmean(values, axis=0)
        scale = np.nanstd(values, axis=0)
        self.scale_ = np.where(scale <= 1e-12, 1.0, scale)

        raw = self._state_score(values)
        self.quantiles_ = np.quantile(raw, np.linspace(0, 1, self.n_states + 1))
        self.train_min_ = float(np.nanmin(raw))
        self.train_max_ = float(np.nanmax(raw))

        states = self._discretise(raw)
        self.transitions_ = self._transition_matrix(states, frame)

    def _score(self, values, frame):
        raw = self._state_score(values)
        states = self._discretise(raw)

        if states.size == 0:
            return np.zeros(0, dtype=float)

        surprise = self._surprise(states, frame)

        # Deviation is normalised with *training* extremes, so scores from
        # different test batches remain comparable -- a per-batch min/max would
        # silently rescale every run and make thresholds meaningless.
        span = max(self.train_max_ - self.train_min_, 1e-9)
        deviation = np.clip((raw - self.train_min_) / span, 0.0, 1.0)

        return (
            self.transition_weight * surprise
            + (1.0 - self.transition_weight) * deviation
        )

    # -- internals ----------------------------------------------------------

    def _state_score(self, values: np.ndarray) -> np.ndarray:
        standardised = (values - self.center_) / self.scale_
        return np.abs(np.nan_to_num(standardised)).mean(axis=1)

    def _discretise(self, raw: np.ndarray) -> np.ndarray:
        raw = np.nan_to_num(np.asarray(raw, dtype=float))
        if raw.size == 0:
            return np.zeros(0, dtype=int)

        if self.quantiles_ is None or np.nanstd(raw) == 0.0:
            return np.zeros(raw.size, dtype=int)

        # Interior edges only, deduplicated: a plane where most features are
        # constant produces repeated quantiles, and np.digitize rejects those.
        edges = np.unique(np.nan_to_num(self.quantiles_[1:-1]))
        edges = edges[np.isfinite(edges)]

        if edges.size == 0:
            return np.zeros(raw.size, dtype=int)

        states = np.digitize(raw, edges, right=True)
        return np.clip(states, 0, self.n_states - 1).astype(int)

    def _segments(self, frame: pd.DataFrame) -> list[tuple[int, int]]:
        """Contiguous row ranges per entity, so transitions stay within a node."""
        rows = len(frame)
        if schema.ENTITY not in frame.columns:
            return [(0, rows)]

        codes = pd.factorize(frame[schema.ENTITY], sort=False)[0]
        edges = np.concatenate(([0], np.flatnonzero(np.diff(codes)) + 1, [rows]))

        return [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:]) if b > a]

    def _transition_matrix(self, states: np.ndarray, frame: pd.DataFrame) -> np.ndarray:
        matrix = np.full((self.n_states, self.n_states), self.smoothing, dtype=float)

        for start, stop in self._segments(frame):
            chunk = states[start:stop]
            if chunk.size < 2:
                continue
            np.add.at(matrix, (chunk[:-1], chunk[1:]), 1.0)

        return matrix / matrix.sum(axis=1, keepdims=True)

    def _surprise(self, states: np.ndarray, frame: pd.DataFrame) -> np.ndarray:
        out = np.zeros(states.size, dtype=float)

        for start, stop in self._segments(frame):
            chunk = states[start:stop]
            if chunk.size == 0:
                continue

            previous = np.concatenate(([chunk[0]], chunk[:-1]))
            out[start:stop] = 1.0 - self.transitions_[previous, chunk]

        return out

    # -- inspection ---------------------------------------------------------

    def transition_matrix(self) -> pd.DataFrame:
        """The learned ``P(next state | current state)``, as a labelled frame."""
        if self.transitions_ is None:
            raise RuntimeError("fit the detector first")

        labels = [f"S{i}" for i in range(self.n_states)]
        return pd.DataFrame(self.transitions_, index=labels, columns=labels)

    def rare_transitions(self, n: int = 10) -> pd.DataFrame:
        """The least likely state moves -- the ones that should alarm you."""
        matrix = self.transition_matrix()

        records = [
            {"from": source, "to": target, "probability": float(matrix.at[source, target])}
            for source in matrix.index
            for target in matrix.columns
        ]

        return (
            pd.DataFrame(records)
            .sort_values("probability")
            .head(n)
            .reset_index(drop=True)
        )


def mine_event_sequences(
    events: pd.DataFrame,
    event_column: str,
    entity: str = schema.ENTITY,
    time: str = schema.TIMESTAMP,
    max_depth: int = 3,
    min_probability: float = 0.60,
    min_support: int = 5,
    top_events: int = 100,
) -> pd.DataFrame:
    """Mine multi-level cause-and-effect chains from a discrete event log.

    Expands sequences layer by layer::

        layer 1:  P(B | A)
        layer 2:  P(C | A, B)
        layer 3:  P(D | A, B, C)

    keeping only continuations that clear ``min_probability`` with at least
    ``min_support`` observations. Repeated events are allowed -- a component
    that throws the same error three times before dying is a real pattern -- but
    identical full sequences are never duplicated.

    Sequences that cross evidence planes are the valuable output: a chain that
    runs ``obs_scrape_missed -> tel_temp_high -> node_down`` says the monitoring
    gap came *first*, which changes both the diagnosis and who gets paged.

    Parameters
    ----------
    events:
        Long event log with entity, timestamp and a discrete event column.
    max_depth:
        Longest chain to expand. Cost grows quickly; 3 is usually enough.

    Returns
    -------
    DataFrame
        ``sequence``, ``next_event``, ``probability``, ``support``, ``depth``,
        ``crosses_planes``.
    """
    frame = events[[entity, time, event_column]].dropna().copy()
    frame[time] = pd.to_datetime(frame[time], utc=True, errors="coerce")
    frame = frame.dropna(subset=[time]).sort_values([entity, time])

    if frame.empty:
        return pd.DataFrame(
            columns=["sequence", "next_event", "probability", "support", "depth", "crosses_planes"]
        )

    vocabulary = set(
        frame[event_column].value_counts().head(top_events).index
    )
    frame = frame[frame[event_column].isin(vocabulary)]

    chains = [
        tuple(values) for _, values in frame.groupby(entity, sort=False)[event_column]
    ]

    records = []
    seen: set[tuple] = set()

    for depth in range(1, max_depth + 1):
        counts: dict[tuple, dict] = defaultdict(lambda: defaultdict(int))

        for chain in chains:
            for i in range(len(chain) - depth):
                prefix = tuple(chain[i : i + depth])
                counts[prefix][chain[i + depth]] += 1

        for prefix, followers in counts.items():
            total = sum(followers.values())
            if total < min_support:
                continue

            for follower, count in followers.items():
                probability = count / total
                if probability < min_probability:
                    continue

                full = prefix + (follower,)
                if full in seen:
                    continue
                seen.add(full)

                planes = {schema.plane_of(str(e)) for e in full}
                records.append(
                    {
                        "sequence": " -> ".join(str(e) for e in prefix),
                        "next_event": str(follower),
                        "probability": probability,
                        "support": count,
                        "depth": depth,
                        "crosses_planes": len(planes) > 1,
                    }
                )

    result = pd.DataFrame(records)
    if result.empty:
        return result

    return result.sort_values(
        ["depth", "probability", "support"], ascending=[True, False, False]
    ).reset_index(drop=True)
