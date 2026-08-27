"""EII Cloud generation.

An *EII Cloud* is the full ``(window x parameter x component)`` tensor of Early
Instability Indicator scores for a system, collapsed to whatever view you need:
a per-window anomaly score, a per-parameter ranking, or the annotated figure
that explains which abnormal situation you are looking at.

The cloud is built from six components (:mod:`tacet.eii.components`), half of
which score the monitoring pipeline rather than the machine. That is deliberate:
a window can enter the cloud because a GPU is overheating *or* because the
exporter that would have told you about it stopped answering.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .. import schema
from . import components as comp

__all__ = ["EIICloud", "EIICloudResult"]

#: Default component weights. Observability-flavoured components carry half the
#: total mass, which is what makes the cloud observability-aware rather than
#: "anomaly detection with a coverage metric bolted on".
DEFAULT_WEIGHTS = {
    "value_deviation": 0.20,
    "observability_degradation": 0.20,
    "change_inconsistency": 0.15,
    "contextual_missingness": 0.15,
    "flatline": 0.15,
    "context_contradiction": 0.15,
}


@dataclass
class EIICloudResult:
    """The generated cloud.

    Attributes
    ----------
    frame:
        The input frame plus aggregated ``eii_*`` columns and ``eii_total``.
    components:
        Long-form ``(row, parameter, component, score)`` table -- the cloud
        proper, and the input to :meth:`tips` and the plotting helpers.
    parameters:
        Parameter columns that were scored.
    entity_column, time_column:
        Column names carried through for annotation and plotting.
    """

    frame: pd.DataFrame
    components: pd.DataFrame
    parameters: list[str] = field(default_factory=list)
    entity_column: str | None = None
    time_column: str | None = None
    #: Fitted ``parameter -> (lower, upper)`` envelope, so plots can show the
    #: band that scoring actually used rather than re-deriving one.
    envelopes: dict = field(default_factory=dict)

    def top_parameters(self, n: int = 10) -> pd.DataFrame:
        """Rank parameters by peak and mean EII contribution."""
        grouped = self.components.groupby("parameter")["score"]
        ranking = pd.DataFrame(
            {
                "max_eii": grouped.max(),
                "mean_eii": grouped.mean(),
                "plane": [
                    schema.plane_of(p) for p in grouped.max().index
                ],
            }
        )
        return ranking.sort_values("max_eii", ascending=False).head(n)

    def dominant_component(self) -> pd.Series:
        """For each window, the component contributing the most EII mass."""
        pivot = self.components.pivot_table(
            index="row", columns="component", values="score", aggfunc="max"
        )
        if pivot.empty:
            return pd.Series(dtype=object)
        return pivot.idxmax(axis=1)

    def tips(self, **kwargs):
        """Interpretation callouts for this cloud. See :func:`tacet.eii.tips.generate_tips`."""
        from .tips import generate_tips

        return generate_tips(self, **kwargs)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"EIICloudResult(windows={len(self.frame)}, "
            f"parameters={len(self.parameters)}, "
            f"points={len(self.components)})"
        )


class EIICloud:
    """Fit expected behaviour, then score deviations *and* blind spots.

    Parameters
    ----------
    parameters:
        Columns to score. ``None`` selects every ``tel_``/``ctx_``/``log_``
        feature in the frame.
    weights:
        Per-component weights; missing keys fall back to :data:`DEFAULT_WEIGHTS`.
        Normalised to sum to 1.
    envelope:
        Lower/upper quantiles defining "expected" during :meth:`fit`.
    expected_present:
        Boolean column that is true when context says the parameter *should*
        have reported. Drives :func:`~tacet.eii.components.contextual_missingness`.
        Without it, a silent node cannot be distinguished from an idle one and
        the cloud degrades to conventional anomaly detection.
    high_load:
        Boolean column that is true when the entity is under load. Drives
        :func:`~tacet.eii.components.context_contradiction`.
    flatline_window:
        Trailing window, in samples, for the variance-collapse check.

    Examples
    --------
    >>> cloud = EIICloud(expected_present="ctx_job_active").fit(train)
    >>> result = cloud.transform(test)
    >>> result.top_parameters(5)
    """

    def __init__(
        self,
        parameters: list[str] | None = None,
        weights: dict[str, float] | None = None,
        envelope: tuple[float, float] = (0.05, 0.95),
        expected_present: str | None = None,
        high_load: str | None = None,
        flatline_window: int = 5,
        entity_column: str = schema.ENTITY,
        time_column: str = schema.TIMESTAMP,
    ):
        self.parameters = parameters
        self.envelope = envelope
        self.expected_present = expected_present
        self.high_load = high_load
        self.flatline_window = flatline_window
        self.entity_column = entity_column
        self.time_column = time_column

        merged = dict(DEFAULT_WEIGHTS)
        merged.update(weights or {})
        unknown = set(merged) - set(comp.COMPONENT_NAMES)
        if unknown:
            raise ValueError(f"unknown EII component(s): {sorted(unknown)}")

        vector = np.array([merged[name] for name in comp.COMPONENT_NAMES], dtype=float)
        if vector.sum() <= 0:
            raise ValueError("EII component weights must sum to a positive value")
        self.weights = dict(zip(comp.COMPONENT_NAMES, vector / vector.sum()))

        self.lower_: dict[str, float] = {}
        self.upper_: dict[str, float] = {}
        self.expected_: dict[str, float] = {}
        self.scale_: dict[str, float] = {}
        self.step_scale_: dict[str, float] = {}
        self.idle_level_: dict[str, float] = {}
        self.busy_level_: dict[str, float] = {}
        self.fitted_ = False

    # -- fitting ------------------------------------------------------------

    def fit(self, frame: pd.DataFrame) -> EIICloud:
        """Learn the expected envelope and level for each parameter."""
        parameters = self._resolve_parameters(frame)
        if not parameters:
            raise ValueError(
                "no parameters to score: pass `parameters=[...]` or use tel_/ctx_/log_ prefixes"
            )

        low_q, high_q = self.envelope
        # Idle/busy reference levels, so context_contradiction can tell a cold
        # plateau from a hot one instead of firing on every steady stretch.
        busy_mask = None
        if self.high_load is not None and self.high_load in frame.columns:
            busy_mask = frame[self.high_load].fillna(0).astype(bool)

        for name in parameters:
            series = pd.to_numeric(frame[name], errors="coerce")
            observed = series.dropna()
            if observed.empty:
                self.lower_[name] = 0.0
                self.upper_[name] = 0.0
                self.expected_[name] = 0.0
                self.scale_[name] = 0.0
                self.step_scale_[name] = 0.0
                continue
            self.lower_[name] = float(observed.quantile(low_q))
            self.upper_[name] = float(observed.quantile(high_q))
            self.expected_[name] = float(observed.median())
            # How much this parameter normally moves. Constant flags land at 0
            # and are exempted from the flatline component.
            self.scale_[name] = float(observed.std()) if observed.size > 1 else 0.0
            # Typical step magnitude, robustly estimated, so the rate-of-change
            # and stillness components have a fixed yardstick.
            steps = observed.diff().dropna()
            self.step_scale_[name] = (
                float(1.4826 * (steps - steps.median()).abs().median())
                if len(steps) > 1
                else 0.0
            )

            if busy_mask is not None:
                idle = series[~busy_mask].dropna()
                busy = series[busy_mask].dropna()
                self.idle_level_[name] = (
                    float(idle.median()) if not idle.empty else float(observed.quantile(0.1))
                )
                self.busy_level_[name] = (
                    float(busy.median()) if not busy.empty else float(observed.quantile(0.9))
                )

        self.parameters_ = parameters
        self.fitted_ = True
        return self

    # -- scoring ------------------------------------------------------------

    def transform(self, frame: pd.DataFrame) -> EIICloudResult:
        """Generate the EII Cloud for ``frame``."""
        if not self.fitted_:
            raise RuntimeError("EIICloud must be fitted before transform()")

        frame = self._sorted(frame)
        parameters = [p for p in self.parameters_ if p in frame.columns]
        if not parameters:
            raise ValueError("none of the fitted parameters are present in the frame")

        n = len(frame)
        expected_present = self._context_mask(frame, self.expected_present, default=1.0)
        high_load = self._context_mask(frame, self.high_load, default=0.0)
        # Trailing-window components must never see across an entity boundary,
        # or node B's first window inherits node A's history.
        segments = self._segments(frame)

        records = []
        totals = np.zeros(n, dtype=float)
        aggregates = {name: np.zeros(n, dtype=float) for name in comp.COMPONENT_NAMES}

        for name in parameters:
            series = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)
            # Absence of a reading *is* the observation. This is the hinge the
            # whole library turns on: never impute here.
            observed = np.isfinite(series).astype(float)

            scores = self._score_parameter(
                series, observed, expected_present, high_load, name, segments
            )

            weighted = np.zeros(n, dtype=float)
            for component, values in scores.items():
                weighted += self.weights[component] * values
                aggregates[component] = np.maximum(aggregates[component], values)

                nonzero = np.flatnonzero(values > 0)
                if nonzero.size:
                    records.append(
                        pd.DataFrame(
                            {
                                "row": nonzero,
                                "parameter": name,
                                "component": component,
                                "score": values[nonzero],
                            }
                        )
                    )

            totals = np.maximum(totals, weighted)

        out = frame.copy()
        for component, values in aggregates.items():
            out[f"{schema.EII}{component}"] = values
        out[f"{schema.EII}total"] = totals

        cloud = (
            pd.concat(records, ignore_index=True)
            if records
            else pd.DataFrame(columns=["row", "parameter", "component", "score"])
        )

        return EIICloudResult(
            frame=out,
            components=cloud,
            parameters=parameters,
            entity_column=self.entity_column if self.entity_column in out else None,
            time_column=self.time_column if self.time_column in out else None,
            envelopes={
                name: (self.lower_[name], self.upper_[name]) for name in parameters
            },
        )

    def fit_transform(self, frame: pd.DataFrame) -> EIICloudResult:
        return self.fit(frame).transform(frame)

    # -- internals ----------------------------------------------------------

    def _score_parameter(
        self, series, observed, expected_present, high_load, name, segments
    ):
        lower = np.full(series.shape, self.lower_.get(name, 0.0))
        upper = np.full(series.shape, self.upper_.get(name, 0.0))
        expected = np.full(series.shape, self.expected_.get(name, 0.0))

        # Pointwise components: safe to evaluate over the whole frame at once.
        scores = {
            "value_deviation": comp.value_deviation(series, lower, upper),
            "observability_degradation": comp.observability_degradation(observed),
            "contextual_missingness": comp.contextual_missingness(observed, expected_present),
        }

        # Sequential components: evaluated one entity at a time.
        for key in ("change_inconsistency", "flatline", "context_contradiction"):
            scores[key] = np.zeros(series.shape, dtype=float)

        for start, stop in segments:
            window = slice(start, stop)
            scores["change_inconsistency"][window] = comp.change_inconsistency(
                series[window],
                expected[window],
                observed[window],
                step_scale=self.step_scale_.get(name, 0.0),
            )
            scores["flatline"][window] = comp.flatline(
                series[window],
                observed[window],
                window=self.flatline_window,
                reference_std=self.scale_.get(name, 0.0),
            )
            scores["context_contradiction"][window] = comp.context_contradiction(
                series[window],
                high_load[window],
                observed[window],
                step_scale=self.step_scale_.get(name, 0.0),
                idle_level=self.idle_level_.get(name),
                busy_level=self.busy_level_.get(name),
            )

        return scores

    def _segments(self, frame: pd.DataFrame) -> list[tuple[int, int]]:
        """Contiguous ``[start, stop)`` row ranges, one per entity.

        ``frame`` is already sorted by ``(entity, time)`` in :meth:`_sorted`, so
        each entity occupies a single slice.
        """
        n = len(frame)
        if self.entity_column is None or self.entity_column not in frame.columns:
            return [(0, n)]

        codes = pd.factorize(frame[self.entity_column], sort=False)[0]
        boundaries = np.flatnonzero(np.diff(codes)) + 1
        edges = np.concatenate(([0], boundaries, [n]))

        return [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:]) if b > a]

    def _resolve_parameters(self, frame: pd.DataFrame) -> list[str]:
        if self.parameters is not None:
            missing = [p for p in self.parameters if p not in frame.columns]
            if missing:
                raise KeyError(f"parameters not in frame: {missing}")
            return list(self.parameters)

        return schema.feature_columns(frame, planes=["telemetry", "context", "log"])

    def _context_mask(self, frame, column, default: float) -> np.ndarray:
        if column is None:
            return np.full(len(frame), default, dtype=float)
        if column not in frame.columns:
            raise KeyError(f"context column {column!r} not in frame")
        return frame[column].fillna(0).astype(bool).to_numpy().astype(float)

    def _sorted(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Order rows so that trailing-window components see real time order."""
        keys = [
            c
            for c in (self.entity_column, self.time_column)
            if c is not None and c in frame.columns
        ]
        if not keys:
            return frame.reset_index(drop=True)
        return frame.sort_values(keys).reset_index(drop=True)
