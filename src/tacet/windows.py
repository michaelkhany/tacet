"""Windowing, and the automatic construction of the observability plane.

:func:`to_windows` turns irregular samples into the fixed ``(entity, window)``
matrix the rest of the library consumes. It differs from every other resampler
you have used in one respect: it computes how much of each window you *should*
have seen, compares that with what you *did* see, and emits the difference as
first-class ``obs_`` features.

That comparison is only possible because the sources refuse to forward-fill. It
is what lets a detector distinguish "this GPU is fine" from "this GPU has not
spoken in forty minutes and nobody noticed".
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import schema
from .sources.base import to_wide

__all__ = ["to_windows", "label_horizon", "label_episodes", "chronological_split"]

#: Aggregations applied to every metric column.
DEFAULT_AGGREGATIONS = ("mean", "std", "min", "max", "last")


def to_windows(
    data,
    window: str = "1h",
    stride: str | None = None,
    expected_interval: str | None = None,
    aggregations: tuple[str, ...] = DEFAULT_AGGREGATIONS,
    entity: str = schema.ENTITY,
    time: str = schema.TIMESTAMP,
    metrics: list[str] | None = None,
) -> pd.DataFrame:
    """Build a window matrix with a derived observability plane.

    Parameters
    ----------
    data:
        A long frame, a wide frame, or any :class:`~tacet.sources.TelemetrySource`.
    window:
        Window length, e.g. ``"1h"``.
    stride:
        Step between window starts. Defaults to ``window`` (tumbling windows).
        A stride shorter than the window produces overlapping windows, which is
        usually what you want for failure prediction.
    expected_interval:
        The nominal sampling period, e.g. ``"30s"``. This is what makes coverage
        meaningful: without it, ``tacet`` infers the interval from the global
        median inter-sample time, which is fine for regular collectors and
        misleading for bursty ones. **Pass it if you know it.**
    aggregations:
        Per-metric aggregations. ``std`` of a single sample is ``NaN`` by
        definition and is left as such.
    metrics:
        Restrict to these metric columns.

    Returns
    -------
    DataFrame
        One row per ``(entity, window)`` with:

        ``tel_<metric>_<agg>``      aggregated telemetry
        ``obs_coverage``            observed / expected samples, in ``[0, 1]``
        ``obs_observed_samples``    samples that actually arrived
        ``obs_expected_samples``    samples the collector should have produced
        ``obs_reporting``           1 if the entity said anything at all
        ``obs_max_gap_seconds``     longest silence inside the window
        ``obs_stale_seconds``       age of the last sample at window close
        ``obs_<metric>_coverage``   per-metric coverage, for partial outages

    Examples
    --------
    >>> windows = to_windows(source, window="1h", stride="10min", expected_interval="30s")
    >>> windows[["obs_coverage", "obs_max_gap_seconds"]].describe()
    """
    frame = _materialise(data, entity=entity, time=time, metrics=metrics)
    if frame.empty:
        raise ValueError("no telemetry to window")

    window_delta = pd.Timedelta(window)
    stride_delta = pd.Timedelta(stride) if stride else window_delta
    if stride_delta <= pd.Timedelta(0) or window_delta <= pd.Timedelta(0):
        raise ValueError("window and stride must be positive durations")
    if window_delta % stride_delta != pd.Timedelta(0):
        raise ValueError(
            f"window ({window}) must be a whole multiple of stride ({stride}) "
            "so that every sample lands in the same number of windows"
        )

    interval = _resolve_interval(frame, expected_interval)
    expanded, origin = _assign_windows(frame, window_delta, stride_delta)

    metric_columns = [
        c for c in expanded.columns
        if c not in (schema.ENTITY, schema.TIMESTAMP, "_window")
    ]
    # Columns that already declare a plane keep it. Prefixing blindly turns an
    # input's `ctx_job_active` into `tel_ctx_job_active`, which silently moves
    # context evidence into the telemetry plane and corrupts every downstream
    # plane-aware calculation.
    resolved = {
        column: column
        if schema.plane_of(column) != "meta"
        else f"{schema.TELEMETRY}{column}"
        for column in metric_columns
    }

    grouped = expanded.groupby([schema.ENTITY, "_window"], sort=True)

    telemetry = grouped[metric_columns].agg(list(aggregations))
    telemetry.columns = [
        f"{resolved[metric]}_{agg}" for metric, agg in telemetry.columns
    ]

    observability = _observability_plane(
        expanded, metric_columns, resolved, window_delta, stride_delta, origin, interval
    )

    matrix = pd.concat([telemetry, observability], axis=1).reset_index()
    matrix[schema.WINDOW_START] = origin + matrix["_window"] * stride_delta
    matrix[schema.WINDOW_END] = matrix[schema.WINDOW_START] + window_delta
    matrix[schema.TIMESTAMP] = matrix[schema.WINDOW_END]
    matrix[schema.WINDOW_ID] = (
        matrix[schema.ENTITY].astype(str) + "#" + matrix["_window"].astype(str)
    )
    matrix = matrix.drop(columns=["_window"])

    ordered = [
        schema.ENTITY,
        schema.WINDOW_ID,
        schema.WINDOW_START,
        schema.WINDOW_END,
        schema.TIMESTAMP,
    ]
    rest = [c for c in matrix.columns if c not in ordered]

    return matrix[ordered + rest].sort_values(
        [schema.ENTITY, schema.WINDOW_START]
    ).reset_index(drop=True)


def _materialise(data, entity: str, time: str, metrics) -> pd.DataFrame:
    """Coerce any accepted input into a wide, time-sorted frame."""
    from .sources import TelemetrySource, open_source

    if isinstance(data, TelemetrySource):
        frame = to_wide(data.read())
    elif isinstance(data, pd.DataFrame) and {"metric", "value"}.issubset(data.columns):
        frame = to_wide(data)
    elif isinstance(data, pd.DataFrame):
        frame = data.rename(
            columns={entity: schema.ENTITY, time: schema.TIMESTAMP}
        ).copy()
    else:
        frame = to_wide(open_source(data).read())

    if schema.ENTITY not in frame.columns:
        raise KeyError(f"entity column {entity!r} not found in input")
    if schema.TIMESTAMP not in frame.columns:
        raise KeyError(f"time column {time!r} not found in input")

    frame[schema.TIMESTAMP] = pd.to_datetime(
        frame[schema.TIMESTAMP], utc=True, errors="coerce"
    )
    frame = frame.dropna(subset=[schema.TIMESTAMP])

    if metrics is not None:
        keep = [schema.ENTITY, schema.TIMESTAMP, *metrics]
        frame = frame[[c for c in keep if c in frame.columns]]

    numeric = frame.select_dtypes(include="number").columns
    keep = [schema.ENTITY, schema.TIMESTAMP, *numeric]

    return frame[keep].sort_values([schema.ENTITY, schema.TIMESTAMP]).reset_index(drop=True)


def _resolve_interval(frame: pd.DataFrame, expected_interval) -> pd.Timedelta:
    if expected_interval is not None:
        return pd.Timedelta(expected_interval)

    deltas = frame.groupby(schema.ENTITY)[schema.TIMESTAMP].diff().dropna()
    if deltas.empty:
        return pd.Timedelta("1min")

    median = deltas.median()
    return median if median > pd.Timedelta(0) else pd.Timedelta("1min")


def _assign_windows(frame, window_delta, stride_delta):
    """Replicate each sample into every overlapping window that contains it."""
    origin = frame[schema.TIMESTAMP].min().floor(stride_delta)
    offsets = (frame[schema.TIMESTAMP] - origin) // stride_delta

    overlap = int(window_delta // stride_delta)

    parts = []
    for back in range(overlap):
        part = frame.copy()
        part["_window"] = offsets - back
        parts.append(part[part["_window"] >= 0])

    return pd.concat(parts, ignore_index=True), origin


def _observability_plane(
    expanded, metric_columns, resolved, window_delta, stride_delta, origin, interval
):
    """Derive the ``obs_`` features from presence, absence and timing.

    Coverage is measured over the **telemetry** plane only. Context and log
    columns are typically joined in from a scheduler or a parser that keeps
    working when the node's exporter dies, so counting them as evidence of
    presence reports full coverage for a window in which the machine said
    nothing at all.
    """
    prefix = schema.OBSERVABILITY
    expected = max(float(window_delta / interval), 1.0)
    keys = [schema.ENTITY, "_window"]

    signal_columns = [
        column
        for column in metric_columns
        if schema.plane_of(resolved[column]) == "telemetry"
    ] or list(metric_columns)

    present_mask = expanded[signal_columns].notna().any(axis=1)
    frame = expanded.assign(_present=present_mask.astype(float))
    grouped = frame.groupby(keys, sort=True)

    observed = grouped["_present"].sum()

    plane = pd.DataFrame(index=observed.index)
    plane[f"{prefix}observed_samples"] = observed
    plane[f"{prefix}expected_samples"] = expected
    plane[f"{prefix}coverage"] = (observed / expected).clip(0.0, 1.0)
    plane[f"{prefix}reporting"] = (observed > 0).astype(float)

    # True window bounds, so edge silence is measured against the window rather
    # than against whichever sample happened to arrive first.
    window_index = plane.index.get_level_values("_window").to_numpy()
    window_start = origin + window_index * stride_delta
    window_end = window_start + window_delta
    span = float(window_delta.total_seconds())

    present = frame[frame["_present"] > 0]

    if present.empty:
        plane[f"{prefix}max_gap_seconds"] = span
        plane[f"{prefix}stale_seconds"] = span
    else:
        present = present.sort_values(keys + [schema.TIMESTAMP])
        interior = (
            present.groupby(keys, sort=True)[schema.TIMESTAMP]
            .diff()
            .dt.total_seconds()
        )
        interior_max = (
            interior.groupby([present[schema.ENTITY], present["_window"]])
            .max()
            .reindex(plane.index)
        )

        times = present.groupby(keys, sort=True)[schema.TIMESTAMP]
        first_seen = times.min().reindex(plane.index)
        last_seen = times.max().reindex(plane.index)

        # Silence before the first sample and after the last one both count.
        lead = (first_seen - pd.Series(window_start, index=plane.index)).dt.total_seconds()
        trail = (pd.Series(window_end, index=plane.index) - last_seen).dt.total_seconds()

        plane[f"{prefix}max_gap_seconds"] = (
            pd.concat([interior_max, lead, trail], axis=1)
            .max(axis=1)
            .fillna(span)
            .clip(0.0, span)
        )
        plane[f"{prefix}stale_seconds"] = trail.fillna(span).clip(0.0, span)

    # Per-metric coverage: catches an exporter losing one field while the rest
    # of the scrape still succeeds, which whole-window coverage would hide.
    for metric in signal_columns:
        counts = (
            frame[metric].notna().astype(float)
            .groupby([frame[schema.ENTITY], frame["_window"]])
            .sum()
            .reindex(plane.index)
        )
        plane[f"{prefix}{resolved[metric]}_coverage"] = (counts / expected).clip(0.0, 1.0)

    return plane


def label_horizon(
    windows: pd.DataFrame,
    events: pd.DataFrame,
    horizon: str = "4h",
    entity: str = schema.ENTITY,
    time: str = schema.TIMESTAMP,
) -> pd.DataFrame:
    """Label windows that fall within ``horizon`` before a known event.

    This is the supervision signal for failure *prediction*: a window is
    positive if the entity failed within the horizon after the window closed.

    Parameters
    ----------
    events:
        One row per event with an entity column and a timestamp column.
    horizon:
        Lead time. ``"4h"`` asks "would we have had four hours of warning?".
    """
    if events.empty:
        return windows.assign(**{schema.LABEL: 0, schema.EVENT_TIME: pd.NaT})

    horizon_delta = pd.Timedelta(horizon)

    events = events.rename(
        columns={entity: schema.ENTITY, time: schema.EVENT_TIME}
    ).copy()
    events[schema.EVENT_TIME] = pd.to_datetime(
        events[schema.EVENT_TIME], utc=True, errors="coerce"
    )
    events = events.dropna(subset=[schema.EVENT_TIME])[
        [schema.ENTITY, schema.EVENT_TIME]
    ].sort_values(schema.EVENT_TIME)

    labelled = windows.sort_values(schema.WINDOW_END).copy()

    # For each window, the first event at or after the window close.
    joined = pd.merge_asof(
        labelled,
        events,
        left_on=schema.WINDOW_END,
        right_on=schema.EVENT_TIME,
        by=schema.ENTITY,
        direction="forward",
        allow_exact_matches=True,
    )

    lead = joined[schema.EVENT_TIME] - joined[schema.WINDOW_END]
    joined[schema.LABEL] = (
        lead.notna() & (lead >= pd.Timedelta(0)) & (lead <= horizon_delta)
    ).astype(int)

    return joined.sort_values([schema.ENTITY, schema.WINDOW_START]).reset_index(drop=True)


def label_episodes(
    windows: pd.DataFrame,
    episodes: pd.DataFrame,
    entity: str = schema.ENTITY,
    start: str = "start",
    end: str = "end",
    kind: str | None = "kind",
) -> pd.DataFrame:
    """Label windows that **overlap** a known fault episode.

    The detection counterpart to :func:`label_horizon`. Use this to ask "did the
    detector notice while it was happening?"; use ``label_horizon`` to ask "did
    it warn beforehand?". Reporting one and calling it the other is a common and
    consequential mistake -- detection scores flatter a method considerably.

    Parameters
    ----------
    episodes:
        One row per episode with entity, start and end columns.
    kind:
        Optional column carried onto matching windows as ``event_kind``.

    Examples
    --------
    >>> windows = label_episodes(windows, truth[truth.family == "observation"])
    """
    result = windows.copy()
    result[schema.LABEL] = 0
    if kind is not None:
        result["event_kind"] = None

    if episodes.empty:
        return result

    for column in (entity, start, end):
        if column not in episodes.columns:
            raise KeyError(f"episodes frame has no {column!r} column")

    starts = pd.to_datetime(episodes[start], utc=True, errors="coerce")
    ends = pd.to_datetime(episodes[end], utc=True, errors="coerce")

    window_start = result[schema.WINDOW_START]
    window_end = result[schema.WINDOW_END]
    entities = result[entity]

    for position, episode in enumerate(episodes.itertuples(index=False)):
        episode_start = starts.iloc[position]
        episode_end = ends.iloc[position]
        if pd.isna(episode_start) or pd.isna(episode_end):
            continue

        # Half-open overlap: the window touches the episode at all.
        overlaps = (
            (entities == getattr(episode, entity))
            & (window_start <= episode_end)
            & (window_end >= episode_start)
        )
        result.loc[overlaps, schema.LABEL] = 1

        if kind is not None and hasattr(episode, kind):
            result.loc[overlaps, "event_kind"] = getattr(episode, kind)

    return result


def chronological_split(
    windows: pd.DataFrame,
    train: float = 0.60,
    validation: float = 0.20,
    time: str = schema.WINDOW_START,
) -> pd.DataFrame:
    """Split by time, never at random.

    Random splits leak the future into the training set and inflate every metric
    you will report. ``tacet`` makes the honest split the easy one.
    """
    if not 0 < train < 1 or not 0 <= validation < 1 or train + validation >= 1:
        raise ValueError("train and validation must be fractions with train + validation < 1")

    ordered = windows.sort_values(time).reset_index(drop=True)
    n = len(ordered)
    train_end = int(n * train)
    validation_end = int(n * (train + validation))

    split = np.full(n, "test", dtype=object)
    split[:train_end] = "train"
    split[train_end:validation_end] = "validation"
    ordered[schema.SPLIT] = split

    return ordered
