"""Telemetry source abstraction.

Every source -- a parquet dump, a live Prometheus endpoint, a socket someone
pushes into -- produces the same thing: a **long frame** of

    ``(entity_id, timestamp, metric, value)``

and nothing else. Widening, windowing and scoring happen downstream, identically
for offline and live data, so a detector developed against a captured dataset
runs unchanged against a production stream.

The contract that matters
-------------------------
A ``tacet`` source **must not invent samples**. No forward-fill, no
interpolation, no dropping of empty scrapes. Conventional ingestion layers do
all three as a courtesy, and in doing so they destroy the single most valuable
signal this library consumes: the fact that a machine stopped answering. If a
sample did not arrive, it must arrive downstream as ``NaN``.
"""

from __future__ import annotations

import abc
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import pandas as pd

from .. import schema

__all__ = ["TelemetrySource", "LONG_COLUMNS", "to_long", "to_wide", "SourceInfo"]

#: The canonical long-frame columns every source yields.
LONG_COLUMNS = (schema.ENTITY, schema.TIMESTAMP, "metric", "value")


@dataclass(frozen=True)
class SourceInfo:
    """Provenance for a source, carried into reports so results stay traceable."""

    kind: str
    location: str
    live: bool = False
    detail: dict | None = None


class TelemetrySource(abc.ABC):
    """Base class for every telemetry input.

    Subclasses implement :meth:`read`. Sources that can deliver data
    incrementally also override :meth:`stream`; the default implementation
    yields a single batch so that offline sources can be dropped into streaming
    pipelines unchanged.
    """

    #: Set by subclasses for provenance reporting.
    info: SourceInfo

    @abc.abstractmethod
    def read(self) -> pd.DataFrame:
        """Return all available data as a long frame (see :data:`LONG_COLUMNS`)."""

    def stream(self) -> Iterator[pd.DataFrame]:
        """Yield long frames as they become available.

        Offline sources yield exactly once. Live sources yield until stopped.
        """
        yield self.read()

    def is_live(self) -> bool:
        return bool(getattr(self, "info", None) and self.info.live)

    # -- convenience --------------------------------------------------------

    def to_wide(self, prefix: str = schema.TELEMETRY) -> pd.DataFrame:
        """Read and pivot to one column per metric, gaps preserved as ``NaN``."""
        return to_wide(self.read(), prefix=prefix)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        info = getattr(self, "info", None)
        if info is None:
            return f"{type(self).__name__}()"
        return f"{type(self).__name__}(location={info.location!r}, live={info.live})"


def to_long(
    frame: pd.DataFrame,
    entity: str = schema.ENTITY,
    time: str = schema.TIMESTAMP,
    metrics: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Melt a wide telemetry frame into the canonical long form.

    Unlike :func:`pandas.melt`, missing values are **kept**. They are the point.
    """
    if entity not in frame.columns:
        raise KeyError(f"entity column {entity!r} not found; got {list(frame.columns)}")
    if time not in frame.columns:
        raise KeyError(f"time column {time!r} not found; got {list(frame.columns)}")

    value_columns = (
        list(metrics)
        if metrics is not None
        else [c for c in frame.columns if c not in (entity, time)]
    )

    long = frame.melt(
        id_vars=[entity, time],
        value_vars=value_columns,
        var_name="metric",
        value_name="value",
    )
    long = long.rename(columns={entity: schema.ENTITY, time: schema.TIMESTAMP})
    long[schema.TIMESTAMP] = pd.to_datetime(long[schema.TIMESTAMP], utc=True, errors="coerce")

    return long[list(LONG_COLUMNS)]


def to_wide(long: pd.DataFrame, prefix: str = schema.TELEMETRY) -> pd.DataFrame:
    """Pivot a long frame to one prefixed column per metric.

    Absent ``(entity, timestamp, metric)`` combinations become ``NaN`` and stay
    that way.
    """
    missing = [c for c in LONG_COLUMNS if c not in long.columns]
    if missing:
        raise KeyError(f"long frame is missing columns: {missing}")

    wide = long.pivot_table(
        index=[schema.ENTITY, schema.TIMESTAMP],
        columns="metric",
        values="value",
        aggfunc="mean",
        dropna=False,
    )
    wide.columns = [
        c if str(c).startswith(prefix) else f"{prefix}{c}" for c in wide.columns
    ]

    return wide.reset_index().sort_values([schema.ENTITY, schema.TIMESTAMP])
