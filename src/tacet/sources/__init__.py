"""Telemetry inputs: offline files, live endpoints, pushed streams.

Every source yields the same long frame, so the rest of ``tacet`` neither knows
nor cares where the data came from -- and, critically, none of them fabricate a
sample that did not arrive.

Quick start
-----------
>>> import tacet
>>> src = tacet.open_source("data/gpu_telemetry.parquet", entity="node", time="ts")
>>> src = tacet.open_source("prometheus://prom.hpc:9090", queries={...})
"""

from __future__ import annotations

import os
from typing import Any

from .base import LONG_COLUMNS, SourceInfo, TelemetrySource, to_long, to_wide
from .live import CallableSource, PushSource, ReplaySource
from .offline import (
    CsvSource,
    DataFrameSource,
    DirectorySource,
    JsonlSource,
    LongFrameSource,
    ParquetSource,
)
from .prometheus import PrometheusSource

__all__ = [
    "TelemetrySource",
    "SourceInfo",
    "LONG_COLUMNS",
    "to_long",
    "to_wide",
    "DataFrameSource",
    "LongFrameSource",
    "CsvSource",
    "ParquetSource",
    "JsonlSource",
    "DirectorySource",
    "CallableSource",
    "PushSource",
    "ReplaySource",
    "PrometheusSource",
    "open_source",
]

_SCHEMES = {
    "prometheus": PrometheusSource,
    "prom": PrometheusSource,
    "csv": CsvSource,
    "parquet": ParquetSource,
    "jsonl": JsonlSource,
    "dir": DirectorySource,
}

_SUFFIXES = (
    ((".parquet", ".pq"), ParquetSource),
    ((".jsonl", ".ndjson"), JsonlSource),
    ((".csv", ".csv.gz", ".tsv"), CsvSource),
)


def open_source(target: Any, **kwargs) -> TelemetrySource:
    """Build a source from a path, URI, callable, or frame.

    ==============================  =========================================
    ``target``                      resolves to
    ==============================  =========================================
    ``pandas.DataFrame``            :class:`~tacet.sources.DataFrameSource`,
                                    or :class:`LongFrameSource` if the frame
                                    already has ``metric``/``value`` columns
    callable                        :class:`~tacet.sources.CallableSource`
    ``"…/x.parquet"``, ``".csv"``   the matching file source
    a directory or a glob           :class:`~tacet.sources.DirectorySource`
    ``"prometheus://host:9090"``    :class:`~tacet.sources.PrometheusSource`
    ``"scheme://rest"``             the source registered for ``scheme``
    ==============================  =========================================

    Examples
    --------
    >>> open_source("runs/*.csv", entity="node_id")
    >>> open_source("prometheus://prom.hpc:9090", queries={"temp": "DCGM_FI_DEV_GPU_TEMP"})
    >>> open_source(lambda: read_nvidia_smi(), interval=15)
    """
    import pandas as pd

    if isinstance(target, TelemetrySource):
        return target

    if isinstance(target, pd.DataFrame):
        if {"metric", "value"}.issubset(target.columns):
            return LongFrameSource(target, **kwargs)
        return DataFrameSource(target, **kwargs)

    if callable(target):
        return CallableSource(target, **kwargs)

    if not isinstance(target, str):
        raise TypeError(f"cannot build a source from {type(target).__name__}")

    if "://" in target:
        scheme, _, rest = target.partition("://")
        factory = _SCHEMES.get(scheme.lower())
        if factory is None:
            raise ValueError(
                f"unknown source scheme {scheme!r}; known: {sorted(_SCHEMES)}"
            )
        if factory is PrometheusSource:
            return factory(f"http://{rest}", **kwargs)
        return factory(rest, **kwargs)

    if os.path.isdir(target) or any(ch in target for ch in "*?["):
        return DirectorySource(target, **kwargs)

    lowered = target.lower()
    for suffixes, factory in _SUFFIXES:
        if lowered.endswith(suffixes):
            return factory(target, **kwargs)

    raise ValueError(
        f"cannot infer a source type for {target!r}; "
        "pass an explicit source class or use a scheme like 'csv://'"
    )
