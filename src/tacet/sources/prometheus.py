"""Prometheus / VictoriaMetrics / Thanos source.

Speaks the standard HTTP query API, so it works against anything that implements
it -- which in practice covers most HPC monitoring stacks, including
DCGM-exporter, node-exporter and Slurm exporters.

Requires the optional dependency ``requests`` (``pip install "tacet[prometheus]"``).

Why not just use the Prometheus client's own range query? Because the API
returns *only the points it has*, with no indication of which scrape intervals
were missed. That is precisely the information ``tacet`` needs. This source
reindexes every series onto the requested step grid and leaves the missed
scrapes as ``NaN``.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence

import pandas as pd

from .. import schema
from .base import SourceInfo, TelemetrySource

__all__ = ["PrometheusSource"]


class PrometheusSource(TelemetrySource):
    """Query a Prometheus-compatible endpoint.

    Parameters
    ----------
    url:
        Base URL, e.g. ``"http://prometheus.hpc.example:9090"``.
    queries:
        Mapping of metric name -> PromQL expression. The metric name becomes the
        ``tel_`` column downstream.
    entity_label:
        The label carrying entity identity. ``"instance"`` on most node
        exporters; often ``"Hostname"`` or ``"hostname"`` for DCGM.
    step:
        Resolution of the range query and of the reindex grid.
    start, end:
        Range bounds. ``None`` with ``lookback`` set means "the last *lookback*".
    lookback:
        Window ending now, used when ``start``/``end`` are not given. Also the
        window fetched by each :meth:`stream` iteration.
    timeout:
        Per-request timeout in seconds. A timed-out query yields an all-``NaN``
        grid for that window rather than raising, so a flaky monitoring backend
        registers as reduced observability instead of crashing the pipeline.

    Examples
    --------
    >>> src = PrometheusSource(
    ...     "http://prom:9090",
    ...     queries={
    ...         "gpu_temp": "DCGM_FI_DEV_GPU_TEMP",
    ...         "sm_clock": "DCGM_FI_DEV_SM_CLOCK",
    ...         "ecc_dbe": "DCGM_FI_DEV_ECC_DBE_AGG_TOTAL",
    ...     },
    ...     entity_label="Hostname",
    ...     step="30s",
    ...     lookback="6h",
    ... )
    >>> long = src.read()
    """

    def __init__(
        self,
        url: str,
        queries: Mapping[str, str],
        entity_label: str = "instance",
        step: str = "30s",
        start=None,
        end=None,
        lookback: str = "1h",
        timeout: float = 30.0,
        headers: Mapping[str, str] | None = None,
        verify: bool = True,
        live: bool = True,
    ):
        self.url = url.rstrip("/")
        self.queries = dict(queries)
        self.entity_label = entity_label
        self.step = step
        self.start = start
        self.end = end
        self.lookback = lookback
        self.timeout = timeout
        self.headers = dict(headers or {})
        self.verify = verify

        self.info = SourceInfo(
            kind="prometheus",
            location=self.url,
            live=live,
            detail={"queries": list(self.queries), "step": step},
        )

    # -- HTTP ---------------------------------------------------------------

    def _session(self):
        try:
            import requests
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ImportError(
                "PrometheusSource needs the 'requests' package. "
                'Install it with: pip install "tacet[prometheus]"'
            ) from exc

        session = requests.Session()
        session.headers.update(self.headers)
        session.verify = self.verify
        return session

    def _range(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        end = pd.Timestamp(self.end, tz="UTC") if self.end is not None else pd.Timestamp.now(tz="UTC")
        if self.start is not None:
            start = pd.Timestamp(self.start, tz="UTC")
        else:
            start = end - pd.Timedelta(self.lookback)
        return start, end

    def _query_range(self, session, expression: str, start, end) -> list[dict]:
        response = session.get(
            f"{self.url}/api/v1/query_range",
            params={
                "query": expression,
                "start": start.timestamp(),
                "end": end.timestamp(),
                "step": self.step,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()

        payload = response.json()
        if payload.get("status") != "success":
            raise RuntimeError(
                f"Prometheus rejected query {expression!r}: {payload.get('error')}"
            )

        return payload["data"]["result"]

    # -- source contract ----------------------------------------------------

    def read(self) -> pd.DataFrame:
        start, end = self._range()
        return self._read_range(start, end)

    def _read_range(self, start, end) -> pd.DataFrame:
        session = self._session()
        grid = pd.date_range(
            start.ceil(self.step), end.floor(self.step), freq=self.step, tz="UTC"
        )

        frames = []
        for metric, expression in self.queries.items():
            try:
                series_list = self._query_range(session, expression, start, end)
            except Exception:
                # The monitoring backend itself is unavailable. Record the whole
                # window as unobserved -- do not drop it, and do not raise.
                frames.append(self._blank(metric, grid, entities=["<unreachable>"]))
                continue

            for series in series_list:
                entity = series["metric"].get(self.entity_label)
                if entity is None:
                    entity = series["metric"].get("instance", "<unlabelled>")

                points = pd.DataFrame(series["values"], columns=[schema.TIMESTAMP, "value"])
                points[schema.TIMESTAMP] = pd.to_datetime(
                    points[schema.TIMESTAMP].astype(float), unit="s", utc=True
                )
                points["value"] = pd.to_numeric(points["value"], errors="coerce")

                # Reindex onto the grid: missed scrapes become NaN and survive.
                aligned = (
                    points.set_index(schema.TIMESTAMP)["value"]
                    .reindex(grid.union(points[schema.TIMESTAMP]))
                    .reindex(grid)
                )

                frames.append(
                    pd.DataFrame(
                        {
                            schema.ENTITY: entity,
                            schema.TIMESTAMP: grid,
                            "metric": metric,
                            "value": aligned.to_numpy(),
                        }
                    )
                )

        if not frames:
            return pd.DataFrame(
                columns=[schema.ENTITY, schema.TIMESTAMP, "metric", "value"]
            )

        return pd.concat(frames, ignore_index=True)

    @staticmethod
    def _blank(metric: str, grid, entities: Sequence[str]) -> pd.DataFrame:
        return pd.concat(
            [
                pd.DataFrame(
                    {
                        schema.ENTITY: entity,
                        schema.TIMESTAMP: grid,
                        "metric": metric,
                        "value": float("nan"),
                    }
                )
                for entity in entities
            ],
            ignore_index=True,
        )

    def stream(self) -> Iterator[pd.DataFrame]:
        """Poll the endpoint every ``lookback``, yielding each fresh window."""
        import time as _time

        interval = pd.Timedelta(self.lookback).total_seconds()
        cursor = pd.Timestamp.now(tz="UTC")

        while True:
            _time.sleep(interval)
            now = pd.Timestamp.now(tz="UTC")
            yield self._read_range(cursor, now)
            cursor = now
