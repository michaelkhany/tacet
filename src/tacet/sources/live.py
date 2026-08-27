"""Live telemetry sources: polling, push, and tailing.

The same long-frame contract as the offline sources, produced incrementally.

Liveness changes one thing that matters here. Offline, a missing sample is
unambiguous -- the export is complete, so the gap is real. Live, "no data yet"
and "no data ever" look identical until enough time has passed. Every live
source therefore emits on a fixed grid and marks a slot missing only once its
collection deadline has passed, so a slow scrape is not mistaken for a dead node.
"""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Iterator, Sequence
from typing import Callable

import pandas as pd

from .. import schema
from .base import SourceInfo, TelemetrySource, to_long

__all__ = ["CallableSource", "PushSource", "ReplaySource"]


class CallableSource(TelemetrySource):
    """Poll any function on an interval.

    The most general live adapter: hand it something that returns the current
    readings and it becomes a ``tacet`` source. The callable may return a wide
    frame, a long frame, or a ``{metric: value}`` mapping per entity.

    Parameters
    ----------
    fetch:
        Zero-argument callable returning the current sample(s).
    interval:
        Seconds between polls.
    max_batches:
        Stop after this many polls. ``None`` runs until interrupted.
    timeout:
        Seconds to allow ``fetch`` before the poll is recorded as a **failed
        collection** -- an empty batch, which downstream reads as a genuine
        observability gap rather than silently skipping the slot.

    Examples
    --------
    >>> src = CallableSource(lambda: nvidia_smi_readings(), interval=15)
    >>> for batch in src.stream():
    ...     ...
    """

    def __init__(
        self,
        fetch: Callable[[], object],
        interval: float = 30.0,
        max_batches: int | None = None,
        timeout: float | None = None,
        entity: str = schema.ENTITY,
        time_column: str = schema.TIMESTAMP,
        name: str = "<callable>",
    ):
        self._fetch = fetch
        self.interval = float(interval)
        self.max_batches = max_batches
        self.timeout = timeout
        self._entity = entity
        self._time = time_column
        self.info = SourceInfo(kind="callable", location=name, live=True)
        self._stop = threading.Event()

    def stop(self) -> None:
        """Ask :meth:`stream` to finish after the current poll."""
        self._stop.set()

    def _normalize(self, payload) -> pd.DataFrame:
        if payload is None:
            return pd.DataFrame(columns=list(schema.META_COLUMNS[:2]) + ["metric", "value"])

        if isinstance(payload, dict):
            payload = pd.DataFrame(payload)

        if not isinstance(payload, pd.DataFrame):
            raise TypeError(
                f"fetch() must return a DataFrame or dict, got {type(payload).__name__}"
            )

        if payload.empty:
            return payload

        if {"metric", "value"}.issubset(payload.columns):
            frame = payload.copy()
            if self._entity in frame.columns and self._entity != schema.ENTITY:
                frame = frame.rename(columns={self._entity: schema.ENTITY})
            if self._time in frame.columns and self._time != schema.TIMESTAMP:
                frame = frame.rename(columns={self._time: schema.TIMESTAMP})
            if schema.TIMESTAMP not in frame.columns:
                frame[schema.TIMESTAMP] = pd.Timestamp.now(tz="UTC")
            frame[schema.TIMESTAMP] = pd.to_datetime(
                frame[schema.TIMESTAMP], utc=True, errors="coerce"
            )
            return frame

        if self._time not in payload.columns:
            payload = payload.assign(**{self._time: pd.Timestamp.now(tz="UTC")})

        return to_long(payload, entity=self._entity, time=self._time)

    def _poll_once(self) -> pd.DataFrame:
        if self.timeout is None:
            return self._normalize(self._fetch())

        result: list = []
        worker = threading.Thread(
            target=lambda: result.append(self._fetch()), daemon=True
        )
        worker.start()
        worker.join(self.timeout)

        if worker.is_alive() or not result:
            # Collection missed its deadline. Emit nothing: the windowing layer
            # will record an unobserved slot, which is exactly what happened.
            return self._normalize(None)

        return self._normalize(result[0])

    def stream(self) -> Iterator[pd.DataFrame]:
        batches = 0
        while not self._stop.is_set():
            started = time.monotonic()

            try:
                batch = self._poll_once()
            except Exception:
                # A collector that raises is itself an observability failure.
                # Surface it as a gap rather than tearing down the stream.
                batch = self._normalize(None)

            yield batch

            batches += 1
            if self.max_batches is not None and batches >= self.max_batches:
                return

            elapsed = time.monotonic() - started
            self._stop.wait(max(0.0, self.interval - elapsed))

    def read(self) -> pd.DataFrame:
        """Collect a single poll. Provided so live sources satisfy the base contract."""
        return self._poll_once()


class PushSource(TelemetrySource):
    """A queue that external code pushes samples into.

    For agents, webhooks, Kafka consumers, MQTT callbacks -- anything that
    delivers data to you rather than being asked for it.

    Examples
    --------
    >>> src = PushSource(batch_seconds=10)
    >>> src.push(entity="gpu-042", metric="sm_clock", value=1410)
    >>> next(src.stream())
    """

    def __init__(self, batch_seconds: float = 10.0, name: str = "<push>", max_queue: int = 100_000):
        self.batch_seconds = float(batch_seconds)
        self._queue: queue.Queue = queue.Queue(maxsize=max_queue)
        self._stop = threading.Event()
        self.info = SourceInfo(kind="push", location=name, live=True)
        #: Samples dropped because the queue was full. Non-zero means the
        #: consumer is falling behind, which is itself worth alerting on.
        self.dropped = 0

    def push(self, entity: str, metric: str, value: float, timestamp=None) -> None:
        record = {
            schema.ENTITY: entity,
            schema.TIMESTAMP: timestamp or pd.Timestamp.now(tz="UTC"),
            "metric": metric,
            "value": value,
        }
        try:
            self._queue.put_nowait(record)
        except queue.Full:
            self.dropped += 1

    def push_many(self, records: Sequence[dict]) -> None:
        for record in records:
            self.push(**record)

    def stop(self) -> None:
        self._stop.set()

    def _drain(self) -> pd.DataFrame:
        records = []
        while True:
            try:
                records.append(self._queue.get_nowait())
            except queue.Empty:
                break

        if not records:
            return pd.DataFrame(columns=[schema.ENTITY, schema.TIMESTAMP, "metric", "value"])

        frame = pd.DataFrame(records)
        frame[schema.TIMESTAMP] = pd.to_datetime(frame[schema.TIMESTAMP], utc=True)
        return frame

    def read(self) -> pd.DataFrame:
        return self._drain()

    def stream(self) -> Iterator[pd.DataFrame]:
        while not self._stop.is_set():
            self._stop.wait(self.batch_seconds)
            yield self._drain()


class ReplaySource(TelemetrySource):
    """Replay an offline source as if it were live.

    The bridge between the two worlds: develop and tune against a captured
    dataset with the exact code path production will run, at ``speed`` times
    real time (``0`` replays as fast as possible).
    """

    def __init__(
        self,
        source: TelemetrySource,
        batch: str = "5min",
        speed: float = 0.0,
    ):
        self._source = source
        self.batch = batch
        self.speed = float(speed)
        self.info = SourceInfo(
            kind="replay", location=source.info.location, live=True,
            detail={"replays": source.info.kind},
        )

    def read(self) -> pd.DataFrame:
        return self._source.read()

    def stream(self) -> Iterator[pd.DataFrame]:
        frame = self._source.read()
        if frame.empty:
            return

        frame = frame.sort_values(schema.TIMESTAMP)
        buckets = frame[schema.TIMESTAMP].dt.floor(self.batch)
        step = pd.Timedelta(self.batch).total_seconds()

        for _, chunk in frame.groupby(buckets, sort=True):
            yield chunk.reset_index(drop=True)
            if self.speed > 0:
                time.sleep(step / self.speed)
