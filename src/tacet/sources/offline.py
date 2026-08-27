"""Offline telemetry sources: frames, files, directories, archives.

These cover captured datasets -- a Zenodo dump, a Slurm accounting export, a
directory of per-node CSVs, a parquet lake written by an ETL job.
"""

from __future__ import annotations

import glob as _glob
import os
from collections.abc import Iterator, Sequence

import pandas as pd

from .. import schema
from .base import SourceInfo, TelemetrySource, to_long

__all__ = [
    "DataFrameSource",
    "CsvSource",
    "ParquetSource",
    "JsonlSource",
    "DirectorySource",
    "LongFrameSource",
]

#: Extension -> pandas reader, for :class:`DirectorySource` and ``from_uri``.
READERS = {
    ".csv": pd.read_csv,
    ".csv.gz": pd.read_csv,
    ".tsv": lambda p, **kw: pd.read_csv(p, sep="\t", **kw),
    ".parquet": pd.read_parquet,
    ".pq": pd.read_parquet,
    ".json": pd.read_json,
    ".jsonl": lambda p, **kw: pd.read_json(p, lines=True, **kw),
    ".ndjson": lambda p, **kw: pd.read_json(p, lines=True, **kw),
    ".feather": pd.read_feather,
}


def _reader_for(path: str):
    lowered = path.lower()
    for extension in sorted(READERS, key=len, reverse=True):
        if lowered.endswith(extension):
            return READERS[extension]
    raise ValueError(
        f"no reader for {path!r}; supported: {', '.join(sorted(READERS))}"
    )


class DataFrameSource(TelemetrySource):
    """Wrap a wide in-memory frame.

    Parameters
    ----------
    frame:
        One row per (entity, timestamp), one column per metric.
    entity, time:
        Column names in ``frame``.
    metrics:
        Metric columns to keep. ``None`` keeps everything else.
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        entity: str = schema.ENTITY,
        time: str = schema.TIMESTAMP,
        metrics: Sequence[str] | None = None,
        name: str = "<dataframe>",
    ):
        self._frame = frame
        self._entity = entity
        self._time = time
        self._metrics = metrics
        self.info = SourceInfo(kind="dataframe", location=name, live=False)

    def read(self) -> pd.DataFrame:
        return to_long(
            self._frame, entity=self._entity, time=self._time, metrics=self._metrics
        )


class LongFrameSource(TelemetrySource):
    """Wrap a frame that is *already* in ``(entity, timestamp, metric, value)`` form."""

    def __init__(self, frame: pd.DataFrame, name: str = "<long-dataframe>"):
        renamed = frame.rename(
            columns={
                "entity": schema.ENTITY,
                "node": schema.ENTITY,
                "node_id": schema.ENTITY,
                "time": schema.TIMESTAMP,
            }
        )
        self._frame = renamed
        self.info = SourceInfo(kind="long-dataframe", location=name, live=False)

    def read(self) -> pd.DataFrame:
        frame = self._frame.copy()
        frame[schema.TIMESTAMP] = pd.to_datetime(
            frame[schema.TIMESTAMP], utc=True, errors="coerce"
        )
        return frame


class _FileSource(TelemetrySource):
    """Shared machinery for single-file offline sources."""

    kind = "file"

    def __init__(
        self,
        path: str,
        entity: str = schema.ENTITY,
        time: str = schema.TIMESTAMP,
        metrics: Sequence[str] | None = None,
        **read_kwargs,
    ):
        self.path = path
        self._entity = entity
        self._time = time
        self._metrics = metrics
        self._read_kwargs = read_kwargs
        self.info = SourceInfo(kind=self.kind, location=path, live=False)

    def _load(self) -> pd.DataFrame:
        raise NotImplementedError

    def read(self) -> pd.DataFrame:
        return to_long(
            self._load(), entity=self._entity, time=self._time, metrics=self._metrics
        )


class CsvSource(_FileSource):
    """A CSV (optionally gzipped) of wide telemetry."""

    kind = "csv"

    def _load(self) -> pd.DataFrame:
        return pd.read_csv(self.path, **self._read_kwargs)


class ParquetSource(_FileSource):
    """A parquet file of wide telemetry."""

    kind = "parquet"

    def _load(self) -> pd.DataFrame:
        return pd.read_parquet(self.path, **self._read_kwargs)


class JsonlSource(_FileSource):
    """Newline-delimited JSON, one object per sample."""

    kind = "jsonl"

    def _load(self) -> pd.DataFrame:
        return pd.read_json(self.path, lines=True, **self._read_kwargs)


class DirectorySource(TelemetrySource):
    """Every matching file under a directory, concatenated.

    Handles the common HPC layout of one file per node or per day. Files are
    read in sorted order so time ordering is stable, and :meth:`stream` yields
    one file at a time so a 400 GB export does not have to fit in memory.

    Parameters
    ----------
    pattern:
        A glob such as ``"/data/exadata/*.parquet"`` or a directory path, in
        which case ``glob_suffix`` is appended.
    entity_from_filename:
        When the files carry the node identity in their name rather than in a
        column, pass a callable mapping basename -> entity id.
    """

    def __init__(
        self,
        pattern: str,
        entity: str = schema.ENTITY,
        time: str = schema.TIMESTAMP,
        metrics: Sequence[str] | None = None,
        glob_suffix: str = "*.csv",
        entity_from_filename=None,
        **read_kwargs,
    ):
        if os.path.isdir(pattern):
            pattern = os.path.join(pattern, glob_suffix)

        self.pattern = pattern
        self._entity = entity
        self._time = time
        self._metrics = metrics
        self._entity_from_filename = entity_from_filename
        self._read_kwargs = read_kwargs
        self.info = SourceInfo(kind="directory", location=pattern, live=False)

    def paths(self) -> list[str]:
        return sorted(_glob.glob(self.pattern))

    def stream(self) -> Iterator[pd.DataFrame]:
        paths = self.paths()
        if not paths:
            raise FileNotFoundError(f"no files matched {self.pattern!r}")

        for path in paths:
            frame = _reader_for(path)(path, **self._read_kwargs)

            if self._entity_from_filename is not None:
                frame[self._entity] = self._entity_from_filename(
                    os.path.basename(path)
                )

            yield to_long(
                frame, entity=self._entity, time=self._time, metrics=self._metrics
            )

    def read(self) -> pd.DataFrame:
        batches = list(self.stream())
        return pd.concat(batches, ignore_index=True)
