"""Input adapters. The contract: never invent a sample that did not arrive."""

import numpy as np
import pandas as pd
import pytest

from tacet import schema
from tacet.sources import (
    CallableSource,
    DataFrameSource,
    LongFrameSource,
    PushSource,
    ReplaySource,
    open_source,
    to_wide,
)


@pytest.fixture
def wide():
    return pd.DataFrame(
        {
            "node": ["n1", "n1", "n2"],
            "ts": pd.date_range("2026-01-01", periods=3, freq="h"),
            "temp": [50.0, np.nan, 61.0],
            "clock": [1400.0, 1410.0, np.nan],
        }
    )


def test_gaps_survive_the_round_trip(wide):
    long = DataFrameSource(wide, entity="node", time="ts").read()
    restored = to_wide(long)

    assert restored["tel_temp"].isna().sum() >= 1
    assert restored["tel_clock"].isna().sum() >= 1


def test_wide_materialises_the_full_grid(wide):
    """Absent (entity, time) combinations must appear as NaN rows, not vanish."""
    restored = to_wide(DataFrameSource(wide, entity="node", time="ts").read())
    assert len(restored) == 6  # 2 entities x 3 timestamps


def test_open_source_dispatch(wide, tmp_path):
    assert isinstance(open_source(wide, entity="node", time="ts"), DataFrameSource)

    long = DataFrameSource(wide, entity="node", time="ts").read()
    assert isinstance(open_source(long), LongFrameSource)

    assert isinstance(open_source(lambda: wide, entity="node"), CallableSource)

    path = tmp_path / "t.csv"
    wide.to_csv(path, index=False)
    source = open_source(str(path), entity="node", time="ts")
    assert not source.read().empty

    with pytest.raises(ValueError, match="unknown source scheme"):
        open_source("nope://x")

    with pytest.raises(ValueError, match="cannot infer"):
        open_source("file.xyz")


def test_callable_source_records_a_failed_poll_as_a_gap():
    """A collector that raises is an observability failure, not a crash."""

    def broken():
        raise RuntimeError("exporter down")

    source = CallableSource(broken, interval=0.01, max_batches=1, entity="node")
    batches = list(source.stream())

    assert len(batches) == 1
    assert batches[0].empty


def test_push_source_batches():
    source = PushSource(batch_seconds=0.01)
    source.push(entity="n1", metric="temp", value=50.0)
    source.push(entity="n1", metric="temp", value=51.0)

    frame = source.read()
    assert len(frame) == 2
    assert set(frame.columns) >= {schema.ENTITY, schema.TIMESTAMP, "metric", "value"}
    assert source.read().empty, "draining twice must not replay"


def test_push_source_counts_drops():
    source = PushSource(batch_seconds=0.01, max_queue=2)
    for i in range(5):
        source.push(entity="n1", metric="m", value=float(i))

    assert source.dropped == 3


def test_replay_source_yields_in_time_order(wide):
    offline = DataFrameSource(wide, entity="node", time="ts")
    batches = list(ReplaySource(offline, batch="1h", speed=0).stream())

    assert len(batches) >= 2
    firsts = [b[schema.TIMESTAMP].min() for b in batches]
    assert firsts == sorted(firsts)


def test_directory_source_needs_matches(tmp_path):
    from tacet.sources import DirectorySource

    with pytest.raises(FileNotFoundError):
        DirectorySource(str(tmp_path / "*.csv")).read()
