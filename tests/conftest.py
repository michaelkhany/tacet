import numpy as np
import pandas as pd
import pytest

import tacet


@pytest.fixture
def raw_telemetry():
    """Two nodes, a 30-minute blind interval, a stuck sensor, an excursion."""
    index = pd.date_range("2026-01-01", periods=240, freq="30s", tz="UTC")
    rng = np.random.default_rng(0)

    frames = []
    for node in ("n1", "n2"):
        temp = rng.normal(50.0, 2.0, len(index))
        frames.append(
            pd.DataFrame(
                {
                    "entity_id": node,
                    "timestamp": index,
                    "tel_temp": temp,
                    "tel_clock": rng.normal(1400.0, 30.0, len(index)),
                    "ctx_job_active": 1.0,
                }
            )
        )

    frame = pd.concat(frames, ignore_index=True)
    # n2 goes dark for 30 minutes, then reports a frozen value.
    dark = (frame.entity_id == "n2") & frame.timestamp.between(
        index[100], index[160]
    )
    frame.loc[dark, ["tel_temp", "tel_clock"]] = np.nan
    frame.loc[(frame.entity_id == "n2") & frame.timestamp.gt(index[200]), "tel_temp"] = 55.0

    return frame


@pytest.fixture
def windows(raw_telemetry):
    return tacet.to_windows(
        raw_telemetry, window="30min", stride="10min", expected_interval="30s"
    )


@pytest.fixture
def cluster():
    """Big enough that the final 20% chronological slice still contains faults.

    A small fixture leaves the test split with no positives at all, and every
    ranking metric silently returns NaN rather than failing.
    """
    return tacet.datasets.make_cluster(
        n_nodes=5,
        n_samples=2400,
        observation_fault_rate=0.05,
        machine_fault_rate=0.03,
        seed=1,
    )
