"""Windowing and the derived observability plane."""

import numpy as np
import pandas as pd
import pytest

import tacet
from tacet import schema


def test_coverage_reflects_real_gaps(windows):
    assert windows["obs_coverage"].min() == 0.0
    assert windows["obs_coverage"].max() == 1.0


def test_existing_plane_prefixes_are_preserved(windows):
    """`ctx_job_active` must not become `tel_ctx_job_active`."""
    assert any(c.startswith("ctx_job_active") for c in windows.columns)
    assert not any(c.startswith("tel_ctx_") for c in windows.columns)
    assert not any(c.startswith("tel_tel_") for c in windows.columns)


def test_context_does_not_count_as_coverage(raw_telemetry):
    """A context column that is always present must not mask a dead exporter."""
    windows = tacet.to_windows(
        raw_telemetry, window="30min", stride="10min", expected_interval="30s"
    )
    dark = windows[windows[schema.ENTITY] == "n2"]["obs_coverage"]

    assert dark.min() == 0.0, "a fully blind window must report zero coverage"


def test_max_gap_includes_window_edges(raw_telemetry):
    """Silence at the start or end of a window is still silence."""
    windows = tacet.to_windows(
        raw_telemetry, window="30min", stride="10min", expected_interval="30s"
    )
    partial = windows[
        (windows["obs_coverage"] > 0) & (windows["obs_coverage"] < 1)
    ]

    assert not partial.empty
    assert partial["obs_max_gap_seconds"].max() > 60.0


def test_window_must_be_multiple_of_stride(raw_telemetry):
    with pytest.raises(ValueError, match="whole multiple"):
        tacet.to_windows(raw_telemetry, window="25min", stride="10min")


def test_label_horizon_marks_only_lead_up(raw_telemetry):
    windows = tacet.to_windows(raw_telemetry, window="30min", stride="30min")
    events = pd.DataFrame(
        {"entity_id": ["n1"], "timestamp": [windows[schema.WINDOW_END].iloc[-1]]}
    )
    labelled = tacet.label_horizon(windows, events, horizon="1h")

    assert labelled[schema.LABEL].sum() > 0
    assert labelled[labelled[schema.ENTITY] == "n2"][schema.LABEL].sum() == 0


def test_label_episodes_marks_overlap(raw_telemetry):
    windows = tacet.to_windows(raw_telemetry, window="30min", stride="30min")
    episodes = pd.DataFrame(
        {
            "entity_id": ["n1"],
            "start": [windows[schema.WINDOW_START].iloc[1]],
            "end": [windows[schema.WINDOW_END].iloc[2]],
            "kind": ["thermal"],
        }
    )
    labelled = tacet.label_episodes(windows, episodes)

    assert labelled[schema.LABEL].sum() >= 2
    assert set(labelled.loc[labelled[schema.LABEL] == 1, "event_kind"]) == {"thermal"}


def test_chronological_split_is_ordered(windows):
    split = tacet.chronological_split(windows)
    order = {"train": 0, "validation": 1, "test": 2}

    codes = split[schema.SPLIT].map(order).to_numpy()
    assert np.all(np.diff(codes) >= 0), "splits must not interleave in time"
