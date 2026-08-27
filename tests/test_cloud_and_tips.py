"""EII Cloud generation, tips, reporting, and the end-to-end path."""

import numpy as np
import pandas as pd
import pytest

import tacet
from tacet import schema
from tacet.eii import RULES, EIICloud


@pytest.fixture
def scenario():
    """One node with a blind interval, an idle gap, a stuck sensor, an excursion."""
    n = 200
    rng = np.random.default_rng(7)
    index = pd.date_range("2026-03-01", periods=n, freq="10min", tz="UTC")

    temp = rng.normal(55.0, 3.0, n)
    job = np.ones(n)

    temp[60:80] = np.nan            # blind: job running, nothing reported
    job[100:115] = 0
    temp[100:115] = np.nan          # expected silence: nothing scheduled
    temp[130:150] = 55.0            # stuck sensor
    temp[170:185] = 78.0            # excursion

    return pd.DataFrame(
        {
            schema.ENTITY: "gpu-042",
            schema.TIMESTAMP: index,
            "tel_temp": temp,
            "ctx_job_active": job,
            "ctx_load_high": job,
        }
    )


@pytest.fixture
def cloud(scenario):
    return (
        EIICloud(
            parameters=["tel_temp"],
            expected_present="ctx_job_active",
            high_load="ctx_load_high",
        )
        .fit(scenario.iloc[:55])
        .transform(scenario)
    )


def test_cloud_emits_component_columns(cloud):
    from tacet.eii.components import COMPONENT_NAMES

    for name in COMPONENT_NAMES:
        assert f"{schema.EII}{name}" in cloud.frame.columns
    assert f"{schema.EII}total" in cloud.frame.columns
    assert cloud.frame[f"{schema.EII}total"].between(0, 1).all()


def test_blind_window_scores_on_observability_only(cloud):
    row = cloud.frame.iloc[65]

    assert row[f"{schema.EII}observability_degradation"] == 1.0
    assert row[f"{schema.EII}contextual_missingness"] == 1.0
    assert row[f"{schema.EII}value_deviation"] == 0.0


def test_idle_gap_is_not_contextually_missing(cloud):
    row = cloud.frame.iloc[105]

    assert row[f"{schema.EII}observability_degradation"] == 1.0
    assert row[f"{schema.EII}contextual_missingness"] == 0.0


def test_tips_locate_the_injected_episodes(cloud):
    tips = cloud.tips()
    located = {tip.code: (tip.start_row, tip.end_row) for tip in tips}

    assert "BLIND_INTERVAL" in located
    assert located["BLIND_INTERVAL"] == (60, 79)

    assert "EXPECTED_SILENCE" in located
    assert located["EXPECTED_SILENCE"] == (100, 114)


def test_tips_stay_few(cloud):
    """Persistence requirements are what keep the chart readable."""
    assert len(cloud.tips()) <= 10


def test_every_rule_has_prose():
    for rule in RULES:
        assert rule.severity in {"critical", "warning", "info"}
        assert len(rule.reading) > 80
        assert len(rule.action) > 20
        assert rule.min_span >= 1


def test_tip_markdown_renders(cloud):
    text = cloud.tips()[0].to_markdown()
    assert "**" in text and "What to do" in text


def test_report_carries_tips_as_metadata(cloud):
    from tacet.viz import render_report

    text = render_report(cloud, title="Test run")
    assert text.startswith("---")
    assert '"tips"' in text
    assert "How to read an EII Cloud" in text


def test_transform_before_fit_raises(scenario):
    with pytest.raises(RuntimeError, match="fitted"):
        EIICloud(parameters=["tel_temp"]).transform(scenario)


def test_unknown_component_weight_rejected():
    with pytest.raises(ValueError, match="unknown EII component"):
        EIICloud(weights={"not_a_component": 1.0})


def test_cloud_does_not_leak_across_entities(scenario):
    """Trailing-window components must reset at an entity boundary."""
    doubled = pd.concat(
        [scenario, scenario.assign(**{schema.ENTITY: "gpu-043"})], ignore_index=True
    )
    result = (
        EIICloud(parameters=["tel_temp"], expected_present="ctx_job_active")
        .fit(scenario.iloc[:55])
        .transform(doubled)
    )
    first = result.frame[result.frame[schema.ENTITY] == "gpu-042"].reset_index(drop=True)
    second = result.frame[result.frame[schema.ENTITY] == "gpu-043"].reset_index(drop=True)

    assert np.allclose(
        first[f"{schema.EII}total"], second[f"{schema.EII}total"], atol=1e-9
    )


def test_end_to_end_pipeline(cluster):
    telemetry, truth = cluster

    windows = tacet.to_windows(
        telemetry, window="30min", stride="10min", expected_interval="1min"
    )
    windows = tacet.label_episodes(
        windows, truth[truth["family"] == "observation"]
    )
    windows = tacet.chronological_split(windows)

    train = windows[windows[schema.SPLIT] == "train"]
    test = windows[windows[schema.SPLIT] == "test"]

    detector = tacet.RobustZScore(planes=["telemetry", "observability"]).fit(train)
    scored = detector.alert(detector.score(test), budget=50)
    report = tacet.score_report(scored, budget=50)

    assert report["alerts"] == 50
    assert 0.0 <= report["blind_alert_rate"] <= 1.0
    assert np.isfinite(report["roc_auc"])


def test_observability_plane_beats_telemetry_alone(cluster):
    """The library's central claim, as an executable assertion."""
    telemetry, truth = cluster

    windows = tacet.to_windows(
        telemetry, window="30min", stride="10min", expected_interval="1min"
    )
    windows = tacet.chronological_split(
        tacet.label_episodes(windows, truth[truth["family"] == "observation"])
    )
    train = windows[windows[schema.SPLIT] == "train"]
    test = windows[windows[schema.SPLIT] == "test"]

    def average_precision(planes):
        detector = tacet.RobustZScore(planes=planes).fit(train)
        scored = detector.alert(detector.score(test), budget=50)
        return tacet.score_report(scored, budget=50)["average_precision"]

    with_observability = average_precision(["telemetry", "observability"])
    telemetry_only = average_precision(["telemetry"])

    assert with_observability > telemetry_only
