"""Correlation mapping, lead/lag, and missingness structure."""

import numpy as np
import pandas as pd
import pytest

from tacet.analysis import analyze_missingness, correlation_map, lead_lag
from tacet.analysis.correlation import METHODS


@pytest.fixture
def coupled():
    rng = np.random.default_rng(0)
    n = 400
    driver = np.sin(np.arange(n) / 9) + rng.normal(0, 0.15, n)
    return pd.DataFrame(
        {
            "entity_id": "n1",
            "tel_a": driver,
            "tel_b": np.roll(driver, 5),
            "obs_coverage": np.clip(1 - 0.4 * np.roll(driver, 3), 0, 1),
            "tel_noise": rng.normal(size=n),
        }
    )


@pytest.mark.parametrize("method", METHODS)
def test_every_method_runs(method, coupled):
    mapping = correlation_map(coupled, method=method, min_periods=20)

    assert mapping.matrix.shape[0] == mapping.matrix.shape[1]
    assert not mapping.top_pairs(3).empty


def test_support_gates_small_samples():
    frame = pd.DataFrame({"tel_a": [1.0, 2.0, 3.0], "tel_b": [3.0, 2.0, 1.0]})
    mapping = correlation_map(frame, method="pearson", min_periods=30)

    assert mapping.matrix.isna().to_numpy().all()


def test_observability_confounding_is_found(coupled):
    mapping = correlation_map(coupled, method="spearman", min_periods=20)
    confounded = mapping.observability_confounding(threshold=0.4)

    assert set(confounded["source"]) >= {"tel_a", "tel_b"}


def test_lead_lag_sign_convention(coupled):
    """tel_a is tel_b shifted back 5 steps, so tel_a leads by +5."""
    result = lead_lag(coupled, target="tel_b", max_lag=10)
    row = result[result.feature == "tel_a"].iloc[0]

    assert row["lead_windows"] == 5
    assert row["coefficient"] == pytest.approx(1.0, abs=1e-6)


def test_lead_lag_does_not_leak_across_entities(coupled):
    doubled = pd.concat(
        [coupled.assign(entity_id="n1"), coupled.assign(entity_id="n2")],
        ignore_index=True,
    )
    single = lead_lag(coupled, target="tel_b", max_lag=8)
    both = lead_lag(doubled, target="tel_b", max_lag=8)

    a_single = single[single.feature == "tel_a"].iloc[0]
    a_both = both[both.feature == "tel_a"].iloc[0]

    assert a_single["lead_windows"] == a_both["lead_windows"]
    assert a_both["n"] == pytest.approx(2 * a_single["n"], rel=0.02)


@pytest.fixture
def gappy():
    rng = np.random.default_rng(3)
    n = 800
    temp = rng.normal(60, 8, n)
    frame = pd.DataFrame(
        {
            "entity_id": np.repeat(["n1", "n2"], n // 2),
            "tel_temp": temp,
            "tel_clock": rng.normal(1400, 50, n),
            "tel_power": rng.normal(250, 20, n),
            "tel_random": rng.normal(size=n),
        }
    )
    frame.loc[rng.choice(n, 40, replace=False), "tel_random"] = np.nan
    hot = np.flatnonzero(temp > 72)
    frame.loc[np.clip(hot + 1, 0, n - 1), "tel_temp"] = np.nan
    for start in (100, 300, 520):
        frame.loc[start : start + 25, ["tel_clock", "tel_power"]] = np.nan
    return frame


def test_mnar_detected_only_where_value_predicts_silence(gappy):
    report = analyze_missingness(gappy)
    mechanism = report.mechanism["mechanism"]

    assert mechanism["tel_temp"] == "MNAR"
    # Bursty collector outages are not value-dependent; a forward-fill probe
    # would call them MNAR purely from run structure.
    assert mechanism["tel_clock"] != "MNAR"
    assert mechanism["tel_power"] != "MNAR"


def test_burstiness_separates_outages_from_dropouts(gappy):
    summary = analyze_missingness(gappy).summary

    assert summary.loc["tel_clock", "burstiness"] > 10
    assert summary.loc["tel_random", "burstiness"] < 3


def test_co_missing_cluster_recovers_shared_collector(gappy):
    clusters = analyze_missingness(gappy).co_missing_clusters()
    assert ["tel_clock", "tel_power"] in clusters


def test_gap_runs_do_not_span_entities(gappy):
    summary = analyze_missingness(gappy, entity="entity_id").summary
    assert summary.loc["tel_clock", "max_gap_run"] <= 26


def test_mcar_test_returns_a_verdict(gappy):
    result = analyze_missingness(gappy).mcar_test
    assert set(result) >= {"statistic", "df", "p_value", "n_patterns"}
    assert np.isfinite(result["statistic"])


def test_label_association_finds_predictive_missingness(gappy):
    frame = gappy.assign(label=gappy["tel_clock"].isna().astype(int))
    association = analyze_missingness(frame, label="label").label_association

    assert association.iloc[0]["column"] in {"tel_clock", "tel_power"}
    assert association.iloc[0]["correlation"] == pytest.approx(1.0, abs=1e-6)
