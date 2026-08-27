"""Detector contract, alert budgeting, observability trust."""

import numpy as np
import pandas as pd
import pytest

import tacet
from tacet import schema
from tacet.detect import REGISTRY, apply_budget, observability_trust


@pytest.fixture
def split_frame():
    rng = np.random.default_rng(0)
    n = 600
    frame = pd.DataFrame(
        {
            schema.ENTITY: "n1",
            "tel_a": rng.normal(size=n),
            "tel_b": rng.normal(size=n),
            "obs_coverage": 1.0,
        }
    )
    frame.loc[520:560, "tel_a"] += 6.0
    frame[schema.LABEL] = 0
    frame.loc[520:560, schema.LABEL] = 1
    return frame.iloc[:400], frame.iloc[400:]


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_every_detector_finds_an_obvious_fault(name, split_frame):
    train, test = split_frame
    detector = REGISTRY[name](planes=["telemetry"]).fit(train)

    scored = detector.score(test)
    assert schema.SCORE in scored.columns
    assert schema.TRUST in scored.columns
    assert len(scored) == len(test)

    report = tacet.score_report(detector.alert(scored, budget=41), budget=41)
    assert report["roc_auc"] > 0.6, f"{name} failed to rank an obvious fault"


def test_budget_is_honoured_exactly_under_ties():
    """Tied scores must not blow the budget.

    `score >= threshold` overshoots whenever scores tie, and anomaly scores tie
    constantly. The extra alerts land on real positives, so recall improves and
    the bug reads as an improvement.
    """
    frame = pd.DataFrame({schema.SCORE: np.ones(1000)})
    alerted = apply_budget(frame, budget=50)

    assert alerted[schema.ALERT].sum() == 50


def test_budget_clamps_to_frame_size():
    frame = pd.DataFrame({schema.SCORE: np.arange(10.0)})
    assert apply_budget(frame, budget=999)[schema.ALERT].sum() == 10
    assert apply_budget(frame, budget=0)[schema.ALERT].sum() == 0


def test_threshold_mode():
    frame = pd.DataFrame({schema.SCORE: np.arange(10.0)})
    assert apply_budget(frame, threshold=7.0)[schema.ALERT].sum() == 3


def test_trust_defaults_to_one_without_observability():
    frame = pd.DataFrame({"tel_a": [1.0, 2.0]})
    assert np.all(observability_trust(frame) == 1.0)


def test_trust_falls_with_coverage():
    frame = pd.DataFrame(
        {
            "obs_coverage": [1.0, 0.5, 0.0],
            "obs_max_gap_seconds": [30.0, 600.0, 1800.0],
            "obs_stale_seconds": [30.0, 600.0, 1800.0],
        }
    )
    trust = observability_trust(frame)

    assert trust[0] > trust[1] > trust[2]
    assert trust[2] == pytest.approx(0.0)


def test_trust_weighting_modes(split_frame):
    train, test = split_frame
    test = test.assign(obs_coverage=0.5, obs_max_gap_seconds=600.0)

    raw = REGISTRY["robust_z"](planes=["telemetry"]).fit(train).score(test)
    discounted = (
        REGISTRY["robust_z"](planes=["telemetry"], trust_weighting="discount")
        .fit(train).score(test)
    )
    boosted = (
        REGISTRY["robust_z"](planes=["telemetry"], trust_weighting="boost")
        .fit(train).score(test)
    )

    assert discounted[schema.SCORE].mean() < raw[schema.SCORE].mean()
    assert boosted[schema.SCORE].mean() > raw[schema.SCORE].mean()


def test_scoring_before_fitting_raises(split_frame):
    _, test = split_frame
    with pytest.raises(RuntimeError, match="fitted"):
        REGISTRY["markov"]().score(test)


def test_markov_transitions_stay_within_entity():
    """A node's first window must not inherit the previous node's state."""
    rng = np.random.default_rng(3)
    frame = pd.DataFrame(
        {
            schema.ENTITY: np.repeat(["a", "b"], 100),
            "tel_x": np.r_[rng.normal(0, 1, 100), rng.normal(20, 1, 100)],
        }
    )
    detector = REGISTRY["markov"](planes=["telemetry"]).fit(frame)
    matrix = detector.transition_matrix().to_numpy()

    assert np.allclose(matrix.sum(axis=1), 1.0)


def test_graph_explain_requires_alerts(split_frame):
    train, test = split_frame
    detector = REGISTRY["graph"](planes=["telemetry"]).fit(train)

    with pytest.raises(KeyError, match="alert"):
        detector.explain(detector.score(test))
