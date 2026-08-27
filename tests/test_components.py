"""The six EII components, and the specific ways they used to be wrong."""

import numpy as np
import pandas as pd
import pytest

from tacet.eii import components as comp


def test_all_components_bounded():
    rng = np.random.default_rng(0)
    values = rng.normal(size=200)
    mask = rng.integers(0, 2, 200).astype(float)

    scores = [
        comp.value_deviation(values, np.full(200, -1.0), np.full(200, 1.0)),
        comp.observability_degradation(mask),
        comp.change_inconsistency(values, np.zeros(200), mask, step_scale=1.0),
        comp.contextual_missingness(mask, np.ones(200)),
        comp.flatline(values, mask, reference_std=1.0),
        comp.context_contradiction(values, np.ones(200), mask, step_scale=1.0),
    ]

    for score in scores:
        assert score.shape == (200,)
        assert np.all((score >= 0.0) & (score <= 1.0))


def test_missing_sample_is_not_an_excursion():
    """A gap must not be scored as a value deviation.

    NaN coerces to 0.0, which sits far below any positive envelope; without an
    explicit guard every blind window scores a maximum thermal excursion.
    """
    values = np.array([50.0, np.nan, 50.0])
    score = comp.value_deviation(values, np.full(3, 45.0), np.full(3, 55.0))

    assert score[1] == 0.0
    assert np.all(score == 0.0)


def test_flatline_ignores_constant_signals():
    """A signal that never varies cannot stop varying."""
    constant = np.ones(50)
    assert np.all(comp.flatline(constant, np.ones(50), reference_std=0.0) == 0.0)


def test_flatline_warmup_is_not_flat():
    """The first sample has no variance to collapse; it must score 0, not 1."""
    values = np.concatenate([np.arange(10.0), np.full(10, 9.0)])
    score = comp.flatline(values, np.ones(20), window=5, reference_std=values.std())

    assert score[0] == 0.0
    assert score[-1] > score[2]


def test_trailing_std_matches_pandas():
    rng = np.random.default_rng(1)
    values = rng.normal(size=500)

    mine, counts = comp._trailing_std(values, 5)
    reference = pd.Series(values).rolling(5, min_periods=2).std().fillna(0.0).to_numpy()

    assert np.allclose(mine, reference, atol=1e-9)
    assert counts[0] == 1 and counts[-1] == 5


def test_contradiction_needs_idle_level_not_just_stillness():
    """Steady load produces steady metrics; stillness alone fires on every plateau."""
    hot_plateau = np.full(60, 70.0)
    load = np.ones(60)

    without_level = comp.context_contradiction(
        hot_plateau, load, np.ones(60), step_scale=2.0
    )
    with_level = comp.context_contradiction(
        hot_plateau, load, np.ones(60), step_scale=2.0, idle_level=38.0, busy_level=70.0
    )

    assert without_level.mean() > 0.9, "stillness-only should fire on the plateau"
    assert with_level.max() == pytest.approx(0.0, abs=1e-9)


def test_contradiction_fires_on_cold_plateau_under_load():
    cold_plateau = np.full(60, 38.0)
    score = comp.context_contradiction(
        cold_plateau, np.ones(60), np.ones(60),
        step_scale=2.0, idle_level=38.0, busy_level=70.0,
    )
    assert score.min() > 0.9


def test_change_inconsistency_uses_fitted_scale():
    """A segment-derived scale rises to meet the data and flags a fixed fraction."""
    rng = np.random.default_rng(2)
    calm = np.cumsum(rng.normal(0, 0.01, 300))

    fitted = comp.change_inconsistency(calm, np.zeros(300), np.ones(300), step_scale=10.0)
    adaptive = comp.change_inconsistency(calm, np.zeros(300), np.ones(300))

    assert fitted.max() < 0.01, "a calm signal should not be inconsistent"
    assert adaptive.mean() > fitted.mean()


def test_empty_input_is_safe():
    empty = np.array([])
    assert comp.flatline(empty, empty).shape == (0,)
    assert comp.change_inconsistency(empty, empty, empty).shape == (0,)
    assert comp.context_contradiction(empty, empty, empty).shape == (0,)
