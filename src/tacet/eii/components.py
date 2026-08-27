"""The six Early Instability Indicator (EII) components.

Each function maps a signal (plus the context needed to judge it) onto a
``[0, 1]`` score, where ``0`` means "nothing to report" and ``1`` means "this is
as abnormal as this component can express".

Three of the six -- :func:`observability_degradation`,
:func:`contextual_missingness` and :func:`context_contradiction` -- score the
*monitoring pipeline*, not the machine. They are the reason a node that has gone
quiet can rank above a node that is merely running hot.

All functions are pure, vectorised, and NaN-tolerant.
"""

from __future__ import annotations

import numpy as np

EPSILON = 1e-9

__all__ = [
    "value_deviation",
    "observability_degradation",
    "change_inconsistency",
    "contextual_missingness",
    "flatline",
    "context_contradiction",
    "COMPONENT_NAMES",
]

#: Canonical component order. Weight vectors follow this order.
COMPONENT_NAMES = (
    "value_deviation",
    "observability_degradation",
    "change_inconsistency",
    "contextual_missingness",
    "flatline",
    "context_contradiction",
)


def _as_float(values) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)


def _as_mask(values) -> np.ndarray:
    return np.asarray(values, dtype=float).astype(bool).astype(float)


def _fill_gaps(values) -> np.ndarray:
    """Linearly interpolate interior NaNs, then edge-fill, without pandas."""
    array = np.asarray(values, dtype=float)
    finite = np.isfinite(array)

    if not finite.any():
        return np.zeros_like(array)
    if finite.all():
        return array

    indices = np.arange(array.size)
    return np.interp(indices, indices[finite], array[finite])


def value_deviation(values, lower, upper) -> np.ndarray:
    """How far the signal escapes its expected envelope, in envelope widths.

    A value inside ``[lower, upper]`` scores ``0``. A value one full envelope
    width outside scores ``1``. This is the only component a conventional
    threshold monitor would also fire on.

    Samples that never arrived score ``0`` here, not ``1``: a gap is not an
    excursion, and pretending otherwise would let missing data masquerade as a
    thermal event. The observability components are what score the gap.
    """
    present = np.isfinite(np.asarray(values, dtype=float))

    values = _as_float(values)
    lower = _as_float(lower)
    upper = _as_float(upper)

    excursion = np.maximum(lower - values, 0.0) + np.maximum(values - upper, 0.0)
    width = np.maximum(upper - lower, EPSILON)

    return np.clip(excursion / width, 0.0, 1.0) * present


def observability_degradation(observed) -> np.ndarray:
    """``1`` wherever the signal was **not** observed.

    The bluntest of the observability components: a sample that never arrived is
    a sample you cannot clear. ``tacet`` scores it rather than imputing it.
    """
    return np.clip(1.0 - _as_mask(observed), 0.0, 1.0)


def change_inconsistency(
    values, expected, observed, step_scale=None, factor: float = 3.0
) -> np.ndarray:
    """Mismatch between how fast the signal moved and how fast it should have.

    Catches drift that stays comfortably inside the envelope -- the signal is at
    a legal *level* but on an illegal *trajectory*. Scored only where the sample
    was actually observed, so a gap cannot masquerade as a trajectory break.

    Parameters
    ----------
    step_scale:
        Typical step magnitude for this signal, learned on training data
        (a robust MAD-style scale of first differences). Pass it.

        Without it the scale is taken from the segment being scored, which is
        self-defeating: the threshold rises to meet whatever the data does, so a
        fixed fraction of every series is always flagged regardless of whether
        anything unusual happened.
    factor:
        Multiples of ``step_scale`` at which the component saturates. ``3.0``
        means "three times a normal step is as inconsistent as we can say".
    """
    values = _fill_gaps(values)
    expected = _fill_gaps(expected)
    observed = _as_mask(observed)

    if values.size == 0:
        return np.zeros(0, dtype=float)

    moved = np.abs(np.diff(values, prepend=values[0]))
    should_have_moved = np.abs(np.diff(expected, prepend=expected[0]))

    raw = np.abs(moved - should_have_moved)

    if step_scale is None:
        step_scale = 1.4826 * np.nanmedian(np.abs(moved - np.nanmedian(moved)))
    step_scale = float(step_scale)

    if not np.isfinite(step_scale) or step_scale <= 1e-12:
        return np.zeros(values.size, dtype=float)

    return np.clip(raw / (factor * step_scale), 0.0, 1.0) * observed


def contextual_missingness(observed, expected_present) -> np.ndarray:
    """``1`` where a sample is missing **and context says it should be there**.

    This is the component that separates a blind spot from an idle machine. A
    node with no job scheduled reporting nothing is normal. A node the scheduler
    believes is running a job reporting nothing is a monitoring failure, a
    hung agent, or a machine on its way out -- and all three deserve attention.
    """
    observed = _as_mask(observed)
    expected_present = _as_mask(expected_present)

    return np.clip((1.0 - observed) * expected_present, 0.0, 1.0)


def flatline(values, observed, window: int = 5, reference_std=None) -> np.ndarray:
    """Local variance collapse: the signal arrives, but it has stopped moving.

    A stuck sensor, a frozen counter, or a cached value replayed by a failing
    exporter all look *perfectly healthy* to a threshold monitor -- the number is
    in range and never late. Scored high here, and scored only where the sample
    was observed, since a gap has no variance to speak of.

    Parameters
    ----------
    reference_std:
        How much this signal *normally* moves, learned on training data. Passing
        it is strongly preferred: judged against its own segment, a signal that
        is flat throughout looks maximally flatlined, and a constant flag column
        scores 1.0 forever. A reference scale of ~0 means the parameter never
        moves anyway, so variance collapse is not a meaningful event and the
        component returns 0.
    """
    values = _fill_gaps(values)
    observed = _as_mask(observed)

    n = values.size
    if n == 0:
        return np.zeros(0, dtype=float)

    if reference_std is None:
        reference_std = values.std()
    reference_std = float(reference_std)

    # Nothing that never varies can stop varying.
    if not np.isfinite(reference_std) or reference_std <= 1e-12:
        return np.zeros(n, dtype=float)

    rolling_std, counts = _trailing_std(values, window)

    normalizer = reference_std + EPSILON
    flatness = 1.0 - np.clip(rolling_std / normalizer, 0.0, 1.0)

    # A window holding fewer than two samples has no variance to collapse. Score
    # it 0 rather than 1 -- otherwise every series opens with a false flatline.
    flatness[counts < 2] = 0.0

    return np.clip(flatness * observed, 0.0, 1.0)


def _trailing_std(values: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Trailing sample standard deviation (``ddof=1``) in O(n).

    Returns the per-position deviation and the number of samples that fed it, so
    callers can tell a genuine zero-variance stretch from an unfilled window.
    """
    n = values.size
    ones = np.ones(n, dtype=float)

    cumulative = np.concatenate(([0.0], np.cumsum(values)))
    cumulative_sq = np.concatenate(([0.0], np.cumsum(values * values)))
    cumulative_n = np.concatenate(([0.0], np.cumsum(ones)))

    end = np.arange(1, n + 1)
    start = np.maximum(end - window, 0)

    total = cumulative[end] - cumulative[start]
    total_sq = cumulative_sq[end] - cumulative_sq[start]
    counts = cumulative_n[end] - cumulative_n[start]

    safe = np.maximum(counts, 1.0)
    # Sum of squared deviations, clipped at 0 to absorb float cancellation.
    ss = np.maximum(total_sq - (total * total) / safe, 0.0)
    variance = np.divide(
        ss,
        np.maximum(counts - 1.0, 1.0),
        out=np.zeros(n, dtype=float),
        where=counts >= 2,
    )

    return np.sqrt(variance), counts


def context_contradiction(
    values,
    high_load,
    observed,
    step_scale=None,
    idle_level=None,
    busy_level=None,
    stillness: float = 0.25,
) -> np.ndarray:
    """The signal claims idle while context insists the entity is busy.

    Physical state and observed state disagree. On an HPC node this is the
    signature of a device that has dropped out of a job without failing the job:
    the scheduler still counts it, the metrics no longer reflect it.

    Two conditions must hold together, and requiring both is what makes the
    component usable:

    **Stillness** -- the signal is not moving, graded against ``step_scale``.
    **Idle level** -- the signal is sitting near where it sits when nothing is
    running, graded between ``idle_level`` and ``busy_level``.

    Stillness alone is not enough. A node under steady load has a steady
    temperature, so a stillness-only test fires continuously through every
    healthy plateau -- which is most of a well-utilised cluster's life. The
    level term is what distinguishes "hot and stable because it is working" from
    "cold and stable while it is supposed to be working".

    Parameters
    ----------
    idle_level, busy_level:
        Typical values when context says idle and busy respectively, learned on
        training data. When they are not supplied the level test is skipped and
        only stillness is scored, which is the looser legacy behaviour.
    stillness:
        Fraction of a normal step below which the signal counts as still.
    """
    values = _fill_gaps(values)
    high_load = _as_mask(high_load)
    observed = _as_mask(observed)

    if values.size == 0:
        return np.zeros(0, dtype=float)

    movement = np.abs(np.diff(values, prepend=values[0]))

    if step_scale is None:
        step_scale = 1.4826 * np.nanmedian(np.abs(movement - np.nanmedian(movement)))
    step_scale = float(step_scale)

    if not np.isfinite(step_scale) or step_scale <= 1e-12:
        return np.zeros(values.size, dtype=float)

    threshold = max(stillness * step_scale, EPSILON)
    stillness_score = 1.0 - np.clip(movement / threshold, 0.0, 1.0)

    level_score = np.ones(values.size, dtype=float)
    if idle_level is not None and busy_level is not None:
        idle_level = float(idle_level)
        busy_level = float(busy_level)
        span = busy_level - idle_level

        if np.isfinite(span) and abs(span) > 1e-12:
            # 1 at the idle level, 0 once the signal reaches its busy level.
            position = (values - idle_level) / span
            level_score = 1.0 - np.clip(position, 0.0, 1.0)

    return np.clip(high_load * observed * stillness_score * level_score, 0.0, 1.0)
