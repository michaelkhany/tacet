"""Synthetic HPC telemetry with *both* kinds of fault injected.

Existing anomaly-detection benchmarks inject faults into the signal and assume
the observation process is perfect. That makes them useless for evaluating the
thing this library is for. :func:`make_cluster` injects two independent fault
families:

**Machine faults** -- thermal excursions, clock throttling, gradual degradation.
Any competent detector finds these.

**Observation faults** -- exporter outages, stuck sensors, partial scrapes, and
silent nodes that context says should be reporting. Conventional detectors score
these as *normal*, because the numbers that arrive are unremarkable and the
numbers that do not arrive are invisible.

Ground truth is emitted for both, separately, so you can report how much of your
detection rate came from each -- which is the comparison the field is missing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import schema

__all__ = ["make_cluster"]


def make_cluster(
    n_nodes: int = 8,
    n_samples: int = 2000,
    freq: str = "1min",
    start: str = "2026-01-01",
    machine_fault_rate: float = 0.02,
    observation_fault_rate: float = 0.02,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate a small GPU cluster's telemetry, with labelled fault episodes.

    Parameters
    ----------
    n_nodes, n_samples, freq, start:
        Shape of the generated data.
    machine_fault_rate:
        Expected share of samples inside a machine-fault episode.
    observation_fault_rate:
        Expected share inside an observation-fault episode.
    seed:
        Reproducibility.

    Returns
    -------
    (telemetry, truth)
        ``telemetry`` is a wide frame ready for :func:`tacet.to_windows`, with
        real ``NaN`` gaps where the observation faults bite. ``truth`` has one
        row per injected episode: ``entity_id``, ``start``, ``end``, ``kind``,
        ``family`` (``"machine"`` or ``"observation"``).

    Examples
    --------
    >>> telemetry, truth = make_cluster(seed=0)
    >>> telemetry["tel_gpu_temp"].isna().mean()   # real gaps, not imputed
    """
    rng = np.random.default_rng(seed)
    index = pd.date_range(start, periods=n_samples, freq=freq, tz="UTC")

    frames = []
    episodes = []

    for node in range(n_nodes):
        name = f"gpu-{node:03d}"
        frame, node_episodes = _one_node(name, index, rng, machine_fault_rate,
                                         observation_fault_rate)
        frames.append(frame)
        episodes.extend(node_episodes)

    telemetry = pd.concat(frames, ignore_index=True)
    truth = pd.DataFrame(
        episodes, columns=[schema.ENTITY, "start", "end", "kind", "family"]
    )

    return telemetry, truth


def _one_node(name, index, rng, machine_rate, observation_rate):
    n = len(index)

    # A duty cycle: jobs arrive, run for a while, and leave.
    job_active = _duty_cycle(n, rng)
    load = job_active * rng.uniform(0.6, 1.0, n)

    temperature = 38 + 34 * load + rng.normal(0, 1.2, n)
    clock = 1980 - 260 * np.clip(temperature - 72, 0, None) / 10 + rng.normal(0, 18, n)
    power = 70 + 320 * load + rng.normal(0, 9, n)
    utilisation = 100 * load + rng.normal(0, 3, n)
    ecc = np.zeros(n)

    episodes = []

    # -- machine faults ------------------------------------------------------
    for start, stop, kind in _episodes(n, machine_rate, rng,
                                       ("thermal", "throttle", "degradation")):
        if kind == "thermal":
            temperature[start:stop] += np.linspace(4, 22, stop - start)
        elif kind == "throttle":
            clock[start:stop] -= np.linspace(50, 620, stop - start)
            power[start:stop] -= np.linspace(10, 130, stop - start)
        else:
            ecc[start:stop] = np.cumsum(rng.poisson(0.4, stop - start))
            temperature[start:stop] += np.linspace(0, 6, stop - start)

        episodes.append((name, index[start], index[stop - 1], kind, "machine"))

    columns = {
        "tel_gpu_temp": temperature,
        "tel_sm_clock": clock,
        "tel_power_draw": power,
        "tel_utilisation": utilisation,
        "tel_ecc_errors": ecc,
    }

    # -- observation faults --------------------------------------------------
    for start, stop, kind in _episodes(n, observation_rate, rng,
                                       ("outage", "stuck_sensor", "partial_scrape")):
        if kind == "outage":
            # The whole exporter stops. Nothing arrives, and the job keeps running.
            for values in columns.values():
                values[start:stop] = np.nan
        elif kind == "stuck_sensor":
            # One reading freezes. In range, on time, and completely wrong.
            frozen = temperature[max(start - 1, 0)]
            columns["tel_gpu_temp"][start:stop] = frozen
        else:
            # Two fields drop out while the rest of the scrape succeeds.
            columns["tel_power_draw"][start:stop] = np.nan
            columns["tel_ecc_errors"][start:stop] = np.nan

        episodes.append((name, index[start], index[stop - 1], kind, "observation"))

    frame = pd.DataFrame(
        {
            schema.ENTITY: name,
            schema.TIMESTAMP: index,
            **columns,
            "ctx_job_active": job_active,
            "ctx_high_load": (load > 0.5).astype(float),
        }
    )

    return frame, episodes


def _duty_cycle(n: int, rng) -> np.ndarray:
    """Alternating idle/busy stretches, so context is not constant."""
    active = np.zeros(n)
    position = 0

    while position < n:
        idle = rng.integers(20, 120)
        busy = rng.integers(80, 400)
        position += idle
        active[position : position + busy] = 1.0
        position += busy

    return active[:n]


def _episodes(n, rate, rng, kinds) -> list[tuple[int, int, str]]:
    """Non-overlapping episodes spread across the series.

    Placement is stratified: the series is divided into as many slots as there
    are episodes and one episode is placed inside each. Kinds are assigned
    round-robin.

    Both choices exist because uniform random placement makes an unreliable
    fixture. At realistic fault rates a series supports only three or four
    episodes, so independent sampling routinely clusters them all in the first
    half and repeats a single kind -- which leaves a chronological test split
    with no positives at all, and every ranking metric silently returns NaN
    instead of failing.
    """
    target = max(int(n * rate), 1)
    mean_length = 37  # midpoint of the 15..60 draw below
    count = max(int(round(target / mean_length)), 1)

    slot = n / count
    offset = int(rng.integers(0, len(kinds)))
    episodes: list[tuple[int, int, str]] = []

    for i in range(count):
        length = int(rng.integers(15, 60))
        low = int(i * slot)
        high = int((i + 1) * slot) - length

        start = int(rng.integers(low, high)) if high > low else low
        stop = min(start + length, n)

        if stop - start < 2:
            continue
        if any(start < e_stop and stop > e_start for e_start, e_stop, _ in episodes):
            continue

        episodes.append((start, stop, kinds[(len(episodes) + offset) % len(kinds)]))

    return episodes
