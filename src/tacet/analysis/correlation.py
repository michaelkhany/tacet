"""Correlation mapping across evidence planes.

Classical association measures, plus the two things a monitoring context
actually needs and general-purpose stats packages do not provide:

* **Cross-plane summaries.** Whether ``tel_gpu_temp`` correlates with
  ``obs_coverage`` is a different kind of fact from whether it correlates with
  ``tel_sm_clock``. The first says your *measurements* depend on your
  *ability to measure*, which invalidates naive inference about the metric.
* **Lead--lag structure.** Which signal moves *first* is the question failure
  prediction turns on; a zero-lag correlation matrix cannot answer it.

Every estimator is missingness-aware: pairs are computed on complete cases and
the supporting sample size is reported alongside the coefficient, so a
correlation of 0.98 backed by nine points cannot quietly become a finding.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .. import schema

__all__ = ["CorrelationMap", "correlation_map", "lead_lag", "METHODS"]

#: Supported association measures.
METHODS = ("pearson", "spearman", "kendall", "partial", "mutual_info", "distance")


@dataclass
class CorrelationMap:
    """Result of :func:`correlation_map`."""

    matrix: pd.DataFrame
    support: pd.DataFrame
    method: str
    min_periods: int

    def to_edges(self, threshold: float = 0.3, absolute: bool = True) -> pd.DataFrame:
        """Flatten to a ranked edge list, annotated with evidence planes."""
        records = []
        columns = list(self.matrix.columns)

        for i, source in enumerate(columns):
            for target in columns[i + 1 :]:
                value = float(self.matrix.at[source, target])
                if not np.isfinite(value):
                    continue

                magnitude = abs(value) if absolute else value
                if magnitude < threshold:
                    continue

                source_plane = schema.plane_of(source)
                target_plane = schema.plane_of(target)

                records.append(
                    {
                        "source": source,
                        "target": target,
                        "coefficient": value,
                        "magnitude": abs(value),
                        "n": int(self.support.at[source, target]),
                        "source_plane": source_plane,
                        "target_plane": target_plane,
                        "cross_plane": source_plane != target_plane,
                    }
                )

        edges = pd.DataFrame(records)
        if edges.empty:
            return edges

        return edges.sort_values("magnitude", ascending=False).reset_index(drop=True)

    def top_pairs(self, n: int = 20, **kwargs) -> pd.DataFrame:
        return self.to_edges(**kwargs).head(n)

    def plane_summary(self) -> pd.DataFrame:
        """Mean absolute association between each pair of evidence planes.

        A high ``telemetry x observability`` cell is the finding to chase: it
        means the values you recorded co-vary with whether you managed to record
        them, so any conclusion drawn from the telemetry alone is confounded.
        """
        edges = self.to_edges(threshold=0.0)
        if edges.empty:
            return pd.DataFrame()

        pairs = edges.assign(
            plane_pair=[
                " x ".join(sorted((row.source_plane, row.target_plane)))
                for row in edges.itertuples()
            ]
        )

        return (
            pairs.groupby("plane_pair")
            .agg(
                mean_abs=("magnitude", "mean"),
                max_abs=("magnitude", "max"),
                pairs=("magnitude", "size"),
            )
            .sort_values("mean_abs", ascending=False)
        )

    def observability_confounding(self, threshold: float = 0.3) -> pd.DataFrame:
        """Telemetry whose values track the health of its own collection.

        Each row is a metric that should not be interpreted at face value.
        """
        edges = self.to_edges(threshold=threshold)
        if edges.empty:
            return edges

        mask = (
            (edges.source_plane == "observability") ^ (edges.target_plane == "observability")
        )
        return edges[mask].reset_index(drop=True)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"CorrelationMap(method={self.method!r}, features={len(self.matrix)})"


def correlation_map(
    frame: pd.DataFrame,
    method: str = "spearman",
    columns: list[str] | None = None,
    min_periods: int = 30,
    bins: int = 12,
    max_samples: int = 2000,
    random_state: int = 0,
) -> CorrelationMap:
    """Compute an association matrix over feature columns.

    Parameters
    ----------
    method:
        ``"pearson"`` linear;
        ``"spearman"`` rank-monotonic (the default -- telemetry is rarely
        linear and frequently heavy-tailed);
        ``"kendall"`` rank concordance, robust on short series;
        ``"partial"`` linear association with all other columns held constant,
        via the precision matrix -- use it to separate a direct relationship
        from one mediated by a third signal;
        ``"mutual_info"`` any dependence, linear or not, normalised to ``[0, 1]``;
        ``"distance"`` distance correlation, zero only under true independence.
    min_periods:
        Pairs with fewer complete observations than this are returned as ``NaN``
        rather than as a confident number computed from nothing.
    bins:
        Quantile bins for ``mutual_info``.
    max_samples:
        ``distance`` correlation builds an n x n distance matrix per pair, so
        both time and memory are O(n^2); larger inputs are subsampled to this
        many rows using ``random_state``.
    """
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; choose from {METHODS}")

    numeric = _numeric_frame(frame, columns)
    if numeric.shape[1] < 2:
        raise ValueError("need at least two numeric columns to correlate")

    support = _pairwise_support(numeric)

    if method in ("pearson", "spearman", "kendall"):
        if method == "kendall":
            _require_scipy("kendall")
        matrix = numeric.corr(method=method, min_periods=min_periods)
    elif method == "partial":
        matrix = _partial_correlation(numeric)
    elif method == "mutual_info":
        matrix = _mutual_information(numeric, bins=bins)
    else:
        matrix = _distance_correlation(
            numeric, max_samples=max_samples, random_state=random_state
        )

    matrix = matrix.mask(support < min_periods)

    return CorrelationMap(
        matrix=matrix, support=support, method=method, min_periods=min_periods
    )


# -- estimators -------------------------------------------------------------


def _numeric_frame(frame: pd.DataFrame, columns) -> pd.DataFrame:
    if columns is not None:
        missing = [c for c in columns if c not in frame.columns]
        if missing:
            raise KeyError(f"columns not in frame: {missing}")
        subset = frame[columns]
    else:
        subset = frame[schema.feature_columns(frame)]

    numeric = subset.select_dtypes(include="number")
    # Constant columns correlate with nothing and produce NaN warnings.
    return numeric.loc[:, numeric.std(numeric_only=True) > 0]


def _pairwise_support(frame: pd.DataFrame) -> pd.DataFrame:
    present = frame.notna().astype(np.int32).to_numpy()
    counts = present.T @ present
    return pd.DataFrame(counts, index=frame.columns, columns=frame.columns)


def _require_scipy(what: str) -> None:
    """Fail with an install hint rather than a bare ModuleNotFoundError."""
    try:
        import scipy  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            f"the {what!r} correlation method needs SciPy. "
            'Install it with: pip install "tacet[stats]"'
        ) from exc


def _series_corr(a: pd.Series, b: pd.Series, method: str) -> float:
    """Pairwise correlation that does not drag SciPy into the default path.

    ``Series.corr`` routes "spearman" and "kendall" through ``scipy.stats``,
    unlike ``DataFrame.corr``, which ranks in Cython. Spearman is this module's
    default, so a numpy+pandas install would otherwise fail on the documented
    path. Spearman is Pearson on ranks, so compute it that way.
    """
    # A constant input has zero standard deviation; numpy warns on the divide
    # before returning the NaN the callers already test for.
    with np.errstate(divide="ignore", invalid="ignore"):
        if method == "spearman":
            return a.rank().corr(b.rank())
        if method == "kendall":
            _require_scipy("kendall")
        return a.corr(b, method=method)


def _partial_correlation(frame: pd.DataFrame) -> pd.DataFrame:
    """Partial correlation from the precision (inverse covariance) matrix."""
    complete = frame.dropna()
    if len(complete) <= frame.shape[1]:
        raise ValueError(
            "partial correlation needs more complete rows than columns; "
            f"got {len(complete)} rows for {frame.shape[1]} columns"
        )

    covariance = np.cov(complete.to_numpy(), rowvar=False)
    # Ridge term keeps near-collinear telemetry (very common) invertible.
    ridge = 1e-8 * np.trace(covariance) / covariance.shape[0]
    precision = np.linalg.pinv(covariance + ridge * np.eye(covariance.shape[0]))

    diagonal = np.sqrt(np.outer(np.diag(precision), np.diag(precision)))
    partial = -precision / np.where(diagonal == 0, 1.0, diagonal)
    np.fill_diagonal(partial, 1.0)

    return pd.DataFrame(partial, index=frame.columns, columns=frame.columns)


def _mutual_information(frame: pd.DataFrame, bins: int) -> pd.DataFrame:
    """Normalised mutual information over quantile-binned values."""
    columns = list(frame.columns)
    codes = {}

    for name in columns:
        series = frame[name]
        try:
            binned = pd.qcut(series, bins, labels=False, duplicates="drop")
        except (ValueError, TypeError):
            binned = pd.cut(series, bins, labels=False)
        codes[name] = binned.to_numpy(dtype=float)

    size = len(columns)
    matrix = np.eye(size)

    for i in range(size):
        for j in range(i + 1, size):
            value = _normalised_mi(codes[columns[i]], codes[columns[j]])
            matrix[i, j] = matrix[j, i] = value

    return pd.DataFrame(matrix, index=columns, columns=columns)


def _normalised_mi(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 2:
        return np.nan

    a, b = a[mask].astype(int), b[mask].astype(int)
    # copy=True: pandas 3 hands back a read-only view.
    joint = pd.crosstab(a, b).to_numpy(dtype=float, copy=True)
    joint /= joint.sum()

    px = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)

    with np.errstate(divide="ignore", invalid="ignore"):
        terms = joint * np.log(joint / (px * py))
    mutual = np.nansum(np.where(joint > 0, terms, 0.0))

    entropy_x = -np.nansum(px * np.log(np.where(px > 0, px, 1.0)))
    entropy_y = -np.nansum(py * np.log(np.where(py > 0, py, 1.0)))
    normaliser = np.sqrt(entropy_x * entropy_y)

    if normaliser <= 0:
        return 0.0

    return float(np.clip(mutual / normaliser, 0.0, 1.0))


def _distance_correlation(frame: pd.DataFrame, max_samples: int, random_state: int):
    """Szekely-Rizzo distance correlation. Zero iff the pair is independent."""
    working = frame
    if len(working) > max_samples:
        working = working.sample(max_samples, random_state=random_state)

    columns = list(working.columns)
    size = len(columns)
    matrix = np.eye(size)

    for i in range(size):
        for j in range(i + 1, size):
            pair = working[[columns[i], columns[j]]].dropna()
            if len(pair) < 4:
                matrix[i, j] = matrix[j, i] = np.nan
                continue
            value = _dcor(pair.iloc[:, 0].to_numpy(), pair.iloc[:, 1].to_numpy())
            matrix[i, j] = matrix[j, i] = value

    return pd.DataFrame(matrix, index=columns, columns=columns)


def _dcor(x: np.ndarray, y: np.ndarray) -> float:
    def centred(v):
        distances = np.abs(v[:, None] - v[None, :])
        return (
            distances
            - distances.mean(axis=0, keepdims=True)
            - distances.mean(axis=1, keepdims=True)
            + distances.mean()
        )

    a, b = centred(x), centred(y)
    n = len(x)

    covariance = (a * b).sum() / (n * n)
    variance_x = (a * a).sum() / (n * n)
    variance_y = (b * b).sum() / (n * n)

    denominator = np.sqrt(variance_x * variance_y)
    if denominator <= 0:
        return 0.0

    return float(np.sqrt(max(covariance, 0.0) / denominator))


# -- lead / lag -------------------------------------------------------------


def lead_lag(
    frame: pd.DataFrame,
    target: str,
    columns: list[str] | None = None,
    max_lag: int = 24,
    method: str = "spearman",
    entity: str = schema.ENTITY,
    min_periods: int = 30,
) -> pd.DataFrame:
    """Find which signals move *before* ``target``, and by how much.

    For each candidate column, the correlation with ``target`` is evaluated at
    every shift from ``-max_lag`` to ``+max_lag`` windows, and the shift with
    the strongest association is reported.

    A **positive** ``lag`` means the candidate leads the target: aligning the
    candidate's value from ``lag`` windows ago against the target now maximises
    the association, so the candidate moved first. That makes it a **precursor**
    and therefore a usable early warning, and ``lead_windows`` is positive. A
    negative lag means the candidate trails the target, which usually makes it a
    consequence rather than a cause.

    Shifting is done per entity, so one node's history never leaks into the next
    node's lag estimate.

    Parameters
    ----------
    target:
        The column being predicted -- often ``eii_total``, an ``obs_`` feature,
        or a labelled failure indicator.
    max_lag:
        Maximum shift in **windows**, not in time units. With 10-minute strides,
        ``max_lag=24`` searches four hours in each direction.

    Returns
    -------
    DataFrame
        Columns ``feature``, ``plane``, ``lag``, ``coefficient``, ``lead_windows``,
        ``n``, sorted by absolute strength.

    Examples
    --------
    >>> lead_lag(windows, target="eii_total", max_lag=24).head()
    """
    if target not in frame.columns:
        raise KeyError(f"target column {target!r} not in frame")

    numeric = _numeric_frame(frame, columns)
    candidates = [c for c in numeric.columns if c != target]
    if not candidates:
        raise ValueError("no candidate columns to test against the target")

    has_entity = entity in frame.columns
    groups = frame[entity] if has_entity else None
    target_series = pd.to_numeric(frame[target], errors="coerce")

    records = []
    for name in candidates:
        candidate = numeric[name]

        best_lag, best_value, best_n = 0, np.nan, 0
        for lag in range(-max_lag, max_lag + 1):
            shifted = (
                candidate.groupby(groups).shift(lag) if has_entity else candidate.shift(lag)
            )

            pair = pd.concat([shifted, target_series], axis=1).dropna()
            if len(pair) < min_periods:
                continue

            value = _series_corr(pair.iloc[:, 0], pair.iloc[:, 1], method)
            if np.isfinite(value) and (
                not np.isfinite(best_value) or abs(value) > abs(best_value)
            ):
                best_lag, best_value, best_n = lag, float(value), len(pair)

        if not np.isfinite(best_value):
            continue

        records.append(
            {
                "feature": name,
                "plane": schema.plane_of(name),
                "lag": best_lag,
                "coefficient": best_value,
                "magnitude": abs(best_value),
                # shift(k) aligns the candidate's value from k windows ago with
                # the target now, so k > 0 means the candidate moved first.
                "lead_windows": best_lag,
                "n": best_n,
            }
        )

    result = pd.DataFrame(records)
    if result.empty:
        return result

    return result.sort_values("magnitude", ascending=False).reset_index(drop=True)
