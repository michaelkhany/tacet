"""Missingness analysis: what your gaps are made of.

Standard practice treats missing telemetry as a data-quality nuisance to be
imputed away before the real analysis begins. In a distributed system it is the
opposite: the pattern of what you failed to record is a measurement of the
system's health, and it carries information no surviving sample does.

This module answers four questions:

1. **How much is missing, and in what shape?** A metric that loses 5% of samples
   uniformly and one that loses 5% in a single 40-minute blackout have the same
   missing rate and completely different meanings. Run-length statistics
   separate them.
2. **What goes missing together?** Columns that fall silent in lockstep share a
   collector, a host, or a rack. :meth:`MissingnessReport.co_missing_clusters`
   recovers that fault domain from the data alone.
3. **Is the missingness random?** The classical MCAR / MAR / MNAR taxonomy
   (Rubin, 1976). MCAR gaps can be ignored. MAR gaps bias any analysis that
   does not condition on the cause. MNAR gaps mean the value is missing
   *because of what it would have been* -- the sensor failed high, the node died
   under load -- which is both the hardest case and the interesting one.
4. **Does missingness predict the outcome?** If it does, your gaps are evidence,
   and dropping incomplete rows discards your best signal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .. import schema

__all__ = ["MissingnessReport", "analyze_missingness", "little_mcar_test"]


@dataclass
class MissingnessReport:
    """Result of :func:`analyze_missingness`."""

    summary: pd.DataFrame
    nullity_correlation: pd.DataFrame
    mechanism: pd.DataFrame
    mcar_test: dict
    patterns: pd.DataFrame
    label_association: pd.DataFrame | None = None

    def co_missing_clusters(self, threshold: float = 0.7) -> list[list[str]]:
        """Group columns that go missing together above ``threshold``.

        Each cluster is a candidate shared failure domain: one exporter, one
        agent, one host. When an alert fires on a whole cluster at once, you are
        looking at a collection outage, not at N independent device faults --
        a distinction that decides whether you page the hardware team or the
        monitoring team.
        """
        matrix = self.nullity_correlation
        if matrix.empty:
            return []

        remaining = set(matrix.columns)
        clusters: list[list[str]] = []

        while remaining:
            seed = remaining.pop()
            group = {seed}
            frontier = [seed]

            while frontier:
                current = frontier.pop()
                for other in list(remaining):
                    value = matrix.at[current, other]
                    if np.isfinite(value) and value >= threshold:
                        group.add(other)
                        remaining.discard(other)
                        frontier.append(other)

            if len(group) > 1:
                clusters.append(sorted(group))

        return sorted(clusters, key=len, reverse=True)

    def worst(self, n: int = 10) -> pd.DataFrame:
        """Columns with the most damaging gaps, ranked by burst length."""
        return self.summary.sort_values(
            ["max_gap_run", "missing_rate"], ascending=False
        ).head(n)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        verdicts = self.mechanism["mechanism"].value_counts().to_dict()
        return f"MissingnessReport(columns={len(self.summary)}, mechanism={verdicts})"


def analyze_missingness(
    frame: pd.DataFrame,
    columns: list[str] | None = None,
    entity: str = schema.ENTITY,
    label: str | None = None,
    mar_threshold: float = 0.15,
    max_patterns: int = 25,
) -> MissingnessReport:
    """Characterise the missingness structure of a window matrix.

    Parameters
    ----------
    columns:
        Columns to analyse. Defaults to every numeric feature column.
    entity:
        Grouping column; run-length statistics are computed within an entity so
        one node's outage is not concatenated onto the next node's.
    label:
        Optional outcome column. When given, each column's missingness is tested
        for association with the label, and the result lands in
        :attr:`MissingnessReport.label_association`.
    mar_threshold:
        Minimum absolute correlation for missingness to count as explained by
        another observed column, i.e. for a MAR rather than MCAR verdict.
    max_patterns:
        Number of distinct missingness patterns to report.

    Examples
    --------
    >>> report = analyze_missingness(windows, label="label")
    >>> report.mechanism.query("mechanism == 'MNAR'")
    >>> report.co_missing_clusters()
    """
    working = _select(frame, columns)
    if working.empty or working.shape[1] == 0:
        raise ValueError("no numeric columns to analyse")

    indicator = working.isna()
    groups = frame[entity] if entity in frame.columns else None

    summary = _summarise(working, indicator, groups)
    nullity = _nullity_correlation(indicator)
    mechanism = _classify_mechanism(
        working, indicator, groups, mar_threshold=mar_threshold
    )
    patterns = _pattern_table(indicator, max_patterns=max_patterns)

    try:
        mcar = little_mcar_test(working)
    except Exception as exc:  # pragma: no cover - degenerate inputs
        mcar = {"statistic": float("nan"), "df": 0, "p_value": float("nan"), "error": str(exc)}

    association = None
    if label is not None:
        if label not in frame.columns:
            raise KeyError(f"label column {label!r} not in frame")
        association = _label_association(indicator, frame[label])

    return MissingnessReport(
        summary=summary,
        nullity_correlation=nullity,
        mechanism=mechanism,
        mcar_test=mcar,
        patterns=patterns,
        label_association=association,
    )


# -- pieces -----------------------------------------------------------------


def _select(frame: pd.DataFrame, columns) -> pd.DataFrame:
    if columns is not None:
        missing = [c for c in columns if c not in frame.columns]
        if missing:
            raise KeyError(f"columns not in frame: {missing}")
        subset = frame[columns]
    else:
        subset = frame[schema.feature_columns(frame)]

    return subset.select_dtypes(include="number")


def _summarise(working, indicator, groups) -> pd.DataFrame:
    records = []

    for name in working.columns:
        flags = indicator[name]
        runs = _gap_runs(flags, groups)

        records.append(
            {
                "column": name,
                "plane": schema.plane_of(name),
                "n": int(len(flags)),
                "missing": int(flags.sum()),
                "missing_rate": float(flags.mean()),
                "n_gaps": len(runs),
                "mean_gap_run": float(np.mean(runs)) if runs else 0.0,
                "max_gap_run": int(max(runs)) if runs else 0,
                # Burstiness > 1 means gaps arrive in clumps rather than
                # sprinkled at random -- outages, not dropped packets.
                "burstiness": _burstiness(runs, float(flags.mean())),
            }
        )

    return pd.DataFrame(records).set_index("column")


def _gap_runs(flags: pd.Series, groups) -> list[int]:
    """Lengths of consecutive-missing runs, computed within each entity."""
    runs: list[int] = []
    chunks = (
        [values for _, values in flags.groupby(groups, sort=False)]
        if groups is not None
        else [flags]
    )

    for chunk in chunks:
        array = chunk.to_numpy(dtype=bool)
        if not array.any():
            continue

        # Boundaries of True runs via padded diff.
        padded = np.concatenate(([False], array, [False]))
        changes = np.diff(padded.astype(np.int8))
        starts = np.flatnonzero(changes == 1)
        ends = np.flatnonzero(changes == -1)
        runs.extend((ends - starts).tolist())

    return runs


def _burstiness(runs: list[int], rate: float) -> float:
    """Mean observed run length over the run length expected if gaps were independent.

    Under MCAR with per-sample probability ``p``, missing runs are geometric with
    mean ``1 / (1 - p)``. Anything materially above that is clustered.
    """
    if not runs or rate <= 0 or rate >= 1:
        return 0.0

    expected = 1.0 / (1.0 - rate)
    return float(np.mean(runs) / expected)


def _nullity_correlation(indicator: pd.DataFrame) -> pd.DataFrame:
    """Correlation between missingness indicators (phi coefficient)."""
    varying = indicator.loc[:, indicator.nunique() > 1]
    if varying.shape[1] < 2:
        return pd.DataFrame()

    return varying.astype(float).corr()


def _classify_mechanism(working, indicator, groups, mar_threshold: float) -> pd.DataFrame:
    """Assign each column an MCAR / MAR / MNAR verdict.

    The MNAR probe is necessarily indirect -- by definition the missing values
    themselves are unavailable -- so it tests the next best thing: among the
    samples that *were* observed, does the value predict whether the very next
    sample goes missing? A sensor that falls silent right after reading extreme
    leaves exactly that fingerprint.

    Restricting the probe to gap **onsets** matters. Correlating missingness
    against a forward-filled series instead would score every long blackout as
    MNAR, because the filled value is pinned constant for exactly as long as the
    flag is set -- an artefact of run length, not evidence of value dependence.
    """
    records = []

    for name in working.columns:
        flags = indicator[name].astype(float)
        rate = float(flags.mean())

        if rate == 0.0 or rate == 1.0:
            records.append(
                {
                    "column": name,
                    "plane": schema.plane_of(name),
                    "missing_rate": rate,
                    "mechanism": "complete" if rate == 0.0 else "never_observed",
                    "max_other_corr": np.nan,
                    "explained_by": None,
                    "onset_corr": np.nan,
                    "confidence": "n/a",
                }
            )
            continue

        # MAR probe: is this column's missingness predicted by another column's
        # observed values?
        best_other, best_value = None, 0.0
        for other in working.columns:
            if other == name:
                continue
            pair = pd.concat([flags, working[other]], axis=1).dropna()
            if len(pair) < 10 or pair.iloc[:, 0].nunique() < 2:
                continue
            # A constant `other` column gives a zero standard deviation, and
            # numpy warns on the divide before returning the NaN we already
            # test for. The NaN is the answer; the warning is noise.
            with np.errstate(divide="ignore", invalid="ignore"):
                value = pair.iloc[:, 0].corr(pair.iloc[:, 1])
            if np.isfinite(value) and abs(value) > abs(best_value):
                best_other, best_value = other, float(value)

        onset_value = _onset_correlation(working[name], indicator[name], groups)

        if (
            np.isfinite(onset_value)
            and abs(onset_value) >= mar_threshold
            and abs(onset_value) >= abs(best_value)
        ):
            mechanism = "MNAR"
        elif abs(best_value) >= mar_threshold:
            mechanism = "MAR"
        else:
            mechanism = "MCAR"

        records.append(
            {
                "column": name,
                "plane": schema.plane_of(name),
                "missing_rate": rate,
                "mechanism": mechanism,
                "max_other_corr": best_value,
                "explained_by": best_other,
                "onset_corr": onset_value,
                "confidence": "low" if flags.sum() < 30 else "ok",
            }
        )

    return pd.DataFrame(records).set_index("column")


def _onset_correlation(values: pd.Series, flags: pd.Series, groups) -> float:
    """Among observed samples, does the value predict that the next one is missing?

    This is the MNAR fingerprint: correlation between an observed reading and
    the onset of silence immediately after it. Computed within entity so one
    node's last sample is never treated as preceding the next node's first gap.
    """
    next_missing = (
        flags.groupby(groups).shift(-1) if groups is not None else flags.shift(-1)
    )

    observed = ~flags.astype(bool)
    pair = pd.concat(
        [values[observed], next_missing[observed].astype(float)], axis=1
    ).dropna()

    if len(pair) < 10 or pair.iloc[:, 1].nunique() < 2 or pair.iloc[:, 0].nunique() < 2:
        return np.nan

    value = pair.iloc[:, 0].corr(pair.iloc[:, 1])
    return float(value) if np.isfinite(value) else np.nan


def _pattern_table(indicator: pd.DataFrame, max_patterns: int) -> pd.DataFrame:
    """Distinct missingness patterns and how often each occurs."""
    signature = indicator.astype(int).astype(str).agg("".join, axis=1)
    counts = signature.value_counts().head(max_patterns)

    return pd.DataFrame(
        {
            "pattern": counts.index,
            "count": counts.to_numpy(),
            "share": counts.to_numpy() / len(indicator),
            "n_missing": [sum(c == "1" for c in p) for p in counts.index],
            "columns_missing": [
                ", ".join(
                    column
                    for column, flag in zip(indicator.columns, pattern)
                    if flag == "1"
                )
                or "(none)"
                for pattern in counts.index
            ],
        }
    ).reset_index(drop=True)


def _label_association(indicator: pd.DataFrame, label: pd.Series) -> pd.DataFrame:
    """Point-biserial association between each column's missingness and the label."""
    target = pd.to_numeric(label, errors="coerce")
    records = []

    for name in indicator.columns:
        flags = indicator[name].astype(float)
        pair = pd.concat([flags, target], axis=1).dropna()

        if len(pair) < 10 or pair.iloc[:, 0].nunique() < 2 or pair.iloc[:, 1].nunique() < 2:
            continue

        value = float(pair.iloc[:, 0].corr(pair.iloc[:, 1]))
        positive = pair[pair.iloc[:, 1] > 0].iloc[:, 0].mean()
        negative = pair[pair.iloc[:, 1] <= 0].iloc[:, 0].mean()

        records.append(
            {
                "column": name,
                "plane": schema.plane_of(name),
                "correlation": value,
                "missing_rate_positive": float(positive),
                "missing_rate_negative": float(negative),
                "lift": float(positive / negative) if negative > 0 else np.inf,
            }
        )

    result = pd.DataFrame(records)
    if result.empty:
        return result

    return result.reindex(
        result["correlation"].abs().sort_values(ascending=False).index
    ).reset_index(drop=True)


def little_mcar_test(frame: pd.DataFrame) -> dict:
    """Little's (1988) chi-square test that data are Missing Completely At Random.

    Groups rows by missingness pattern and asks whether the observed means
    differ across patterns by more than sampling noise. A small p-value rejects
    MCAR: your gaps carry information.

    .. note::
       The population moments are estimated by available-case (pairwise)
       covariance rather than by the EM maximum-likelihood fit of the original
       paper. This keeps the test dependency-free and fast on wide telemetry
       matrices, at the cost of some accuracy when missingness is heavy. Treat
       the p-value as indicative, and prefer the per-column verdicts in
       :attr:`MissingnessReport.mechanism` for decisions about individual
       metrics.

    Returns
    -------
    dict
        ``statistic``, ``df``, ``p_value``, ``n_patterns``, ``method``.
    """
    working = frame.select_dtypes(include="number")
    working = working.loc[:, working.notna().any()]

    if working.shape[1] == 0:
        raise ValueError("no numeric columns")

    means = working.mean()
    covariance = working.cov()  # pairwise-complete by default
    columns = list(working.columns)

    indicator = working.isna()
    signature = indicator.astype(int).astype(str).agg("".join, axis=1)

    statistic = 0.0
    degrees = 0
    patterns = 0

    for key, rows in working.groupby(signature, sort=False):
        observed = [c for c, flag in zip(columns, key) if flag == "0"]
        if not observed or len(rows) < 2:
            continue

        block = covariance.loc[observed, observed].to_numpy(dtype=float)
        block = np.nan_to_num(block)

        difference = (rows[observed].mean() - means[observed]).to_numpy(dtype=float)
        difference = np.nan_to_num(difference)

        try:
            inverse = np.linalg.pinv(block)
        except np.linalg.LinAlgError:  # pragma: no cover - singular block
            continue

        statistic += len(rows) * float(difference @ inverse @ difference)
        degrees += len(observed)
        patterns += 1

    degrees = max(degrees - len(columns), 1)

    try:
        from scipy.stats import chi2

        p_value = float(chi2.sf(statistic, degrees))
    except ImportError:
        # Wilson-Hilferty cube-root normal approximation, so the test still
        # returns a usable p-value without SciPy installed.
        z = ((statistic / degrees) ** (1 / 3) - (1 - 2 / (9 * degrees))) / np.sqrt(
            2 / (9 * degrees)
        )
        p_value = float(0.5 * math.erfc(z / math.sqrt(2))) if np.isfinite(z) else float("nan")

    return {
        "statistic": float(statistic),
        "df": int(degrees),
        "p_value": p_value,
        "n_patterns": patterns,
        "method": "little-mcar (available-case moments)",
    }
