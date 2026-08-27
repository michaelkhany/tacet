"""Evaluation, including metrics that account for what you could not see.

Two families here. The first is standard and threshold-free. The second exists
because standard metrics silently assume the observations were complete, and in
a distributed system they never are.

On F1
-----
When positives vastly outnumber the alert budget -- 5000 pre-failure windows and
an operator who will look at 200 -- recall is capped at 4% before the model does
anything, and F1 inherits that cap. Reporting it as a model quality score is
close to meaningless. ``tacet`` computes it because reviewers ask for it, and
puts ROC-AUC and average precision next to it because those are the numbers that
actually rank methods. :func:`evaluate` reports all of them and flags the case.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import schema

__all__ = ["evaluate", "lead_time", "compare"]


def evaluate(
    scored: pd.DataFrame,
    label: str = schema.LABEL,
    budget: int | None = None,
    trust_threshold: float = 0.5,
) -> dict:
    """Score a detector's output.

    Parameters
    ----------
    scored:
        Frame with ``anomaly_score``, a label column, and ideally ``alert`` and
        ``observability_trust``.
    budget:
        Apply this alert budget first, if ``scored`` is not already alerted.
    trust_threshold:
        Below this observability trust, a window counts as **blind** for the
        observability-aware metrics.

    Returns
    -------
    dict
        Standard counts and rates, plus:

        ``roc_auc``, ``average_precision``
            Budget-independent ranking quality. Prefer these.
        ``blind_alert_rate``
            Share of alerts raised on windows we could barely see. High values
            mean the detector is largely reacting to monitoring failures. That
            may be exactly right -- but you should know it, not discover it in
            production.
        ``blind_positive_rate``
            Share of true positives that occurred while blind: failures whose
            evidence was never collected. This is the ceiling on what *any*
            detector could have found, and if it is high, the fix is
            instrumentation rather than modelling.
        ``mean_trust_at_alert``
            Average observability trust across raised alerts.
        ``f1_is_capped``
            True when positives exceed the budget, i.e. when recall and F1 are
            bounded by the budget rather than by the model.
    """
    from .. import detect

    if label not in scored.columns:
        raise KeyError(f"label column {label!r} not in frame")

    frame = scored
    if schema.ALERT not in frame.columns:
        if budget is None:
            raise ValueError("frame has no 'alert' column; pass `budget`")
        frame = detect.apply_budget(frame, budget=budget)

    truth = pd.to_numeric(frame[label], errors="coerce").fillna(0).astype(int).to_numpy()
    alerts = pd.to_numeric(frame[schema.ALERT], errors="coerce").fillna(0).astype(int).to_numpy()

    tp = int(((truth == 1) & (alerts == 1)).sum())
    fp = int(((truth == 0) & (alerts == 1)).sum())
    fn = int(((truth == 1) & (alerts == 0)).sum())
    tn = int(((truth == 0) & (alerts == 0)).sum())

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    positives = int((truth == 1).sum())
    raised = int(alerts.sum())

    result = {
        "windows": int(len(frame)),
        "positives": positives,
        "alerts": raised,
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f1_is_capped": bool(positives > raised > 0),
        **_ranking(frame, truth),
        **_observability(frame, truth, alerts, trust_threshold),
    }

    if schema.ENTITY in frame.columns:
        detected = frame[(truth == 1) & (alerts == 1)]
        result["entities_detected"] = int(detected[schema.ENTITY].nunique())
        result["entities_with_events"] = int(
            frame[truth == 1][schema.ENTITY].nunique()
        )

    return result


def _ranking(frame: pd.DataFrame, truth: np.ndarray) -> dict:
    """ROC-AUC and average precision, which no alert budget can distort."""
    if schema.SCORE not in frame.columns:
        return {"roc_auc": float("nan"), "average_precision": float("nan")}

    scores = (
        pd.to_numeric(frame[schema.SCORE], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .to_numpy(dtype=float)
    )

    positives = int((truth == 1).sum())
    # Undefined with a single class present; say so rather than returning 0.5.
    if positives == 0 or positives == len(truth):
        return {"roc_auc": float("nan"), "average_precision": float("nan")}

    try:
        from sklearn.metrics import average_precision_score, roc_auc_score

        return {
            "roc_auc": float(roc_auc_score(truth, scores)),
            "average_precision": float(average_precision_score(truth, scores)),
        }
    except ImportError:
        return {
            "roc_auc": _roc_auc(truth, scores),
            "average_precision": _average_precision(truth, scores),
        }


def _average_precision(truth: np.ndarray, scores: np.ndarray) -> float:
    """Average precision without sklearn: the sum of (delta recall x precision).

    Returning NaN when sklearn is absent would be the one thing this library
    exists to argue against -- a number that quietly goes missing while the run
    still reports success.
    """
    order = np.argsort(-scores, kind="mergesort")
    labels = truth[order] == 1
    ordered = scores[order]

    # One cut per distinct score, so tied scores are resolved as a group rather
    # than in whatever order the sort happened to leave them.
    cuts = np.r_[np.where(np.diff(ordered))[0], len(labels) - 1]

    hits = np.cumsum(labels)[cuts]
    precision = hits / (cuts + 1.0)
    recall = hits / hits[-1]

    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def _roc_auc(truth: np.ndarray, scores: np.ndarray) -> float:
    """Rank-based ROC-AUC, so the metric survives without SciPy or sklearn."""
    ranks = pd.Series(scores).rank().to_numpy()
    positives = truth == 1

    n_pos = int(positives.sum())
    n_neg = len(truth) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _observability(frame, truth, alerts, trust_threshold: float) -> dict:
    """Metrics that ask what the detector could actually see."""
    if schema.TRUST not in frame.columns:
        return {
            "blind_alert_rate": float("nan"),
            "blind_positive_rate": float("nan"),
            "mean_trust_at_alert": float("nan"),
            "blind_windows": 0,
        }

    trust = pd.to_numeric(frame[schema.TRUST], errors="coerce").fillna(1.0).to_numpy()
    blind = trust < trust_threshold

    raised = alerts == 1
    positives = truth == 1

    return {
        "blind_windows": int(blind.sum()),
        "blind_window_rate": float(blind.mean()),
        "blind_alert_rate": float(blind[raised].mean()) if raised.any() else 0.0,
        "blind_positive_rate": float(blind[positives].mean()) if positives.any() else 0.0,
        "mean_trust_at_alert": float(trust[raised].mean()) if raised.any() else float("nan"),
    }


def lead_time(
    scored: pd.DataFrame,
    entity: str = schema.ENTITY,
    time: str = schema.WINDOW_END,
    event_time: str = schema.EVENT_TIME,
) -> pd.DataFrame:
    """How much warning each detected event actually got.

    The metric operators care about and papers usually omit. A detector with
    perfect recall that fires ninety seconds before the node dies has not
    bought anyone anything; one with mediocre recall that fires four hours early
    lets you drain the node and migrate the job.

    Returns one row per detected event with ``lead_minutes`` measured from the
    **first** alert on that entity within the event's horizon.
    """
    required = {schema.ALERT, event_time, entity, time}
    missing = required - set(scored.columns)
    if missing:
        raise KeyError(f"lead_time needs columns: {sorted(missing)}")

    flagged = scored[
        (scored[schema.ALERT] == 1) & scored[event_time].notna()
    ].copy()

    if flagged.empty:
        return pd.DataFrame(columns=[entity, event_time, "first_alert", "lead_minutes"])

    first = (
        flagged.groupby([entity, event_time], as_index=False)[time]
        .min()
        .rename(columns={time: "first_alert"})
    )
    first["lead_minutes"] = (
        first[event_time] - first["first_alert"]
    ).dt.total_seconds() / 60.0

    return first.sort_values("lead_minutes", ascending=False).reset_index(drop=True)


def compare(results: dict[str, dict], sort_by: str = "average_precision") -> pd.DataFrame:
    """Tabulate several :func:`evaluate` outputs side by side.

    Examples
    --------
    >>> compare({name: evaluate(scored[name], budget=200) for name in scored})
    """
    table = pd.DataFrame(results).T
    table.index.name = "method"

    if sort_by in table.columns:
        table = table.sort_values(sort_by, ascending=False)

    return table
