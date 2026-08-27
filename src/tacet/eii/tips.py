"""Interpretation tips: the annotations that tell you how to read the cloud.

An EII Cloud plot without tips shows you *that* something is abnormal. With
tips, it tells you *which kind* of abnormal, which is the part that decides what
you do next -- and, more often than anyone admits, decides whether you page the
hardware team or the monitoring team.

Each rule below recognises one characteristic signature in the component mix and
emits a :class:`Tip`: a title, a plain-language reading of what that region of
the chart means, and a suggested action. Tips render as callouts on the figure
(:func:`tacet.viz.plot_cloud`) and as annotated tables in the Markdown report
(:func:`tacet.viz.write_report`).

The signature that matters most is :data:`UNTRUSTWORTHY_CALM`. Every other rule
fires on something visibly wrong. That one fires on a region that looks
*healthy* and is not -- a stretch of chart where the metric sat calmly inside
its envelope while a third of the samples behind it never arrived. Conventional
monitoring has no way to draw that, because to a conventional monitor it is
indistinguishable from a good day.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from .. import schema

__all__ = ["Tip", "TipRule", "RULES", "generate_tips", "tips_to_frame"]


@dataclass
class Tip:
    """One interpretation callout anchored to a region of the cloud."""

    code: str
    severity: str
    title: str
    reading: str
    action: str
    parameter: str | None = None
    entity: str | None = None
    start_row: int = 0
    end_row: int = 0
    start_time: object = None
    end_time: object = None
    score: float = 0.0
    evidence: dict = field(default_factory=dict)

    @property
    def span(self) -> int:
        """Number of consecutive windows this tip covers."""
        return self.end_row - self.start_row + 1

    def to_markdown(self) -> str:
        marker = {"critical": "🔴", "warning": "🟠", "info": "🔵"}.get(self.severity, "•")
        where = f"`{self.parameter}`" if self.parameter else "the cloud"
        when = ""
        if self.start_time is not None:
            when = f" from {self.start_time} to {self.end_time}"

        return (
            f"{marker} **{self.title}** — {where}{when} "
            f"({self.span} window{'s' if self.span != 1 else ''})\n\n"
            f"> {self.reading}\n\n"
            f"**What to do:** {self.action}\n"
        )

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return f"[{self.severity}] {self.title} ({self.parameter}, {self.span}w)"


@dataclass(frozen=True)
class TipRule:
    """A named signature in the component mix.

    ``test`` receives a dict of component name -> value for one window and
    returns True when the signature is present.
    """

    code: str
    severity: str
    title: str
    reading: str
    action: str
    test: object
    min_span: int = 1
    """Consecutive windows required before the signature is reported.

    Single-window blips are the dominant false positive for the rate-of-change
    and stillness rules: on any noisy signal some fraction of steps is unusually
    large by chance. Requiring persistence is what separates a trend from a
    tick, and it is the difference between a chart with four annotations and a
    chart with forty."""

    def matches(self, components: dict) -> bool:
        return bool(self.test(components))


def _get(components: dict, name: str) -> float:
    value = components.get(name, 0.0)
    return 0.0 if value is None or not np.isfinite(value) else float(value)


#: The rule set, evaluated in order. The first match wins for a given window,
#: so the most specific and most consequential signatures come first.
RULES: tuple[TipRule, ...] = (
    TipRule(
        code="BLIND_INTERVAL",
        severity="critical",
        title="Blind interval — the entity went quiet while it was supposed to be working",
        reading=(
            "The cloud is dense here but the underlying metric line is absent. Samples "
            "stopped arriving while context said this entity had work in flight. This is "
            "not an idle period: it is a stretch of time you have no evidence about, and "
            "a conventional dashboard would render it as a flat or interpolated line that "
            "looks perfectly healthy."
        ),
        action=(
            "Treat this window as unassessed rather than as normal. Check whether the "
            "exporter, the agent, or the node itself stopped responding, and whether the "
            "job scheduled here completed."
        ),
        test=lambda c: _get(c, "contextual_missingness") >= 0.5
        and _get(c, "observability_degradation") >= 0.5,
    ),
    TipRule(
        code="QUIET_UNDER_LOAD",
        severity="critical",
        title="Contradiction — telemetry says idle, context says busy",
        reading=(
            "Readings arrive on schedule but barely move, while the workload plane "
            "insists this entity is under load. The observed state and the physical "
            "state disagree. On a GPU node this is the signature of a device that has "
            "dropped out of its job without failing it: the scheduler still counts the "
            "device, the metrics no longer reflect it."
        ),
        action=(
            "Compare against sibling devices on the same node running the same job. If "
            "they move and this one does not, the device is a candidate for draining."
        ),
        test=lambda c: _get(c, "context_contradiction") >= 0.5,
        min_span=3,
    ),
    TipRule(
        code="STUCK_SENSOR",
        severity="warning",
        title="Variance collapse — the signal arrives but has stopped changing",
        reading=(
            "The line is flat, in range, and on time. That combination passes every "
            "threshold check ever written, and it is what a frozen counter, a stuck "
            "sensor, or an exporter replaying a cached value looks like. Genuine "
            "telemetry from a live system is never this still."
        ),
        action=(
            "Verify the value at the source. If it matches, the sensor is fine and the "
            "component is genuinely inactive; if the source has moved on, the collection "
            "path is stale and everything downstream of it is fiction."
        ),
        test=lambda c: _get(c, "flatline") >= 0.7,
        min_span=3,
    ),
    TipRule(
        code="UNTRUSTWORTHY_CALM",
        severity="warning",
        title="Absence of evidence — this looks calm, but you were partly blind",
        reading=(
            "The metric stayed inside its envelope, so this region reads as healthy. It "
            "should not. A meaningful share of the samples behind it never arrived, so "
            "the calm you are looking at is computed from partial data. Nothing here "
            "says the entity was fine; it says you did not see enough to tell."
        ),
        action=(
            "Do not clear this period on the strength of the chart. Recover the missing "
            "samples if they were buffered, or widen the window until coverage is high "
            "enough to support a conclusion."
        ),
        test=lambda c: _get(c, "value_deviation") < 0.2
        and 0.3 <= _get(c, "observability_degradation") < 1.0,
        min_span=2,
    ),
    TipRule(
        code="DRIFT",
        severity="warning",
        title="Trajectory break — legal value, illegal rate of change",
        reading=(
            "The value is still comfortably inside its envelope, so no threshold has "
            "been crossed, but it is moving at a rate that does not match its own "
            "history. Degradation shows up here long before it shows up as an excursion "
            "— this is the part of the chart where warning time is bought."
        ),
        action=(
            "Extend the view backwards and check whether the slope is sustained. A "
            "sustained trajectory break with a known failure downstream is your lead-time "
            "signal; feed it to `tacet.analysis.lead_lag` to quantify the warning."
        ),
        test=lambda c: _get(c, "change_inconsistency") >= 0.6
        and _get(c, "value_deviation") < 0.3,
        min_span=3,
    ),
    TipRule(
        code="COMPOUND",
        severity="critical",
        title="Compound failure — the machine and its monitoring are both degrading",
        reading=(
            "Value deviation and observability degradation are elevated together. Either "
            "the fault is taking the collection path down with it, or a collection "
            "problem is corrupting the readings. Both orderings are serious and they are "
            "hard to tell apart from the chart alone."
        ),
        action=(
            "Establish ordering before diagnosing. `tacet.analysis.lead_lag` against the "
            "observability plane will say which moved first, and that determines whether "
            "this is a hardware incident or a monitoring incident."
        ),
        test=lambda c: _get(c, "value_deviation") >= 0.5
        and _get(c, "observability_degradation") >= 0.5,
    ),
    TipRule(
        code="TRUE_EXCURSION",
        severity="warning",
        title="Clean excursion — genuinely out of envelope, and well observed",
        reading=(
            "The value left its expected envelope while coverage stayed complete. This "
            "is the one signature on the chart you can take at face value: the "
            "measurement is trustworthy and the deviation is real."
        ),
        action=(
            "Handle as an ordinary threshold event. Because observability is intact "
            "here, the magnitude and duration on the chart can be used directly."
        ),
        test=lambda c: _get(c, "value_deviation") >= 0.6
        and _get(c, "observability_degradation") < 0.2
        and _get(c, "contextual_missingness") < 0.2,
    ),
    TipRule(
        code="EXPECTED_SILENCE",
        severity="info",
        title="Silent, and expected to be — no cause for concern",
        reading=(
            "No samples arrived, and context agrees none were due: nothing was scheduled "
            "on this entity. The gap in the chart is real but benign. This rule exists so "
            "that ordinary idle time does not compete for attention with genuine blind "
            "intervals, which look identical until context is consulted."
        ),
        action="None. Shown so the gap is explained rather than merely absent.",
        test=lambda c: _get(c, "observability_degradation") >= 0.5
        and _get(c, "contextual_missingness") < 0.2,
    ),
)


def generate_tips(
    result,
    min_score: float = 0.0,
    max_tips: int = 50,
    merge_gap: int = 1,
    codes: list[str] | None = None,
) -> list[Tip]:
    """Derive interpretation callouts from a generated cloud.

    Consecutive windows matching the same rule for the same parameter are merged
    into a single tip covering the whole episode -- one callout reading "blind
    for 40 minutes" rather than eighty identical markers stacked on one region of
    the chart.

    Parameters
    ----------
    result:
        An :class:`~tacet.eii.cloud.EIICloudResult`.
    min_score:
        Ignore windows whose total EII falls below this.
    max_tips:
        Cap on returned tips, most severe and longest-running first.
    merge_gap:
        Windows of separation tolerated when merging an episode. ``1`` merges
        runs broken by a single quiet window.
    codes:
        Restrict to these rule codes.

    Returns
    -------
    list[Tip]
        Sorted by severity, then by span.

    Examples
    --------
    >>> tips = cloud.tips()
    >>> print(tips[0].to_markdown())
    """
    active = [rule for rule in RULES if codes is None or rule.code in codes]
    if not active:
        return []

    cloud = result.components
    if cloud.empty:
        return []

    frame = result.frame
    total_column = f"{schema.EII}total"

    wide = cloud.pivot_table(
        index=["row", "parameter"], columns="component", values="score", aggfunc="max"
    ).fillna(0.0)

    if min_score > 0 and total_column in frame.columns:
        keep = frame.index[frame[total_column] >= min_score]
        wide = wide[wide.index.get_level_values("row").isin(keep)]

    # (parameter, code) -> ordered list of matching rows
    hits: dict[tuple, list[tuple[int, float]]] = {}

    for (row, parameter), values in zip(wide.index, wide.to_dict("records")):
        for rule in active:
            if rule.matches(values):
                strength = max(values.values()) if values else 0.0
                hits.setdefault((parameter, rule.code), []).append((int(row), strength))
                break  # first match wins: rules are ordered by specificity

    by_code = {rule.code: rule for rule in active}
    tips: list[Tip] = []

    for (parameter, code), matches in hits.items():
        rule = by_code[code]
        for episode in _episodes([row for row, _ in matches], merge_gap):
            if len(episode) < rule.min_span:
                continue
            strengths = [s for r, s in matches if episode[0] <= r <= episode[-1]]
            tips.append(
                _build_tip(rule, parameter, episode, strengths, frame, result, wide)
            )

    order = {"critical": 0, "warning": 1, "info": 2}
    tips.sort(key=lambda t: (order.get(t.severity, 9), -t.span, -t.score))

    return tips[:max_tips]


def _episodes(rows: list[int], merge_gap: int) -> list[list[int]]:
    """Split sorted rows into runs, tolerating ``merge_gap`` windows of silence."""
    if not rows:
        return []

    ordered = sorted(rows)
    runs = [[ordered[0]]]

    for row in ordered[1:]:
        if row - runs[-1][-1] <= merge_gap + 1:
            runs[-1].append(row)
        else:
            runs.append([row])

    return runs


def _build_tip(rule, parameter, episode, strengths, frame, result, wide) -> Tip:
    start, end = episode[0], episode[-1]

    entity = None
    if result.entity_column and start in frame.index:
        entity = str(frame.at[start, result.entity_column])

    start_time = end_time = None
    if result.time_column:
        if start in frame.index:
            start_time = frame.at[start, result.time_column]
        if end in frame.index:
            end_time = frame.at[end, result.time_column]

    evidence = {}
    if (start, parameter) in wide.index:
        evidence = {
            key: round(float(value), 3)
            for key, value in wide.loc[(start, parameter)].items()
            if value > 0
        }

    return Tip(
        code=rule.code,
        severity=rule.severity,
        title=rule.title,
        reading=rule.reading,
        action=rule.action,
        parameter=parameter,
        entity=entity,
        start_row=start,
        end_row=end,
        start_time=start_time,
        end_time=end_time,
        score=float(np.mean(strengths)) if strengths else 0.0,
        evidence=evidence,
    )


def tips_to_frame(tips: list[Tip]) -> pd.DataFrame:
    """Tabulate tips for export into a report or notebook."""
    if not tips:
        return pd.DataFrame(
            columns=["code", "severity", "parameter", "entity", "span", "score"]
        )

    records = []
    for tip in tips:
        record = asdict(tip)
        record["span"] = tip.span
        record.pop("evidence", None)
        records.append(record)

    return pd.DataFrame(records)
