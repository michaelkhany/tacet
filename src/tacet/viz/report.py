"""Markdown reporting, with tips carried as front-matter metadata.

The report is the second surface the tips render on. On the figure they are
callouts; here they are structured metadata plus prose, so a finding can be
pasted into a ticket, an incident review, or a paper without being retyped.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd

__all__ = ["write_report", "render_report"]


def render_report(
    result,
    tips=None,
    title: str = "EII Cloud findings",
    top_parameters: int = 10,
    include_metadata: bool = True,
    extra: dict | None = None,
) -> str:
    """Render a findings report as Markdown text.

    Parameters
    ----------
    result:
        An :class:`~tacet.eii.cloud.EIICloudResult`.
    tips:
        Tips to include. ``None`` generates them.
    include_metadata:
        Emit a YAML front-matter block carrying the tips as structured data, so
        downstream tooling can consume the findings without parsing prose.
    extra:
        Extra key/values for the front matter -- run id, dataset, git sha.
    """
    tips = result.tips() if tips is None else tips
    generated = datetime.now(timezone.utc).isoformat(timespec="seconds")

    parts: list[str] = []

    if include_metadata:
        parts.append(_front_matter(result, tips, generated, extra))

    parts.append(f"# {title}\n")
    parts.append(
        f"Generated {generated} · {len(result.frame)} windows · "
        f"{len(result.parameters)} parameters · {len(tips)} tips\n"
    )

    parts.append(_summary_section(result, tips))
    parts.append(_tips_section(tips))
    parts.append(_parameters_section(result, top_parameters))
    parts.append(_reading_guide())

    return "\n".join(parts)


def write_report(result, path: str, **kwargs) -> str:
    """Render and write the report. Returns the path written."""
    text = render_report(result, **kwargs)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    return path


# -- sections ---------------------------------------------------------------


def _front_matter(result, tips, generated, extra) -> str:
    payload = {
        "generated": generated,
        "windows": int(len(result.frame)),
        "parameters": list(result.parameters),
        "tip_count": len(tips),
        "tips": [
            {
                "code": tip.code,
                "severity": tip.severity,
                "parameter": tip.parameter,
                "entity": tip.entity,
                "span_windows": tip.span,
                "start": str(tip.start_time) if tip.start_time is not None else None,
                "end": str(tip.end_time) if tip.end_time is not None else None,
                "evidence": tip.evidence,
            }
            for tip in tips
        ],
    }
    if extra:
        payload.update(extra)

    return "---\n" + json.dumps(payload, indent=2, default=str) + "\n---\n"


def _summary_section(result, tips) -> str:
    counts: dict[str, int] = {}
    for tip in tips:
        counts[tip.severity] = counts.get(tip.severity, 0) + 1

    lines = ["## Summary\n"]

    if not tips:
        lines.append(
            "No interpretation tips fired. Either the system behaved, or the "
            "observability plane was too sparse to support a verdict — check "
            "`obs_coverage` before concluding the former.\n"
        )
        return "\n".join(lines)

    order = ["critical", "warning", "info"]
    badge = {"critical": "🔴 critical", "warning": "🟠 warning", "info": "🔵 info"}
    lines.append(
        " · ".join(f"**{counts[s]}** {badge[s]}" for s in order if s in counts) + "\n"
    )

    observability_codes = {
        "BLIND_INTERVAL", "UNTRUSTWORTHY_CALM", "STUCK_SENSOR", "QUIET_UNDER_LOAD"
    }
    from_monitoring = sum(1 for tip in tips if tip.code in observability_codes)

    if from_monitoring:
        share = from_monitoring / len(tips)
        lines.append(
            f"\n**{from_monitoring} of {len(tips)} findings ({share:.0%}) concern what "
            "could not be observed rather than what was measured.** Conventional "
            "threshold monitoring would not have raised them, because in every case "
            "the numbers that did arrive were unremarkable.\n"
        )

    return "\n".join(lines)


def _tips_section(tips) -> str:
    if not tips:
        return ""

    lines = ["## Findings\n"]
    for i, tip in enumerate(tips, start=1):
        lines.append(f"### {i}. {tip.title}\n")

        facts = [f"`{tip.code}`", f"severity **{tip.severity}**", f"{tip.span} windows"]
        if tip.parameter:
            facts.append(f"parameter `{tip.parameter}`")
        if tip.entity:
            facts.append(f"entity `{tip.entity}`")
        if tip.start_time is not None:
            facts.append(f"{tip.start_time} → {tip.end_time}")
        lines.append(" · ".join(facts) + "\n")

        lines.append(f"**How to read this:** {tip.reading}\n")
        lines.append(f"**What to do:** {tip.action}\n")

        if tip.evidence:
            evidence = ", ".join(f"`{k}` = {v}" for k, v in sorted(tip.evidence.items()))
            lines.append(f"**Evidence at onset:** {evidence}\n")

    return "\n".join(lines)


def _parameters_section(result, top: int) -> str:
    ranking = result.top_parameters(top)
    if ranking.empty:
        return ""

    lines = ["## Parameters ranked by peak EII\n"]
    lines.append(_markdown_table(ranking.round(4)))
    lines.append("")

    return "\n".join(lines)


def _markdown_table(frame: pd.DataFrame) -> str:
    """Render a frame as a Markdown table.

    ``DataFrame.to_markdown`` needs ``tabulate``, which is an optional extra.
    A findings report is core output and must not fail on a formatting
    dependency, so fall back to building the table directly.
    """
    try:
        return frame.to_markdown()
    except ImportError:
        pass

    index_name = frame.index.name or ""
    header = [index_name, *(str(c) for c in frame.columns)]

    rows = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for key, record in frame.iterrows():
        cells = [str(key), *(f"{v}" for v in record.to_numpy())]
        rows.append("| " + " | ".join(cells) + " |")

    return "\n".join(rows)


def _reading_guide() -> str:
    return """## How to read an EII Cloud

The cloud scores six things. Three describe the machine, three describe your
ability to watch it:

| Component | What a high score means | Whose problem |
|---|---|---|
| `value_deviation` | reading left its expected envelope | the machine |
| `change_inconsistency` | legal value, abnormal rate of change | the machine |
| `flatline` | value arrives but has stopped moving | the sensor |
| `observability_degradation` | samples did not arrive | the pipeline |
| `contextual_missingness` | samples did not arrive *and were due* | the pipeline |
| `context_contradiction` | telemetry says idle, context says busy | either — establish order first |

A high total driven by the top two rows is an incident. A high total driven by
the bottom three is a **blind spot**: not evidence that something is wrong, but
evidence that you would not know if it were. The two are routinely confused, and
they call for opposite responses — the first one pages the hardware team, the
second one means your monitoring cannot currently answer the question.
"""
