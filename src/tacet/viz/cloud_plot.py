"""Plot an EII Cloud, with or without interpretation tips.

``tips=False`` gives the raw picture: what happened.
``tips=True`` adds the callouts: what it means, and what to do about it.

The plot makes one editorial choice worth knowing about. Regions with no data
are drawn as **shaded blind bands**, never as a connected line across the gap.
Every plotting library joins the points either side of a gap by default, which
renders a monitoring outage as a smooth interpolation through it -- the single
most misleading convention in operational dashboards, and the reason silent
failures survive review.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import schema

__all__ = ["plot_cloud", "plot_components"]

SEVERITY_COLORS = {
    "critical": "#c1121f",
    "warning": "#e07a1f",
    "info": "#2a6f97",
}


def _pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ImportError(
            "plotting needs matplotlib. Install it with: pip install \"tacet[viz]\""
        ) from exc
    return plt


def plot_cloud(
    result,
    parameter: str | None = None,
    tips: bool | list = True,
    entity: str | None = None,
    max_tips: int = 6,
    figsize: tuple[float, float] = (14, 9),
    path: str | None = None,
    title: str | None = None,
    dpi: int = 160,
):
    """Render the cloud for one parameter as a three-panel figure.

    Panels, top to bottom:

    1. the raw signal with its fitted envelope and shaded blind bands;
    2. the six EII components as a heatmap, so you can see *which* kind of
       abnormality is driving the score at each moment;
    3. total EII against observability trust -- the two lines that, read
       together, separate "nothing is wrong" from "we cannot tell".

    Parameters
    ----------
    result:
        An :class:`~tacet.eii.cloud.EIICloudResult`.
    parameter:
        Which parameter to draw. Defaults to the highest-scoring one.
    tips:
        ``True`` generates tips and annotates them; ``False`` draws the bare
        cloud; a list of :class:`~tacet.eii.tips.Tip` annotates those.
    entity:
        Restrict to one entity. Required in spirit when the frame holds several
        -- otherwise windows from different nodes are drawn as one series.
    path:
        Save here instead of returning an interactive figure.

    Returns
    -------
    matplotlib.figure.Figure

    Examples
    --------
    >>> plot_cloud(cloud, tips=True, path="eii_cloud.png")
    >>> plot_cloud(cloud, tips=False)          # the bare cloud
    """
    plt = _pyplot()

    frame, cloud, rows = _slice(result, entity)
    parameter = parameter or _busiest(cloud, result)

    tip_list = _resolve_tips(result, tips, parameter, entity, max_tips)

    # Everything is drawn against the integer window index, then the bottom
    # axis is relabelled with timestamps. Mixing a datetime axis with imshow's
    # numeric `extent` under sharex makes matplotlib read the extent as epoch
    # nanoseconds, which smears the shared range from 1970 to the present.
    x = np.arange(len(frame))
    figure, (ax_signal, ax_components, ax_total) = plt.subplots(
        3, 1, figsize=figsize, sharex=True,
        gridspec_kw={"height_ratios": [3, 2.4, 1.8], "hspace": 0.18},
        dpi=dpi,
    )

    _draw_signal(ax_signal, x, frame, parameter, result)
    _draw_components(ax_components, x, cloud, parameter, rows, figure)
    _draw_total(ax_total, x, frame)

    if tip_list:
        _annotate(ax_signal, x, tip_list, rows)

    _time_ticks(ax_total, frame, result)

    heading = title or f"EII Cloud — {parameter}" + (f" — {entity}" if entity else "")
    if tip_list:
        heading += f"  ({len(tip_list)} interpretation tip{'s' if len(tip_list) != 1 else ''})"
    figure.suptitle(heading, fontsize=13, fontweight="bold", y=0.98)

    figure.align_ylabels()

    if path:
        figure.savefig(path, bbox_inches="tight", dpi=dpi)
        plt.close(figure)

    return figure


# -- panels -----------------------------------------------------------------


def _draw_signal(ax, x, frame, parameter, result):
    values = pd.to_numeric(frame[parameter], errors="coerce").to_numpy(dtype=float)
    observed = np.isfinite(values)

    # Blind bands first, so the line draws over them.
    for start, stop in _runs(~observed):
        ax.axvspan(
            x[start],
            x[min(stop, len(x) - 1)],
            color="#b0b0b0",
            alpha=0.35,
            zorder=0,
            label="_nolegend_",
        )

    # The band that scoring used, not one re-derived from the data being shown.
    # Re-deriving stretches the envelope to include the very excursions the
    # chart is meant to highlight, so they stop looking like excursions.
    envelope = getattr(result, "envelopes", {}).get(parameter)
    if envelope is None:
        finite = values[observed]
        envelope = tuple(np.quantile(finite, [0.05, 0.95])) if finite.size else None

    if envelope is not None:
        low, high = envelope
        ax.axhspan(low, high, color="#4c956c", alpha=0.10, zorder=0)
        ax.axhline(low, color="#4c956c", lw=0.8, ls="--", alpha=0.6)
        ax.axhline(high, color="#4c956c", lw=0.8, ls="--", alpha=0.6,
                   label="fitted envelope")

    # Masked array breaks the line at gaps instead of interpolating across them.
    ax.plot(x, np.ma.masked_invalid(values), lw=1.4, color="#1d3557", label=parameter)
    ax.scatter(
        x[~observed],
        np.full((~observed).sum(), np.nanmin(values) if observed.any() else 0),
        marker="|", s=40, color="#6c757d", label="no data",
    )

    ax.set_ylabel(parameter, fontsize=9)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.25)


def _draw_components(ax, x, cloud, parameter, rows, figure):
    from ..eii.components import COMPONENT_NAMES

    subset = cloud[cloud["parameter"] == parameter]
    matrix = np.zeros((len(COMPONENT_NAMES), len(rows)))

    position = {row: i for i, row in enumerate(rows)}
    for record in subset.itertuples():
        if record.row in position and record.component in COMPONENT_NAMES:
            matrix[COMPONENT_NAMES.index(record.component), position[record.row]] = record.score

    image = ax.imshow(
        matrix, aspect="auto", origin="lower", cmap="magma_r", vmin=0, vmax=1,
        extent=(-0.5, len(rows) - 0.5, -0.5, len(COMPONENT_NAMES) - 0.5),
    )
    ax.set_yticks(range(len(COMPONENT_NAMES)))
    ax.set_yticklabels([c.replace("_", " ") for c in COMPONENT_NAMES], fontsize=8)
    ax.set_ylabel("EII component", fontsize=9)

    bar = figure.colorbar(image, ax=ax, pad=0.01, fraction=0.025)
    bar.set_label("component score", fontsize=8)
    bar.ax.tick_params(labelsize=7)


def _draw_total(ax, x, frame):
    total = f"{schema.EII}total"
    if total in frame.columns:
        ax.plot(x, frame[total].to_numpy(dtype=float), lw=1.6,
                color="#9d0208", label="total EII")
        ax.fill_between(x, 0, frame[total].to_numpy(dtype=float),
                        color="#9d0208", alpha=0.15)

    coverage = f"{schema.OBSERVABILITY}coverage"
    if coverage in frame.columns:
        ax.plot(x, frame[coverage].to_numpy(dtype=float), lw=1.2, ls="--",
                color="#2a6f97", label="observability coverage")

    ax.set_ylim(-0.02, 1.05)
    ax.set_ylabel("score", fontsize=9)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.25)


def _annotate(ax, x, tips, rows):
    """Place tip callouts, staggered so overlapping episodes stay readable."""
    position = {row: i for i, row in enumerate(rows)}
    low, high = ax.get_ylim()
    span = high - low

    for i, tip in enumerate(tips):
        start = position.get(tip.start_row)
        end = position.get(tip.end_row, start)
        if start is None:
            continue

        color = SEVERITY_COLORS.get(tip.severity, "#333333")
        ax.axvspan(x[start], x[min(end, len(x) - 1)], color=color, alpha=0.12, zorder=1)

        # Stagger vertically; short titles only, full text lives in the report.
        level = high - span * (0.08 + 0.15 * (i % 4))
        label = tip.title.split("—")[0].strip()

        ax.annotate(
            f"{label}\n({tip.span}w)",
            xy=(x[start], level),
            xytext=(8, 0),
            textcoords="offset points",
            fontsize=7.5,
            color="white",
            va="center",
            bbox={"boxstyle": "round,pad=0.35", "facecolor": color, "alpha": 0.92,
                  "edgecolor": "none"},
            arrowprops={"arrowstyle": "-", "color": color, "lw": 1.0},
            zorder=5,
        )


# -- helpers ----------------------------------------------------------------


def _slice(result, entity):
    frame = result.frame
    if entity is not None:
        if result.entity_column is None:
            raise ValueError("this cloud has no entity column to filter on")
        frame = frame[frame[result.entity_column] == entity]
        if frame.empty:
            raise ValueError(f"no windows for entity {entity!r}")

    rows = list(frame.index)
    cloud = result.components[result.components["row"].isin(rows)]

    return frame, cloud, rows


def _busiest(cloud, result):
    if cloud.empty:
        return result.parameters[0]
    return cloud.groupby("parameter")["score"].max().idxmax()


def _time_ticks(ax, frame, result, count: int = 8):
    """Relabel the shared integer axis with timestamps."""
    n = len(frame)
    positions = np.linspace(0, max(n - 1, 0), min(count, max(n, 1))).astype(int)
    ax.set_xticks(positions)

    if not (result.time_column and result.time_column in frame.columns):
        ax.set_xticklabels([str(p) for p in positions])
        return

    stamps = pd.to_datetime(frame[result.time_column]).iloc[positions]
    span = stamps.max() - stamps.min()
    fmt = "%H:%M" if span < pd.Timedelta("2d") else "%m-%d %H:%M"

    ax.set_xticklabels([s.strftime(fmt) for s in stamps], rotation=30, ha="right",
                       fontsize=8)
    ax.set_xlabel(f"window  ({stamps.min():%Y-%m-%d} → {stamps.max():%Y-%m-%d})",
                  fontsize=9)


def _resolve_tips(result, tips, parameter, entity, max_tips):
    if tips is False:
        return []

    tip_list = result.tips(max_tips=200) if tips is True else list(tips)
    filtered = [t for t in tip_list if t.parameter in (None, parameter)]
    if entity is not None:
        filtered = [t for t in filtered if t.entity in (None, entity)]

    return filtered[:max_tips]


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous ``[start, stop]`` index ranges where ``mask`` is True."""
    if not mask.any():
        return []

    padded = np.concatenate(([False], mask, [False]))
    changes = np.diff(padded.astype(np.int8))

    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1) - 1

    return list(zip(starts.tolist(), stops.tolist()))


def plot_components(result, parameter: str | None = None, path: str | None = None,
                    figsize=(11, 4), dpi: int = 160):
    """Bar chart of mean component contribution -- which kind of abnormality dominates."""
    plt = _pyplot()
    from ..eii.components import COMPONENT_NAMES

    cloud = result.components
    if parameter is not None:
        cloud = cloud[cloud["parameter"] == parameter]

    means = (
        cloud.groupby("component")["score"].mean()
        .reindex(COMPONENT_NAMES).fillna(0.0)
    )

    figure, ax = plt.subplots(figsize=figsize, dpi=dpi)
    colors = ["#1d3557" if c in ("value_deviation", "change_inconsistency", "flatline")
              else "#c1121f" for c in means.index]

    ax.barh([c.replace("_", " ") for c in means.index], means.to_numpy(), color=colors)
    ax.set_xlabel("mean component score")
    ax.set_title(
        "EII component mix"
        + (f" — {parameter}" if parameter else "")
        + "\n(red = the monitoring pipeline, navy = the machine)",
        fontsize=10,
    )
    ax.grid(alpha=0.25, axis="x")
    figure.tight_layout()

    if path:
        figure.savefig(path, bbox_inches="tight", dpi=dpi)
        plt.close(figure)

    return figure
