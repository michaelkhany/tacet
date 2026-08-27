"""Canonical frame contract shared by every ``tacet`` component.

``tacet`` works on *window matrices*: one row per (entity, time window), with
feature columns namespaced by **evidence plane**. The plane prefix is not
cosmetic -- it is what lets the library reason about *where* evidence came from
and therefore how much to trust it.

Planes
------
``tel_``  telemetry ......... what the sensor reported
``ctx_``  context ........... scheduler / workload / topology state
``log_``  log ............... event and template counts
``obs_``  observability ..... health of the *monitoring pipeline itself*
``eii_``  early instability .. derived EII Cloud components (see :mod:`tacet.eii`)

The ``obs_`` plane is what separates ``tacet`` from conventional anomaly
detection stacks: it carries the answer to "could we even see this entity?",
which is treated as evidence rather than as a gap to impute over.
"""

from __future__ import annotations

from typing import Final

# --- identity / time -------------------------------------------------------
ENTITY: Final = "entity_id"
DEVICE: Final = "device_id"
TIMESTAMP: Final = "timestamp"
WINDOW_ID: Final = "window_id"
WINDOW_START: Final = "window_start"
WINDOW_END: Final = "window_end"

# --- supervision / results -------------------------------------------------
LABEL: Final = "label"
EVENT_TIME: Final = "event_time"
SPLIT: Final = "split"
SCORE: Final = "anomaly_score"
ALERT: Final = "alert"
TRUST: Final = "observability_trust"

#: Columns that are never treated as model features.
META_COLUMNS: Final = (
    ENTITY,
    DEVICE,
    TIMESTAMP,
    WINDOW_ID,
    WINDOW_START,
    WINDOW_END,
    LABEL,
    EVENT_TIME,
    SPLIT,
    SCORE,
    ALERT,
    TRUST,
)

# --- evidence planes -------------------------------------------------------
TELEMETRY: Final = "tel_"
CONTEXT: Final = "ctx_"
LOG: Final = "log_"
OBSERVABILITY: Final = "obs_"
EII: Final = "eii_"

#: Plane name -> column prefix.
PLANES: Final = {
    "telemetry": TELEMETRY,
    "context": CONTEXT,
    "log": LOG,
    "observability": OBSERVABILITY,
    "eii": EII,
}

#: Reverse lookup, longest prefix first so ``eii_`` never shadows a shorter one.
_PREFIX_TO_PLANE: Final = {prefix: name for name, prefix in PLANES.items()}


def plane_of(column: str) -> str:
    """Return the evidence plane a column belongs to, or ``"meta"``.

    >>> plane_of("obs_scrape_gap")
    'observability'
    >>> plane_of("entity_id")
    'meta'
    """
    for prefix, name in _PREFIX_TO_PLANE.items():
        if column.startswith(prefix):
            return name
    return "meta"


def feature_columns(frame, planes=None) -> list[str]:
    """Feature columns of ``frame``, optionally restricted to ``planes``.

    Parameters
    ----------
    frame:
        A window matrix.
    planes:
        Plane names to keep (e.g. ``["telemetry", "observability"]``). ``None``
        keeps every non-meta column.
    """
    if planes is None:
        return [c for c in frame.columns if c not in META_COLUMNS]

    prefixes = tuple(PLANES[p] for p in planes)
    return [
        c
        for c in frame.columns
        if c not in META_COLUMNS and c.startswith(prefixes)
    ]
