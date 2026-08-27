"""tacet — observability-aware anomaly detection for HPC and distributed systems.

*tacet* (Latin, "it is silent") is the direction in a musical score telling an
instrument not to play. In a monitoring system, silence is rarely an
instruction. It is usually the most important thing the system said, and almost
every tool built for the problem is designed not to hear it.

The premise
-----------
Conventional anomaly detection asks *is this value abnormal?* and, when the
value is missing, imputes something plausible and carries on. That is a
reasonable choice for a sensor on a bench and an actively harmful one for a
distributed system, where the failure of the observer and the failure of the
observed are correlated events. A node that overheats and takes its exporter
down produces no anomalous readings at all.

``tacet`` asks a second question alongside the first: *could we see this at all,
and how much do we trust what came back?* Missing samples are scored, not
filled. Coverage, gap structure and staleness become first-class features. Every
detector reports observability trust next to its anomaly score, so "quiet and
clearly observed" and "quiet and half blind" -- the same number, opposite
conclusions -- stop being the same answer.

What is here
------------
``tacet.open_source``       offline files, live endpoints, pushed streams
``tacet.to_windows``        windowing that manufactures the observability plane
``tacet.EIICloud``          Early Instability Indicator scoring, six components
``cloud.tips()``            interpretation callouts: how to read each anomaly
``tacet.detect``            Markov, feature-graph, and classical detectors
``tacet.analysis``          correlation mapping, lead/lag, missingness structure
``tacet.evaluate``          budget-aware and observability-aware metrics
``tacet.viz``               annotated figures and Markdown findings reports

Quick start
-----------
>>> import tacet
>>> telemetry, truth = tacet.datasets.make_cluster(seed=0)
>>> windows = tacet.to_windows(telemetry, window="30min", stride="10min",
...                            expected_interval="1min")
>>> cloud = tacet.EIICloud(expected_present="ctx_job_active_last",
...                        high_load="ctx_high_load_last").fit_transform(windows)
>>> for tip in cloud.tips()[:3]:
...     print(tip)
"""

from __future__ import annotations

__version__ = "0.1.0"

from . import analysis, datasets, detect, eii, evaluate, schema, sources, viz
from .analysis import analyze_missingness, correlation_map, lead_lag
from .detect import (
    CUSUMDetector,
    EWMADetector,
    FeatureGraphDetector,
    MahalanobisDetector,
    MarkovDetector,
    RobustZScore,
    apply_budget,
    observability_trust,
)
from .eii import EIICloud, EIICloudResult, Tip, generate_tips
from .evaluate import compare, lead_time
from .evaluate import evaluate as score_report
from .sources import TelemetrySource, open_source, to_long, to_wide
from .windows import chronological_split, label_episodes, label_horizon, to_windows

__all__ = [
    "__version__",
    # sub-packages
    "schema",
    "sources",
    "eii",
    "detect",
    "analysis",
    "evaluate",
    "viz",
    "datasets",
    # input
    "open_source",
    "TelemetrySource",
    "to_long",
    "to_wide",
    "to_windows",
    "label_horizon",
    "label_episodes",
    "chronological_split",
    # EII cloud
    "EIICloud",
    "EIICloudResult",
    "Tip",
    "generate_tips",
    # detection
    "MarkovDetector",
    "FeatureGraphDetector",
    "RobustZScore",
    "EWMADetector",
    "CUSUMDetector",
    "MahalanobisDetector",
    "apply_budget",
    "observability_trust",
    # analysis
    "correlation_map",
    "lead_lag",
    "analyze_missingness",
    # evaluation
    "score_report",
    "compare",
    "lead_time",
]
