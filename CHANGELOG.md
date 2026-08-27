# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-08-27

First public release.

### Added

- **Sources** (`tacet.sources`) — offline (`CsvSource`, `ParquetSource`,
  `JsonlSource`, `DirectorySource`), live (`CallableSource`, `PushSource`,
  `ReplaySource`), and `PrometheusSource` for any Prometheus-compatible
  endpoint. All share one long-frame contract, and none of them fabricate a
  sample that did not arrive.
- **Windowing** (`tacet.to_windows`) — builds the `obs_` observability plane
  automatically: coverage, expected vs observed sample counts, longest interior
  gap, staleness, and per-metric coverage.
- **EII Cloud** (`tacet.EIICloud`) — Early Instability Indicator scoring across
  six components, three of which measure the monitoring pipeline rather than the
  machine.
- **Tips** (`cloud.tips()`) — interpretation callouts that name the abnormal
  situation, explain how to read it on the chart, and suggest an action.
  Rendered on figures and as report front matter.
- **Detectors** (`tacet.detect`) — `MarkovDetector`, `FeatureGraphDetector`,
  `RobustZScore`, `EWMADetector`, `CUSUMDetector`, `MahalanobisDetector`, all
  behind one `fit`/`score`/`alert` interface with observability trust reported
  alongside every score.
- **Analysis** (`tacet.analysis`) — correlation mapping across six estimators,
  cross-plane summaries, `lead_lag` precursor search, and a missingness module
  covering run-length structure, nullity correlation, co-missing clusters,
  MCAR/MAR/MNAR classification, and Little's MCAR test.
- **Evaluation** (`tacet.evaluate`) — exact alert budgeting, ROC-AUC and average
  precision, plus `blind_alert_rate`, `blind_positive_rate` and
  `mean_trust_at_alert`, and `lead_time` for warning horizons.
- **Synthetic data** (`tacet.datasets.make_cluster`) — injects machine faults
  *and* observation faults separately, with per-episode ground truth.
