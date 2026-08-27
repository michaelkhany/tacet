"""End-to-end tacet walkthrough on synthetic cluster telemetry.

Run: python examples/quickstart.py
"""

import tacet

# 1. Data. Any source works the same way -- swap in open_source("prometheus://...")
#    or open_source("/data/*.parquet") and nothing below changes.
telemetry, truth = tacet.datasets.make_cluster(n_nodes=6, n_samples=3000, seed=0)
print(f"{len(telemetry):,} samples, {telemetry['tel_gpu_temp'].isna().mean():.1%} never arrived")

# 2. Window it. This is where the observability plane gets built.
windows = tacet.to_windows(
    telemetry, window="30min", stride="10min", expected_interval="1min"
)
print(f"{len(windows):,} windows · mean coverage {windows['obs_coverage'].mean():.2f}")

# 3. Label the observation faults -- exporter outages, stuck sensors, partial
#    scrapes. These are the failures conventional detectors score as *normal*,
#    which makes them the honest test of what this library is for.
windows = tacet.label_episodes(windows, truth[truth["family"] == "observation"])
windows = tacet.chronological_split(windows)

train = windows[windows["split"] == "train"]
test = windows[windows["split"] == "test"]

# 4. Missingness first -- before modelling, understand the gaps.
report = tacet.analyze_missingness(windows, label="label")
print("\nmissingness mechanisms:", report.mechanism["mechanism"].value_counts().to_dict())
print("co-missing clusters:", report.co_missing_clusters()[:2])

# 5. The EII Cloud, and the tips that explain it.
#    Scoring every aggregate of every metric produces a lot of near-duplicate
#    findings, so name the parameters you actually care about.
cloud = tacet.EIICloud(
    parameters=["tel_gpu_temp_mean", "tel_sm_clock_mean", "tel_power_draw_mean"],
    expected_present="ctx_job_active_last",
    high_load="ctx_high_load_last",
).fit(train).transform(test)

tips = cloud.tips(max_tips=8)
print(f"\n{len(tips)} interpretation tips:")
for tip in tips:
    print(f"  [{tip.severity:8}] {tip.code:18} {tip.parameter} ({tip.span}w)")

# 6. Detect, with every method sharing one interface.
results = {}
for name, factory in tacet.detect.REGISTRY.items():
    detector = factory(planes=["telemetry", "observability"]).fit(train)
    scored = detector.alert(detector.score(test), budget=200)
    results[name] = tacet.score_report(scored, budget=200)

# ...and the same detector on telemetry alone, to show what the observability
# plane is actually worth.
for name in ("robust_z", "graph"):
    detector = tacet.detect.REGISTRY[name](planes=["telemetry"]).fit(train)
    scored = detector.alert(detector.score(test), budget=200)
    results[f"{name} (telemetry only)"] = tacet.score_report(scored, budget=200)

table = tacet.compare(results)
print("\n" + table[["roc_auc", "average_precision", "precision", "blind_alert_rate"]]
      .round(3).to_string())

# 7. Correlation structure across planes.
mapping = tacet.correlation_map(test, method="spearman")
print("\nplane-level association:")
print(mapping.plane_summary().round(3).to_string())

confounded = mapping.observability_confounding(threshold=0.4)
if not confounded.empty:
    print(f"\n{len(confounded)} metrics co-vary with their own collection health "
          "-- do not read these at face value.")
