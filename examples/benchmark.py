"""What is the observability plane actually worth?

Runs every detector twice on the same data -- once on telemetry alone, once with
the derived `obs_` plane included -- against observation faults: exporter
outages, stuck sensors, partial scrapes. These are the failures a conventional
detector scores as normal, because every number that arrives is unremarkable.

Averaged over several seeds, because a single run of anything is an anecdote.

Run: python examples/benchmark.py
"""

import pandas as pd

import tacet
from tacet import schema

SEEDS = range(6)
BUDGET = 200


def run(seed):
    telemetry, truth = tacet.datasets.make_cluster(
        n_nodes=6, n_samples=3000, seed=seed
    )
    windows = tacet.to_windows(
        telemetry, window="30min", stride="10min", expected_interval="1min"
    )
    windows = tacet.chronological_split(
        tacet.label_episodes(windows, truth[truth["family"] == "observation"])
    )

    train = windows[windows[schema.SPLIT] == "train"]
    test = windows[windows[schema.SPLIT] == "test"]

    if test[schema.LABEL].sum() == 0:
        return []

    rows = []
    for name, factory in tacet.detect.REGISTRY.items():
        for label, planes in (
            ("telemetry only", ["telemetry"]),
            ("+ observability", ["telemetry", "observability"]),
        ):
            detector = factory(planes=planes).fit(train)
            scored = detector.alert(detector.score(test), budget=BUDGET)
            report = tacet.score_report(scored, budget=BUDGET)

            rows.append(
                {
                    "detector": name,
                    "features": label,
                    "roc_auc": report["roc_auc"],
                    "average_precision": report["average_precision"],
                    "seed": seed,
                }
            )
    return rows


def main():
    records = [row for seed in SEEDS for row in run(seed)]
    frame = pd.DataFrame(records)

    table = (
        frame.groupby(["detector", "features"])[["roc_auc", "average_precision"]]
        .mean()
        .unstack("features")
        .round(3)
    )
    print(f"Mean over {len(SEEDS)} seeds, observation faults, budget {BUDGET}\n")
    print(table.to_string())

    wide = frame.pivot_table(
        index=["detector", "seed"], columns="features", values="average_precision"
    )
    gain = (wide["+ observability"] - wide["telemetry only"]).groupby("detector").mean()

    print("\nAverage-precision gain from the observability plane:")
    print(gain.round(3).sort_values(ascending=False).to_string())
    print(f"\nmean gain across all detectors: {gain.mean():+.3f}")
    print(f"detectors improved: {(gain > 0).sum()}/{len(gain)}")


if __name__ == "__main__":
    main()
