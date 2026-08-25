#!/usr/bin/env python3
"""Plot recommendation quality and stopping output across user-provided eta."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
ETAS = (0.1, 0.2, 0.4, 0.7, 1.0)
BENCHMARKS = (("hotpotqa", "HotpotQA"), ("mathqa", "MathQA"))


def slug(value: float) -> str:
    return f"{value:.8g}".replace(".", "p").replace("-", "m")


def mean_ci(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    mean = float(np.mean(array))
    ci = 1.96 * float(np.std(array, ddof=1)) / np.sqrt(array.size) if array.size > 1 else 0.0
    return mean, ci


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--indir", type=Path,
        default=ROOT / "analysis/vs/gittins_eta_pareto_coverage_grid129",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    indir = args.indir if args.indir.is_absolute() else ROOT / args.indir
    output = args.output or indir / "radial_gittins_eta_pareto_coverage.png"

    rows: list[dict[str, float | int | str]] = []
    metrics: dict[tuple[str, float], dict[str, list[float]]] = {}
    for benchmark, _ in BENCHMARKS:
        for eta in ETAS:
            values = {key: [] for key in ("recall", "precision", "false_positives", "size", "cost")}
            for path in sorted((indir / "raw").glob(f"{benchmark}_eta-{slug(eta)}_seed-*.npz")):
                with np.load(path, allow_pickle=False) as data:
                    if "stop_recommended_arm_indices" not in data:
                        continue
                    recommended = set(map(int, data["stop_recommended_arm_indices"]))
                    truth = set(map(int, data["true_pareto_arm_indices"]))
                    true_positive = len(recommended & truth)
                    values["recall"].append(true_positive / len(truth) if truth else 1.0)
                    values["precision"].append(true_positive / len(recommended) if recommended else 1.0)
                    values["false_positives"].append(float(len(recommended - truth)))
                    values["size"].append(float(len(recommended)))
                    values["cost"].append(float(data["stop_cost_fraction"]))
            metrics[(benchmark, eta)] = values
            row: dict[str, float | int | str] = {"benchmark": benchmark, "eta": eta,
                                                  "n_seeds": len(values["recall"])}
            for key, samples in values.items():
                if samples:
                    row[f"mean_{key}"], row[f"ci95_{key}"] = mean_ci(samples)
            rows.append(row)

    if not any(int(row["n_seeds"]) for row in rows):
        raise SystemExit(f"no coverage-aware runs found under {indir / 'raw'}")

    plt.rcParams.update({"font.size": 11, "axes.titlesize": 15, "axes.labelsize": 12})
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.2), sharex="col")
    colors = plt.cm.viridis(np.linspace(0.12, 0.88, len(ETAS)))
    panels = (
        ("recall", "True Pareto frontier coverage (recall)", (0, 1.05)),
        ("precision", "Recommendation precision", (0, 1.05)),
        ("false_positives", "False-positive recommendations", None),
        ("cost", "Stopping cumulative cost fraction", (0, 1.0)),
    )
    for ax, (metric, ylabel, ylim) in zip(axes.flat, panels):
        for offset, (benchmark, label) in enumerate(BENCHMARKS):
            means, cis = [], []
            for eta in ETAS:
                samples = metrics[(benchmark, eta)][metric]
                mean, ci = mean_ci(samples) if samples else (np.nan, np.nan)
                means.append(mean); cis.append(ci)
            x = np.arange(len(ETAS), dtype=float) + (offset - 0.5) * 0.10
            ax.errorbar(x, means, yerr=cis, marker="o", linewidth=2, capsize=3,
                        label=label)
        ax.set_ylabel(ylabel)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.grid(alpha=0.22)
        ax.set_xticks(range(len(ETAS)), [str(eta) for eta in ETAS])
    for ax in axes[-1]:
        ax.set_xlabel("User-provided trade-off parameter η")
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Radial-Gittins stopping recommendations across η (mean ± 95% CI)",
                 fontsize=17)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)

    csv_path = indir / "eta_pareto_coverage_summary.csv"
    fields = sorted({key for row in rows for key in row}, key=lambda x: (x not in ("benchmark", "eta", "n_seeds"), x))
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    print(output.resolve())
    print(csv_path.resolve())


if __name__ == "__main__":
    main()
