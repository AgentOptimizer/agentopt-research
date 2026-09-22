#!/usr/bin/env python3
"""Plot 10% and 30% 20-seed frequency frontiers for every baseline."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_rgb


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = ROOT / "analysis/20seed_results"
DEFAULT_OUTPUT = DEFAULT_RESULTS / "figures/baseline_frequency_frontiers"
CHECKPOINT_INDEX = DEFAULT_RESULTS / "recommendation_checkpoints.csv"
LEGACY_PARETO_10PCT = (
    ROOT / "analysis/continuous_seeds_42_61/data/pareto_baselines_10pct/recommendations.csv"
)
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from experiments.combined_objective.plot_gittins_20seed_frequency_frontiers import (  # noqa: E402
    DATASETS,
    _draw_panel,
    _load_landscape,
)


METHODS = (
    "ege_sh",
    "ege_sr",
    "ape_k",
    "qnehvi",
    "random_configurations",
    "random_questions",
)
METHOD_LABELS = {
    "ege_sh": "EGE-SH",
    "ege_sr": "EGE-SR",
    "ape_k": "APE-k",
    "qnehvi": "qNEHVI",
    "random_configurations": "Random configurations",
    "random_questions": "Random questions",
}
METHOD_COLORS = {
    "ege_sh": "tab:green",
    "ege_sr": "tab:blue",
    "ape_k": "tab:purple",
    "qnehvi": "tab:pink",
    "random_configurations": "tab:brown",
    "random_questions": "tab:cyan",
}
EXPECTED_SEEDS = set(range(42, 62))


def _mix(color: tuple[float, float, float], other: tuple[float, float, float], amount: float):
    return tuple((1.0 - amount) * value + amount * target for value, target in zip(color, other))


def _method_colormap(method: str) -> LinearSegmentedColormap:
    base = to_rgb(METHOD_COLORS[method])
    return LinearSegmentedColormap.from_list(
        f"{method}_frequency",
        (_mix(base, (1.0, 1.0, 1.0), 0.82), _mix(base, (1.0, 1.0, 1.0), 0.45), base,
         _mix(base, (0.0, 0.0, 0.0), 0.38)),
        N=256,
    )


def _indices(text: str) -> set[int]:
    return {int(value) for value in text.split(";") if value.strip()}


def _latest_trajectory_rows(method_dir: Path, target: float) -> dict[int, dict[str, str]]:
    chosen: dict[int, dict[str, str]] = {}
    for path in sorted(method_dir.glob("seed-*/cost_trajectory.csv")):
        if not path.exists():
            continue
        seed = int(path.parent.name.removeprefix("seed-"))
        latest: dict[str, str] | None = None
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if float(row["budget_fraction"]) <= target + 1e-12:
                    latest = row
        if latest is not None:
            chosen[seed] = latest
    return chosen


def _aggregate_recommendation_rows(
    method_dir: Path, method: str, dataset: str, target: float,
) -> dict[int, dict[str, str]]:
    path = (
        LEGACY_PARETO_10PCT
        if method in {"ege_sh", "ape_k"}
        and dataset in {"hotpotqa", "mathqa"}
        and np.isclose(target, 0.10)
        else method_dir / "aggregate_recommendations.csv"
    )
    if not path.exists():
        return {}
    chosen: dict[int, dict[str, str]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("method", method).lower() != method:
                continue
            if row.get("benchmark", dataset).lower() != dataset:
                continue
            # The legacy qNEHVI recommendation table contains only its 10%
            # export and therefore has no explicit budget_fraction column.
            row_target = float(row.get("budget_fraction") or 0.10)
            if np.isclose(row_target, target):
                chosen[int(row["seed"])] = row
    return chosen


def _random_rows(method_dir: Path, method: str, target: float) -> dict[int, dict[str, str]]:
    path = method_dir / "multi_seed_results.csv"
    if not path.exists():
        return {}
    chosen: dict[int, dict[str, str]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["version"] == method and np.isclose(float(row["budget_fraction"]), target):
                chosen[int(row["seed"])] = row
    return chosen


def _summary(
    *,
    results_root: Path,
    method: str,
    dataset: str,
    target: float,
    landscape: dict[str, object],
) -> dict[str, object]:
    method_dir = results_root / method / dataset
    rows: dict[int, dict[str, str]] = {}
    checkpoint_index = results_root / CHECKPOINT_INDEX.name
    if checkpoint_index.is_file() and method in {"ege_sh", "ege_sr", "ape_k", "qnehvi"}:
        with checkpoint_index.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if (
                    row["method"] == method
                    and row["dataset"] == dataset
                    and np.isclose(float(row["target_cost_fraction"]), target)
                ):
                    rows[int(row["seed"])] = row
    source = "checkpoint_index" if rows else "trajectory"
    if not rows:
        rows = _latest_trajectory_rows(method_dir, target)
    if not rows and method in {"random_configurations", "random_questions"}:
        rows = _random_rows(method_dir, method, target)
        source = "random"
    if not rows:
        rows = _aggregate_recommendation_rows(method_dir, method, dataset, target)
        source = "aggregate"

    truth = np.asarray(landscape["truth"])
    true_frontier = set(np.asarray(landscape["frontier"], dtype=int).tolist())
    model_names = list(landscape["model_names"])
    model_index = {name: index for index, name in enumerate(model_names)}
    full_search_cost = float(landscape["full_search_cost"])

    counts: Counter[int] = Counter()
    recalls: list[float] = []
    spends: list[float] = []
    exact = 0
    for seed, row in sorted(rows.items()):
        if seed not in EXPECTED_SEEDS:
            continue
        if source in {"trajectory", "checkpoint_index"}:
            selected = _indices(row["selected_arm_indices"])
            spend = (
                float(row["cumulative_search_cost_usd"])
                if source == "checkpoint_index"
                else target * full_search_cost
            )
        else:
            names = json.loads(row["selected_models"])
            missing_names = [name for name in names if name not in model_index]
            if missing_names:
                raise ValueError(
                    f"{method}/{dataset}/seed-{seed} contains unknown models: {missing_names[:3]}"
                )
            selected = {model_index[name] for name in names}
            if source == "random":
                spend = float(row["total_search_cost_usd"])
            else:
                spend = float(row["actual_cost_fraction"]) * full_search_cost
        if any(index < 0 or index >= len(truth) for index in selected):
            raise ValueError(f"invalid selected index in {method}/{dataset}/seed-{seed}")
        counts.update(selected)
        recalls.append(len(selected & true_frontier) / len(true_frontier))
        exact += int(selected == true_frontier)
        spends.append(spend)

    available = len(spends)
    if not available:
        return {
            "counts": counts,
            "available_seed_count": 0,
            "mean_spend_usd": float("nan"),
            "spend_iqr_usd": (float("nan"), float("nan")),
            "mean_recall": float("nan"),
            "exact": 0,
        }
    return {
        "counts": counts,
        "available_seed_count": available,
        "mean_spend_usd": float(np.mean(spends)),
        "spend_iqr_usd": tuple(float(value) for value in np.quantile(spends, (0.25, 0.75))),
        "mean_recall": float(np.mean(recalls)),
        "exact": exact,
    }


def make_figure(
    *,
    method: str,
    target: float,
    results_root: Path,
    output_dir: Path,
    landscapes: dict[str, dict[str, object]],
) -> tuple[Path, Path]:
    cmap = _method_colormap(method)
    norm = mpl.colors.Normalize(vmin=1, vmax=20)
    fig, axes = plt.subplots(2, 4, figsize=(20.0, 10.5))
    for ax, dataset in zip(axes.flat, DATASETS):
        summary = _summary(
            results_root=results_root,
            method=method,
            dataset=dataset,
            target=target,
            landscape=landscapes[dataset],
        )
        _draw_panel(
            ax,
            dataset=dataset,
            landscape=landscapes[dataset],
            summary=summary,
            target=target,
            cmap=cmap,
            norm=norm,
        )

    fig.supxlabel(
        "Mean deployment cost (USD per query, log scale)",
        fontsize=24,
        fontweight="normal",
        y=0.035,
    )
    fig.supylabel(
        "Mean accuracy",
        fontsize=24,
        fontweight="normal",
        x=0.022,
    )
    fig.suptitle(
        f"{METHOD_LABELS[method]} recommendations across 20 seeds ({target:.0%} checkpoint)",
        fontsize=27,
        y=0.985,
    )
    fig.subplots_adjust(left=0.075, right=0.885, bottom=0.105, top=0.84, wspace=0.25, hspace=0.48)
    colorbar_ax = fig.add_axes([0.915, 0.165, 0.014, 0.64])
    scalar = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    colorbar = fig.colorbar(scalar, cax=colorbar_ax)
    colorbar.set_label("Recommendation frequency (out of 20 seeds)", fontsize=18, labelpad=12)
    colorbar.set_ticks((1, 5, 10, 15, 20))
    colorbar.ax.tick_params(labelsize=14)

    method_output = output_dir / method
    method_output.mkdir(parents=True, exist_ok=True)
    stem = method_output / f"{method}_20seed_frequency_frontiers_{int(round(100 * target))}pct"
    png_path = stem.with_suffix(".png")
    pdf_path = stem.with_suffix(".pdf")
    fig.savefig(png_path, dpi=240, facecolor="white")
    fig.savefig(pdf_path, facecolor="white")
    plt.close(fig)
    return png_path, pdf_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    args = parser.parse_args()

    landscapes = {
        dataset: _load_landscape(args.results_root, dataset) for dataset in DATASETS
    }
    for method in args.methods:
        for target in (0.10, 0.30):
            png_path, pdf_path = make_figure(
                method=method,
                target=target,
                results_root=args.results_root,
                output_dir=args.output_dir,
                landscapes=landscapes,
            )
            print(f"wrote {png_path}")
            print(f"wrote {pdf_path}")


if __name__ == "__main__":
    main()
