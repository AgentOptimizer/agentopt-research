#!/usr/bin/env python3
"""Draw the paper-style eight-dataset HV/GD/IGD comparison figures.

The two random baselines have ten saved nominal cell-budget checkpoints per
seed.  At each checkpoint ordinal (10%, 20%, ..., 100% nominal cell budget),
the displayed x coordinate is the mean realized-USD cost fraction over all 20
seeds, while y and its uncertainty band are computed from the corresponding 20
metric values.  Thus the horizontal axis remains a genuine cost axis and every
displayed random point has identical 20-seed support.  Random curves are drawn
as ten-point step trajectories to match the original paper figure.

HotpotQA/MathQA random results are stored in legacy aggregate CSVs, while the
six SCOPE datasets use per-seed trajectory directories.  Both inputs are
converted to the same per-seed representation and all three metrics are
recomputed from the saved recommendation membership in the same reciprocal-
cost desirability space before aggregation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MaxNLocator
from matplotlib.transforms import ScaledTranslation


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from experiments.combined_objective.offline_multiobjective_random_search import (  # noqa: E402
    common_question_ids,
    mean_raw_vectors,
    normalized_truth_vectors,
    pareto_min_cost_indices,
)
from experiments.combined_objective.offline_radial_gittins import (  # noqa: E402
    front_quality_metrics,
    hypervolume_2d,
    nondominated_indices,
)
from experiments.single_objective.offline_selector_sim import load_pickle  # noqa: E402


RESULTS_ROOT = ROOT / "analysis/20seed_results"
PREVIOUS_FIGURE_ROOT = (
    RESULTS_ROOT / "figures/eight_dataset_curves_unified_saved_trajectory"
)
DEFAULT_OUTPUT_ROOT = (
    RESULTS_ROOT / "figures/eight_dataset_curves_paper_mean_cost_checkpoints"
)

DATASETS = (
    "hotpotqa",
    "mathqa",
    "restaurant_test",
    "stackoverflow",
    "bird_dev",
    "restaurant_valid",
    "bing_querylogs",
    "bird_mini_dev",
)
QA_DATASETS = frozenset(("hotpotqa", "mathqa"))
DATASET_LABELS = {
    "hotpotqa": "HotpotQA",
    "mathqa": "MathQA",
    "restaurant_test": "Restaurant Test",
    "stackoverflow": "Stack Overflow",
    "bird_dev": "BIRD Dev",
    "restaurant_valid": "Restaurant Valid",
    "bing_querylogs": "Bing Query Logs",
    "bird_mini_dev": "BIRD Mini Dev",
}
SEEDS = tuple(range(42, 62))
RANDOM_METHODS = ("random_configurations", "random_questions")
METHODS = (
    "radial_gittins",
    "ege_sh",
    "ape_k",
    "qnehvi",
    *RANDOM_METHODS,
)
METRICS = (
    ("hv_regret", "Hypervolume Regret", "eight_datasets_hv_regret"),
    (
        "generational_distance",
        "Generational Distance",
        "eight_datasets_generational_distance",
    ),
    (
        "inverted_generational_distance",
        "Inverted Generational Distance",
        "eight_datasets_inverted_generational_distance",
    ),
)
METHOD_STYLES = {
    "radial_gittins": ("tab:orange", "Radial Gittins", 3.0),
    "ege_sh": ("tab:green", "EGE-SH", 2.4),
    "ape_k": ("tab:purple", "APE-k", 2.4),
    "qnehvi": ("tab:pink", "qNEHVI", 2.4),
    "random_configurations": ("tab:brown", "Random configurations", 2.25),
    "random_questions": ("tab:cyan", "Random questions", 2.25),
}
GRID_STEP = 0.005


mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.9,
    }
)


@dataclass(frozen=True)
class RunTrajectory:
    seed: int
    x: np.ndarray
    values: dict[str, np.ndarray]


@dataclass(frozen=True)
class MetricSpace:
    truth: np.ndarray
    truth_front: np.ndarray
    ground_truth_hv: float


def _metric_space(truth: np.ndarray) -> MetricSpace:
    truth = np.asarray(truth, dtype=np.float64)
    front = truth[np.asarray(nondominated_indices(truth), dtype=int)]
    return MetricSpace(truth, front, float(hypervolume_2d(front)))


def _quality_values(space: MetricSpace, selected: Sequence[int]) -> dict[str, float]:
    obtained = (
        space.truth[list(selected)]
        if selected
        else np.empty((0, 2), dtype=np.float64)
    )
    quality = front_quality_metrics(
        obtained,
        space.truth_front,
        (0.0, 0.0),
        space.ground_truth_hv,
    )
    return {
        "hv_regret": float(quality.hypervolume_regret),
        "generational_distance": float(quality.generational_distance),
        "inverted_generational_distance": float(
            quality.inverted_generational_distance
        ),
    }


def _qa_truth(dataset: str) -> tuple[MetricSpace, dict[str, int]]:
    models, datapoints, table = load_pickle(
        str(ROOT / f"experiments/data/lookup/{dataset}_lookup.pkl")
    )
    questions = common_question_ids(models, datapoints, table)
    raw = mean_raw_vectors(models, questions, table)
    positive_costs = raw[:, 1][raw[:, 1] > 0.0]
    reference = float(np.median(positive_costs))
    truth = normalized_truth_vectors(raw, reference)
    # Check that raw-space and transformed-space Pareto memberships agree.  It
    # catches accidental changes to the cost transform before metrics are drawn.
    expected = set(pareto_min_cost_indices(raw))
    actual = set(nondominated_indices(truth))
    if actual != expected:
        raise ValueError(f"Pareto membership changed under normalization: {dataset}")
    return _metric_space(truth), {model: index for index, model in enumerate(models)}


def _load_qa_random(dataset: str, method: str) -> list[RunTrajectory]:
    path = RESULTS_ROOT / method / dataset / "multi_seed_results.csv"
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    rows = [row for row in rows if row["version"] == method]
    space, model_index = _qa_truth(dataset)
    trajectories = []
    for seed in SEEDS:
        seed_rows = sorted(
            (row for row in rows if int(row["seed"]) == seed),
            key=lambda row: float(row["budget_fraction"]),
        )
        if len(seed_rows) != 10:
            raise ValueError(f"expected 10 QA random checkpoints: {path}, seed={seed}")
        full_cost = float(seed_rows[-1]["total_search_cost_usd"])
        values = {metric: [] for metric, _, _ in METRICS}
        xs = []
        for row in seed_rows:
            selected_models = json.loads(row["selected_models"])
            selected = tuple(model_index[name] for name in selected_models)
            quality = _quality_values(space, selected)
            xs.append(float(row["total_search_cost_usd"]) / full_cost)
            for metric in values:
                values[metric].append(quality[metric])
            # Preserve the legacy HV as an audit check while recomputing every
            # displayed metric through the unified membership-based path.
            if not np.isclose(
                quality["hv_regret"],
                float(row["hypervolume_regret"]),
                atol=1e-12,
                rtol=1e-10,
            ):
                raise ValueError(f"QA HV mismatch: {path}, seed={seed}")
        trajectories.append(
            RunTrajectory(
                seed,
                np.asarray(xs, dtype=np.float64),
                {
                    metric: np.asarray(metric_values, dtype=np.float64)
                    for metric, metric_values in values.items()
                },
            )
        )
    return trajectories


def _indices(text: str) -> tuple[int, ...]:
    return tuple(int(value) for value in text.split(";") if value.strip())


def _load_scope_random(dataset: str, method: str) -> list[RunTrajectory]:
    trajectories = []
    reference_truth: np.ndarray | None = None
    for seed in SEEDS:
        directory = RESULTS_ROOT / method / dataset / f"seed-{seed}"
        with (directory / "cost_trajectory.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            rows = sorted(
                csv.DictReader(handle), key=lambda row: float(row["budget_fraction"])
            )
        if len(rows) != 10:
            raise ValueError(
                f"expected 10 SCOPE random checkpoints: {directory}, seed={seed}"
            )
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        full_cost = float(summary["total_search_cost_usd"])
        with np.load(directory / "vectors.npz") as vectors:
            truth = np.asarray(vectors["truth_vectors"], dtype=np.float64)
        if reference_truth is None:
            reference_truth = truth
        elif not np.allclose(reference_truth, truth, atol=0.0, rtol=0.0):
            raise ValueError(f"truth vectors differ across seeds: {dataset}/{method}")
        space = _metric_space(truth)
        values = {metric: [] for metric, _, _ in METRICS}
        xs = []
        for row in rows:
            quality = _quality_values(space, _indices(row["selected_arm_indices"]))
            xs.append(float(row["cumulative_search_cost_usd"]) / full_cost)
            for metric in values:
                values[metric].append(quality[metric])
            stored = {
                "hv_regret": float(row["hv_regret"]),
                "generational_distance": float(row["generational_distance"]),
                "inverted_generational_distance": float(
                    row["inverted_generational_distance"]
                ),
            }
            if any(
                not np.isclose(quality[name], stored[name], atol=1e-12, rtol=1e-10)
                for name in stored
            ):
                raise ValueError(
                    f"stored metric mismatch: {directory}, seed={seed}, "
                    f"budget={row['budget_fraction']}"
                )
        trajectories.append(
            RunTrajectory(
                seed,
                np.asarray(xs, dtype=np.float64),
                {
                    metric: np.asarray(metric_values, dtype=np.float64)
                    for metric, metric_values in values.items()
                },
            )
        )
    return trajectories


def load_random_trajectories(dataset: str, method: str) -> list[RunTrajectory]:
    trajectories = (
        _load_qa_random(dataset, method)
        if dataset in QA_DATASETS
        else _load_scope_random(dataset, method)
    )
    if [run.seed for run in trajectories] != list(SEEDS):
        raise ValueError(f"random seed mismatch: {dataset}/{method}")
    return trajectories


def aggregate_random_trajectories(
    trajectories: Sequence[RunTrajectory], metric: str, *, grid_step: float = GRID_STEP,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Align random runs with 20-seed support from their mean starting x.

    Before a run's first saved checkpoint, its first y is extended left.  Once
    the run has started, its latest saved y at or before each grid location is
    used.  No intermediate recommendation membership is invented.
    """
    if not trajectories:
        raise ValueError("at least one trajectory is required")
    if not math.isfinite(grid_step) or grid_step <= 0.0:
        raise ValueError("grid_step must be finite and positive")
    for run in trajectories:
        if len(run.x) == 0 or len(run.x) != len(run.values[metric]):
            raise ValueError("invalid random trajectory")
        if np.any(np.diff(run.x) < -1e-12):
            raise ValueError("random x positions must be nondecreasing")

    start_x = float(np.mean([run.x[0] for run in trajectories]))
    first_regular = math.ceil((start_x - 1e-12) / grid_step) * grid_step
    regular = np.arange(first_regular, 1.0 + grid_step / 2.0, grid_step)
    regular = regular[regular > start_x + 1e-12]
    xs = np.concatenate(([start_x], regular))
    if xs[-1] < 1.0 - 1e-12:
        xs = np.append(xs, 1.0)
    else:
        xs[-1] = 1.0

    aligned = np.empty((len(trajectories), len(xs)), dtype=np.float64)
    for row, run in enumerate(trajectories):
        positions = np.searchsorted(run.x, xs, side="right") - 1
        # The requested left extension makes every run available at start_x.
        positions = np.maximum(positions, 0)
        aligned[row] = run.values[metric][positions]

    counts = np.sum(np.isfinite(aligned), axis=0)
    if not np.all(counts == len(trajectories)):
        raise ValueError("every corrected random point must contain every seed")
    means = np.mean(aligned, axis=0)
    two_se = (
        2.0 * np.std(aligned, axis=0, ddof=1) / np.sqrt(len(trajectories))
        if len(trajectories) > 1
        else np.zeros(len(xs), dtype=np.float64)
    )
    return xs, means, two_se, counts, start_x


def aggregate_random_checkpoints(
    trajectories: Sequence[RunTrajectory], metric: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate like-for-like nominal checkpoints on the realized-cost axis.

    Column ``j`` contains the ``j``-th saved checkpoint from every seed.  Its x
    coordinate is the mean realized cost fraction, and its y coordinate and
    band are the mean and two standard errors over those same runs.
    """
    if not trajectories:
        raise ValueError("at least one trajectory is required")
    checkpoint_count = len(trajectories[0].x)
    if checkpoint_count == 0:
        raise ValueError("random trajectories must not be empty")
    for run in trajectories:
        if len(run.x) != checkpoint_count or len(run.values[metric]) != checkpoint_count:
            raise ValueError("random trajectories must share checkpoint ordinals")
        if np.any(np.diff(run.x) < -1e-12):
            raise ValueError("random x positions must be nondecreasing")

    x_matrix = np.vstack([run.x for run in trajectories])
    y_matrix = np.vstack([run.values[metric] for run in trajectories])
    xs = np.mean(x_matrix, axis=0)
    means = np.mean(y_matrix, axis=0)
    two_se = (
        2.0 * np.std(y_matrix, axis=0, ddof=1) / np.sqrt(len(trajectories))
        if len(trajectories) > 1
        else np.zeros(checkpoint_count, dtype=np.float64)
    )
    counts = np.sum(np.isfinite(y_matrix), axis=0)
    if not np.all(counts == len(trajectories)):
        raise ValueError("every random checkpoint must contain every seed")
    return xs, means, two_se, counts


def _load_previous_nonrandom_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    retained = [row for row in rows if row["method"] not in RANDOM_METHODS]
    required = {
        (dataset, method, metric)
        for dataset in DATASETS
        for method in METHODS
        if method not in RANDOM_METHODS
        for metric, _, _ in METRICS
    }
    available = {(row["dataset"], row["method"], row["metric"]) for row in retained}
    missing = required - available
    if missing:
        raise ValueError(f"missing saved non-random curves: {sorted(missing)}")
    return retained


def build_curves(
    previous_summary: Path,
) -> tuple[list[dict[str, object]], dict[str, dict[str, float]]]:
    rows: list[dict[str, object]] = list(_load_previous_nonrandom_rows(previous_summary))
    starts: dict[str, dict[str, float]] = {}
    for dataset in DATASETS:
        starts[dataset] = {}
        for method in RANDOM_METHODS:
            trajectories = load_random_trajectories(dataset, method)
            for metric, _, _ in METRICS:
                xs, means, spreads, counts = aggregate_random_checkpoints(
                    trajectories, metric
                )
                starts[dataset][method] = float(xs[0])
                for x, mean, spread, count in zip(xs, means, spreads, counts):
                    rows.append(
                        {
                            "dataset": dataset,
                            "method": method,
                            "metric": metric,
                            "cost_fraction": float(x),
                            "mean": float(mean),
                            "two_se": float(spread),
                            "n_runs": int(count),
                        }
                    )
    dataset_order = {name: index for index, name in enumerate(DATASETS)}
    method_order = {name: index for index, name in enumerate(METHODS)}
    metric_order = {name: index for index, (name, _, _) in enumerate(METRICS)}
    rows.sort(
        key=lambda row: (
            dataset_order[str(row["dataset"])],
            method_order[str(row["method"])],
            metric_order[str(row["metric"])],
            float(row["cost_fraction"]),
        )
    )
    return rows, starts


def _write_csv(rows: Sequence[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "dataset",
                "method",
                "metric",
                "cost_fraction",
                "mean",
                "two_se",
                "n_runs",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)


def _plot_metric(
    rows: Sequence[dict[str, object]], metric: str, ylabel: str, stem: Path,
) -> None:
    figure, axes = plt.subplots(2, 4, figsize=(22.2, 10.8), sharex=True)
    for ax, dataset in zip(axes.flat, DATASETS):
        for method in METHODS:
            series = [
                row
                for row in rows
                if row["dataset"] == dataset
                and row["method"] == method
                and row["metric"] == metric
            ]
            series.sort(key=lambda row: float(row["cost_fraction"]))
            if not series:
                continue
            xs = np.asarray([float(row["cost_fraction"]) for row in series])
            ys = np.asarray([float(row["mean"]) for row in series])
            spread = np.asarray([float(row["two_se"]) for row in series])
            color, label, linewidth = METHOD_STYLES[method]
            is_random = method in RANDOM_METHODS
            ax.plot(
                xs,
                ys,
                color=color,
                linewidth=linewidth,
                label=label,
                drawstyle="steps-post" if is_random else "default",
            )
            fill_kwargs = {"step": "post"} if is_random else {}
            ax.fill_between(
                xs,
                np.maximum(0.0, ys - spread),
                ys + spread,
                color=color,
                alpha=0.13,
                linewidth=0,
                **fill_kwargs,
            )
        ax.set_title(DATASET_LABELS[dataset], fontsize=27, pad=8)
        ax.set_xlim(0.0, 1.02)
        ax.set_ylim(bottom=0.0)
        ax.grid(color="#d9dde3", linewidth=0.6, alpha=0.55)
        ax.set_axisbelow(True)
        # The paper layout uses a complete rectangular frame for every panel.
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.9)
        ax.set_xticks((0.0, 0.5, 1.0))
        ax.tick_params(
            axis="both",
            which="major",
            direction="out",
            bottom=True,
            left=True,
            top=False,
            right=False,
            labelbottom=True,
            length=6.5,
            width=1.15,
            labelsize=25,
            pad=5,
        )
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0%}"))
        # Move only the rendered "0%" text slightly right.  The zero tick,
        # axis range, and every trajectory x coordinate remain unchanged.
        zero_label = ax.get_xticklabels()[0]
        zero_label.set_transform(
            zero_label.get_transform()
            + ScaledTranslation(4.0 / 72.0, 0.0, figure.dpi_scale_trans)
        )
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))

    figure.supxlabel(
        "Percentage of Exhaustive Evaluation Cost", fontsize=35, x=0.51, y=0.145
    )
    figure.supylabel(ylabel, fontsize=35, x=0.020, y=0.57)
    handles = [
        Line2D(
            [], [], color=METHOD_STYLES[method][0],
            linewidth=METHOD_STYLES[method][2], label=METHOD_STYLES[method][1],
        )
        for method in METHODS
    ]
    handles.append(Patch(facecolor="#808080", alpha=0.16, label=r"$\pm 2$ SE"))
    figure.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.51, 0.010),
        ncol=4,
        frameon=False,
        fontsize=27,
        columnspacing=1.8,
        handlelength=3.0,
        handletextpad=0.7,
    )
    figure.subplots_adjust(
        left=0.090, right=0.985, bottom=0.265, top=0.95, wspace=0.34, hspace=0.27
    )
    for suffix, kwargs in (
        (".png", {"dpi": 250}),
        (".pdf", {}),
        (".svg", {}),
    ):
        figure.savefig(stem.with_suffix(suffix), facecolor="white", **kwargs)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument(
        "--previous-summary",
        type=Path,
        default=PREVIOUS_FIGURE_ROOT / "curve_summary.csv",
        help="Saved non-random curves from the current eight-dataset figure.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    if args.results_root.resolve() != RESULTS_ROOT.resolve():
        raise SystemExit(
            "--results-root override is not yet supported because indexed input "
            "paths are intentionally audited against analysis/20seed_results"
        )
    output = args.output_root
    output.mkdir(parents=True, exist_ok=True)

    rows, starts = build_curves(args.previous_summary)
    _write_csv(rows, output / "curve_summary.csv")
    for metric, ylabel, filename in METRICS:
        _plot_metric(rows, metric, ylabel, output / filename)
        print(f"wrote {output / filename}.[png|pdf|svg]", flush=True)

    metadata = {
        "results_root": str(RESULTS_ROOT),
        "previous_nonrandom_summary": str(args.previous_summary),
        "datasets": list(DATASETS),
        "methods": list(METHODS),
        "expected_seeds": list(SEEDS),
        "cost_axis": (
            "realized cumulative search cost divided by exhaustive search cost"
        ),
        "random_source": {
            "hotpotqa_mathqa": "legacy aggregate multi_seed_results.csv",
            "scope": "per-seed cost_trajectory.csv",
        },
        "random_metric_policy": (
            "recompute HV regret, GD, and IGD from saved recommendation membership "
            "in reciprocal-cost desirability space for every dataset"
        ),
        "random_alignment": (
            "checkpoint-ordinal aggregation: at each of the ten nominal cell-budget "
            "checkpoints, x is the mean realized USD fraction and y is the mean of "
            "the corresponding 20 metric values"
        ),
        "random_mean_start_cost_fraction": starts,
        "random_n_runs_at_every_displayed_point": 20,
        "random_saved_checkpoints_per_seed": 10,
        "random_rendering": "ten-point steps-post trajectory",
        "uncertainty": "mean +/- 2 standard errors",
    }
    (output / "plot_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    (output / "method_styles.json").write_text(
        json.dumps(
            {
                method: {
                    "color": color,
                    "label": label,
                    "linewidth": linewidth,
                    "linestyle": "-",
                }
                for method, (color, label, linewidth) in METHOD_STYLES.items()
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {output / 'curve_summary.csv'}", flush=True)
    print(f"wrote {output / 'plot_metadata.json'}", flush=True)


if __name__ == "__main__":
    main()
