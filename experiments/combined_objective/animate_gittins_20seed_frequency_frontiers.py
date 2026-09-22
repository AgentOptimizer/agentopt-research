#!/usr/bin/env python3
"""Animate 20-seed Gittins recommendation frequencies over search spend.

Each run is reconstructed from its event-driven recommendation snapshots.  At
budget ``b``, the displayed recommendation is the most recent snapshot whose
cost fraction is at most ``b``.  Before a run's warm start snapshot, that run
does not contribute a recommendation.  After a run terminates, its final
recommendation is held fixed.
"""

from __future__ import annotations

import argparse
import bisect
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import imageio_ffmpeg
import matplotlib as mpl
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, MaxNLocator, NullFormatter

from plot_gittins_20seed_frequency_frontiers import (
    DATASET_LABELS,
    DATASETS,
    DEFAULT_RESULTS,
    EXPECTED_SEEDS,
    _frequency_colormap,
    _load_landscape,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_ROOT = (
    ROOT
    / "experiments/combined_objective/results/gauss_radau_8bench_20seed_independent"
)
DEFAULT_OUTPUT = DEFAULT_RESULTS / "figures/gittins_frequency_frontiers"
PAIR_NAME = "gauss_radau_accuracy_endpoint"


@dataclass(frozen=True)
class Snapshot:
    cost_fraction: float
    selected_arm_indices: tuple[int, ...]
    pareto_recall: float
    exact_frontier: bool


@dataclass(frozen=True)
class RunTimeline:
    fractions: tuple[float, ...]
    snapshots: tuple[Snapshot, ...]

    def at(self, budget: float) -> Snapshot | None:
        position = bisect.bisect_right(self.fractions, budget + 1e-12) - 1
        return None if position < 0 else self.snapshots[position]


def _load_timeline(path: Path, model_count: int) -> RunTimeline:
    if not path.is_file():
        raise FileNotFoundError(f"missing Gittins result: {path}")
    run = json.loads(path.read_text(encoding="utf-8"))["run"]
    snapshots: list[Snapshot] = []
    for point in run["points"]:
        indices = tuple(int(index) for index in point["selected_arm_indices"])
        if any(index < 0 or index >= model_count for index in indices):
            raise ValueError(f"invalid selected arm index in {path}")
        snapshots.append(
            Snapshot(
                cost_fraction=float(point["cost_fraction"]),
                selected_arm_indices=indices,
                pareto_recall=float(point["pareto_recall"]),
                exact_frontier=bool(point["exact_frontier"]),
            )
        )
    snapshots.sort(key=lambda snapshot: snapshot.cost_fraction)
    if not snapshots:
        raise ValueError(f"no recommendation snapshots in {path}")
    fractions = tuple(snapshot.cost_fraction for snapshot in snapshots)
    if any(right < left for left, right in zip(fractions, fractions[1:])):
        raise ValueError(f"non-monotone snapshot costs in {path}")
    return RunTimeline(fractions=fractions, snapshots=tuple(snapshots))


def _load_all(
    results_root: Path, run_root: Path
) -> tuple[dict[str, dict[str, object]], dict[str, list[RunTimeline]], int]:
    landscapes: dict[str, dict[str, object]] = {}
    timelines: dict[str, list[RunTimeline]] = {}
    event_count = 0
    for dataset in DATASETS:
        landscape = _load_landscape(results_root, dataset)
        landscapes[dataset] = landscape
        model_count = len(np.asarray(landscape["truth"]))
        dataset_timelines = []
        for seed in sorted(EXPECTED_SEEDS):
            path = run_root / f"seed-{seed}" / PAIR_NAME / dataset / "result.json"
            timeline = _load_timeline(path, model_count)
            dataset_timelines.append(timeline)
            event_count += len(timeline.snapshots)
        timelines[dataset] = dataset_timelines
    return landscapes, timelines, event_count


def _summary(timelines: list[RunTimeline], budget: float) -> dict[str, object]:
    snapshots = [timeline.at(budget) for timeline in timelines]
    available = [snapshot for snapshot in snapshots if snapshot is not None]
    counts: Counter[int] = Counter()
    for snapshot in available:
        counts.update(snapshot.selected_arm_indices)
    return {
        "counts": counts,
        "available": len(available),
        "exact": sum(snapshot.exact_frontier for snapshot in available),
        "median_recall": (
            float(np.median([snapshot.pareto_recall for snapshot in available]))
            if available
            else None
        ),
    }


def _frame_budgets(frame_count: int, fps: int) -> np.ndarray:
    core = np.linspace(0.0, 1.0, frame_count)
    return np.concatenate(
        (
            np.repeat(core[0], max(1, fps // 2)),
            core,
            np.repeat(core[-1], max(1, fps)),
        )
    )


def make_animation(
    *,
    results_root: Path,
    run_root: Path,
    output_dir: Path,
    frame_count: int,
    fps: int,
    dpi: int,
    poster_budget: float,
) -> tuple[Path, Path, int]:
    landscapes, timelines, event_count = _load_all(results_root, run_root)
    cmap = _frequency_colormap()
    norm = mpl.colors.Normalize(vmin=1, vmax=len(EXPECTED_SEEDS))
    mpl.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()

    fig, axes = plt.subplots(2, 4, figsize=(16.0, 9.0))
    dynamic_scatters: dict[str, mpl.collections.PathCollection] = {}
    status_texts: dict[str, mpl.text.Text] = {}
    spend_texts: dict[str, mpl.text.Text] = {}

    for ax, dataset in zip(axes.flat, DATASETS):
        truth = np.asarray(landscapes[dataset]["truth"])
        frontier = np.asarray(landscapes[dataset]["frontier"], dtype=int)
        frontier = frontier[np.argsort(truth[frontier, 1])]
        ax.scatter(
            truth[:, 1],
            truth[:, 0],
            s=10,
            color="#cfd3d8",
            alpha=0.58,
            edgecolors="none",
            zorder=1,
        )
        ax.plot(
            truth[frontier, 1],
            truth[frontier, 0],
            color="#25282c",
            linewidth=1.35,
            zorder=2,
        )
        ax.scatter(
            truth[frontier, 1],
            truth[frontier, 0],
            s=30,
            facecolors="white",
            edgecolors="#25282c",
            linewidths=1.0,
            zorder=3,
        )
        dynamic_scatters[dataset] = ax.scatter(
            [],
            [],
            c=[],
            cmap=cmap,
            norm=norm,
            s=68,
            edgecolors="#6f4a22",
            linewidths=0.75,
            zorder=4,
        )
        ax.set_title(DATASET_LABELS[dataset], fontsize=15.5, pad=21, fontweight="semibold")
        spend_texts[dataset] = ax.text(
            0.5,
            1.012,
            "Search spend: 0% of brute force",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=10.5,
            color="#4e5359",
        )
        status_texts[dataset] = ax.text(
            0.025,
            0.025,
            "Exact frontier: 0/20   |   Median recall: n/a",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=9.4,
            color="#292d32",
            bbox={
                "boxstyle": "round,pad=0.22",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.82,
            },
            zorder=6,
        )
        if np.all(truth[:, 1] > 0.0):
            ax.set_xscale("log")
        ax.grid(color="#d9dde3", linewidth=0.5, alpha=0.55)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="both", labelsize=10.5)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        ax.xaxis.set_minor_formatter(NullFormatter())

    fig.supxlabel("Mean deployment cost (USD per query, log scale)", fontsize=16, y=0.035)
    fig.supylabel("Mean accuracy", fontsize=16, x=0.022)
    title = fig.suptitle(
        "Gauss–Radau Gittins recommendations across 20 seeds  |  Search spend: 0%",
        fontsize=20,
        y=0.983,
    )
    fig.subplots_adjust(
        left=0.075,
        right=0.885,
        bottom=0.105,
        top=0.87,
        wspace=0.25,
        hspace=0.38,
    )
    colorbar_ax = fig.add_axes([0.915, 0.165, 0.014, 0.64])
    scalar = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    colorbar = fig.colorbar(scalar, cax=colorbar_ax)
    colorbar.set_label("Recommendation frequency (out of 20 seeds)", fontsize=13, labelpad=10)
    colorbar.set_ticks((1, 5, 10, 15, 20))
    colorbar.ax.tick_params(labelsize=11)

    def update(budget: float) -> list[mpl.artist.Artist]:
        percent = 100.0 * float(budget)
        title.set_text(
            "Gauss–Radau Gittins recommendations across 20 seeds"
            f"  |  Search spend: {percent:.1f}%"
        )
        artists: list[mpl.artist.Artist] = [title]
        for dataset in DATASETS:
            truth = np.asarray(landscapes[dataset]["truth"])
            summary = _summary(timelines[dataset], float(budget))
            counts = summary["counts"]
            shown_indices = np.asarray(sorted(counts), dtype=int)
            shown_counts = np.asarray(
                [counts[int(index)] for index in shown_indices], dtype=float
            )
            scatter = dynamic_scatters[dataset]
            if len(shown_indices):
                scatter.set_offsets(truth[shown_indices][:, [1, 0]])
                scatter.set_array(shown_counts)
            else:
                scatter.set_offsets(np.empty((0, 2)))
                scatter.set_array(np.asarray([], dtype=float))
            recall = summary["median_recall"]
            recall_label = "n/a" if recall is None else f"{recall:.2f}"
            available = int(summary["available"])
            status_texts[dataset].set_text(
                f"Exact frontier: {summary['exact']}/20   |   Median recall: {recall_label}"
                + ("" if available == 20 else f"   |   Active seeds: {available}/20")
            )
            spend_texts[dataset].set_text(
                f"Search spend: {percent:.1f}% of brute force"
            )
            artists.extend(
                (scatter, status_texts[dataset], spend_texts[dataset])
            )
        return artists

    output_dir.mkdir(parents=True, exist_ok=True)
    poster_path = output_dir / "gittins_20seed_frequency_animation_poster.png"
    update(float(poster_budget))
    fig.savefig(poster_path, dpi=dpi, facecolor="white")

    budgets = _frame_budgets(frame_count, fps)
    movie = animation.FuncAnimation(
        fig,
        update,
        frames=budgets,
        interval=1000.0 / fps,
        blit=False,
    )
    mp4_path = output_dir / "gittins_20seed_frequency_frontiers.mp4"
    writer = animation.FFMpegWriter(
        fps=fps,
        codec="libx264",
        bitrate=5000,
        extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
        metadata={
            "title": "Gittins Pareto frontier recommendations over search spend",
            "artist": "AgentOpt",
        },
    )
    movie.save(mp4_path, writer=writer, dpi=dpi)
    plt.close(fig)
    return mp4_path, poster_path, event_count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--frames", type=int, default=151)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument("--poster-budget", type=float, default=0.30)
    args = parser.parse_args()
    if args.frames < 2:
        parser.error("--frames must be at least 2")
    if args.fps < 1:
        parser.error("--fps must be positive")
    if not 0.0 <= args.poster_budget <= 1.0:
        parser.error("--poster-budget must lie in [0, 1]")

    mp4_path, poster_path, event_count = make_animation(
        results_root=args.results_root,
        run_root=args.run_root,
        output_dir=args.output_dir,
        frame_count=args.frames,
        fps=args.fps,
        dpi=args.dpi,
        poster_budget=args.poster_budget,
    )
    print(f"loaded {event_count:,} event checkpoints across 160 runs")
    print(f"wrote {mp4_path}")
    print(f"wrote {poster_path}")


if __name__ == "__main__":
    main()
