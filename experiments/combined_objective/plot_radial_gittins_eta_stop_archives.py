#!/usr/bin/env python3
"""Plot seed-42 stopping recommendations against the full-data Pareto front."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[2]
ETAS = (0.1, 0.2, 0.4, 0.7, 1.0)
BENCHMARKS = (("hotpotqa", "HotpotQA"), ("mathqa", "MathQA"))


def slug(value: float) -> str:
    return f"{value:.8g}".replace(".", "p").replace("-", "m")


def legend_handles() -> list[Line2D]:
    return [
        Line2D([], [], linestyle="", marker="o", color="#CBD4DF",
               label="all configurations"),
        Line2D([], [], color="#65748A", marker="o", markersize=4,
               label="true full-data Pareto front"),
        Line2D([], [], linestyle="", marker="*", markersize=12,
               color="#279060", label="correctly recommended"),
        Line2D([], [], linestyle="", marker="D", color="#D76B27",
               label="missed Pareto arm"),
        Line2D([], [], linestyle="", marker="s", color="#22577A",
               label="false-positive recommendation"),
    ]


def draw_panel(
    ax: plt.Axes,
    path: Path,
    title: str,
    *,
    compact: bool = False,
    recommendation_key: str = "stop_recommended_arm_indices",
    cost_key: str = "stop_cost_fraction",
    checkpoint_label: str = "stop",
    correctly_recommended_scale: float = 1.0,
) -> None:
    if not path.exists():
        ax.text(0.5, 0.5, "run unavailable", ha="center", va="center")
        ax.set_title(title)
        return
    with np.load(path, allow_pickle=False) as data:
        points = np.asarray(data["raw_truth_vectors"], dtype=float)
        truth = set(map(int, data["true_pareto_arm_indices"]))
        if recommendation_key not in data:
            ax.text(0.5, 0.5, "checkpoint unavailable", ha="center", va="center")
            ax.set_title(title)
            return
        recommended = set(map(int, data[recommendation_key]))
        checkpoint_cost = float(data[cost_key])
    true_positive = truth & recommended
    false_positive = recommended - truth
    missed = truth - recommended
    recall = len(true_positive) / len(truth) if truth else 1.0
    precision = len(true_positive) / len(recommended) if recommended else 1.0

    scale = 0.72 if compact else 1.0
    ax.scatter(
        points[:, 1], points[:, 0], s=28 * scale, color="#CBD4DF",
        alpha=0.62, edgecolors="none", zorder=1,
    )
    ordered_front = sorted(truth, key=lambda index: points[index, 1])
    ax.plot(
        points[ordered_front, 1], points[ordered_front, 0], color="#65748A",
        marker="o", markersize=4 * scale, linewidth=1.8, zorder=2,
    )
    if missed:
        idx = sorted(missed)
        ax.scatter(
            points[idx, 1], points[idx, 0], marker="D", s=70 * scale,
            color="#D76B27", edgecolor="white", linewidth=0.6, zorder=4,
        )
    if false_positive:
        idx = sorted(false_positive)
        ax.scatter(
            points[idx, 1], points[idx, 0], marker="s", s=72 * scale,
            color="#22577A", edgecolor="white", linewidth=0.6, zorder=5,
        )
    if true_positive:
        idx = sorted(true_positive)
        ax.scatter(
            points[idx, 1], points[idx, 0], marker="*",
            s=150 * scale * correctly_recommended_scale, color="#279060",
            edgecolor="white", linewidth=0.6, zorder=6,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Mean deployment cost (USD, log scale)")
    ax.set_ylabel("Accuracy (full data)")
    ax.grid(alpha=0.22)
    ax.set_title(
        f"{title}: {checkpoint_label}, cost={checkpoint_cost:.1%}\n"
        f"Pareto={len(truth)}, recommended={len(recommended)}, "
        f"recall={recall:.2f}, precision={precision:.2f}"
    )


def write_presentation(indir: Path, output_dir: Path, seed: int) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(14.5, 6.8))
    for ax, (benchmark, title) in zip(axes, BENCHMARKS):
        path = indir / "raw" / f"{benchmark}_eta-1_seed-{seed}.npz"
        draw_panel(
            ax, path, title, checkpoint_label="stop",
            correctly_recommended_scale=1.65,
        )
    figure.legend(
        handles=legend_handles(), loc="lower center", ncol=5, frameon=False,
        bbox_to_anchor=(0.5, -0.015),
    )
    figure.suptitle("Radial-Gittins recommendations at stopping", fontsize=18)
    figure.tight_layout(rect=(0, 0.14, 1, 0.93))
    output = output_dir / "radial_gittins_stop_recommendations.png"
    figure.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(figure)
    print(output.resolve())
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--indir", type=Path,
        default=ROOT / "analysis/vs/gittins_eta_pareto_coverage_grid129",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--presentation-only", action="store_true")
    parser.add_argument(
        "--presentation-dir", type=Path, default=ROOT / "analysis/vs/give",
    )
    args = parser.parse_args()
    indir = args.indir if args.indir.is_absolute() else ROOT / args.indir
    presentation_dir = (
        args.presentation_dir
        if args.presentation_dir.is_absolute()
        else ROOT / args.presentation_dir
    )

    plt.rcParams.update({"font.size": 12, "axes.titlesize": 15, "axes.labelsize": 13})
    if not args.presentation_only:
        outdir = indir / "stop_archive_figures"
        outdir.mkdir(parents=True, exist_ok=True)
        for eta in ETAS:
            fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))
            for ax, (benchmark, title) in zip(axes, BENCHMARKS):
                path = indir / "raw" / f"{benchmark}_eta-{slug(eta)}_seed-{args.seed}.npz"
                draw_panel(ax, path, title)
            fig.legend(
                handles=legend_handles(), loc="lower center", ncol=3,
                frameon=False, bbox_to_anchor=(0.5, -0.02),
            )
            fig.suptitle(
                f"Radial-Gittins recommendations at stopping — η={eta:g}, "
                f"seed={args.seed}, grid=129", fontsize=17,
            )
            fig.tight_layout(rect=(0, 0.12, 1, 0.93))
            output = outdir / f"eta-{slug(eta)}_seed-{args.seed}_stop_archive.png"
            fig.savefig(output, dpi=220, bbox_inches="tight")
            plt.close(fig)
            print(output.resolve())

        stages = (
            ("stop", "stop_recommended_arm_indices", "stop_cost_fraction", "at stopping"),
            ("50pct", "halfway_recommended_arm_indices", "halfway_cost_fraction",
             "at 50% observed cells"),
            ("100pct", "end_recommended_arm_indices", "end_cost_fraction",
             "at 100% observed cells"),
        )
        for stage_slug, recommendation_key, cost_key, stage_title in stages:
            combined, axes = plt.subplots(len(ETAS), 2, figsize=(14.5, 23.5))
            for row, eta in enumerate(ETAS):
                for ax, (benchmark, title) in zip(axes[row], BENCHMARKS):
                    path = indir / "raw" / f"{benchmark}_eta-{slug(eta)}_seed-{args.seed}.npz"
                    draw_panel(
                        ax, path, f"η={eta:g} — {title}", compact=True,
                        recommendation_key=recommendation_key, cost_key=cost_key,
                        checkpoint_label=stage_title,
                    )
            combined.legend(
                handles=legend_handles(), loc="lower center", ncol=5,
                frameon=False, bbox_to_anchor=(0.5, 0.012),
            )
            combined.suptitle(
                f"Radial-Gittins recommendations {stage_title} across η — "
                f"seed={args.seed}, grid=129", fontsize=19, y=0.995,
            )
            combined.tight_layout(rect=(0, 0.04, 1, 0.985), h_pad=2.0)
            output = outdir / f"all_etas_seed-{args.seed}_{stage_slug}_archives.png"
            combined.savefig(output, dpi=220, bbox_inches="tight")
            plt.close(combined)
            print(output.resolve())

    write_presentation(indir, presentation_dir, args.seed)


if __name__ == "__main__":
    main()
