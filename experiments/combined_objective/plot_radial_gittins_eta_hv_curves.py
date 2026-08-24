#!/usr/bin/env python3
"""Plot provisional pre-stop and completed-only post-stop eta HV curves."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[2]
COLORS = {
    0.1: "#2a9d8f",
    0.2: "#3f6fb6",
    0.4: "#6a3d9a",
    0.7: "#9b4f96",
    1.0: "#c45c26",
}


def _align_step(xs: np.ndarray, ys: np.ndarray, grid: np.ndarray) -> np.ndarray:
    positions = np.searchsorted(xs, grid, side="right") - 1
    result = np.full_like(grid, np.nan)
    valid = positions >= 0
    result[valid] = ys[positions[valid]]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--indir", type=Path, default=Path("analysis/vs/gittins_eta_hv_curves")
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--points", type=int, default=201)
    args = parser.parse_args()

    indir = args.indir if args.indir.is_absolute() else ROOT / args.indir
    grouped: dict[tuple[str, float], list[dict]] = defaultdict(list)
    for path in sorted((indir / "raw").glob("*.npz")):
        with np.load(path) as payload:
            if bool(payload["halted_at_gittins_stop"]):
                continue
            grouped[(str(payload["benchmark"]), float(payload["eta"]))].append(
                {
                    "x": payload["cost_fraction"].copy(),
                    "deployable": payload["deployable_hv_regret"].copy(),
                    "provisional": payload["provisional_hv_regret"].copy(),
                    "stop": float(payload["stop_cost_fraction"]),
                    "stop_y": float(payload["stop_deployable_hv_regret"]),
                }
            )
    if not grouped:
        raise SystemExit(f"no full-trajectory runs found under {indir / 'raw'}")

    grid = np.linspace(0.0, 1.0, args.points)
    figure, axes = plt.subplots(1, 2, figsize=(12.2, 4.5), sharey=False)
    rows = []
    for ax, benchmark in zip(axes, ("hotpotqa", "mathqa")):
        for eta in sorted({key[1] for key in grouped if key[0] == benchmark}):
            runs = grouped[(benchmark, eta)]
            stops = np.asarray([run["stop"] for run in runs])
            stop_regrets = np.asarray([run["stop_y"] for run in runs])
            deployable = np.vstack(
                [
                    _align_step(run["x"], run["deployable"], grid)
                    for run in runs
                ]
            )
            provisional = np.vstack(
                [
                    _align_step(run["x"], run["provisional"], grid)
                    for run in runs
                ]
            )
            mean_stop = float(np.mean(stops))
            mean_stop_regret = float(np.mean(stop_regrets))
            provisional_mean = np.mean(provisional, axis=0)
            deployable_mean_full = np.mean(deployable, axis=0)
            deployable_ci95_full = (
                1.96
                * np.std(deployable, axis=0, ddof=1)
                / np.sqrt(len(runs))
                if len(runs) > 1
                else np.zeros_like(deployable_mean_full)
            )
            deployable_display = grid + 1e-12 >= mean_stop
            color = COLORS.get(eta, None)
            label = (
                f"η={eta:g} (n={len(runs)}, stop={mean_stop:.1%}, "
                f"regret@stop={mean_stop_regret:.4f})"
            )
            ax.plot(
                grid,
                provisional_mean,
                color=color,
                linewidth=1.2,
                linestyle="--",
                alpha=0.72,
            )
            ax.plot(
                grid[deployable_display],
                deployable_mean_full[deployable_display],
                color=color,
                linewidth=2.0,
                label=label,
            )
            ax.fill_between(
                grid[deployable_display],
                np.maximum(
                    0.0,
                    deployable_mean_full[deployable_display]
                    - deployable_ci95_full[deployable_display],
                ),
                deployable_mean_full[deployable_display]
                + deployable_ci95_full[deployable_display],
                color=color,
                alpha=0.10,
                linewidth=0,
            )
            ax.axvline(
                mean_stop,
                color=color,
                linestyle=":",
                linewidth=1.3,
                alpha=0.9,
            )
            ax.scatter(
                [mean_stop],
                [mean_stop_regret],
                color=color,
                marker="o",
                s=42,
                edgecolors="white",
                linewidths=0.7,
                zorder=5,
            )
            rows.append(
                {
                    "benchmark": benchmark,
                    "eta": eta,
                    "n_runs": len(runs),
                    "mean_stop_cost_fraction": float(np.mean(stops)),
                    "max_stop_cost_fraction": float(np.max(stops)),
                    "mean_stop_deployable_hv_regret": mean_stop_regret,
                }
            )
        ax.set_title("HotpotQA" if benchmark == "hotpotqa" else "MathQA")
        ax.set_xlabel("Cumulative search cost fraction")
        ax.set_ylabel("Normalized-desirability HV regret")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(bottom=0.0)
        ax.grid(True, alpha=0.25)
        eta_handles, eta_labels = ax.get_legend_handles_labels()
        meaning_handles = [
            Line2D(
                [0],
                [0],
                color="#667085",
                linestyle="--",
                linewidth=1.4,
                label="Provisional mean (full trajectory)",
            ),
            Line2D(
                [0],
                [0],
                color="#667085",
                linestyle="-",
                linewidth=2.0,
                label="Completed-only mean (from mean stop)",
            ),
            Patch(
                facecolor="#667085",
                alpha=0.14,
                label="Completed-only 95% CI (mean ± 1.96 SE)",
            ),
            Line2D(
                [0],
                [0],
                color="#667085",
                linestyle=":",
                marker="o",
                markeredgecolor="white",
                label="Mean stop and mean regret@stop",
            ),
        ]
        ax.legend(
            eta_handles + meaning_handles,
            eta_labels + [handle.get_label() for handle in meaning_handles],
            fontsize=6.8,
            loc="upper right",
        )
    figure.suptitle(
        "Radial-Gittins across η: provisional full trajectory (dashed), "
        "completed-only from mean stop (solid)\n"
        "Dotted vertical lines and circles mark mean stopping positions"
    )
    figure.tight_layout()
    output = args.output or indir / "radial_gittins_eta_hv_curves.png"
    output = output if output.is_absolute() else ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)

    summary = indir / "eta_hv_curve_summary.csv"
    with summary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {output}")
    print(f"wrote {summary}")


if __name__ == "__main__":
    main()
