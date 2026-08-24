#!/usr/bin/env python3
"""Plot cached Radial-Gittins eta sweep runs without rerunning the policy."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_COLORS = {
    0.01: "#6a3d9a",
    0.03: "#3f6fb6",
    0.1: "#2a9d8f",
    0.3: "#e9a23b",
    1.0: "#c45c26",
}


def _mean_ci(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return np.nan, np.nan
    mean = float(np.mean(finite))
    ci95 = (
        float(1.96 * np.std(finite, ddof=1) / np.sqrt(len(finite)))
        if len(finite) > 1
        else 0.0
    )
    return mean, ci95


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--indir", type=Path, default=Path("analysis/vs/gittins_eta_sweep"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    indir = args.indir if args.indir.is_absolute() else ROOT / args.indir
    files = sorted((indir / "raw").glob("*.npz"))
    if not files:
        raise SystemExit(f"no cached runs found under {indir / 'raw'}")

    runs: dict[tuple[str, float], list[dict]] = defaultdict(list)
    for path in files:
        with np.load(path) as payload:
            benchmark = str(payload["benchmark"])
            eta = float(payload["eta"])
            stop = float(payload["stop_cost_fraction"])
            if "stop_deployable_hv_regret" in payload:
                stop_regret = float(payload["stop_deployable_hv_regret"])
            else:
                positions = np.flatnonzero(payload["cost_fraction"] + 1e-12 >= stop)
                position = int(positions[0]) if len(positions) else -1
                stop_regret = float(payload["deployable_hv_regret"][position])
            runs[(benchmark, eta)].append(
                {
                    "seed": int(payload["seed"]),
                    "stop": stop,
                    "stop_regret": stop_regret,
                }
            )

    benchmarks = [name for name in ("hotpotqa", "mathqa") if any(k[0] == name for k in runs)]
    etas = sorted({key[1] for key in runs})
    figure, axes = plt.subplots(2, len(benchmarks), figsize=(6.1 * len(benchmarks), 7.2))
    if len(benchmarks) == 1:
        axes = np.asarray(axes).reshape(2, 1)
    summary_rows = []

    for column, benchmark in enumerate(benchmarks):
        stop_ax, regret_ax = axes[:, column]
        for eta in etas:
            eta_runs = runs.get((benchmark, eta), [])
            if not eta_runs:
                continue
            color = DEFAULT_COLORS.get(eta, plt.cm.viridis(np.log10(eta + 1.01) / 2.0))
            stops = np.asarray([run["stop"] for run in eta_runs], dtype=np.float64)
            regrets = np.asarray(
                [run["stop_regret"] for run in eta_runs], dtype=np.float64
            )
            stop_mean, stop_ci = _mean_ci(stops)
            regret_mean, regret_ci = _mean_ci(regrets)
            stop_ax.errorbar(
                eta,
                stop_mean,
                yerr=stop_ci,
                fmt="o",
                markersize=7,
                capsize=4,
                color=color,
            )
            regret_ax.errorbar(
                eta,
                regret_mean,
                yerr=regret_ci,
                fmt="o",
                markersize=7,
                capsize=4,
                color=color,
            )
            summary_rows.append(
                {
                    "benchmark": benchmark,
                    "eta": eta,
                    "n_runs": len(eta_runs),
                    "mean_stop_cost_fraction": stop_mean,
                    "ci95_stop_cost_fraction": stop_ci,
                    "min_stop_cost_fraction": float(np.nanmin(stops)),
                    "max_stop_cost_fraction": float(np.nanmax(stops)),
                    "mean_stop_deployable_hv_regret": regret_mean,
                    "ci95_stop_deployable_hv_regret": regret_ci,
                }
            )

        stop_ax.set_title(benchmark.replace("qa", "QA").replace("hotpot", "Hotpot"))
        stop_ax.set_xscale("log")
        stop_ax.set_xticks(etas, [f"{eta:g}" for eta in etas])
        stop_ax.set_xlabel("Trade-off parameter η (log scale)")
        stop_ax.set_ylabel("Stopping cumulative cost fraction")
        stop_ax.set_ylim(0.0, 1.0)
        stop_ax.grid(True, which="both", alpha=0.25)
        regret_ax.set_xscale("log")
        regret_ax.set_xticks(etas, [f"{eta:g}" for eta in etas])
        regret_ax.set_xlabel("Trade-off parameter η (log scale)")
        regret_ax.set_ylabel("Completed-only HV regret at stop")
        regret_ax.set_ylim(bottom=0.0)
        regret_ax.grid(True, which="both", alpha=0.25)

    figure.suptitle("Radial-Gittins sensitivity to search-cost trade-off η (mean ± 95% CI)")
    figure.tight_layout()
    output = args.output or indir / "radial_gittins_eta_comparison.png"
    output = output if output.is_absolute() else ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)

    summary_path = indir / "eta_sweep_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"wrote {output}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
