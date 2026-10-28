"""Render the paper's three-objective method comparison from server-latency runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.combined_objective import plot_three_objective_corrected_gd_igd as paper


def export_random(baseline_root: Path, result_root: Path) -> None:
    for dataset in paper.DATASETS:
        payload = json.loads((baseline_root / dataset / "baselines.json").read_text(encoding="utf-8"))
        for method in ("random_questions", "random_configurations"):
            for run in payload["groups"][method]:
                seed = int(run["seed"])
                path = result_root / "three_objective_random_20seed" / f"seed_{seed}" / dataset / method / "result.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({"config": {"seed": seed}, "run": run},
                                           separators=(",", ":")) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    export_random(args.result_root / "random_baselines", args.result_root)
    original = paper.run_path

    def run_path(source: Path, dataset: str, method: str, seed: int) -> Path:
        if method == "cc_gittins":
            return source / "three_ablation_warm" / "four_direction_real_cost" / f"seed-{seed}" / dataset / "result.json"
        return original(source, dataset, method, seed)

    paper.run_path = run_path
    paper.configure_matplotlib()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = args.output_dir / "three_objective_method_curve_summary.csv"
    paper.build_summary(args.result_root, summary)
    output = args.output_dir / "three_objective_four_hv_igd"
    paper.plot(summary, output)
    print(output.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
