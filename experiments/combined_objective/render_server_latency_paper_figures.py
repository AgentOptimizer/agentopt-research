"""Render six manuscript figure previews without changing manuscript files."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def call(name: str, *arguments: object) -> None:
    command = [sys.executable, str(HERE / name), *(str(value) for value in arguments)]
    print("Running", name, flush=True)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(filter(None, (
        str(ROOT), str(HERE), environment.get("PYTHONPATH", ""),
    )))
    subprocess.run(command, cwd=ROOT, env=environment, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path,
                        default=ROOT / "tmp/server_latency_runs")
    parser.add_argument("--output-root", type=Path,
                        default=ROOT / "output/figures/server_latency_paper_preview_20261028")
    parser.add_argument("--main-quality-prior-mode", choices=("fixed", "warm"),
                        default="warm")
    args = parser.parse_args()
    result_root = args.result_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    ablation = result_root / "three_ablation_warm"
    call("plot_three_objective_ablation_families.py", "--source", ablation,
         "--output-root", output_root)
    for family in ("cost_mechanism", "continuation"):
        call("plot_three_objective_ablations.py", "--root", output_root,
             "--family", family)
    call("plot_three_objective_direction_ablation.py", "--source", ablation,
         "--output-dir", output_root / "direction_ablation")
    call("plot_server_latency_prior_factorial.py", "--ablation-root", ablation,
         "--factorial-root", result_root / "three_prior_factorial",
         "--output-dir", output_root / "prior_variance",
         "--main-quality-prior-mode", args.main_quality_prior_mode)
    call("plot_server_latency_main_comparison.py", "--result-root", result_root,
         "--output-dir", output_root / "method_comparison")
    call("plot_three_objective_direction_runtime.py", "--source",
         result_root / "three_direction_runtime", "--output-dir",
         output_root / "direction_runtime")
    mapping = {
        "three_objective_four_hv_igd":
            output_root / "method_comparison/three_objective_four_hv_igd",
        "three_objective_cost_mechanism":
            output_root / "cost_mechanism_ablation/three_objective_cost_mechanism_ablation_hv_gd_igd",
        "three_objective_directions":
            output_root / "direction_ablation/three_objective_direction_ablation_hv_gd_igd",
        "three_objective_direction_runtime_seven_settings":
            output_root / "direction_runtime/three_objective_direction_policy_runtime_fixed_30pct_entries_20seed",
        "three_objective_continuation":
            output_root / "continuation_ablation/three_objective_continuation_ablation_hv_gd_igd",
        "three_objective_prior_variance":
            output_root / "prior_variance/three_objective_prior_variance",
    }
    paper_figures = output_root / "paper_figures"
    paper_figures.mkdir(exist_ok=True)
    hashes = {}
    for name, source in mapping.items():
        for extension in ("png", "pdf", "svg"):
            candidate = source.with_suffix("." + extension)
            if not candidate.is_file():
                raise FileNotFoundError(candidate)
            target = paper_figures / f"{name}.{extension}"
            shutil.copy2(candidate, target)
            hashes[target.name] = hashlib.sha256(target.read_bytes()).hexdigest()
    provenance = {
        "objective": "Q maximize, cumulative successful-call server latency minimize, USD deployment cost minimize",
        "datasets": ["hotpotqa", "mathqa"], "seeds": list(range(42, 62)),
        "main_prior_variance_modes": [args.main_quality_prior_mode, "warm", "warm"],
        "source_matrices": {dataset: json.loads((ROOT / "data" / dataset / "server_latency_manifest.json").read_text())
                            for dataset in ("hotpotqa", "mathqa")},
        "result_root": str(result_root), "paper_files_unchanged": True,
        "figure_sha256": hashes,
    }
    (output_root / "preview_manifest.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(f"Six figure previews: {paper_figures}")


if __name__ == "__main__":
    main()
