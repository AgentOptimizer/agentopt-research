"""Check one saved three-objective run against the current engine on its original input."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.combined_objective.offline_three_objective_gittins import (
    simulate_three_objective_gittins,
)
from experiments.combined_objective.three_objective_metrics import (
    load_three_objective_benchmark,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--saved-result", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    saved = json.loads(args.saved_result.read_text(encoding="utf-8"))
    config = saved["config"]
    old = saved["run"]
    models, questions, table, hashes = load_three_objective_benchmark(args.input_dir)
    by_name = lambda values: {Path(name).name: digest for name, digest in values.items()}
    if by_name(hashes) != by_name(config["input_sha256"]):
        raise ValueError("original input hashes do not match the saved run")
    replay = simulate_three_objective_gittins(
        models, questions, table,
        seed=int(config["seed"]),
        directions=config["directions"],
        batch_size=int(config["batch_size"]),
        warm_start_batch_size=int(config["warm_start_batch_size"]),
        grid_size=int(config["grid_size"]),
        question_order=config["question_order"],
        warm_start_question_order=config["warm_start_question_order"],
        lambda_initial=float(config["eta_initial"]),
        lambda_decay=float(config["eta_decay"]),
        acquisition_cost_mode=config["acquisition_cost_mode"],
        continuation_mode=config["continuation_mode"],
        recommendation_beta=float(config["recommendation_beta"]),
        quality_prior_variance_mode="fixed",
        latency_prior_variance_mode="warm",
        cost_prior_variance_mode="warm",
        observation_budget_fraction=float(config["observation_budget_fraction"]),
    )
    fields = ("event", "total_evaluations", "counts", "selected_arm_indices")
    first_difference = None
    for index, (earlier, current) in enumerate(zip(old["points"], replay["points"])):
        for field in fields:
            if earlier[field] != current[field]:
                first_difference = {"point": index, "field": field,
                                    "saved": earlier[field], "replay": current[field]}
                break
        if first_difference is not None:
            break
    report = {
        "input_hashes_matched": True,
        "saved_points": len(old["points"]),
        "replay_points": len(replay["points"]),
        "saved_stop_reason": old["stop_reason"],
        "replay_stop_reason": replay["stop_reason"],
        "saved_evaluations": old["total_evaluations"],
        "replay_evaluations": replay["total_evaluations"],
        "first_checkpoint_difference": first_difference,
    }
    report["matching_path"] = (
        first_difference is None
        and report["saved_points"] == report["replay_points"]
        and report["saved_stop_reason"] == report["replay_stop_reason"]
        and report["saved_evaluations"] == report["replay_evaluations"]
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
