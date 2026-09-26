"""Current and historical two-axis outputs must be comparable and auditable."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest

from experiments.combined_objective import aggregate_gittins_ablation_v2_20seed as collector
from experiments.combined_objective import gittins_ablation_v2 as protocol
from experiments.combined_objective.ablation_metrics import evaluation_space, score_selection
from experiments.combined_objective.run_two_direction_ablation import PAIRS


def _write_grid(monkeypatch, tmp_path: Path, *, legacy_g0: bool) -> Path:
    root = tmp_path / "results"
    monkeypatch.setattr(collector, "BENCHMARKS", ("toy",))
    monkeypatch.setattr(collector, "SEEDS", (42,))
    monkeypatch.setattr(protocol, "G0_SOURCE_ROOT", tmp_path / "historical_g2")
    for name, settings in protocol.CONFIGURATIONS.items():
        is_legacy = legacy_g0 and name == protocol.G0_CONFIGURATION
        if is_legacy:
            path = protocol.result_path(name, "toy", 42, output_root=root)
        else:
            path = root / name / "seed-42" / settings["pair_name"] / "toy" / "result.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        config = {
            **settings,
            "ablation_name": "g2_exact_axes" if is_legacy else name,
            "benchmark": "toy",
            "seed": 42,
            "directions": [list(direction) for direction in PAIRS[settings["pair_name"]]],
        }
        costs = [1.0, 1.0] if settings["acquisition_cost_mode"] == "unit" else [0.1, 0.2]
        payload = {
            "config": config,
            "parameters": {"expected_batch_costs_usd": costs},
            "run": {
                "raw_truth_vectors": [[0.5, 0.1], [0.8, 0.2]],
                "model_names": ["cheap", "accurate"],
                "bruteforce_search_cost_usd": 10.0,
            },
            "summary": {"final_recall": 1.0},
        }
        path.write_text(json.dumps(payload))
        if not is_legacy:
            path.with_name("slurm_task_metadata.json").write_text(json.dumps({"exit_code": 0}))
    monkeypatch.setattr(sys, "argv", ["collect", str(root)])
    return root


@pytest.mark.parametrize("legacy_g0", [False, True])
def test_collector_accepts_native_and_historical_two_axis_controls(monkeypatch, tmp_path, legacy_g0):
    root = _write_grid(monkeypatch, tmp_path, legacy_g0=legacy_g0)
    collector.main()
    summary = json.loads((root / "aggregate/collection_summary.json").read_text())
    assert summary["expected_run_count"] == 10
    assert summary["completed_valid_run_count"] == 10
    assert summary["missing_run_count"] == summary["invalid_run_count"] == 0
    assert summary["reused_g0_run_count"] == int(legacy_g0)


@pytest.mark.parametrize("mutation", ["unit_cost", "truth", "failed_task"])
def test_collector_rejects_incompatible_results(monkeypatch, tmp_path, mutation):
    root = _write_grid(monkeypatch, tmp_path, legacy_g0=False)
    name = "g2_q_only_unit_cost"
    path = protocol.result_path(name, "toy", 42, output_root=root)
    payload = json.loads(path.read_text())
    if mutation == "unit_cost":
        payload["parameters"]["expected_batch_costs_usd"] = [0.1, 0.2]
    elif mutation == "truth":
        payload["run"]["raw_truth_vectors"][0][0] = 0.7
    else:
        path.with_name("slurm_task_metadata.json").write_text(json.dumps({"exit_code": 2}))
    path.write_text(json.dumps(payload))
    with pytest.raises(SystemExit, match="1"):
        collector.main()
    summary = json.loads((root / "aggregate/collection_summary.json").read_text())
    assert summary["invalid_run_count"] == 1
    assert summary["invalid"][0]["configuration"] == name


def test_frontier_metrics_preserve_full_frontier_and_penalize_missing_points():
    raw = np.asarray([[0.3, 0.1], [0.7, 0.3], [0.9, 0.8], [0.2, 0.4]])
    truth, front, _, hv = evaluation_space(raw)
    complete = score_selection((0, 1, 2), truth, front, hv)
    assert complete == {"hv_regret": 0.0, "generational_distance": 0.0, "inverted_generational_distance": 0.0}
    partial = score_selection((1,), truth, front, hv)
    assert partial["hv_regret"] > 0
    assert partial["generational_distance"] == 0
    assert partial["inverted_generational_distance"] > 0
