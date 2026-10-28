"""Canonical three-objective CC-Gittins ablation configurations."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any


BENCHMARKS = ("mathqa", "hotpotqa")
SEEDS = tuple(range(42, 62))

DIRECTION_SETS = {
    "q_only": ((1.0, 0.0, 0.0),),
    "l_only": ((0.0, 1.0, 0.0),),
    "d_only": ((0.0, 0.0, 1.0),),
    "pure_axes": ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    "near_axis_three": ((0.9, 0.1, 0.0), (0.1, 0.9, 0.0), (0.0, 0.0, 1.0)),
    "four_direction_main": (
        (1.0, 0.0, 0.0),
        (0.5, 0.5, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    ),
    "six_ql_dense": (
        (1.0, 0.0, 0.0),
        (0.75, 0.25, 0.0),
        (0.5, 0.5, 0.0),
        (0.25, 0.75, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    ),
    "six_three_edges": (
        (1.0, 0.0, 0.0),
        (0.5, 0.5, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.5, 0.5),
        (0.0, 0.0, 1.0),
        (0.5, 0.0, 0.5),
    ),
}


def _configuration(
    direction_set: str,
    *,
    acquisition_cost_mode: str = "real",
    continuation_mode: str = "eta_decay",
    label: str,
) -> dict[str, Any]:
    return {
        "direction_set": direction_set,
        "directions": DIRECTION_SETS[direction_set],
        "acquisition_cost_mode": acquisition_cost_mode,
        "continuation_mode": continuation_mode,
        "direction_scheduler": "round_robin",
        "label": label,
    }


CONFIGURATIONS = {
    "q_only_real_cost": _configuration("q_only", label="Q-only, real cost"),
    "q_only_unit_cost": _configuration(
        "q_only", acquisition_cost_mode="unit", label="Q-only, unit cost"
    ),
    "pure_axes_real_cost": _configuration("pure_axes", label="Pure axes, real cost"),
    "pure_axes_unit_cost": _configuration(
        "pure_axes", acquisition_cost_mode="unit", label="Pure axes, unit cost"
    ),
    "four_direction_real_cost": _configuration(
        "four_direction_main", label="Four-direction main, real cost"
    ),
    "four_direction_unit_cost": _configuration(
        "four_direction_main",
        acquisition_cost_mode="unit",
        label="Four-direction main, unit cost",
    ),
    "l_only_real_cost": _configuration("l_only", label="L-only, real cost"),
    "d_only_real_cost": _configuration("d_only", label="D-only, real cost"),
    "near_axis_three_real_cost": _configuration(
        "near_axis_three", label="Near-axis three, real cost"
    ),
    "six_ql_dense_real_cost": _configuration(
        "six_ql_dense", label="Six directions, Q-L dense, real cost"
    ),
    "six_three_edges_real_cost": _configuration(
        "six_three_edges", label="Six directions, three edges, real cost"
    ),
    "four_direction_fixed_eta": _configuration(
        "four_direction_main",
        continuation_mode="fixed_eta_no_stop",
        label="Four-direction fixed eta, forced continuation",
    ),
}

RUN_CONFIGURATIONS = tuple(CONFIGURATIONS)
FAMILIES = {
    "cost_mechanism": (
        "q_only_real_cost",
        "q_only_unit_cost",
        "pure_axes_real_cost",
        "pure_axes_unit_cost",
        "four_direction_real_cost",
        "four_direction_unit_cost",
    ),
    "directions": (
        "q_only_real_cost",
        "l_only_real_cost",
        "d_only_real_cost",
        "pure_axes_real_cost",
        "near_axis_three_real_cost",
        "four_direction_real_cost",
        "six_ql_dense_real_cost",
        "six_three_edges_real_cost",
    ),
    "continuation": (
        "four_direction_real_cost",
        "four_direction_fixed_eta",
    ),
}

DEFAULT_OUTPUT_ROOT = Path(
    os.environ.get(
        "AGENTOPT_THREE_OBJECTIVE_ABLATION_ROOT",
        "/scratch/yh6415/project/agentopt-research/analysis/"
        "usd_cost_checkpoints_latest_under_20seed/new_qa_figures/"
        "gittins_g2_main_figures/source_data/ablation/three_objective_20seed",
    )
)


def task_count() -> int:
    return len(RUN_CONFIGURATIONS) * len(BENCHMARKS) * len(SEEDS)


def task_mapping(task_id: int) -> tuple[str, str, int]:
    if isinstance(task_id, bool) or not isinstance(task_id, int) or not 0 <= task_id < task_count():
        raise ValueError(f"task_id must be in [0, {task_count()})")
    runs_per_configuration = len(BENCHMARKS) * len(SEEDS)
    configuration = RUN_CONFIGURATIONS[task_id // runs_per_configuration]
    within = task_id % runs_per_configuration
    benchmark = BENCHMARKS[within // len(SEEDS)]
    seed = SEEDS[within % len(SEEDS)]
    return configuration, benchmark, seed


def result_path(configuration: str, benchmark: str, seed: int, output_root: Path) -> Path:
    return output_root / configuration / f"seed-{seed}" / benchmark / "result.json"
