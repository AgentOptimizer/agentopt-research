"""Canonical one-factor-at-a-time ablations around the current Gittins method.

The method previously called ``g2_exact_axes`` is the paper method and is G0
in this protocol. G0 can be run directly; saved historical G2 results are
a fallback when no new G0 result exists. All other groups change one
conceptual factor relative to G0.
"""
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS = (
    "restaurant_test",
    "hotpotqa",
    "mathqa",
    "stackoverflow",
    "bird_dev",
    "restaurant_valid",
    "bing_querylogs",
    "bird_mini_dev",
)
SEEDS = tuple(range(42, 62))

G0_CONFIGURATION = "g0_current_gittins"
G0_SOURCE_ROOT = (
    ROOT
    / "analysis/usd_cost_checkpoints_latest_under_20seed/new_qa_runs/combined"
    / "gittins_ablation_8bench_20seed/g2_exact_axes"
)
DEFAULT_OUTPUT_ROOT = (
    ROOT
    / "experiments/combined_objective/results"
    / "gittins_ablation_v2_8bench_20seed"
)
DEFAULT_FIGURE_ROOT = (
    ROOT
    / "analysis/usd_cost_checkpoints_latest_under_20seed/new_qa_figures"
    / "gittins_g2_main_figures/ablation"
)

# Unique configurations after removing overlaps across the four requested
# ablation families.  In particular, Q-only real cost belongs to both the cost
# and direction panels, while G0 belongs to every family.
CONFIGURATIONS = {
    G0_CONFIGURATION: {
        "pair_name": "exact_axes",
        "acquisition_cost_mode": "real",
        "continuation_mode": "eta_decay",
        "eta_decay_schedule": "direction_stop",
        "direction_scheduler": "round_robin",
        "label": "G0 Current (axes, real cost)",
        "source_configuration": "g2_exact_axes",
    },
    "g1_q_only_real_cost": {
        "pair_name": "quality_only",
        "acquisition_cost_mode": "real",
        "continuation_mode": "eta_decay",
        "eta_decay_schedule": "direction_stop",
        "direction_scheduler": "round_robin",
        "label": "G1 Q-only, real cost",
    },
    "g2_q_only_unit_cost": {
        "pair_name": "quality_only",
        "acquisition_cost_mode": "unit",
        "continuation_mode": "eta_decay",
        "eta_decay_schedule": "direction_stop",
        "direction_scheduler": "round_robin",
        "label": "G2 Q-only, unit cost",
    },
    "g3_two_axis_unit_cost": {
        "pair_name": "exact_axes",
        "acquisition_cost_mode": "unit",
        "continuation_mode": "eta_decay",
        "eta_decay_schedule": "direction_stop",
        "direction_scheduler": "round_robin",
        "label": "G3 Axes, unit cost",
    },
    "g4_d_only_real_cost": {
        "pair_name": "deployment_only",
        "acquisition_cost_mode": "real",
        "continuation_mode": "eta_decay",
        "eta_decay_schedule": "direction_stop",
        "direction_scheduler": "round_robin",
        "label": "G4 D-only, real cost",
    },
    "g5_axes_midpoint_real_cost": {
        "pair_name": "axes_midpoint",
        "acquisition_cost_mode": "real",
        "continuation_mode": "eta_decay",
        "eta_decay_schedule": "direction_stop",
        "direction_scheduler": "round_robin",
        "label": "G5 Axes + midpoint",
    },
    "g6_five_directions_real_cost": {
        "pair_name": "five_directions",
        "acquisition_cost_mode": "real",
        "continuation_mode": "eta_decay",
        "eta_decay_schedule": "direction_stop",
        "direction_scheduler": "round_robin",
        "label": "G6 Five directions",
    },
    "g7_fixed_eta_no_stop": {
        "pair_name": "exact_axes",
        "acquisition_cost_mode": "real",
        "continuation_mode": "fixed_eta_no_stop",
        "eta_decay_schedule": "global_stop",
        "direction_scheduler": "round_robin",
        "label": "G7 Fixed eta (no stop)",
    },
    "g8_q_then_d": {
        "pair_name": "exact_axes",
        "acquisition_cost_mode": "real",
        "continuation_mode": "eta_decay",
        "eta_decay_schedule": "direction_stop",
        "direction_scheduler": "quality_then_deployment",
        "label": "G8 Q then D",
    },
    "g9_d_then_q": {
        "pair_name": "exact_axes",
        "acquisition_cost_mode": "real",
        "continuation_mode": "eta_decay",
        "eta_decay_schedule": "direction_stop",
        "direction_scheduler": "deployment_then_quality",
        "label": "G9 D then Q",
    },
}

RUN_CONFIGURATIONS = tuple(CONFIGURATIONS)

FAMILIES = {
    "cost_mechanism": (
        G0_CONFIGURATION,
        "g1_q_only_real_cost",
        "g2_q_only_unit_cost",
        "g3_two_axis_unit_cost",
    ),
    "directions": (
        "g1_q_only_real_cost",
        "g4_d_only_real_cost",
        G0_CONFIGURATION,
        "g5_axes_midpoint_real_cost",
        "g6_five_directions_real_cost",
    ),
    "continuation": (
        G0_CONFIGURATION,
        "g7_fixed_eta_no_stop",
    ),
    "scheduler": (
        G0_CONFIGURATION,
        "g8_q_then_d",
        "g9_d_then_q",
    ),
}


def result_path(
    configuration: str,
    benchmark: str,
    seed: int,
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> Path:
    settings = CONFIGURATIONS[configuration]
    pair_name = str(settings["pair_name"])
    local_path = (
        output_root
        / configuration
        / f"seed-{seed}"
        / pair_name
        / benchmark
        / "result.json"
    )
    if configuration == G0_CONFIGURATION and not local_path.is_file():
        return (
            G0_SOURCE_ROOT
            / f"seed-{seed}"
            / pair_name
            / benchmark
            / "result.json"
        )
    return local_path
