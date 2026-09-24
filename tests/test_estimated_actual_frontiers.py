from __future__ import annotations

import unittest

from experiments.combined_objective.estimated_actual_frontiers import (
    build_frontier_artifact,
)


def _pull(
    arm: int,
    score: float,
    cost: float,
    evaluations: int,
    cumulative_cost: float,
    *,
    atomic_update_index: int | None = None,
) -> dict[str, object]:
    result: dict[str, object] = {
        "arm_index": arm,
        "question_ids": [evaluations - 1],
        "batch_score_mean": score,
        "actual_batch_search_cost_usd": cost,
        "cumulative_evaluations": evaluations,
        "cumulative_search_cost_usd": cumulative_cost,
    }
    if atomic_update_index is not None:
        result["atomic_update_index"] = atomic_update_index
    return result


class EstimatedActualFrontierTests(unittest.TestCase):
    def test_causal_costs_and_membership_frames_are_frozen(self):
        artifact = build_frontier_artifact(
            dataset="tiny",
            method="ape_k",
            seed=42,
            models=("a", "b"),
            truth_raw_vectors=((0.7, 0.4), (0.8, 0.8)),
            full_data_pareto_arm_indices=(0, 1),
            full_search_cost_usd=3.0,
            trace=(
                _pull(0, 0.5, 0.4, 1, 0.4),
                _pull(1, 0.6, 0.6, 2, 1.0),
                _pull(0, 0.9, 0.8, 3, 1.8),
                _pull(1, 0.8, 1.2, 4, 3.0),
            ),
            snapshots=(
                {"event": "recommendation_initial", "cumulative_evaluations": 1,
                 "selected_arm_indices": (0,)},
                {"event": "recommendation_changed", "cumulative_evaluations": 2,
                 "selected_arm_indices": (0, 1)},
                {"event": "terminal", "cumulative_evaluations": 4,
                 "selected_arm_indices": (1,)},
            ),
            cost_estimator="causal_observed_arm_mean",
            target_fractions=(0.10, 0.30),
        )
        self.assertEqual(len(artifact["frontier_change_frames"]), 3)
        ten = artifact["cost_checkpoints"]["10pct"]
        thirty = artifact["cost_checkpoints"]["30pct"]
        self.assertEqual(ten["cumulative_evaluations"], 1)
        self.assertEqual(ten["selected_arm_indices"], [0])
        self.assertEqual(thirty["cumulative_evaluations"], 2)
        self.assertEqual(thirty["selected_arm_indices"], [0, 1])
        # First pull initializes at actual cost. The unseen second arm uses the
        # pooled observed mean, so estimated cumulative spend is 0.8, not 1.0.
        self.assertAlmostEqual(thirty["estimated_search_cost_usd"], 0.8)
        self.assertEqual(
            thirty["actual_raw_archive_vectors"], [[0.7, 0.4], [0.8, 0.8]]
        )

    def test_crossing_mid_atomic_update_uses_the_completed_update(self):
        artifact = build_frontier_artifact(
            dataset="tiny",
            method="random_questions",
            seed=42,
            models=("a", "b"),
            truth_raw_vectors=((0.7, 0.4), (0.8, 0.8)),
            full_data_pareto_arm_indices=(0, 1),
            full_search_cost_usd=2.0,
            trace=(
                _pull(0, 0.5, 0.3, 1, 0.3, atomic_update_index=0),
                _pull(1, 0.6, 0.3, 2, 0.6, atomic_update_index=0),
                _pull(0, 0.7, 0.3, 3, 0.9, atomic_update_index=1),
                _pull(1, 0.8, 0.3, 4, 1.2, atomic_update_index=1),
            ),
            snapshots=(
                {"event": "recommendation_initial", "cumulative_evaluations": 2,
                 "selected_arm_indices": (0,)},
                {"event": "recommendation_changed", "cumulative_evaluations": 4,
                 "selected_arm_indices": (1,)},
            ),
            cost_estimator="causal_observed_arm_mean",
            target_fractions=(0.20, 0.50),
        )
        twenty = artifact["cost_checkpoints"]["20pct"]
        fifty = artifact["cost_checkpoints"]["50pct"]
        self.assertEqual(twenty["cumulative_evaluations"], 2)
        self.assertEqual(twenty["selected_arm_indices"], [0])
        self.assertEqual(fifty["cumulative_evaluations"], 4)
        self.assertEqual(fifty["selected_arm_indices"], [1])


if __name__ == "__main__":
    unittest.main()
