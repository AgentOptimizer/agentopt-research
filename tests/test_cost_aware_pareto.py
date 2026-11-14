import numpy as np

from experiments.combined_objective.run_cost_aware_pareto import (
    COHV_UCB,
    _copsi_classify,
    replay,
)
from experiments.single_objective.offline_selector_sim import SampleResult


def test_copsi_prefers_rejection_on_equal_classification_gaps():
    assert _copsi_classify(np.array([[0.8, 0.8], [0.4, 0.4]])) == (1, False)


def test_budget_crossing_batch_is_charged_and_excluded_from_scores():
    models = ["fast", "accurate", "dominated"]
    questions = list(range(8))
    table = {
        model: {
            question: SampleResult(
                score=score, cost=cost, latency_seconds=0, input_tokens={}, output_tokens={},
            )
            for question in questions
        }
        for model, score, cost in zip(models, (0.5, 1.0, 0.25), (0.01, 0.03, 0.02))
    }
    result = replay(models, questions, table, method=COHV_UCB, seed=42, budget_fraction=0.13,
                    batch_size=1)
    assert result["stop_reason"] == "budget_crossed"
    assert result["final"]["cost_usd"] <= result["budget_usd"] < result["charged_cost_usd"]
    assert result["final"]["evaluations"] + 1 == result["charged_evaluations"]
    assert result["final"]["evaluations"] == 3
