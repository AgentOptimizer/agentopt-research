import json

import numpy as np

from experiments.combined_objective import run_heldout_pareto as heldout
from experiments.combined_objective.summarize_heldout_pareto import summarize
from experiments.single_objective.offline_selector_sim import SampleResult


def test_heldout_outcomes_change_scores_without_changing_acquisition(monkeypatch, tmp_path):
    models = ("cheap", "middle", "expensive")
    questions = list(range(30))
    shuffled = np.random.default_rng(heldout.SPLIT_SEED).permutation(questions)
    reserved = set(np.array_split(shuffled, len(heldout.FOLDS))[0])

    def benchmark(reverse_heldout_quality):
        table = {
            model: {
                q: SampleResult(
                    score=1 - quality if reverse_heldout_quality and q in reserved else quality,
                    cost=cost, latency_seconds=0, input_tokens={}, output_tokens={},
                )
                for q in questions
            }
            for model, quality, cost in zip(models, (0.25, 0.5, 0.75), (0.01, 0.02, 0.03))
        }
        return models, questions, table, {}

    methods = (heldout.RANDOM_QUESTIONS, heldout.COHV_UCB, heldout.SCALARIZATION_UCB)
    runs = []
    for reverse in (False, True):
        outdir = tmp_path / str(reverse)
        monkeypatch.setattr(heldout, "load_benchmark", lambda _: benchmark(reverse))
        heldout._run_benchmark("hotpotqa", (1,), (42,), methods, outdir)
        run = json.loads((outdir / "hotpotqa-fold1-seed42.json").read_text(encoding="utf-8"))
        assert set(run["heldout_queries"]) == reserved
        assert set(run["search_queries"]).isdisjoint(reserved)
        runs.append(run)

    original, reversed_quality = runs
    assert original["warm_start_by_method"] == reversed_quality["warm_start_by_method"]
    fields = ("actual_usd", "evaluations", "recommendations")
    for method in methods:
        baseline = original["curve_by_method"][method]
        changed = reversed_quality["curve_by_method"][method]
        assert [[row[field] for field in fields] for row in baseline] == [
            [row[field] for field in fields] for row in changed
        ]
    assert original["heldout_oracle"] != reversed_quality["heldout_oracle"]
    assert [row["gd"] for row in original["rows"]] != [row["gd"] for row in reversed_quality["rows"]]

    curves = summarize(tmp_path / "False")["benchmarks"]["hotpotqa"]["methods"]
    for scores in curves.values():
        assert scores["hv"][0] == 0
        assert scores["hv_regret"][0] == original["heldout_oracle_hv"]
        assert scores["gd"][0] is None and scores["igd"][0] is None
        assert scores["gd"][-1] is not None and scores["igd"][-1] is not None
