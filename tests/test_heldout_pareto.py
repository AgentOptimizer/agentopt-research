import csv
import json

import pytest

from experiments.combined_objective import run_heldout_pareto as heldout


def write_cache(directory, n_questions, rows):
    directory.mkdir(parents=True, exist_ok=True)
    configs = [
        {"model_name": name, "config": roles}
        for name, roles, _, _ in rows
    ]
    (directory / "configs.json").write_text(json.dumps(configs))
    header = ["model_name", *(f"question_{q}" for q in range(n_questions))]
    for filename, field in (("accuracy_matrix.csv", 2), ("cost_matrix_usd.csv", 3)):
        with (directory / filename).open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            for row in rows:
                writer.writerow([row[0], *row[field]])


def test_transfer_matches_roles_and_keeps_search_independent_of_test_scores(monkeypatch, tmp_path):
    monkeypatch.setattr(heldout, "ROOT", tmp_path)
    scope = tmp_path / "data/scope"
    cheap = {"plan": "a", "answer": "b"}
    expensive = {"plan": "b", "answer": "a"}
    extra = {"plan": "c", "answer": "b"}
    search_questions = 25
    write_cache(scope / "restaurant_valid", search_questions, [
        ("search-a", cheap, [0.2 + 0.1 * (q % 3) for q in range(search_questions)], [0.01] * search_questions),
        ("search-b", expensive, [0.6 + 0.1 * (q % 3) for q in range(search_questions)], [0.02] * search_questions),
        ("search-c", extra, [0.8] * search_questions, [0.03] * search_questions),
    ])
    runs = []
    for reverse in (False, True):
        write_cache(scope / "restaurant_test", 3, [
            ("test-b", expensive, [0.1 if reverse else 0.8] * 3, [0.02] * 3),
            ("test-a", dict(reversed(tuple(cheap.items()))), [0.6] * 3, [0.01] * 3),
            ("test-c", extra, [0.8, 0.9, None], [0.03, 0.03, None]),
        ])
        outdir = tmp_path / str(reverse)
        heldout.run_pair("restaurant", (42,), (0.01, 0.3, 0.8), outdir)
        runs.append(json.loads((outdir / "restaurant-seed42.json").read_text()))

    first, second = runs
    assert first["policy"]["acquisition_target"] == "finite_benchmark_mean"
    assert [(c["search_model_name"], c["test_model_name"]) for c in first["configurations"]] == [
        ("search-b", "test-b"), ("search-a", "test-a"),
    ]
    assert first["exhaustive_search_cost_usd"] == pytest.approx(0.75)
    assert first["test_pareto_indices"] == [0, 1]
    assert second["test_pareto_indices"] == [1]
    assert not first["checkpoints"][0]["available"]
    for left, right in zip(first["checkpoints"][1:], second["checkpoints"][1:]):
        assert left["search_cost_usd"] <= left["budget_usd"] + 1e-12
        assert left["search_cost_usd"] == right["search_cost_usd"]
        assert left["search_evaluations"] == right["search_evaluations"]
        assert left["selected_indices"] == right["selected_indices"]
    assert first["checkpoints"][-1]["test_metrics"] != second["checkpoints"][-1]["test_metrics"]


def test_bird_excludes_every_dev_occurrence_of_search_questions(tmp_path):
    mapping = tmp_path / "mapping.json"
    mapping.write_text(json.dumps({
        "source": "test cache source", "mini_to_dev": {"0": [8, 10], "1": [8]},
    }))
    test_ids, provenance = heldout.exclude_bird_search_questions(mapping, [0, 1], [0, 1, 7, 8, 10])
    assert test_ids == [0, 1, 7]
    assert provenance["excluded_dev_question_ids"] == [8, 10]
    with pytest.raises(ValueError, match="cover exactly"):
        heldout.exclude_bird_search_questions(mapping, [0, 1, 2], [0, 1, 7, 8, 10])
    with pytest.raises(ValueError, match="valid, distinct Dev matches"):
        heldout.exclude_bird_search_questions(mapping, [0, 1], [0, 1, 7, 8])
