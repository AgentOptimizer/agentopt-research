"""Server latency provenance, alignment, and regeneration regression checks."""
import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from data import build_server_latency_matrices as server
from data import extract_response_matrices as extract
from experiments.combined_objective.three_objective_metrics import benchmark_input_hashes


ROOT = Path(__file__).resolve().parents[1]


def _write_lookup(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def inputs(tmp_path):
    for filename, values in (("accuracy_matrix.csv", [1, .5]),
                             ("cost_matrix_usd.csv", [.01, .02])):
        with (tmp_path / filename).open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerows([
                ["model_name", "question_0", "question_1"], ["model", *values],
            ])
    (tmp_path / "metadata.json").write_text('{"shape": [1, 2]}', encoding="utf-8")
    rows = [
        dict(benchmark="mathqa", configuration_id="model", question_id="1",
             score="0.5", cost="0.02", server_latency_ms="2345",
             record_source="context_overflow_adjudicated_incorrect"),
        dict(benchmark="mathqa", configuration_id="model", question_id="0",
             score="1", cost="0.01", server_latency_ms="1000", record_source="historical"),
    ]
    lookup = tmp_path / "mathqa_lookup.10x10.csv"
    _write_lookup(lookup, rows)
    return tmp_path, lookup, rows


def test_conversion_aligns_by_cell_and_records_lower_bounds(inputs):
    target, lookup, _ = inputs
    manifest = server.build("mathqa", lookup, target)
    with (target / "latency_matrix_seconds.csv").open(newline="") as handle:
        assert list(csv.reader(handle)) == [
            ["model_name", "question_0", "question_1"], ["model", "1", "2.345"],
        ]
    assert manifest["source_lookup_sha256"] == server.sha256(lookup)
    assert manifest["output_matrix_sha256"] == server.sha256(target / "latency_matrix_seconds.csv")
    assert manifest["context_overflow_partial_cells"] == [{
        "configuration_id": "model", "question_id": 1,
        "successful_calls_server_latency_seconds_lower_bound": 2.345,
    }]
    assert json.loads((target / "metadata.json").read_text())["latency_definition"] == server.LATENCY_DEFINITION
    assert any(key.endswith("server_latency_manifest.json") for key in benchmark_input_hashes(target))


@pytest.mark.parametrize("field,value", [
    ("benchmark", "hotpotqa"), ("configuration_id", "unknown"), ("question_id", "7"),
    ("score", "0.7"), ("cost", "0.1"), ("server_latency_ms", "-1"),
    ("server_latency_ms", "nan"), ("server_latency_ms", "inf"),
])
def test_bad_source_leaves_existing_latency_untouched(inputs, field, value):
    target, lookup, rows = inputs
    matrix = target / "latency_matrix_seconds.csv"
    matrix.write_text("existing latency\n", encoding="utf-8")
    rows[0][field] = value
    _write_lookup(lookup, rows)
    with pytest.raises(ValueError):
        server.build("mathqa", lookup, target)
    assert matrix.read_text() == "existing latency\n"


@pytest.mark.parametrize("duplicate", [False, True])
def test_requires_each_cell_exactly_once(inputs, duplicate):
    target, lookup, rows = inputs
    _write_lookup(lookup, rows + [rows[0]] if duplicate else rows[:1])
    with pytest.raises(ValueError, match="duplicate|Missing"):
        server.build("mathqa", lookup, target)


def test_full_extraction_uses_server_totals_instead_of_pickle_latency(inputs, monkeypatch):
    target, lookup, _ = inputs
    table = {"model": {
        q: SimpleNamespace(score=score, cost=cost, latency_seconds=999999,
                           input_tokens={"llm": 5}, output_tokens={"llm": 2})
        for q, score, cost in ((0, 1., .01), (1, .5, .02))
    }}
    monkeypatch.setattr(extract, "load_pickle", lambda _: (["model"], [0, 1], table))
    metadata = extract.extract_benchmark("mathqa", ROOT / "fixture.pkl", target, lookup)
    with (target / "latency_matrix_seconds.csv").open(newline="") as handle:
        assert list(csv.reader(handle))[1] == ["model", "1", "2.345"]
    assert metadata["latency_definition"] == server.LATENCY_DEFINITION
    assert metadata["observed_cells"] == 2


@pytest.mark.parametrize("dataset,partial_count", [("hotpotqa", 0), ("mathqa", 4)])
def test_committed_matrices_match_manifests(dataset, partial_count):
    target = ROOT / "data" / dataset
    manifest = json.loads((target / "server_latency_manifest.json").read_text())
    matrix = target / "latency_matrix_seconds.csv"
    assert hashlib.sha256(matrix.read_bytes()).hexdigest() == manifest["output_matrix_sha256"]
    assert manifest["shape"] == [100, 200]
    assert manifest["cell_count"] == sum(manifest["record_sources"].values()) == 20000
    assert len(manifest["context_overflow_partial_cells"]) == partial_count
    with matrix.open(newline="") as handle:
        rows = list(csv.reader(handle))
    cells = {(row[0], int(q.removeprefix("question_"))): float(value)
             for row in rows[1:] for q, value in zip(rows[0][1:], row[1:])}
    for partial in manifest["context_overflow_partial_cells"]:
        assert cells[(partial["configuration_id"], partial["question_id"])] == partial[
            "successful_calls_server_latency_seconds_lower_bound"]
