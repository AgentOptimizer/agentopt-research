"""Regeneration keeps portable provenance and the original measured cells."""

import csv
import gzip
import json
import sqlite3

import pytest

from data.extract_scope_matrices import convert_dataset, load_supplemental_prices


@pytest.mark.parametrize("include_partial", [False, True])
def test_export_omits_private_directories_and_preserves_cells(tmp_path, include_partial):
    private_root = tmp_path / "private-checkout"
    manifest_path = private_root / "matrices" / "demo_configs.json"
    manifest_path.parent.mkdir(parents=True)
    configuration = {"solver": "demo-model"}
    manifest_path.write_text(json.dumps({
        "name": "demo", "prefix": "demo", "shape": [1, 2],
        "source_db": "workspace/demo.db",
        "rows": [{"config_id": 7, "config": configuration}],
    }), encoding="utf-8")
    price_path = private_root / "custom_prices.json"
    price_path.write_text(json.dumps({
        "as_of": "2026-01-01", "models": {
            "demo-model": {"input_price": 2.0, "output_price": 3.0},
        },
    }), encoding="utf-8")
    prices, pricing_metadata = load_supplemental_prices(price_path)
    database_path = private_root / "demo.db"
    observations = [(0, 10, 2, 0.5)]
    if not include_partial:
        observations.append((1, 20, 4, 1.0))
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "CREATE TABLE cache_configs (id INTEGER, config_key TEXT, model_config TEXT)"
        )
        connection.execute("INSERT INTO cache_configs VALUES (?, ?, ?)",
                           (7, "demo|config_id=7", json.dumps(configuration)))
        connection.execute(
            "CREATE TABLE cache_queries (config_id INTEGER, query_index INTEGER, "
            "input_tokens TEXT, output_tokens TEXT, performance_score REAL)"
        )
        connection.executemany("INSERT INTO cache_queries VALUES (?, ?, ?, ?, ?)", [
            (7, question, json.dumps({"solver": inputs}),
             json.dumps({"solver": outputs}), score)
            for question, inputs, outputs, score in observations
        ])

    output_root = tmp_path / "export"
    convert_dataset(manifest_path, database_path, output_root, prices,
                    pricing_metadata, include_partial=include_partial)
    output = output_root / "demo"
    metadata_text = (output / "metadata.json").read_text(encoding="utf-8")
    metadata = json.loads(metadata_text)
    assert metadata["source"] == "matrices/demo_configs.json"
    assert metadata["supplemental_pricing_file"] == "custom_prices.json"
    assert metadata["source_database"] == "workspace/demo.db"
    assert "private-checkout" not in metadata_text
    assert metadata["shape"] == [1, 2]
    assert metadata["observed_cells"] == len(observations)
    with gzip.open(output / "records.csv.gz", "rt", encoding="utf-8", newline="") as f:
        records = list(csv.DictReader(f))
    assert [int(row["question_id"]) for row in records] == [x[0] for x in observations]
    assert [float(row["accuracy"]) for row in records] == [x[3] for x in observations]
    assert [float(row["cost_usd"]) for row in records] == pytest.approx([
        inputs * 2e-6 + outputs * 3e-6 for _, inputs, outputs, _ in observations
    ])
    with (output / "accuracy_matrix.csv").open(encoding="utf-8", newline="") as f:
        row = list(csv.reader(f))[1]
    assert row == ["demo|config_id=7", "0.5", "" if include_partial else "1.0"]
