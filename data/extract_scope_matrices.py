#!/usr/bin/env python3
"""Convert SCOPE cached evaluations into AgentOpt matrix datasets."""

from __future__ import annotations

import argparse
import ast
import csv
import gzip
import io
import json
import sqlite3
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Optional


DEFAULT_SUPPLEMENTAL_PRICES = Path(__file__).with_name(
    "scope_supplemental_prices.json"
)


def load_scope_prices(path: Path) -> dict[str, dict[str, float]]:
    """Read SCOPE's per-token ``ALL_MODELS`` price table without importing it."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(
            isinstance(target, ast.Name) and target.id == "ALL_MODELS"
            for target in node.targets
        ):
            models = ast.literal_eval(node.value)
            return {
                name: {
                    "input_price": float(payload["input_price"]),
                    "output_price": float(payload["output_price"]),
                }
                for name, payload in models.items()
            }
    raise ValueError(f"ALL_MODELS not found in {path}")


def load_supplemental_prices(
    path: Path,
) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["pricing_file"] = str(path)
    per_token = {
        name: {
            "input_price": float(price["input_price"]) / 1_000_000,
            "output_price": float(price["output_price"]) / 1_000_000,
        }
        for name, price in payload["models"].items()
    }
    return per_token, payload


MatrixValue = Optional[float]


def write_matrix(path: Path, names: list[str], rows: list[list[MatrixValue]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model_name", *(f"question_{i}" for i in range(len(rows[0])))])
        writer.writerows([name, *values] for name, values in zip(names, rows))


def convert_dataset(
    config_path: Path,
    database_path: Path,
    output_root: Path,
    prices: dict[str, dict[str, float]],
    pricing_metadata: dict[str, Any],
    include_partial: bool = False,
) -> None:
    source = json.loads(config_path.read_text(encoding="utf-8"))
    n_questions = int(source["shape"][1])
    connection = sqlite3.connect(database_path)
    if include_partial:
        prefix = source["prefix"]
        cached_configs = connection.execute(
            "SELECT id,config_key,model_config FROM cache_configs "
            "WHERE substr(config_key,1,?)=? AND substr(config_key,?,1)='|' "
            "ORDER BY id",
            (len(prefix), prefix, len(prefix) + 1),
        ).fetchall()
        candidate_rows = [
            {
                "config_id": config_id,
                "config_key": config_key,
                "config": json.loads(model_config),
            }
            for config_id, config_key, model_config in cached_configs
        ]
    else:
        candidate_rows = source["rows"]

    accepted: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for row in candidate_rows:
        missing = sorted(set(row["config"].values()) - prices.keys())
        (excluded if missing else accepted).append(
            {**row, "missing_price_models": missing} if missing else row
        )

    if excluded:
        missing_models = sorted(
            {model for row in excluded for model in row["missing_price_models"]}
        )
        raise ValueError(
            f"{source['name']} still has {len(excluded)} unpriced configurations: "
            f"{missing_models}"
        )

    outdir = output_root / source["name"]
    outdir.mkdir(parents=True, exist_ok=True)
    names: list[str] = []
    accuracy_rows: list[list[MatrixValue]] = []
    cost_rows: list[list[MatrixValue]] = []
    input_rows: list[list[MatrixValue]] = []
    output_rows: list[list[MatrixValue]] = []
    config_manifest: list[dict[str, Any]] = []
    complete_count = 0
    partial_count = 0
    observed_cells = 0
    records_path = outdir / "records.csv.gz"
    with (
        records_path.open("wb") as raw_records,
        gzip.GzipFile(fileobj=raw_records, mode="wb", mtime=0) as compressed_records,
        io.TextIOWrapper(
            compressed_records, newline="", encoding="utf-8"
        ) as records_handle,
    ):
        records = csv.writer(records_handle)
        records.writerow(
            [
                "task", "config_id", "config_name", "question_id",
                "model_config", "input_tokens", "output_tokens", "cost_usd",
                "accuracy",
            ]
        )
        for row in accepted:
            config = row["config"]
            name = f"{source['name']}|config_id={row['config_id']}"
            query_rows = connection.execute(
                "SELECT query_index,input_tokens,output_tokens,performance_score "
                "FROM cache_queries WHERE config_id=? ORDER BY query_index",
                (row["config_id"],),
            ).fetchall()
            question_ids = [int(query[0]) for query in query_rows]
            if not query_rows or any(
                question_id < 0 or question_id >= n_questions
                for question_id in question_ids
            ):
                raise ValueError(
                    f"config {row['config_id']} has no observations or an out-of-range question"
                )
            is_complete = question_ids == list(range(n_questions))
            if not include_partial and not is_complete:
                raise ValueError(
                    f"config {row['config_id']} is not a complete 0..{n_questions - 1} row"
                )

            accuracy: list[MatrixValue] = [None] * n_questions
            costs: list[MatrixValue] = [None] * n_questions
            inputs: list[MatrixValue] = [None] * n_questions
            outputs: list[MatrixValue] = [None] * n_questions
            config_json = json.dumps(config, sort_keys=True, separators=(",", ":"))
            for question_id, input_json, output_json, score in query_rows:
                input_by_module = json.loads(input_json)
                output_by_module = json.loads(output_json)
                input_total = sum(int(value) for value in input_by_module.values())
                output_total = sum(int(value) for value in output_by_module.values())
                cost = sum(
                    int(input_by_module.get(module, 0)) * prices[model]["input_price"]
                    + int(output_by_module.get(module, 0)) * prices[model]["output_price"]
                    for module, model in config.items()
                )
                accuracy[question_id] = float(score)
                costs[question_id] = float(cost)
                inputs[question_id] = input_total
                outputs[question_id] = output_total
                records.writerow(
                    [
                        source["name"], row["config_id"], name, question_id,
                        config_json, input_total, output_total, format(cost, ".12g"),
                        float(score),
                    ]
                )
            names.append(name)
            accuracy_rows.append(accuracy)
            cost_rows.append(costs)
            input_rows.append(inputs)
            output_rows.append(outputs)
            observed_count = len(query_rows)
            observed_cells += observed_count
            complete_count += int(is_complete)
            partial_count += int(not is_complete)
            config_manifest.append(
                {
                    "model_name": name,
                    "config_id": row["config_id"],
                    "config": config,
                    "observed_count": observed_count,
                    "coverage_fraction": observed_count / n_questions,
                    "is_complete": is_complete,
                }
            )
    connection.close()

    write_matrix(outdir / "accuracy_matrix.csv", names, accuracy_rows)
    write_matrix(outdir / "cost_matrix_usd.csv", names, cost_rows)
    write_matrix(outdir / "input_token_matrix.csv", names, input_rows)
    write_matrix(outdir / "output_token_matrix.csv", names, output_rows)
    write_matrix(
        outdir / "total_token_matrix.csv",
        names,
        [
            [
                None if a is None or b is None else a + b
                for a, b in zip(left, right)
            ]
            for left, right in zip(input_rows, output_rows)
        ],
    )
    (outdir / "configs.json").write_text(
        json.dumps(config_manifest, indent=2) + "\n", encoding="utf-8"
    )
    supplemental_used = sorted(
        set(pricing_metadata["models"])
        & {model for row in accepted for model in row["config"].values()}
    )
    metadata = {
        "benchmark": source["name"],
        "source": str(config_path),
        "source_database": source["source_db"],
        "shape": [len(accepted), n_questions],
        "orientation": "rows=workflow configurations; columns=question IDs",
        "accuracy_definition": "SCOPE performance_score (0/1 per query)",
        "cost_definition": (
            "sum(module input tokens * model input price + module output tokens * "
            "model output price)"
        ),
        "records_file": "records.csv.gz",
        "matrix_kind": "sparse" if include_partial else "complete-only",
        "missing_cell_representation": "empty CSV field" if include_partial else None,
        "cache_configurations": len(candidate_rows),
        "original_complete_configurations": len(source["rows"]),
        "included_configurations": len(accepted),
        "complete_configurations": complete_count,
        "partial_configurations": partial_count,
        "observed_cells": observed_cells,
        "possible_cells": len(accepted) * n_questions,
        "coverage_fraction": (
            observed_cells / (len(accepted) * n_questions) if accepted else 0.0
        ),
        "excluded_configurations": 0,
        "excluded_missing_price_models": [],
        "excluded_config_ids": [],
        "supplemental_pricing_as_of": pricing_metadata["as_of"],
        "supplemental_price_models_used": supplemental_used,
        "supplemental_pricing_file": pricing_metadata["pricing_file"],
    }
    (outdir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"{source['name']}: wrote {len(accepted)}x{n_questions}; "
        f"complete {complete_count}, partial {partial_count}, excluded 0"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parent / "scope")
    parser.add_argument(
        "--supplemental-prices", type=Path, default=DEFAULT_SUPPLEMENTAL_PRICES
    )
    parser.add_argument(
        "--include-partial",
        action="store_true",
        help="include cache configurations with missing questions as sparse rows",
    )
    args = parser.parse_args()
    scope_root = args.scope_root.resolve()
    output_root = args.output_root.resolve()
    prices = load_scope_prices(scope_root / "src/workflows/models.py")
    supplemental, pricing_metadata = load_supplemental_prices(
        args.supplemental_prices.resolve()
    )
    prices.update(supplemental)
    configs = sorted((scope_root / "matrices").glob("*_configs.json"))
    source_databases = {
        json.loads(path.read_text(encoding="utf-8"))["source_db"] for path in configs
    }
    with tempfile.TemporaryDirectory(prefix="agentopt-scope-") as temp:
        with tarfile.open(scope_root / "data_workspace.tar.gz", "r:gz") as archive:
            for member_name in source_databases:
                archive.extract(member_name, path=temp, filter="data")
        for config_path in configs:
            source = json.loads(config_path.read_text(encoding="utf-8"))
            convert_dataset(
                config_path,
                Path(temp) / source["source_db"],
                output_root,
                prices,
                pricing_metadata,
                include_partial=args.include_partial,
            )


if __name__ == "__main__":
    main()
