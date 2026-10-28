"""Build aligned QA latency matrices from recorded per-question server totals.

This is an offline transformation: no LLM requests are made. The source lookup
tables contain cumulative ``server_latency_ms`` for each configuration/question.
The four MathQA context-overflow entries retain their successful-call subtotal
and are identified in the accompanying manifest.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATASETS = ("hotpotqa", "mathqa")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(dataset: str, lookup: Path) -> None:
    target = ROOT / "data" / dataset
    with (target / "accuracy_matrix.csv").open(newline="", encoding="utf-8") as handle:
        source = list(csv.reader(handle))
    header = source[0]
    models = [row[0] for row in source[1:]]
    questions = header[1:]
    expected = {(model, question.removeprefix("question_")) for model in models for question in questions}
    accuracy = {(row[0], question.removeprefix("question_")): float(value)
                for row in source[1:] for question, value in zip(questions, row[1:])}
    with (target / "cost_matrix_usd.csv").open(newline="", encoding="utf-8") as handle:
        cost_rows = list(csv.reader(handle))
    if cost_rows[0] != header or [row[0] for row in cost_rows[1:]] != models:
        raise ValueError("Cost matrix does not align with accuracy matrix")
    cost = {(row[0], question.removeprefix("question_")): float(value)
            for row in cost_rows[1:] for question, value in zip(questions, row[1:])}
    values: dict[tuple[str, str], float] = {}
    exceptional = []
    sources: dict[str, int] = {}
    with lookup.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["benchmark"] != dataset:
                raise ValueError(f"Wrong benchmark in {lookup}: {row['benchmark']}")
            key = (row["configuration_id"], row["question_id"])
            if key not in expected or key in values:
                raise ValueError(f"Unknown or duplicate cell: {key}")
            if not math.isclose(float(row["score"]), accuracy[key], abs_tol=1e-9) or not math.isclose(
                float(row["cost"]), cost[key], abs_tol=1e-8):
                raise ValueError(f"Lookup score or cost differs from research matrix: {key}")
            value = float(row["server_latency_ms"])
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid latency for {key}: {value}")
            values[key] = value / 1000.0
            sources[row["record_source"]] = sources.get(row["record_source"], 0) + 1
            if row["record_source"] == "context_overflow_adjudicated_incorrect":
                exceptional.append({"configuration_id": key[0], "question_id": int(key[1]),
                                    "successful_calls_server_latency_seconds_lower_bound": values[key]})
    if values.keys() != expected:
        raise ValueError(f"Missing {len(expected - values.keys())} cells")
    matrix = target / "latency_matrix_seconds.csv"
    with matrix.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        for model in models:
            writer.writerow([model, *(format(values[(model, q.removeprefix("question_"))], ".12g")
                                      for q in questions)])
    manifest = {
        "dataset": dataset,
        "metric": "sum of successful LLM invocation response metrics.latencyMs per workflow, in seconds",
        "source_lookup": lookup.name,
        "source_lookup_sha256": sha256(lookup),
        "output_matrix_sha256": sha256(matrix),
        "shape": [len(models), len(questions)],
        "cell_count": len(values),
        "record_sources": sources,
        "context_overflow_partial_cells": exceptional,
        "limitations": [
            "Historical per-question call lists are unavailable; the lookup aggregate cannot independently prove that every historical call was included.",
            "Rejected context-overflow requests have no server latency metric; the four listed cells contain successful-call lower bounds.",
            "Server latency excludes client execution, network transfer, tool execution, and thread-pool wait.",
        ],
    }
    (target / "server_latency_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"{dataset}: {len(values)} cells, {len(exceptional)} partial lower bounds, {matrix}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lookup-dir", type=Path, required=True,
                        help="Directory containing hotpotqa_lookup.10x10.csv and mathqa_lookup.10x10.csv")
    args = parser.parse_args()
    for dataset in DATASETS:
        build(dataset, args.lookup_dir / f"{dataset}_lookup.10x10.csv")


if __name__ == "__main__":
    main()
