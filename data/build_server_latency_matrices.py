"""Build aligned QA latency matrices from per-question server invocation totals.

This offline conversion makes no model API calls. Latency is the sum of
successful invocation response metrics, converted from milliseconds to seconds.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("hotpotqa", "mathqa")
LATENCY_DEFINITION = (
    "sum of successful LLM invocation response metrics.latencyMs per workflow, in seconds"
)
SOURCE_NOTE = (
    "Accuracy, cost, and token matrices come from the research lookup pickle. "
    "Latency comes from per-question server invocation totals; see server_latency_manifest.json."
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_reference(path: Path) -> tuple[list[str], list[str], dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    if not rows or len(rows[0]) < 2 or rows[0][0] != "model_name":
        raise ValueError(f"Invalid matrix header: {path}")
    header = rows[0]
    questions = [q.removeprefix("question_") for q in header[1:]]
    if any(not q.startswith("question_") or not q[9:].isdigit() for q in header[1:]):
        raise ValueError(f"Invalid question IDs: {path}")
    models = [row[0] if row else "" for row in rows[1:]]
    if (not models or any(not model.strip() for model in models)
            or len(set(models)) != len(models) or len(set(questions)) != len(questions)):
        raise ValueError(f"Missing or duplicate matrix labels: {path}")
    values = {}
    for row in rows[1:]:
        if len(row) != len(header):
            raise ValueError(f"Wrong matrix row length: {path}")
        for question, value in zip(questions, row[1:]):
            parsed = float(value)
            if not math.isfinite(parsed):
                raise ValueError(f"Non-finite matrix value: {path}")
            values[(row[0], question)] = parsed
    return header, models, values


def build(dataset: str, lookup: Path, target: Path, *, source_revision: str | None = None) -> dict:
    header, models, accuracy = _read_reference(target / "accuracy_matrix.csv")
    cost_header, cost_models, cost = _read_reference(target / "cost_matrix_usd.csv")
    if cost_header != header or cost_models != models:
        raise ValueError("Cost matrix does not align with accuracy matrix")
    values = {}
    exceptional = []
    sources: dict[str, int] = {}
    with lookup.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["benchmark"] != dataset:
                raise ValueError(f"Wrong benchmark in {lookup}: {row['benchmark']}")
            key = (row["configuration_id"], row["question_id"])
            if key not in accuracy or key in values:
                raise ValueError(f"Unknown or duplicate cell: {key}")
            if not math.isclose(float(row["score"]), accuracy[key], abs_tol=1e-9) or not math.isclose(
                float(row["cost"]), cost[key], abs_tol=1e-8
            ):
                raise ValueError(f"Lookup score or cost differs from research matrix: {key}")
            value = float(row["server_latency_ms"])
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid latency for {key}: {value}")
            values[key] = value / 1000.0
            source = row["record_source"]
            sources[source] = sources.get(source, 0) + 1
            if source == "context_overflow_adjudicated_incorrect":
                exceptional.append({
                    "configuration_id": key[0], "question_id": int(key[1]),
                    "successful_calls_server_latency_seconds_lower_bound": values[key],
                })
    if values.keys() != accuracy.keys():
        raise ValueError(f"Missing {len(accuracy.keys() - values.keys())} cells")

    matrix = target / "latency_matrix_seconds.csv"
    with matrix.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        for model in models:
            writer.writerow([model, *(format(values[(model, q.removeprefix("question_"))], ".12g")
                                      for q in header[1:])])
    manifest = {
        "dataset": dataset,
        "metric": LATENCY_DEFINITION,
        "source_lookup": lookup.name,
        "source_lookup_sha256": sha256(lookup),
        "output_matrix_sha256": sha256(matrix),
        "shape": [len(models), len(header) - 1],
        "cell_count": len(values),
        "record_sources": sources,
        "context_overflow_partial_cells": exceptional,
        "limitations": [
            "Historical per-question call lists are unavailable; the lookup aggregate cannot independently prove that every historical call was included.",
            "Rejected context-overflow requests have no server latency metric; the listed cells contain successful-call lower bounds.",
            "Server latency excludes client execution, network transfer, tool execution, and thread-pool wait.",
        ],
    }
    if source_revision is not None:
        manifest.update(
            source_repository="https://github.com/AgentOptimizer/agentopt",
            source_revision=source_revision,
            source_path=f"experiments/missing_cells/completed/{lookup.name}",
        )
    (target / "server_latency_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    metadata_path = target / "metadata.json"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata.update(latency_definition=LATENCY_DEFINITION,
                        latency_source="server_latency_manifest.json", source_note=SOURCE_NOTE)
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8", newline="\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lookup-dir", type=Path, required=True,
                        help="Directory containing {hotpotqa,mathqa}_lookup.10x10.csv")
    parser.add_argument("--outdir", type=Path, default=ROOT / "data")
    parser.add_argument("--source-revision", help="Source agentopt commit, when known")
    args = parser.parse_args()
    for dataset in DATASETS:
        manifest = build(dataset, args.lookup_dir / f"{dataset}_lookup.10x10.csv",
                         args.outdir / dataset, source_revision=args.source_revision)
        print(f"{dataset}: {manifest['cell_count']} server latency cells")


if __name__ == "__main__":
    main()
