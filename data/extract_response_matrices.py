#!/usr/bin/env python3
"""Regenerate the committed response matrices from the research lookup pickles.

Writes, per benchmark, six configuration-by-question matrices plus a
``metadata.json`` describing shape, missingness and the highest-cost cells.

The source is the current research lookup pickle under
``experiments/data/lookup/``. It is deliberately not merged with or corrected
from the separate aggregated JSONL run.

    PYTHONPATH=. .venv/bin/python data/extract_response_matrices.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from experiments.single_objective.offline_selector_sim import (  # noqa: E402
    SampleResult,
    load_pickle,
)

BENCHMARKS = ("hotpotqa", "mathqa")
LOOKUP_DIR = ROOT / "experiments/data/lookup"
TOP_COST_CELLS = 20

MATRIX_NAMES = (
    "accuracy_matrix",
    "cost_matrix_usd",
    "latency_matrix_seconds",
    "input_token_matrix",
    "output_token_matrix",
    "total_token_matrix",
)
TOKEN_DEFINITIONS = {
    "input_token_definition": "sum of input tokens across workflow calls",
    "output_token_definition": "sum of output tokens across workflow calls",
    "total_token_definition": "input plus output tokens",
}
SOURCE_NOTE = (
    "These matrices intentionally reflect the current research lookup pickle. "
    "They are not merged with or corrected from a separate JSONL run."
)

Cell = Dict[str, object]


def _token_total(counts: Optional[Mapping[str, int]]) -> int:
    return sum((counts or {}).values())


def _cell_values(sample: SampleResult) -> Cell:
    input_tokens = _token_total(sample.input_tokens)
    output_tokens = _token_total(sample.output_tokens)
    return {
        "accuracy_matrix": sample.score,
        "cost_matrix_usd": sample.cost,
        "latency_matrix_seconds": sample.latency_seconds,
        "input_token_matrix": input_tokens,
        "output_token_matrix": output_tokens,
        "total_token_matrix": input_tokens + output_tokens,
    }


def _write_matrix(
    path: Path,
    models: Sequence[str],
    datapoints: Sequence[int],
    cells: Mapping[str, Mapping[int, object]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model_name"] + [f"question_{dp}" for dp in datapoints])
        for model in models:
            row = cells.get(model, {})
            # A blank cell means the pickle has no observation, never zero.
            writer.writerow([model] + [row.get(dp, "") for dp in datapoints])


def extract_benchmark(benchmark: str, pickle_path: Path, outdir: Path) -> Cell:
    models, datapoints, table = load_pickle(str(pickle_path))
    known_datapoints = frozenset(datapoints)

    matrices: Dict[str, Dict[str, Dict[int, object]]] = {
        name: {} for name in MATRIX_NAMES
    }
    cost_cells: List[Cell] = []

    for model in models:
        for datapoint, sample in table.get(model, {}).items():
            if datapoint not in known_datapoints:
                continue
            values = _cell_values(sample)
            for name, value in values.items():
                matrices[name].setdefault(model, {})[datapoint] = value
            cost_cells.append(
                {
                    "model_name": model,
                    "question_id": datapoint,
                    "cost_usd": sample.cost,
                    "input_tokens": values["input_token_matrix"],
                    "output_tokens": values["output_token_matrix"],
                    "total_tokens": values["total_token_matrix"],
                }
            )

    outdir.mkdir(parents=True, exist_ok=True)
    for name in MATRIX_NAMES:
        _write_matrix(outdir / f"{name}.csv", models, datapoints, matrices[name])

    cost_cells.sort(
        key=lambda cell: (-cell["cost_usd"], cell["model_name"], cell["question_id"])
    )
    observed = len(cost_cells)
    metadata: Cell = {
        "benchmark": benchmark,
        "source": pickle_path.relative_to(ROOT).as_posix(),
        "shape": [len(models), len(datapoints)],
        "observed_cells": observed,
        "missing_cells": len(models) * len(datapoints) - observed,
        "cost_definition": "sum(model input tokens * input price + model output tokens * output price)",
        "pricing_file": "../aws_bedrock_prices_10x10.json",
        "orientation": "rows=model configurations; columns=question IDs",
        "blank_cells": "missing lookup-table observations, not zero",
        "latency_definition": "end-to-end wall-clock seconds for the workflow cell",
        **TOKEN_DEFINITIONS,
        "source_note": SOURCE_NOTE,
        "top_20_cost_cells": cost_cells[:TOP_COST_CELLS],
    }
    with (outdir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmarks", nargs="+", default=list(BENCHMARKS))
    parser.add_argument("--lookup-dir", type=Path, default=LOOKUP_DIR)
    parser.add_argument("--outdir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()

    for benchmark in args.benchmarks:
        pickle_path = args.lookup_dir / f"{benchmark}_lookup.pkl"
        if not pickle_path.is_file():
            raise SystemExit(
                f"Lookup pickle not found: {pickle_path}\n"
                f"{args.lookup_dir}/ is gitignored; place "
                f"{{{','.join(BENCHMARKS)}}}_lookup.pkl there and retry."
            )
        metadata = extract_benchmark(benchmark, pickle_path, args.outdir / benchmark)
        rows, cols = metadata["shape"]
        print(
            f"{benchmark}: {rows}x{cols}, {metadata['observed_cells']} observed, "
            f"{metadata['missing_cells']} missing"
        )


if __name__ == "__main__":
    main()
