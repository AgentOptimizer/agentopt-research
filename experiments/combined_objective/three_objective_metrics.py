"""Strict Q/latency/USD data loading and post-run three-objective diagnostics.

Raw vectors always have order ``(quality, mean_latency_seconds, mean_cost_usd)``.
Quality is maximized and both latency and deployment cost are minimized.  The
positive reciprocal coordinates below are used only to score finished replays;
their full-data reference scales must never enter acquisition or calibration.
"""
from __future__ import annotations

import csv
import hashlib
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from experiments.single_objective.offline_selector_sim import LookupTable, SampleResult
from experiments.combined_objective.anonymous_metadata import artifact_reference


ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS = {name: ROOT / "data" / name for name in ("mathqa", "hotpotqa")}
MATRIX_FILENAMES = (
    "accuracy_matrix.csv",
    "cost_matrix_usd.csv",
    "latency_matrix_seconds.csv",
)


def _benchmark_directory(benchmark_or_path: str | Path) -> Path:
    return BENCHMARKS.get(str(benchmark_or_path).lower(), Path(benchmark_or_path))


def benchmark_input_hashes(benchmark_or_path: str | Path) -> dict[str, str]:
    """Fingerprint every loaded matrix, including latency, and its metadata."""
    directory = _benchmark_directory(benchmark_or_path)
    files = [directory / name for name in MATRIX_FILENAMES]
    metadata = directory / "metadata.json"
    if metadata.is_file():
        files.append(metadata)
    hashes = {}
    for path in files:
        if not path.is_file():
            raise ValueError(f"missing three-objective matrix: {path}")
        key = artifact_reference(path, root=ROOT)
        hashes[key] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def _read_matrix(path: Path) -> tuple[list[str], list[str], list[int], np.ndarray]:
    if not path.is_file():
        raise ValueError(f"missing three-objective matrix: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if not header:
            raise ValueError(f"empty three-objective matrix: {path}")
        if header[0] != "model_name":
            raise ValueError(f"{path} must start with a model_name column")
        columns = header[1:]
        if not columns or any(not name.startswith("question_") for name in columns):
            raise ValueError(f"{path} requires question_<id> columns")
        try:
            questions = [int(name.removeprefix("question_")) for name in columns]
        except ValueError as error:
            raise ValueError(f"invalid question column in {path}") from error
        if len(set(questions)) != len(questions):
            raise ValueError(f"duplicate question IDs in {path}")
        models: list[str] = []
        seen: set[str] = set()
        values: list[list[float]] = []
        for line, row in enumerate(reader, start=2):
            if len(row) != len(header):
                raise ValueError(f"{path}:{line} has {len(row)} columns; expected {len(header)}")
            model = row[0]
            if not model.strip() or model in seen:
                raise ValueError(f"missing or duplicate model_name at {path}:{line}")
            if any(not value.strip() for value in row[1:]):
                raise ValueError(f"missing cell at {path}:{line}; three-objective matrices must be complete")
            try:
                parsed = [float(value) for value in row[1:]]
            except ValueError as error:
                raise ValueError(f"non-numeric value at {path}:{line}") from error
            if not all(math.isfinite(value) for value in parsed):
                raise ValueError(f"non-finite value at {path}:{line}")
            seen.add(model)
            models.append(model)
            values.append(parsed)
    if not models:
        raise ValueError(f"three-objective matrix has no configurations: {path}")
    return models, columns, questions, np.asarray(values, dtype=np.float64)


def load_three_objective_benchmark(
    benchmark_or_path: str | Path,
) -> tuple[list[str], list[int], LookupTable, dict[str, str]]:
    """Load exactly aligned complete accuracy, USD cost, and latency matrices.

    Accepts ``mathqa``/``hotpotqa`` or a directory, preserving the CSV row and
    question order.  Missing, reordered, duplicate, or malformed cells/labels
    fail explicitly; there is no zero-latency fallback or silent intersection.
    """
    directory = _benchmark_directory(benchmark_or_path)
    matrices = [_read_matrix(directory / name) for name in MATRIX_FILENAMES]
    models, columns, questions, accuracy = matrices[0]
    for name, (other_models, other_columns, _, _) in zip(MATRIX_FILENAMES[1:], matrices[1:]):
        if other_models != models:
            raise ValueError(f"accuracy and {name} matrices have different model rows")
        if other_columns != columns:
            raise ValueError(f"accuracy and {name} matrices have different questions (column labels or order)")
    cost = matrices[1][3]
    latency = matrices[2][3]
    if np.any((accuracy < 0.0) | (accuracy > 1.0)):
        raise ValueError("accuracy outside [0, 1]")
    if np.any(cost < 0.0):
        raise ValueError("negative cost in cost_matrix_usd.csv")
    if np.any(latency < 0.0):
        raise ValueError("negative latency in latency_matrix_seconds.csv")
    table: LookupTable = {
        model: {
            question: SampleResult(
                score=float(accuracy[row, column]),
                latency_seconds=float(latency[row, column]),
                input_tokens={},
                output_tokens={},
                cost=float(cost[row, column]),
            )
            for column, question in enumerate(questions)
        }
        for row, model in enumerate(models)
    }
    return models, questions, table, benchmark_input_hashes(directory)


def _three_points(points: Any) -> np.ndarray:
    array = np.asarray(points, dtype=np.float64)
    if array.ndim == 1 and array.size == 0:
        array = array.reshape(0, 3)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError("points must have shape (n, 3)")
    if not np.all(np.isfinite(array)):
        raise ValueError("points must contain only finite values")
    return array


def nondominated_indices(points: Any) -> np.ndarray:
    """Return all undominated 3D indices when every coordinate is maximized.

    Exact duplicates remain separate arms.  Dominance must be strict in at
    least one coordinate; no tolerance blurs close but distinct configurations.
    """
    array = _three_points(points)
    keep = np.ones(len(array), dtype=bool)
    for index, point in enumerate(array):
        keep[index] = not np.any(np.all(array >= point, axis=1) & np.any(array > point, axis=1))
    return np.flatnonzero(keep)


def raw_pareto_indices(points: Any) -> list[int]:
    """Pareto indices for raw ``(Q, L seconds, D USD)``; max Q, min L/D."""
    array = _three_points(points)
    return nondominated_indices(array * np.asarray([1.0, -1.0, -1.0])).tolist()


def raw_truth_vectors(
    models: Sequence[str], questions: Sequence[int], table: Mapping[str, Mapping[int, SampleResult]],
) -> np.ndarray:
    """Full-matrix mean Q/L/D vectors for diagnostic use after replay only."""
    if not questions:
        raise ValueError("at least one question is required")
    vectors = []
    for model in models:
        try:
            samples = [table[model][question] for question in questions]
        except KeyError as error:
            raise ValueError(f"incomplete lookup table for {model}") from error
        vectors.append(np.mean([(sample.score, sample.latency_seconds, sample.cost) for sample in samples], axis=0))
    return _three_points(vectors)


def _hypervolume_2d(points: np.ndarray) -> float:
    """Union of origin-anchored rectangles with nonnegative upper corners."""
    if not len(points):
        return 0.0
    ordered = points[np.argsort(points[:, 0])]
    suffix_y = np.maximum.accumulate(ordered[::-1, 1])[::-1]
    widths = np.diff(np.concatenate(([0.0], ordered[:, 0])))
    return float(np.sum(widths * suffix_y))


def hypervolume_3d(points: Any, reference: Sequence[float] | None = None) -> float:
    """Exact dominated volume for maximization, with an origin default reference.

    A sweep of distinct x coordinates integrates the union of yz rectangles.
    Points below the reference in any coordinate contribute no volume.
    """
    array = _three_points(points)
    origin = np.zeros(3) if reference is None else np.asarray(reference, dtype=np.float64)
    if origin.shape != (3,) or not np.all(np.isfinite(origin)):
        raise ValueError("reference must contain three finite coordinates")
    shifted = array - origin
    eligible = shifted[np.all(shifted > 0.0, axis=1)]
    if not len(eligible):
        return 0.0
    front = eligible[nondominated_indices(eligible)]
    volume = 0.0
    previous_x = 0.0
    for x in np.unique(front[:, 0]):
        slice_points = front[front[:, 0] >= x, 1:]
        volume += float(x - previous_x) * _hypervolume_2d(slice_points)
        previous_x = float(x)
    return float(volume)


def front_distance(obtained: Any, reference: Any) -> float:
    """Mean nearest-front Euclidean distance in the 3D scoring coordinates."""
    obtained_array = _three_points(obtained)
    reference_array = _three_points(reference)
    obtained_front = obtained_array[nondominated_indices(obtained_array)]
    reference_front = reference_array[nondominated_indices(reference_array)]
    if not len(obtained_front):
        return math.inf if len(reference_front) else 0.0
    if not len(reference_front):
        return math.inf
    differences = obtained_front[:, None, :] - reference_front[None, :, :]
    return float(np.sqrt(np.sum(differences ** 2, axis=-1)).min(axis=1).mean())


def evaluation_space(raw_truth: Any) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    """Post-run reciprocal Q/L/D coordinates and full-data median references.

    Each reference is the median strictly positive per-arm full-data mean,
    matching the existing USD scoring convention.  If an entire axis is zero,
    a unit reference maps that constant axis to one without changing dominance.
    """
    raw = _three_points(raw_truth)
    if not len(raw):
        raise ValueError("truth vectors must contain at least one arm")
    if np.any((raw[:, 0] < 0.0) | (raw[:, 0] > 1.0)) or np.any(raw[:, 1:] < 0.0):
        raise ValueError("truth vectors require Q in [0, 1] and nonnegative latency/cost")
    positive_latency = raw[:, 1][raw[:, 1] > 0.0]
    positive_cost = raw[:, 2][raw[:, 2] > 0.0]
    latency_reference = float(np.median(positive_latency)) if len(positive_latency) else 1.0
    cost_reference = float(np.median(positive_cost)) if len(positive_cost) else 1.0
    metric_truth = np.column_stack((
        raw[:, 0],
        latency_reference / (latency_reference + raw[:, 1]),
        cost_reference / (cost_reference + raw[:, 2]),
    ))
    truth_front = metric_truth[nondominated_indices(metric_truth)]
    return metric_truth, truth_front, latency_reference, cost_reference, hypervolume_3d(truth_front)


def score_selection(
    selected: Sequence[int], metric_truth: Any, truth_front: Any, truth_hv: float,
) -> dict[str, float]:
    metric = _three_points(metric_truth)
    indices = np.asarray(selected, dtype=int)
    if indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= len(metric)):
        raise ValueError("selected arm indices are outside the truth matrix")
    obtained = metric[indices]
    obtained_hv = hypervolume_3d(obtained)
    regret = max(0.0, float(truth_hv) - obtained_hv)
    return {
        "hypervolume": obtained_hv,
        "hv_regret": regret,
        "relative_hv_regret": regret / truth_hv if truth_hv > 0.0 else 0.0,
        "generational_distance": front_distance(obtained, truth_front),
        "inverted_generational_distance": front_distance(truth_front, obtained),
    }


def enrich_run(run: dict[str, Any]) -> dict[str, Any]:
    """Add oracle metrics to an already finished run and its checkpoints in place.

    Search fractions use actual USD divided by actual brute-force USD.  The
    all-zero-cost case maps to zero.  Undefined empty-front distances are saved
    as null, allowing strict JSON serialization instead of nonstandard Infinity.
    """
    raw = _three_points(run["raw_truth_vectors"])
    metric, truth_front, latency_reference, cost_reference, truth_hv = evaluation_space(raw)
    frontier = raw_pareto_indices(raw)
    true_set = set(frontier)
    best_quality = set(np.flatnonzero(raw[:, 0] == np.max(raw[:, 0])).tolist())
    best_latency = set(np.flatnonzero(raw[:, 1] == np.min(raw[:, 1])).tolist())
    best_cost = set(np.flatnonzero(raw[:, 2] == np.min(raw[:, 2])).tolist())
    brute_force = float(run["bruteforce_search_cost_usd"])
    if not math.isfinite(brute_force) or brute_force < 0.0:
        raise ValueError("bruteforce_search_cost_usd must be finite and nonnegative")
    run.update(
        full_data_pareto_arm_indices=frontier,
        true_best_accuracy_arm_indices=sorted(best_quality),
        true_best_latency_arm_indices=sorted(best_latency),
        true_best_cost_arm_indices=sorted(best_cost),
        metric_truth_vectors=metric.tolist(),
        metric_latency_reference_seconds=latency_reference,
        metric_cost_reference_usd=cost_reference,
        metric_reference_point=[0.0, 0.0, 0.0],
        metric_objective_order=["Q", "L", "D"],
        metric_transform="Q, median_positive_mean_L/(median_positive_mean_L+mean_L), median_positive_mean_D/(median_positive_mean_D+mean_D)",
        ground_truth_hypervolume=truth_hv,
    )
    for point in run["points"]:
        selected = set(map(int, point["selected_arm_indices"]))
        true_positive = selected & true_set
        metrics = score_selection(sorted(selected), metric, truth_front, truth_hv)
        point.update({key: value if math.isfinite(value) else None for key, value in metrics.items()})
        cost = float(point["cumulative_search_cost_usd"])
        point.update(
            cost_fraction=cost / brute_force if brute_force > 0.0 else 0.0,
            pareto_true_positive_count=len(true_positive),
            pareto_false_positive_count=len(selected - true_set),
            pareto_false_negative_count=len(true_set - selected),
            pareto_precision=len(true_positive) / len(selected) if selected else 1.0,
            pareto_recall=len(true_positive) / len(true_set) if true_set else 1.0,
            exact_frontier=selected == true_set,
            covers_frontier=true_set.issubset(selected),
            contains_accuracy_endpoint=bool(selected & best_quality),
            contains_latency_endpoint=bool(selected & best_latency),
            contains_cost_endpoint=bool(selected & best_cost),
        )
    total_cost = float(run.get("cumulative_search_cost_usd", run.get("total_cost", 0.0)))
    run["cost_fraction"] = total_cost / brute_force if brute_force > 0.0 else 0.0
    return run
