#!/usr/bin/env python3
"""
Offline Selector Simulator — Single Objective
==============================================
Accuracy-only replay of selector decision logic on frozen brute-force
results (no API calls). This is a baseline; the main protocol is
multi-objective (see ``combined_objective/``).

Preferred input: pickle lookup tables under ``data/lookup/``
(gitignored — keep them locally). JSONL is still supported.

Usage:
    python single_objective/offline_selector_sim.py \\
        --pickle data/lookup/gpqa_lookup.pkl \\
        --selectors all --seeds 50

    python single_objective/offline_selector_sim.py \\
        --pickle data/lookup/mathqa_lookup.pkl \\
        --selectors random_search,matrix_ucb --seeds 20
"""

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

_EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent
_REPO_ROOT = _EXPERIMENTS_DIR.parent
_SRC_DIR = _REPO_ROOT / "src"
_DEFAULT_PICKLE_DIR = _EXPERIMENTS_DIR / "data" / "lookup"
_GITTINS_ROOTS_DISK_CACHE = (
    Path(__file__).resolve().parent / "results" / "cache_gittins_roots"
)

# Reused across seeds in plotting sweeps. Building the JAX lookup is the
# dominant fixed cost, while it is identical for a given matrix/configuration.
_GITTINS_ROOTS_CACHE: Dict[Tuple[Any, ...], Any] = {}

if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

try:
    from dotenv import load_dotenv
    load_dotenv()
    load_dotenv(_EXPERIMENTS_DIR / ".env")
    load_dotenv(_REPO_ROOT / ".env")
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Bedrock pricing ($/MTok) — keyed by ARN suffix (profile ID)
# Updated to official rates as of March 2026
# ---------------------------------------------------------------------------
_BEDROCK_PRICES_BY_SUFFIX: Dict[str, Tuple[float, float]] = {
    "58ii6j0n0zhw": (0.25, 1.25),     # Claude 3 Haiku
    "4ax1twcuwbfk": (1.00, 5.00),     # Claude Haiku 4.5
    "vqhud2pxz4wy": (5.00, 25.00),    # Claude Opus 4.6
    "fkpdj71utboq": (0.07, 0.30),     # gpt-oss-20b
    "d9uiuyipu5b2": (0.15, 0.60),     # gpt-oss-120b
    "nrqbxznvrt7p": (0.60, 3.00),     # Kimi K2.5
    "uj2ujdo7k1qe": (0.15, 0.15),     # Ministral 3 8B
    "d6kuf8xcphsl": (0.15, 0.60),     # Qwen3 32B
    "a6jppcyeu4ms": (0.15, 1.20),     # Qwen3 Next 80B A3B
}


def _compute_sample_cost(input_tokens: Dict[str, int],
                         output_tokens: Dict[str, int]) -> float:
    """Compute cost in USD from token dicts using Bedrock pricing."""
    total = 0.0
    for key, count in input_tokens.items():
        suffix = key.rsplit("/", 1)[-1] if "/" in key else key
        prices = _BEDROCK_PRICES_BY_SUFFIX.get(suffix)
        if prices:
            total += count * prices[0] / 1_000_000
    for key, count in output_tokens.items():
        suffix = key.rsplit("/", 1)[-1] if "/" in key else key
        prices = _BEDROCK_PRICES_BY_SUFFIX.get(suffix)
        if prices:
            total += count * prices[1] / 1_000_000
    return total


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@dataclass
class SampleResult:
    score: float
    latency_seconds: float
    input_tokens: Dict[str, int]
    output_tokens: Dict[str, int]
    cost: float = 0.0


# Historical pickles record this class as ``offline_selector_sim_v2.SampleResult``.
sys.modules.setdefault("offline_selector_sim_v2", sys.modules[__name__])

# Lookup: model_name -> {datapoint_index -> SampleResult}
LookupTable = Dict[str, Dict[int, SampleResult]]


def load_jsonl(path: str) -> Tuple[List[str], List[int], LookupTable]:
    """Load brute-force JSONL into a lookup table.

    Returns (model_names_sorted, datapoint_indices_sorted, table).
    """
    table: LookupTable = {}
    models: Set[str] = set()
    datapoints: Set[int] = set()

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            combo = json.loads(line)
            model = combo["model_name"]
            models.add(model)

            for dp in combo.get("datapoint_results", []):
                dp_idx = dp["datapoint_index"]
                score = dp["score"]
                latency = dp["latency_seconds"]
                input_tokens = dp.get("input_tokens", {})
                output_tokens = dp.get("output_tokens", {})
                cost = _compute_sample_cost(input_tokens, output_tokens)

                if model not in table:
                    table[model] = {}
                table[model][dp_idx] = SampleResult(
                    score=score,
                    latency_seconds=latency,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost=cost,
                )
                datapoints.add(dp_idx)

    return sorted(models), sorted(datapoints), table


def load_pickle(path: str) -> Tuple[List[str], List[int], LookupTable]:
    """Load from pickle lookup table under ``data/lookup/``."""
    import pickle

    with open(path, "rb") as f:
        data = pickle.load(f)

    models = data["model_names"]
    datapoints = data["datapoints"]
    raw_table = data["table"]

    table: LookupTable = {}
    for model_name, dp_dict in raw_table.items():
        table[model_name] = {}
        for dp_idx, sr in dp_dict.items():
            table[model_name][dp_idx] = SampleResult(
                score=sr.score,
                latency_seconds=sr.latency_seconds,
                input_tokens=getattr(sr, "input_tokens", {}) or {},
                output_tokens=getattr(sr, "output_tokens", {}) or {},
                cost=getattr(sr, "cost", 0.0) or 0.0,
            )

    return models, datapoints, table


def load_scope(path: str) -> Tuple[List[str], List[int], LookupTable]:
    """Load a SCOPE benchmark directory from paired accuracy/cost matrices.

    The directory must contain ``accuracy_matrix.csv`` and
    ``cost_matrix_usd.csv``. Both files use rows as workflow configurations,
    columns named ``question_<id>``, and a leading ``model_name`` column.
    """
    scope_dir = Path(path)
    accuracy_path = scope_dir / "accuracy_matrix.csv"
    cost_path = scope_dir / "cost_matrix_usd.csv"
    for matrix_path in (accuracy_path, cost_path):
        if not matrix_path.is_file():
            raise ValueError(f"missing SCOPE matrix: {matrix_path}")

    def read_matrix(matrix_path: Path) -> Tuple[List[str], List[int], Dict[str, List[float]]]:
        with matrix_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            try:
                header = next(reader)
            except StopIteration as exc:
                raise ValueError(f"empty SCOPE matrix: {matrix_path}") from exc
            if not header or header[0] != "model_name":
                raise ValueError(
                    f"{matrix_path} must start with a model_name column"
                )
            try:
                question_ids = [
                    int(column.removeprefix("question_"))
                    for column in header[1:]
                    if column.startswith("question_")
                ]
            except ValueError as exc:
                raise ValueError(
                    f"invalid question column in {matrix_path}"
                ) from exc
            if len(question_ids) != len(header) - 1 or not question_ids:
                raise ValueError(
                    f"{matrix_path} requires question_<id> columns"
                )
            if len(set(question_ids)) != len(question_ids):
                raise ValueError(f"duplicate question IDs in {matrix_path}")

            models: List[str] = []
            values: Dict[str, List[float]] = {}
            for line_number, row in enumerate(reader, start=2):
                if len(row) != len(header):
                    raise ValueError(
                        f"{matrix_path}:{line_number} has {len(row)} columns; "
                        f"expected {len(header)}"
                    )
                model_name = row[0]
                if not model_name or model_name in values:
                    raise ValueError(
                        f"missing or duplicate model_name at "
                        f"{matrix_path}:{line_number}"
                    )
                try:
                    parsed = [float(value) for value in row[1:]]
                except ValueError as exc:
                    raise ValueError(
                        f"non-numeric value at {matrix_path}:{line_number}"
                    ) from exc
                if not all(math.isfinite(value) for value in parsed):
                    raise ValueError(
                        f"non-finite value at {matrix_path}:{line_number}"
                    )
                models.append(model_name)
                values[model_name] = parsed
        if not models:
            raise ValueError(f"SCOPE matrix has no configurations: {matrix_path}")
        return models, question_ids, values

    models, datapoints, accuracies = read_matrix(accuracy_path)
    cost_models, cost_datapoints, costs = read_matrix(cost_path)
    if cost_models != models:
        raise ValueError("SCOPE accuracy and cost matrices have different model rows")
    if cost_datapoints != datapoints:
        raise ValueError("SCOPE accuracy and cost matrices have different questions")

    table: LookupTable = {}
    for model_name in models:
        table[model_name] = {}
        for column, question_id in enumerate(datapoints):
            score = accuracies[model_name][column]
            cost = costs[model_name][column]
            if not 0.0 <= score <= 1.0:
                raise ValueError(
                    f"accuracy outside [0, 1] for {model_name}, question {question_id}"
                )
            if cost < 0.0:
                raise ValueError(
                    f"negative cost for {model_name}, question {question_id}"
                )
            table[model_name][question_id] = SampleResult(
                score=score,
                latency_seconds=0.0,
                input_tokens={},
                output_tokens={},
                cost=cost,
            )
    return models, datapoints, table


def _require_data_path(path: str) -> str:
    """Fail clearly when gitignored benchmark data is missing locally."""
    p = Path(path)
    if p.is_file():
        return str(p)
    hint = (
        f"Data file not found: {path}\n"
        f"Expected local pickles in {_DEFAULT_PICKLE_DIR}/ "
        f"(gitignored — not uploaded). Place "
        f"{{gpqa,bfcl,hotpotqa,mathqa}}_lookup.pkl there and retry."
    )
    raise SystemExit(hint)


# ---------------------------------------------------------------------------
# Stats helpers (matching base.py exactly)
# ---------------------------------------------------------------------------

def _compute_stats(scores: List[float]) -> Tuple[float, float]:
    """Return (mean, sample_std). Matches BaseModelSelector._compute_stats."""
    n = len(scores)
    if n == 0:
        return 0.0, 0.5
    mean = sum(scores) / n
    if n < 2:
        return mean, 0.5
    variance = sum((s - mean) ** 2 for s in scores) / (n - 1)
    return mean, math.sqrt(variance)


@dataclass
class ModelSummary:
    model_name: str
    accuracy: float
    latency_seconds: float
    cost: float
    n_samples_evaluated: int
    is_best: bool = False


@dataclass
class SimulationResult:
    selector: str
    seed: int
    params: Dict[str, Any]
    best_model: Optional[str]
    best_accuracy: float
    total_evaluations: int
    total_cost: float
    n_models_evaluated: int
    found_true_best: bool
    compute_time_seconds: float
    model_results: List[ModelSummary] = field(default_factory=list)


def compute_ground_truth(models: List[str], datapoints: List[int],
                         table: LookupTable) -> Tuple[str, float]:
    """Compute brute-force best model (accuracy, tiebreak latency then cost)."""
    best_name = None
    best_acc = float("-inf")
    best_lat = float("inf")
    best_cost = float("inf")
    tol = 1e-9

    for model in models:
        samples = table.get(model, {})
        available = [samples[dp] for dp in datapoints if dp in samples]
        if not available:
            continue
        acc = sum(s.score for s in available) / len(available)
        lat = sum(s.latency_seconds for s in available) / len(available)
        cost = sum(s.cost for s in available)

        if acc > best_acc + tol:
            best_name, best_acc, best_lat, best_cost = model, acc, lat, cost
        elif abs(acc - best_acc) <= tol and lat < best_lat - tol:
            best_name, best_acc, best_lat, best_cost = model, acc, lat, cost
        elif (abs(acc - best_acc) <= tol and abs(lat - best_lat) <= tol
              and cost < best_cost - tol):
            best_name, best_acc, best_lat, best_cost = model, acc, lat, cost

    return best_name, best_acc
def _evaluate_model_full(idx: int, models: List[str], datapoints: List[int],
                         table: LookupTable) -> Tuple[float, float, float, int, float]:
    """Evaluate a model on all datapoints. Returns (acc, lat, cost, n_eval, compute_time)."""
    model = models[idx]
    samples = table.get(model, {})
    available = [samples[dp] for dp in datapoints if dp in samples]
    n_eval = len(available)
    acc = sum(s.score for s in available) / n_eval if n_eval else 0.0
    lat = sum(s.latency_seconds for s in available) / n_eval if n_eval else 0.0
    cost = sum(s.cost for s in available)
    ct = sum(s.latency_seconds for s in available)
    return acc, lat, cost, n_eval, ct


# ---------------------------------------------------------------------------
# Selector: Random Search
# ---------------------------------------------------------------------------

def simulate_random_search(
    models: List[str], datapoints: List[int], table: LookupTable,
    sample_fraction: float = 0.25, seed: int = 42,
) -> SimulationResult:
    n_total = len(models)
    sample_size = max(1, min(n_total, math.ceil(n_total * sample_fraction)))

    rng = random.Random(seed)
    sampled = sorted(rng.sample(range(n_total), sample_size)) if sample_size < n_total else list(range(n_total))

    total_evals = 0
    total_cost = 0.0
    compute_time = 0.0
    model_results = []
    best_name, best_acc, best_lat = None, float("-inf"), float("inf")
    tol = 1e-9

    for idx in sampled:
        acc, lat, cost, n_eval, ct = _evaluate_model_full(idx, models, datapoints, table)
        total_evals += n_eval
        total_cost += cost
        compute_time += ct
        model_results.append(ModelSummary(models[idx], acc, lat, cost, n_eval))
        if acc > best_acc + tol or (abs(acc - best_acc) <= tol and lat < best_lat - tol):
            best_name, best_acc, best_lat = models[idx], acc, lat

    for mr in model_results:
        mr.is_best = (mr.model_name == best_name)

    gt_name, _ = compute_ground_truth(models, datapoints, table)
    return SimulationResult("random_search", seed,
        {"sample_fraction": sample_fraction},
        best_name, best_acc, total_evals, total_cost,
        len(sampled), best_name == gt_name, compute_time, model_results)
def _check_botorch():
    try:
        import torch  # noqa: F401
        from botorch.models.gp_regression_mixed import MixedSingleTaskGP  # noqa: F401
        return True
    except ImportError:
        return False


def simulate_bayesian_optimization(
    models: List[str], datapoints: List[int], table: LookupTable,
    n_initial_random: Optional[int] = None, n_iterations: Optional[int] = None,
    seed: int = 42,
) -> SimulationResult:
    import torch
    from botorch.acquisition.analytic import LogExpectedImprovement
    from botorch.fit import fit_gpytorch_mll
    from botorch.models.gp_regression_mixed import MixedSingleTaskGP
    from gpytorch.mlls import ExactMarginalLogLikelihood

    n_models = len(models)
    if n_initial_random is None:
        n_initial_random = min(4, n_models)
    if n_iterations is None:
        n_iterations = max(0, int(0.2 * n_models))

    rng = random.Random(seed)
    evaluated: Set[int] = set()
    X_list, Y_list = [], []
    model_data: Dict[int, Tuple[float, float, float, int, float]] = {}
    total_evals = 0

    def eval_model(idx):
        nonlocal total_evals
        acc, lat, cost, n_eval, ct = _evaluate_model_full(idx, models, datapoints, table)
        model_data[idx] = (acc, lat, cost, n_eval, ct)
        total_evals += n_eval
        return acc

    # Initial random
    pool = list(range(n_models))
    rng.shuffle(pool)
    for idx in pool[:n_initial_random]:
        evaluated.add(idx)
        acc = eval_model(idx)
        X_list.append([idx])
        Y_list.append(acc)

    # BO loop
    for _ in range(n_iterations):
        unseen = [c for c in range(n_models) if c not in evaluated]
        if not unseen:
            break

        if len(X_list) < 2:
            idx = rng.choice(unseen)
        else:
            train_X = torch.tensor(X_list, dtype=torch.float64)
            train_Y = torch.tensor(Y_list, dtype=torch.float64).unsqueeze(-1)
            model_gp = MixedSingleTaskGP(train_X=train_X, train_Y=train_Y, cat_dims=[0])
            mll = ExactMarginalLogLikelihood(model_gp.likelihood, model_gp)
            fit_gpytorch_mll(mll)
            cand_X = torch.tensor([[c] for c in unseen], dtype=torch.float64)
            acq = LogExpectedImprovement(model=model_gp, best_f=train_Y.max().item())
            with torch.no_grad():
                ei = acq(cand_X.unsqueeze(1))
            idx = unseen[ei.argmax().item()]

        evaluated.add(idx)
        acc = eval_model(idx)
        X_list.append([idx])
        Y_list.append(acc)

    model_results = []
    best_name, best_acc, best_lat = None, float("-inf"), float("inf")
    tol = 1e-9
    for idx in range(n_models):
        if idx in model_data:
            acc, lat, cost, n_eval, ct = model_data[idx]
        else:
            acc, lat, cost, n_eval = 0.0, 0.0, 0.0, 0
        model_results.append(ModelSummary(models[idx], acc, lat, cost, n_eval))
        if idx in evaluated:
            if acc > best_acc + tol or (abs(acc - best_acc) <= tol and lat < best_lat - tol):
                best_name, best_acc, best_lat = models[idx], acc, lat

    for mr in model_results:
        mr.is_best = (mr.model_name == best_name)

    gt_name, _ = compute_ground_truth(models, datapoints, table)
    total_cost = sum(d[2] for d in model_data.values())
    compute_time = sum(d[4] for d in model_data.values())

    return SimulationResult("bayesian_optimization", seed,
        {"n_initial_random": n_initial_random, "n_iterations": n_iterations},
        best_name, best_acc, total_evals, total_cost,
        len(evaluated), best_name == gt_name, compute_time, model_results)


def run_multi_seed(selector_fn, models, datapoints, table,
                   n_seeds, base_seed=0, **kwargs):
    results = []
    for i in range(n_seeds):
        result = selector_fn(models, datapoints, table, seed=base_seed + i, **kwargs)
        results.append(result)
    return results


def summarize_multi_seed(results, gt_name, gt_acc, models=None,
                         datapoints=None, table=None):
    if models is not None and datapoints is not None and table is not None:
        true_accs = {}
        for model in models:
            samples = table.get(model, {})
            available = [samples[dp] for dp in datapoints if dp in samples]
            true_accs[model] = sum(s.score for s in available) / len(available) if available else 0.0
        accuracies = [true_accs.get(r.best_model, 0.0) for r in results]
    else:
        accuracies = [r.best_accuracy for r in results]

    evals = [r.total_evaluations for r in results]
    costs = [r.total_cost for r in results]
    found_best = [r.found_true_best for r in results]
    compute_times = [r.compute_time_seconds for r in results]
    n = len(results)
    mean_acc = sum(accuracies) / n
    std_acc = (sum((a - mean_acc)**2 for a in accuracies) / max(n - 1, 1)) ** 0.5

    return {
        "selector": results[0].selector,
        "n_seeds": n,
        "ground_truth_best": gt_name,
        "ground_truth_accuracy": gt_acc,
        "mean_accuracy": mean_acc,
        "std_accuracy": std_acc,
        "min_accuracy": min(accuracies),
        "max_accuracy": max(accuracies),
        "found_true_best_pct": sum(found_best) / n * 100,
        "mean_evaluations": sum(evals) / n,
        "mean_cost": sum(costs) / n,
        "mean_compute_time": sum(compute_times) / n,
        "params": results[0].params,
    }


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def print_single_result(result, gt_name, gt_acc):
    print(f"\n{'─'*60}")
    print(f"  {result.selector}  (seed={result.seed})")
    print(f"  Params: {result.params}")
    print(f"{'─'*60}")
    print(f"  {'Model':<50} {'Acc':>7} {'Samples':>8} {'Cost':>10}")
    print(f"  {'─'*50} {'─'*7} {'─'*8} {'─'*10}")
    for mr in sorted(result.model_results, key=lambda x: (-x.accuracy, x.latency_seconds)):
        marker = " *" if mr.is_best else ""
        print(f"  {mr.model_name:<50} {mr.accuracy:>7.3f} {mr.n_samples_evaluated:>8}"
              f" ${mr.cost:>8.4f}{marker}")
    print(f"\n  Best: {result.best_model}  ({result.best_accuracy:.3f})")
    print(f"  Total evaluations: {result.total_evaluations}")
    print(f"  Total cost: ${result.total_cost:.4f}")
    print(f"  Total wall time: {result.compute_time_seconds:.3f}s")
    print(f"  Found true best ({gt_name}): {'YES' if result.found_true_best else 'NO'}")


def print_summary(summary):
    print(f"\n{'='*60}")
    print(f"  {summary['selector']}  —  {summary['n_seeds']} seeds")
    print(f"  Params: {summary['params']}")
    print(f"{'='*60}")
    print(f"  Ground truth best: {summary['ground_truth_best']}"
          f"  ({summary['ground_truth_accuracy']:.3f})")
    print(f"  Mean accuracy found:  {summary['mean_accuracy']:.4f}"
          f"  +/- {summary['std_accuracy']:.4f}")
    print(f"  Accuracy range:       [{summary['min_accuracy']:.4f},"
          f" {summary['max_accuracy']:.4f}]")
    print(f"  Found true best:      {summary['found_true_best_pct']:.1f}%"
          f"  ({int(summary['found_true_best_pct'] * summary['n_seeds'] / 100)}"
          f"/{summary['n_seeds']})")
    print(f"  Mean evaluations:     {summary['mean_evaluations']:.0f}")
    print(f"  Mean total cost:      ${summary['mean_cost']:.4f}")


# ---------------------------------------------------------------------------
# Selector: Matrix UCB (plain)
# ---------------------------------------------------------------------------

def simulate_matrix_ucb(
    models: List[str], datapoints: List[int], table: LookupTable,
    a: float = 1.0, observation_budget_fraction: float = 1.0,
    max_original_cost: Optional[float] = None,
    accounting_cost_per_arm: Optional[Sequence[float]] = None,
    seed: int = 42,
    history: Optional[List[Dict[str, Any]]] = None,
) -> SimulationResult:
    """Offline simulation of MatrixUCBModelSelector.

    Replays the UCB cell-selection logic from matrix_ucb.py using the lookup
    table instead of live API calls. Each step picks the combo with the highest
    UCB (mean + sqrt(a/count)), then evaluates a batch of unseen datapoints
    for that combo.

    observation_budget_fraction: fraction of available cells to observe before
    stopping (default 1.0 = full grid = same as brute force).
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    n_combos = len(models)
    n_dp = len(datapoints)
    max_cells = 20  # match default max_concurrent

    # Build availability mask: True if cell exists in lookup table
    available = np.zeros((n_combos, n_dp), dtype=bool)
    for i, model_name in enumerate(models):
        model_data = table.get(model_name, {})
        for j, dp_id in enumerate(datapoints):
            if dp_id in model_data:
                available[i, j] = True
    n_available = int(available.sum())
    true_means = np.asarray([
        np.mean([table[m][dp].score for dp in datapoints if dp in table.get(m, {})])
        if table.get(m) else float("nan")
        for m in models
    ], dtype=np.float64)
    mu_star = float(np.nanmax(true_means))

    # Budget: stop after observing this many cells
    budget = (
        n_available if max_original_cost is not None else
        max(1, int(math.ceil(observation_budget_fraction * n_available)))
        if observation_budget_fraction < 1.0 else n_available
    )
    accounting_costs = (
        np.asarray(accounting_cost_per_arm, dtype=np.float64)
        if accounting_cost_per_arm is not None else None
    )
    if accounting_costs is not None and accounting_costs.shape != (n_combos,):
        raise ValueError(f"accounting_cost_per_arm must have shape ({n_combos},)")

    # observed tracks scores; unavailable cells use -inf sentinel (excluded from UCB)
    observed = np.full((n_combos, n_dp), np.nan, dtype=np.float64)
    cell_costs: Dict[Tuple[int, int], float] = {}
    total_evals = 0
    total_cost = 0.0

    # Per-combo count of available datapoints (for "fully observed" check)
    available_per_combo = available.sum(axis=1)

    while True:
        filled = int(np.sum(~np.isnan(observed)))
        if filled >= budget:
            break
        if max_original_cost is not None and total_cost >= float(max_original_cost):
            break

        # UCB selection (same as _ucb_plain_next_batch)
        bounds = np.full(n_combos, np.inf, dtype=np.float64)
        with np.errstate(invalid="ignore"):
            mus = np.nanmean(observed, axis=1)
        counts = np.sum(~np.isnan(observed), axis=1)
        mask = counts > 0
        bounds[mask] = mus[mask] + np.sqrt(a / counts[mask])
        fully_observed = counts >= available_per_combo
        bounds[fully_observed] = -np.inf
        if bool(np.all(fully_observed)):
            break

        best_combo = int(np.argmax(bounds))
        # Only pick from available AND unobserved cells
        unobserved_dp = np.where(np.isnan(observed[best_combo]) & available[best_combo])[0]
        if len(unobserved_dp) == 0:
            break

        remaining = budget - total_evals
        k = min(max_cells, len(unobserved_dp), remaining)
        if k <= 0:
            break
        pick = rng.permutation(len(unobserved_dp))[:k]
        dp_indices = unobserved_dp[pick]

        model_name = models[best_combo]
        model_data = table.get(model_name, {})
        for dp_local_idx in dp_indices:
            if total_evals >= budget:
                break
            dp_id = datapoints[dp_local_idx]
            sr = model_data[dp_id]
            observed[best_combo, dp_local_idx] = sr.score
            cell_costs[(best_combo, dp_local_idx)] = sr.cost
            total_evals += 1
            total_cost += (
                float(accounting_costs[best_combo])
                if accounting_costs is not None else sr.cost
            )

        if history is not None:
            empirical_means = np.nanmean(observed, axis=1)
            scores = np.where(np.isnan(empirical_means), -np.inf, empirical_means)
            recommended_arm = int(np.argmax(scores))
            history.append({
                "cum_eval": total_evals,
                "cum_cost": total_cost,
                "recommended_arm": recommended_arm,
                "recommended_model": models[recommended_arm],
                "simple_regret": mu_star - float(true_means[recommended_arm]),
            })

    # Find best model from observed data
    best_name = None
    best_acc = float("-inf")
    best_lat = float("inf")
    tol = 1e-9
    model_results = []
    for i, model_name in enumerate(models):
        valid = ~np.isnan(observed[i])
        if not np.any(valid):
            continue
        acc = float(np.nanmean(observed[i]))
        model_data = table.get(model_name, {})
        lats = [model_data[datapoints[j]].latency_seconds
                for j in range(n_dp) if not np.isnan(observed[i, j]) and datapoints[j] in model_data]
        lat = sum(lats) / len(lats) if lats else 0.0
        cost_i = sum(cell_costs.get((i, j), 0.0) for j in range(n_dp))
        n_eval_i = int(np.sum(valid))
        model_results.append(ModelSummary(model_name, acc, lat, cost_i, n_eval_i))
        if acc > best_acc + tol or (abs(acc - best_acc) <= tol and lat < best_lat - tol):
            best_name, best_acc, best_lat = model_name, acc, lat

    for mr in model_results:
        mr.is_best = (mr.model_name == best_name)

    # Compute TRUE full-dataset accuracy of the selected combo
    # (not the partial-observation estimate which is biased at low budgets)
    if best_name is not None:
        model_data = table.get(best_name, {})
        all_scores = [model_data[dp].score for dp in datapoints if dp in model_data]
        true_best_acc = sum(all_scores) / len(all_scores) if all_scores else 0.0
    else:
        true_best_acc = 0.0

    gt_name, _ = compute_ground_truth(models, datapoints, table)
    return SimulationResult(
        "matrix_ucb", seed, {"a": a, "observation_budget_fraction": observation_budget_fraction},
        best_name, true_best_acc, total_evals, total_cost,
        len(model_results), best_name == gt_name, 0.0, model_results,
    )


def simulate_gittins(
    models: List[str], datapoints: List[int], table: LookupTable,
    observation_budget_fraction: float = 1.0,
    batch_size: int = 20,
    prior_mean: float = 0.5,
    prior_variance: float = 0.04,
    obs_noise_variance: Optional[float] = None,
    cost_per_transition: float | Sequence[float] = 1.0,
    cost_scaling_factor: float = 1e-4,
    max_original_cost: Optional[float] = None,
    accounting_cost_per_arm: Optional[Sequence[float]] = None,
    allow_early_stop: bool = False,
    seed: int = 42,
    history: Optional[List[Dict[str, Any]]] = None,
    run_metadata: Optional[Dict[str, Any]] = None,
) -> SimulationResult:
    """Offline simulation of GittinsModelSelector on frozen lookup tables."""
    wall_t0 = time.perf_counter()
    import torch
    from agentopt.model_selection.gittins_lookup import compute_roots_lookup_table
    from agentopt.model_selection.gittins_policy import (
        gittins_index_exploration,
        gittins_post_pull_update,
    )
    from agentopt.model_selection.gittins_shrinking_posterior import (
        transition_stds_shrinking_gaussian_posterior,
    )
    import jax
    import jax.numpy as jnp
    import numpy as np

    torch.manual_seed(seed)
    np.random.seed(seed)

    n_combos = len(models)
    n_dp = len(datapoints)
    bsz = max(int(batch_size), 1)
    tau_sq_batch = (
        float(obs_noise_variance)
        if obs_noise_variance is not None
        else 1.0 / (4.0 * bsz)
    )
    tau_sq_cell = tau_sq_batch * float(bsz)

    available = np.zeros((n_combos, n_dp), dtype=bool)
    for i, model_name in enumerate(models):
        model_data = table.get(model_name, {})
        for j, dp_id in enumerate(datapoints):
            if dp_id in model_data:
                available[i, j] = True
    n_available = int(available.sum())
    true_means = np.asarray([
        np.mean([table[m][dp].score for dp in datapoints if dp in table.get(m, {})])
        if table.get(m) else float("nan")
        for m in models
    ], dtype=np.float64)
    mu_star = float(np.nanmax(true_means))
    budget = (
        n_available if max_original_cost is not None else
        max(1, int(math.ceil(observation_budget_fraction * n_available)))
        if observation_budget_fraction < 1.0 else n_available
    )
    if isinstance(cost_per_transition, (int, float)):
        decision_costs = np.full((n_combos,), float(cost_per_transition), dtype=np.float64)
    else:
        decision_costs = np.asarray(list(cost_per_transition), dtype=np.float64)
        if decision_costs.shape != (n_combos,):
            raise ValueError(f"cost_per_transition must have shape ({n_combos},)")
    accounting_costs = (
        np.asarray(accounting_cost_per_arm, dtype=np.float64)
        if accounting_cost_per_arm is not None else None
    )
    if accounting_costs is not None and accounting_costs.shape != (n_combos,):
        raise ValueError(f"accounting_cost_per_arm must have shape ({n_combos},)")

    # Missing and not-yet-observed cells both stay NaN.  ``available_t`` is the
    # sole authority for whether a cell exists, matching Matrix UCB semantics.
    observed_np = np.full((n_combos, n_dp), np.nan, dtype=np.float64)
    observed_t = torch.full((n_combos, n_dp), float("nan"), dtype=torch.float32)
    available_t = torch.from_numpy(available.copy()).to(torch.bool)

    cell_costs: Dict[Tuple[int, int], float] = {}
    total_evals = 0
    total_cost = 0.0
    cached_scores = torch.full((n_combos,), float("inf"), dtype=torch.float32)

    # A ragged response matrix gives each arm a different finite horizon. Build
    # one lookup row per arm; padding is never indexed because counts cannot
    # exceed that arm's number of available cells.
    horizons = available.sum(axis=1).astype(int)
    cache_key = (
        float(prior_variance), float(tau_sq_cell), int(n_dp),
        tuple(int(x) for x in horizons),
        tuple(float(x) for x in decision_costs), float(cost_scaling_factor),
    )
    cached_roots = _GITTINS_ROOTS_CACHE.get(cache_key)
    cache_digest = hashlib.sha256(repr(cache_key).encode("utf-8")).hexdigest()
    roots_cache_path = _GITTINS_ROOTS_DISK_CACHE / f"{cache_digest}.npy"
    if cached_roots is None and roots_cache_path.exists():
        cached_roots = torch.from_numpy(np.load(roots_cache_path)).to(torch.float32)
        _GITTINS_ROOTS_CACHE[cache_key] = cached_roots.clone()
    if cached_roots is not None:
        roots_lookup_table = cached_roots.clone()
    else:
        roots_lookup_table = torch.zeros((n_combos, n_dp + 1), dtype=torch.float32)
        for horizon_np in np.unique(horizons):
            horizon = int(horizon_np)
            arm_indices = np.flatnonzero(horizons == horizon)
            transition_stds = transition_stds_shrinking_gaussian_posterior(
                jnp.float32(prior_variance), jnp.float32(tau_sq_cell), horizon,
            )
            roots = compute_roots_lookup_table(
                transition_stds=transition_stds,
                costs_per_arm=jnp.asarray(
                    decision_costs[arm_indices] * float(cost_scaling_factor),
                    dtype=jnp.float32,
                ),
                n_points=2**10 + 1,
            )
            roots_group = torch.from_numpy(
                np.array(jax.device_get(roots), copy=True)
            ).to(torch.float32)
            if roots_group.ndim == 1:
                roots_group = roots_group.unsqueeze(0)
            for row, arm_idx in enumerate(arm_indices):
                roots_lookup_table[int(arm_idx), :horizon + 1] = roots_group[row]
        _GITTINS_ROOTS_CACHE[cache_key] = roots_lookup_table.clone()
        _GITTINS_ROOTS_DISK_CACHE.mkdir(parents=True, exist_ok=True)
        np.save(roots_cache_path, roots_lookup_table.cpu().numpy())

    last_pulled: Optional[List[int]] = None
    natural_stop: List[Optional[int]] = [None]
    rec_aware_stop: List[Optional[int]] = [None]
    natural_stop_cost: Optional[float] = None
    rec_aware_stop_cost: Optional[float] = None

    while True:
        if total_evals >= budget:
            break
        if max_original_cost is not None and total_cost >= float(max_original_cost):
            break
        step_bsz = min(bsz, budget - total_evals)
        out = gittins_index_exploration(
            observed_t,
            prior_mean=prior_mean,
            prior_variance=prior_variance,
            obs_noise_variance=tau_sq_batch,
            cost_per_transition=cost_per_transition,
            cost_scaling_factor=cost_scaling_factor,
            batch_size=bsz,
            cached_scores=cached_scores,
            recompute_arms=None if last_pulled is None else last_pulled,
            allow_early_stop=False,
            roots_lookup_table=roots_lookup_table,
            batch_observation_model=True,
            availability_mask=available_t,
        )
        if out is None:
            break

        combo_idxs = out[0].tolist()
        dp_idxs = out[1].tolist()
        pulled = set()
        for ci, di in zip(combo_idxs, dp_idxs):
            if total_evals >= budget:
                break
            if not available[ci, di]:
                continue
            if not math.isnan(float(observed_t[ci, di])):
                continue
            model_name = models[ci]
            dp_id = datapoints[di]
            sr = table[model_name][dp_id]
            observed_t[ci, di] = float(sr.score)
            observed_np[ci, di] = float(sr.score)
            cell_costs[(ci, di)] = sr.cost
            total_evals += 1
            total_cost += (
                float(accounting_costs[ci])
                if accounting_costs is not None else sr.cost
            )
            pulled.add(int(ci))

        if not pulled:
            break
        last_pulled = sorted(pulled)
        natural_before = natural_stop[0]
        rec_before = rec_aware_stop[0]
        _, cached_scores = gittins_post_pull_update(
            observed_t,
            cached_scores=cached_scores,
            recompute_arms=last_pulled,
            prior_mean=prior_mean,
            prior_variance=prior_variance,
            obs_noise_variance=tau_sq_batch,
            cost_per_transition=cost_per_transition,
            cost_scaling_factor=cost_scaling_factor,
            batch_size=bsz,
            roots_lookup_table=roots_lookup_table,
            batch_observation_model=True,
            availability_mask=available_t,
            sim_cum_eval=total_evals,
            natural_stop_cum_eval_holder=natural_stop,
            recommendation_aware_stop_cum_eval_holder=rec_aware_stop,
        )
        if natural_before is None and natural_stop[0] is not None:
            natural_stop_cost = float(total_cost)
        if rec_before is None and rec_aware_stop[0] is not None:
            rec_aware_stop_cost = float(total_cost)
        if history is not None:
            observed_mask = available_t & ~observed_t.isnan()
            counts = observed_mask.sum(dim=1).to(torch.float64)
            obs_sum = torch.where(
                observed_mask, observed_t, torch.zeros_like(observed_t)
            ).sum(dim=1).to(torch.float64)
            precision = 1.0 / float(prior_variance) + counts / float(tau_sq_cell)
            posterior_means = (
                float(prior_mean) / float(prior_variance)
                + obs_sum / float(tau_sq_cell)
            ) / precision
            posterior_means[counts == 0] = float(prior_mean)
            recommended_arm = int(torch.argmax(posterior_means).item())
            history.append({
                "cum_eval": total_evals,
                "cum_cost": total_cost,
                "recommended_arm": recommended_arm,
                "recommended_model": models[recommended_arm],
                "simple_regret": mu_star - float(true_means[recommended_arm]),
            })

    best_name = None
    best_acc = float("-inf")
    best_lat = float("inf")
    tol = 1e-9
    model_results = []
    for i, model_name in enumerate(models):
        # Only count truly evaluated (available) cells
        scores = []
        for j in range(n_dp):
            if available[i, j] and not math.isnan(float(observed_np[i, j])):
                scores.append(float(observed_np[i, j]))
        if not scores:
            continue
        acc = sum(scores) / len(scores)
        model_data = table.get(model_name, {})
        lats = [
            model_data[datapoints[j]].latency_seconds
            for j in range(n_dp)
            if available[i, j] and not math.isnan(float(observed_np[i, j]))
            and datapoints[j] in model_data
        ]
        lat = sum(lats) / len(lats) if lats else 0.0
        cost_i = sum(cell_costs.get((i, j), 0.0) for j in range(n_dp))
        n_eval_i = len(scores)
        model_results.append(ModelSummary(model_name, acc, lat, cost_i, n_eval_i))
        if acc > best_acc + tol or (abs(acc - best_acc) <= tol and lat < best_lat - tol):
            best_name, best_acc, best_lat = model_name, acc, lat

    for mr in model_results:
        mr.is_best = (mr.model_name == best_name)

    if best_name is not None:
        model_data = table.get(best_name, {})
        all_scores = [model_data[dp].score for dp in datapoints if dp in model_data]
        true_best_acc = sum(all_scores) / len(all_scores) if all_scores else 0.0
    else:
        true_best_acc = 0.0

    gt_name, _ = compute_ground_truth(models, datapoints, table)
    total_wall_time_s = float(time.perf_counter() - wall_t0)
    if run_metadata is not None:
        run_metadata.update({
            "natural_stop_cum_eval": natural_stop[0],
            "natural_stop_cum_cost": natural_stop_cost,
            "recommendation_aware_stop_cum_eval": rec_aware_stop[0],
            "recommendation_aware_stop_cum_cost": rec_aware_stop_cost,
            "actual_final_cum_eval": total_evals,
            "actual_final_cum_cost": total_cost,
            "total_wall_time_s": total_wall_time_s,
            "tau_sq_batch": tau_sq_batch,
            "tau_sq_cell": tau_sq_cell,
        })
    return SimulationResult(
        "gittins", seed,
        {
            "observation_budget_fraction": observation_budget_fraction,
            "batch_size": bsz,
            "prior_mean": prior_mean,
            "prior_variance": prior_variance,
            "obs_noise_variance": tau_sq_batch,
            "tau_sq_batch": tau_sq_batch,
            "tau_sq_cell": tau_sq_cell,
            "batch_observation_model": True,
            "allow_early_stop": False,
            "max_original_cost": max_original_cost,
            "cost_per_transition": decision_costs.tolist(),
            "natural_stop_cum_eval": natural_stop[0],
            "recommendation_aware_stop_cum_eval": rec_aware_stop[0],
            "total_wall_time_s": total_wall_time_s,
        },
        best_name, true_best_acc, total_evals, total_cost,
        len(model_results), best_name == gt_name, total_wall_time_s, model_results,
    )


ALL_SELECTORS = [
    "random_search", "matrix_ucb", "gittins", "bayesian_optimization",
]


def main():
    parser = argparse.ArgumentParser(
        description="Offline selector simulator — accuracy-only replay on frozen benchmark data",
    )
    parser.add_argument("--pickle", default=None, help="Path to pickle lookup table (.pkl)")
    parser.add_argument("--jsonl", default=None, help="Path to brute-force JSONL file")
    parser.add_argument("--selectors", default="all",
                        help=f"Comma-separated: {','.join(ALL_SELECTORS)},all")
    parser.add_argument("--seeds", type=int, default=1, help="Number of random seeds (default 1)")
    parser.add_argument("--base-seed", type=int, default=42, help="Starting seed (default 42)")
    parser.add_argument("--output", default=None, help="Path to write summary CSV")

    # Selector-specific params
    parser.add_argument("--rs-fraction", type=float, default=0.25)

    parser.add_argument("--ucb-budget", type=float, default=0.2,
                        help="Matrix UCB / Gittins observation budget fraction (default 0.2)")
    parser.add_argument("--gittins-batch", type=int, default=20,
                        help="Gittins per-step example batch size (default 20)")
    parser.add_argument("--gittins-no-early-stop", action="store_true",
                        help="Deprecated compatibility flag; Gittins always runs to budget")

    args = parser.parse_args()

    if args.pickle:
        source = _require_data_path(args.pickle)
        print(f"Loading pickle: {source}")
        models, datapoints, table = load_pickle(source)
    elif args.jsonl:
        source = _require_data_path(args.jsonl)
        print(f"Loading JSONL: {source}")
        models, datapoints, table = load_jsonl(source)
    else:
        parser.error("One of --pickle or --jsonl is required")

    print(f"  Models: {len(models)}")
    print(f"  Samples: {len(datapoints)}")
    print(f"  Total entries: {sum(len(v) for v in table.values())}")

    gt_name, gt_acc = compute_ground_truth(models, datapoints, table)
    print(f"\n  Ground truth best: {gt_name}  ({gt_acc:.4f})")

    bf_evals = sum(len(v) for v in table.values())
    bf_cost = sum(s.cost for m in table.values() for s in m.values())
    print(f"  Brute force: {bf_evals} evaluations, ${bf_cost:.4f} total cost")

    if args.selectors == "all":
        selectors = list(ALL_SELECTORS)
        if not _check_botorch():
            selectors.remove("bayesian_optimization")
            print("\n  (Skipping bayesian_optimization — torch/botorch not installed)")

    else:
        selectors = [s.strip() for s in args.selectors.split(",")]

    summaries = []

    for sel in selectors:
        if sel == "random_search":
            fn = simulate_random_search
            kwargs = {"sample_fraction": args.rs_fraction}
        elif sel == "bayesian_optimization":
            if not _check_botorch():
                print(f"\n  Skipping {sel} — torch/botorch not installed")
                continue
            fn = simulate_bayesian_optimization
            kwargs = {}
        elif sel == "matrix_ucb":
            fn = simulate_matrix_ucb
            kwargs = {"observation_budget_fraction": args.ucb_budget}
        elif sel == "gittins":
            fn = simulate_gittins
            kwargs = {
                "observation_budget_fraction": args.ucb_budget,
                "batch_size": args.gittins_batch,
                "allow_early_stop": False,
            }
        else:
            print(f"\n  Unknown selector: {sel}")
            continue

        if args.seeds > 1:
            results = run_multi_seed(fn, models, datapoints, table,
                                     n_seeds=args.seeds, base_seed=args.base_seed, **kwargs)
            summary = summarize_multi_seed(results, gt_name, gt_acc, models, datapoints, table)
            print_summary(summary)
            summaries.append(summary)
        else:
            result = fn(models, datapoints, table, seed=args.base_seed, **kwargs)
            print_single_result(result, gt_name, gt_acc)
            summaries.append(summarize_multi_seed([result], gt_name, gt_acc, models, datapoints, table))

    if args.output and summaries:
        import csv as csv_mod
        fields = ["selector", "n_seeds", "ground_truth_best", "ground_truth_accuracy",
                  "mean_accuracy", "std_accuracy", "found_true_best_pct",
                  "mean_evaluations", "mean_cost", "mean_compute_time", "params"]
        with open(args.output, "w", newline="") as f:
            writer = csv_mod.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for s in summaries:
                row = {k: s[k] for k in fields}
                row["params"] = json.dumps(row["params"])
                writer.writerow(row)
        print(f"\nSummary CSV written to: {args.output}")


if __name__ == "__main__":
    main()
