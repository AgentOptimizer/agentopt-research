#!/usr/bin/env python3
"""
Offline Selector Simulator — Combined Objective
================================================
Scalarized multi-objective replay:

    objective = accuracy - lambda_cost * cost - lambda_latency * latency

Preferred input: pickle lookup tables under ``../data/lookup/``
(gitignored — keep them locally).

Usage:
    python combined_objective/offline_selector_sim.py \\
        --pickle data/lookup/gpqa_lookup.pkl \\
        --selectors all --seeds 50 \\
        --lambda-cost 0.1 --lambda-latency 0.05
"""

import argparse
import csv
import json
import math
import os
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

_EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent
_REPO_ROOT = _EXPERIMENTS_DIR.parent
_SRC_DIR = _REPO_ROOT / "src"
_DEFAULT_PICKLE_DIR = _EXPERIMENTS_DIR / "data" / "lookup"

# Expose src/agentopt; pickle alias is registered after SampleResult.
sys.path.insert(0, str(_EXPERIMENTS_DIR))
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

try:
    from dotenv import load_dotenv
    load_dotenv()
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Bedrock pricing ($/MTok) — keyed by ARN suffix (profile ID)
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

LookupTable = Dict[str, Dict[int, SampleResult]]


def load_jsonl(path: str) -> Tuple[List[str], List[int], LookupTable]:
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
    """Load from pickle lookup table under ``data/lookup/``.

    Historical pickles store ``offline_selector_sim_v2.SampleResult``;
    fields match this module's ``SampleResult``.
    """
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
# Combined objective with min-max normalization
# ---------------------------------------------------------------------------

@dataclass
class NormStats:
    """Min-max normalization stats computed once from the full dataset."""
    cost_min: float = 0.0
    cost_max: float = 1.0
    latency_min: float = 0.0
    latency_max: float = 1.0

    @property
    def cost_range(self) -> float:
        r = self.cost_max - self.cost_min
        return r if r > 1e-12 else 1.0

    @property
    def latency_range(self) -> float:
        r = self.latency_max - self.latency_min
        return r if r > 1e-12 else 1.0


def compute_norm_stats(table: LookupTable, datapoints: List[int]) -> NormStats:
    """Compute min/max cost and latency across all models and datapoints."""
    all_costs = []
    all_lats = []
    for model_data in table.values():
        for dp in datapoints:
            if dp in model_data:
                sr = model_data[dp]
                all_costs.append(sr.cost)
                all_lats.append(sr.latency_seconds)
    if not all_costs:
        return NormStats()
    return NormStats(
        cost_min=min(all_costs), cost_max=max(all_costs),
        latency_min=min(all_lats), latency_max=max(all_lats),
    )


_NORM: Optional[NormStats] = None


def set_norm_stats(table: LookupTable, datapoints: List[int]) -> NormStats:
    """Compute and set the global normalization stats. Call once after loading data."""
    global _NORM
    _NORM = compute_norm_stats(table, datapoints)
    return _NORM


def compute_sample_objective(sr: SampleResult, lambda_cost: float,
                             lambda_latency: float) -> float:
    """Compute per-sample objective with min-max normalized cost/latency.

    objective = score - λ_cost * norm_cost - λ_latency * norm_latency

    Where norm_cost = (cost - min) / (max - min), putting it in [0, 1].
    Lambdas are now interpretable: 0.1 = "10% as important as accuracy".
    Uses global _NORM (set by set_norm_stats). If not set, uses raw values.
    """
    if _NORM is None:
        return sr.score - lambda_cost * sr.cost - lambda_latency * sr.latency_seconds
    norm_cost = (sr.cost - _NORM.cost_min) / _NORM.cost_range
    norm_lat = (sr.latency_seconds - _NORM.latency_min) / _NORM.latency_range
    return sr.score - lambda_cost * norm_cost - lambda_latency * norm_lat


def compute_model_objective(model_name: str, datapoints: List[int],
                            table: LookupTable, lambda_cost: float,
                            lambda_latency: float) -> float:
    """Compute mean objective for a model across all available datapoints."""
    samples = table.get(model_name, {})
    available = [samples[dp] for dp in datapoints if dp in samples]
    if not available:
        return float("-inf")
    objectives = [compute_sample_objective(s, lambda_cost, lambda_latency) for s in available]
    return sum(objectives) / len(objectives)


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def _compute_stats(scores: List[float]) -> Tuple[float, float]:
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
    objective: float
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
    best_objective: float
    total_evaluations: int
    total_cost: float
    n_models_evaluated: int
    found_true_best: bool
    compute_time_seconds: float
    model_results: List[ModelSummary] = field(default_factory=list)


def compute_ground_truth(models: List[str], datapoints: List[int],
                         table: LookupTable, lambda_cost: float = 0.0,
                         lambda_latency: float = 0.0) -> Tuple[str, float, float]:
    """Compute brute-force best model by combined objective.

    Returns (best_name, best_accuracy, best_objective).
    """
    best_name = None
    best_obj = float("-inf")
    best_acc = 0.0
    tol = 1e-9

    for model in models:
        samples = table.get(model, {})
        available = [samples[dp] for dp in datapoints if dp in samples]
        if not available:
            continue
        acc = sum(s.score for s in available) / len(available)
        obj = sum(compute_sample_objective(s, lambda_cost, lambda_latency)
                  for s in available) / len(available)

        if obj > best_obj + tol:
            best_name, best_obj, best_acc = model, obj, acc

    return best_name, best_acc, best_obj


def _evaluate_model_full(idx: int, models: List[str], datapoints: List[int],
                         table: LookupTable, lambda_cost: float = 0.0,
                         lambda_latency: float = 0.0
                         ) -> Tuple[float, float, float, float, int, float]:
    """Evaluate a model on all datapoints.

    Returns (accuracy, objective, latency, cost, n_eval, compute_time).
    """
    model = models[idx]
    samples = table.get(model, {})
    available = [samples[dp] for dp in datapoints if dp in samples]
    n_eval = len(available)
    acc = sum(s.score for s in available) / n_eval if n_eval else 0.0
    obj = sum(compute_sample_objective(s, lambda_cost, lambda_latency)
              for s in available) / n_eval if n_eval else float("-inf")
    lat = sum(s.latency_seconds for s in available) / n_eval if n_eval else 0.0
    cost = sum(s.cost for s in available)
    ct = sum(s.latency_seconds for s in available)
    return acc, obj, lat, cost, n_eval, ct


# ---------------------------------------------------------------------------
# Selector: Brute Force (reference baseline)
# ---------------------------------------------------------------------------

def simulate_brute_force(
    models: List[str], datapoints: List[int], table: LookupTable,
    lambda_cost: float = 0.0, lambda_latency: float = 0.0,
    seed: int = 42,
) -> SimulationResult:
    """Evaluate all models on all datapoints — the reference upper bound."""
    total_evals = 0
    total_cost = 0.0
    compute_time = 0.0
    model_results = []
    best_name = None
    best_obj = float("-inf")
    best_acc = 0.0
    tol = 1e-9

    for idx in range(len(models)):
        acc, obj, lat, cost, n_eval, ct = _evaluate_model_full(
            idx, models, datapoints, table, lambda_cost, lambda_latency)
        total_evals += n_eval
        total_cost += cost
        compute_time += ct
        model_results.append(ModelSummary(models[idx], acc, obj, lat, cost, n_eval))
        if obj > best_obj + tol:
            best_name, best_obj, best_acc = models[idx], obj, acc

    for mr in model_results:
        mr.is_best = (mr.model_name == best_name)

    return SimulationResult("brute_force", seed,
        {"lambda_cost": lambda_cost, "lambda_latency": lambda_latency},
        best_name, best_acc, best_obj, total_evals, total_cost,
        len(models), True, compute_time, model_results)


# ---------------------------------------------------------------------------
# Selector: Random Search
# ---------------------------------------------------------------------------

def simulate_random_search(
    models: List[str], datapoints: List[int], table: LookupTable,
    lambda_cost: float = 0.0, lambda_latency: float = 0.0,
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
    best_name = None
    best_obj = float("-inf")
    best_acc = 0.0
    tol = 1e-9

    for idx in sampled:
        acc, obj, lat, cost, n_eval, ct = _evaluate_model_full(
            idx, models, datapoints, table, lambda_cost, lambda_latency)
        total_evals += n_eval
        total_cost += cost
        compute_time += ct
        model_results.append(ModelSummary(models[idx], acc, obj, lat, cost, n_eval))
        if obj > best_obj + tol:
            best_name, best_obj, best_acc = models[idx], obj, acc

    for mr in model_results:
        mr.is_best = (mr.model_name == best_name)

    gt_name, _, _ = compute_ground_truth(models, datapoints, table, lambda_cost, lambda_latency)
    return SimulationResult("random_search", seed,
        {"sample_fraction": sample_fraction,
         "lambda_cost": lambda_cost, "lambda_latency": lambda_latency},
        best_name, best_acc, best_obj, total_evals, total_cost,
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
    lambda_cost: float = 0.0, lambda_latency: float = 0.0,
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
    model_data: Dict[int, Tuple[float, float, float, float, int, float]] = {}
    total_evals = 0

    def eval_model(idx):
        nonlocal total_evals
        acc, obj, lat, cost, n_eval, ct = _evaluate_model_full(
            idx, models, datapoints, table, lambda_cost, lambda_latency)
        model_data[idx] = (acc, obj, lat, cost, n_eval, ct)
        total_evals += n_eval
        return obj

    pool = list(range(n_models))
    rng.shuffle(pool)
    for idx in pool[:n_initial_random]:
        evaluated.add(idx)
        obj = eval_model(idx)
        X_list.append([idx])
        Y_list.append(obj)

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
        obj = eval_model(idx)
        X_list.append([idx])
        Y_list.append(obj)

    model_results = []
    best_name = None
    best_obj = float("-inf")
    best_acc = 0.0
    tol = 1e-9
    for idx in range(n_models):
        if idx in model_data:
            acc, obj, lat, cost, n_eval, ct = model_data[idx]
        else:
            acc, obj, lat, cost, n_eval = 0.0, float("-inf"), 0.0, 0.0, 0
        model_results.append(ModelSummary(models[idx], acc, obj, lat, cost, n_eval))
        if idx in evaluated:
            if obj > best_obj + tol:
                best_name, best_obj, best_acc = models[idx], obj, acc

    for mr in model_results:
        mr.is_best = (mr.model_name == best_name)

    gt_name, _, _ = compute_ground_truth(models, datapoints, table, lambda_cost, lambda_latency)
    total_cost = sum(d[3] for d in model_data.values())
    compute_time = sum(d[5] for d in model_data.values())

    return SimulationResult("bayesian_optimization", seed,
        {"n_initial_random": n_initial_random, "n_iterations": n_iterations,
         "lambda_cost": lambda_cost, "lambda_latency": lambda_latency},
        best_name, best_acc, best_obj, total_evals, total_cost,
        len(evaluated), best_name == gt_name, compute_time, model_results)


def simulate_matrix_ucb(
    models: List[str], datapoints: List[int], table: LookupTable,
    lambda_cost: float = 0.0, lambda_latency: float = 0.0,
    a: float = 1.0, observation_budget_fraction: float = 1.0,
    seed: int = 42,
) -> SimulationResult:
    import numpy as np
    rng = np.random.default_rng(seed)
    n_combos = len(models)
    n_dp = len(datapoints)
    max_cells = 20

    available = np.zeros((n_combos, n_dp), dtype=bool)
    for i, model_name in enumerate(models):
        model_data = table.get(model_name, {})
        for j, dp_id in enumerate(datapoints):
            if dp_id in model_data:
                available[i, j] = True
    n_available = int(available.sum())

    budget = max(1, int(math.ceil(observation_budget_fraction * n_available))) if observation_budget_fraction < 1.0 else n_available

    observed = np.full((n_combos, n_dp), np.nan, dtype=np.float64)
    cell_costs: Dict[Tuple[int, int], float] = {}
    total_evals = 0
    total_cost = 0.0

    available_per_combo = available.sum(axis=1)

    while True:
        filled = int(np.sum(~np.isnan(observed)))
        if filled >= budget:
            break

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
            obj = compute_sample_objective(sr, lambda_cost, lambda_latency)
            observed[best_combo, dp_local_idx] = obj
            cell_costs[(best_combo, dp_local_idx)] = sr.cost
            total_evals += 1
            total_cost += sr.cost

    best_name = None
    best_obj = float("-inf")
    best_acc = 0.0
    tol = 1e-9
    model_results = []
    for i, model_name in enumerate(models):
        valid = ~np.isnan(observed[i])
        if not np.any(valid):
            continue
        obj_mean = float(np.nanmean(observed[i]))
        model_data_i = table.get(model_name, {})
        lats = [model_data_i[datapoints[j]].latency_seconds
                for j in range(n_dp) if not np.isnan(observed[i, j]) and datapoints[j] in model_data_i]
        lat = sum(lats) / len(lats) if lats else 0.0
        scores = [model_data_i[datapoints[j]].score
                  for j in range(n_dp) if not np.isnan(observed[i, j]) and datapoints[j] in model_data_i]
        acc = sum(scores) / len(scores) if scores else 0.0
        cost_i = sum(cell_costs.get((i, j), 0.0) for j in range(n_dp))
        n_eval_i = int(np.sum(valid))
        model_results.append(ModelSummary(model_name, acc, obj_mean, lat, cost_i, n_eval_i))
        if obj_mean > best_obj + tol:
            best_name, best_obj, best_acc = model_name, obj_mean, acc

    for mr in model_results:
        mr.is_best = (mr.model_name == best_name)

    # True full-dataset objective of selected combo
    if best_name is not None:
        true_best_obj = compute_model_objective(
            best_name, datapoints, table, lambda_cost, lambda_latency)
        model_data_best = table.get(best_name, {})
        all_scores = [model_data_best[dp].score for dp in datapoints if dp in model_data_best]
        true_best_acc = sum(all_scores) / len(all_scores) if all_scores else 0.0
    else:
        true_best_obj = float("-inf")
        true_best_acc = 0.0

    gt_name, _, _ = compute_ground_truth(models, datapoints, table, lambda_cost, lambda_latency)
    return SimulationResult(
        "matrix_ucb", seed,
        {"a": a, "observation_budget_fraction": observation_budget_fraction,
         "lambda_cost": lambda_cost, "lambda_latency": lambda_latency},
        best_name, true_best_acc, true_best_obj, total_evals, total_cost,
        len(model_results), best_name == gt_name, 0.0, model_results,
    )


def simulate_gittins(
    models: List[str], datapoints: List[int], table: LookupTable,
    lambda_cost: float = 0.0, lambda_latency: float = 0.0,
    observation_budget_fraction: float = 1.0,
    batch_size: int = 20,
    prior_mean: float = 0.5,
    prior_variance: float = 0.04,
    obs_noise_variance: Optional[float] = None,
    cost_per_transition: float = 1.0,
    cost_scaling_factor: float = 1e-4,
    allow_early_stop: bool = True,
    seed: int = 42,
) -> SimulationResult:
    """Offline Gittins on combined-objective cell values."""
    import torch
    import numpy as np
    import jax
    import jax.numpy as jnp
    from agentopt.model_selection.gittins_lookup import compute_roots_lookup_table
    from agentopt.model_selection.gittins_policy import (
        gittins_index_exploration,
        gittins_post_pull_update,
    )
    from agentopt.model_selection.gittins_shrinking_posterior import (
        transition_stds_shrinking_gaussian_posterior,
    )

    torch.manual_seed(seed)
    np.random.seed(seed)

    n_combos = len(models)
    n_dp = len(datapoints)
    bsz = max(int(batch_size), 1)
    tau_sq = (
        float(obs_noise_variance)
        if obs_noise_variance is not None
        else 1.0 / (4.0 * bsz)
    )

    available = np.zeros((n_combos, n_dp), dtype=bool)
    for i, model_name in enumerate(models):
        model_data = table.get(model_name, {})
        for j, dp_id in enumerate(datapoints):
            if dp_id in model_data:
                available[i, j] = True
    n_available = int(available.sum())
    budget = (
        max(1, int(math.ceil(observation_budget_fraction * n_available)))
        if observation_budget_fraction < 1.0
        else n_available
    )

    observed_np = np.full((n_combos, n_dp), np.nan, dtype=np.float64)
    observed_t = torch.full((n_combos, n_dp), float("nan"), dtype=torch.float32)
    for i in range(n_combos):
        for j in range(n_dp):
            if not available[i, j]:
                observed_t[i, j] = float(prior_mean)

    cell_costs: Dict[Tuple[int, int], float] = {}
    total_evals = 0
    total_cost = 0.0
    cached_scores = torch.full((n_combos,), float("inf"), dtype=torch.float32)

    transition_stds = transition_stds_shrinking_gaussian_posterior(
        jnp.float32(prior_variance), jnp.float32(tau_sq), n_dp,
    )
    roots_all = compute_roots_lookup_table(
        transition_stds=transition_stds,
        costs_per_arm=jnp.float32(float(cost_per_transition) * float(cost_scaling_factor)),
        n_points=2**10 + 1,
    )
    roots_lookup_table = torch.from_numpy(np.array(jax.device_get(roots_all), copy=True)).to(torch.float32)
    if roots_lookup_table.ndim == 1:
        roots_lookup_table = roots_lookup_table.unsqueeze(0)

    last_pulled: Optional[List[int]] = None
    natural_stop: List[Optional[int]] = [None]
    rec_aware_stop: List[Optional[int]] = [None]

    while True:
        if total_evals >= budget:
            break
        step_bsz = min(bsz, budget - total_evals)
        out = gittins_index_exploration(
            observed_t,
            prior_mean=prior_mean,
            prior_variance=prior_variance,
            obs_noise_variance=tau_sq,
            cost_per_transition=cost_per_transition,
            cost_scaling_factor=cost_scaling_factor,
            batch_size=step_bsz,
            cached_scores=cached_scores,
            recompute_arms=None if last_pulled is None else last_pulled,
            allow_early_stop=allow_early_stop,
            roots_lookup_table=roots_lookup_table,
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
            val = compute_sample_objective(sr, lambda_cost, lambda_latency)
            observed_t[ci, di] = float(val)
            observed_np[ci, di] = float(val)
            cell_costs[(ci, di)] = sr.cost
            total_evals += 1
            total_cost += sr.cost
            pulled.add(int(ci))

        if not pulled:
            break
        last_pulled = sorted(pulled)
        _, cached_scores = gittins_post_pull_update(
            observed_t,
            cached_scores=cached_scores,
            recompute_arms=last_pulled,
            prior_mean=prior_mean,
            prior_variance=prior_variance,
            obs_noise_variance=tau_sq,
            cost_per_transition=cost_per_transition,
            cost_scaling_factor=cost_scaling_factor,
            batch_size=step_bsz,
            roots_lookup_table=roots_lookup_table,
            sim_cum_eval=total_evals,
            natural_stop_cum_eval_holder=natural_stop,
            recommendation_aware_stop_cum_eval_holder=rec_aware_stop,
        )
        if allow_early_stop and natural_stop[0] is not None and observation_budget_fraction >= 1.0:
            break

    best_name = None
    best_obj = float("-inf")
    best_acc = 0.0
    tol = 1e-9
    model_results = []
    for i, model_name in enumerate(models):
        idxs = [
            j for j in range(n_dp)
            if available[i, j] and not math.isnan(float(observed_np[i, j]))
        ]
        if not idxs:
            continue
        obj_mean = float(np.mean([observed_np[i, j] for j in idxs]))
        model_data_i = table.get(model_name, {})
        scores = [model_data_i[datapoints[j]].score for j in idxs if datapoints[j] in model_data_i]
        acc = sum(scores) / len(scores) if scores else 0.0
        lats = [model_data_i[datapoints[j]].latency_seconds for j in idxs if datapoints[j] in model_data_i]
        lat = sum(lats) / len(lats) if lats else 0.0
        cost_i = sum(cell_costs.get((i, j), 0.0) for j in idxs)
        model_results.append(ModelSummary(model_name, acc, obj_mean, lat, cost_i, len(idxs)))
        if obj_mean > best_obj + tol:
            best_name, best_obj, best_acc = model_name, obj_mean, acc

    for mr in model_results:
        mr.is_best = (mr.model_name == best_name)

    if best_name is not None:
        true_best_obj = compute_model_objective(
            best_name, datapoints, table, lambda_cost, lambda_latency)
        model_data_best = table.get(best_name, {})
        all_scores = [model_data_best[dp].score for dp in datapoints if dp in model_data_best]
        true_best_acc = sum(all_scores) / len(all_scores) if all_scores else 0.0
    else:
        true_best_obj = float("-inf")
        true_best_acc = 0.0

    gt_name, _, _ = compute_ground_truth(models, datapoints, table, lambda_cost, lambda_latency)
    return SimulationResult(
        "gittins", seed,
        {
            "observation_budget_fraction": observation_budget_fraction,
            "batch_size": bsz,
            "lambda_cost": lambda_cost,
            "lambda_latency": lambda_latency,
            "obs_noise_variance": tau_sq,
            "allow_early_stop": allow_early_stop,
        },
        best_name, true_best_acc, true_best_obj, total_evals, total_cost,
        len(model_results), best_name == gt_name, 0.0, model_results,
    )


def run_multi_seed(selector_fn, models, datapoints, table,
                   n_seeds, base_seed=0, **kwargs):
    results = []
    for i in range(n_seeds):
        result = selector_fn(models, datapoints, table, seed=base_seed + i, **kwargs)
        results.append(result)
    return results


def summarize_multi_seed(results, gt_name, gt_acc, gt_obj,
                         models=None, datapoints=None, table=None,
                         lambda_cost=0.0, lambda_latency=0.0):
    if models is not None and datapoints is not None and table is not None:
        true_accs = {}
        true_objs = {}
        for model in models:
            samples = table.get(model, {})
            available = [samples[dp] for dp in datapoints if dp in samples]
            true_accs[model] = sum(s.score for s in available) / len(available) if available else 0.0
            true_objs[model] = (sum(compute_sample_objective(s, lambda_cost, lambda_latency)
                                    for s in available) / len(available)) if available else float("-inf")
        accuracies = [true_accs.get(r.best_model, 0.0) for r in results]
        objectives = [true_objs.get(r.best_model, float("-inf")) for r in results]
    else:
        accuracies = [r.best_accuracy for r in results]
        objectives = [r.best_objective for r in results]

    evals = [r.total_evaluations for r in results]
    costs = [r.total_cost for r in results]
    found_best = [r.found_true_best for r in results]
    compute_times = [r.compute_time_seconds for r in results]
    n = len(results)
    mean_acc = sum(accuracies) / n
    std_acc = (sum((a - mean_acc)**2 for a in accuracies) / max(n - 1, 1)) ** 0.5
    mean_obj = sum(objectives) / n
    std_obj = (sum((o - mean_obj)**2 for o in objectives) / max(n - 1, 1)) ** 0.5

    return {
        "selector": results[0].selector,
        "n_seeds": n,
        "ground_truth_best": gt_name,
        "ground_truth_accuracy": gt_acc,
        "ground_truth_objective": gt_obj,
        "mean_accuracy": mean_acc,
        "std_accuracy": std_acc,
        "mean_objective": mean_obj,
        "std_objective": std_obj,
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
    print(f"  {'Model':<50} {'Acc':>7} {'Obj':>9} {'Samples':>8} {'Cost':>10}")
    print(f"  {'─'*50} {'─'*7} {'─'*9} {'─'*8} {'─'*10}")
    for mr in sorted(result.model_results, key=lambda x: (-x.objective, x.latency_seconds)):
        marker = " *" if mr.is_best else ""
        print(f"  {mr.model_name:<50} {mr.accuracy:>7.3f} {mr.objective:>9.4f}"
              f" {mr.n_samples_evaluated:>8} ${mr.cost:>8.4f}{marker}")
    print(f"\n  Best: {result.best_model}  (acc={result.best_accuracy:.3f}, obj={result.best_objective:.4f})")
    print(f"  Total evaluations: {result.total_evaluations}")
    print(f"  Total cost: ${result.total_cost:.4f}")
    print(f"  Found true best ({gt_name}): {'YES' if result.found_true_best else 'NO'}")


def print_summary(summary):
    print(f"\n{'='*60}")
    print(f"  {summary['selector']}  —  {summary['n_seeds']} seeds")
    print(f"  Params: {summary['params']}")
    print(f"{'='*60}")
    print(f"  Ground truth best: {summary['ground_truth_best']}"
          f"  (acc={summary['ground_truth_accuracy']:.3f},"
          f" obj={summary['ground_truth_objective']:.4f})")
    print(f"  Mean accuracy found:   {summary['mean_accuracy']:.4f}"
          f"  +/- {summary['std_accuracy']:.4f}")
    print(f"  Mean objective found:  {summary['mean_objective']:.4f}"
          f"  +/- {summary['std_objective']:.4f}")
    print(f"  Accuracy range:        [{summary['min_accuracy']:.4f},"
          f" {summary['max_accuracy']:.4f}]")
    print(f"  Found true best:       {summary['found_true_best_pct']:.1f}%"
          f"  ({int(summary['found_true_best_pct'] * summary['n_seeds'] / 100)}"
          f"/{summary['n_seeds']})")
    print(f"  Mean evaluations:      {summary['mean_evaluations']:.0f}")
    print(f"  Mean total cost:       ${summary['mean_cost']:.4f}")


ALL_SELECTORS = [
    "brute_force", "random_search", "matrix_ucb", "gittins", "bayesian_optimization",
]


def main():
    parser = argparse.ArgumentParser(
        description="Offline selector simulator — combined objective (accuracy - λ*cost - λ*latency)",
    )
    parser.add_argument("--jsonl", default=None, help="Path to brute-force JSONL file")
    parser.add_argument("--pickle", default=None, help="Path to pickle lookup table (.pkl)")
    parser.add_argument("--selectors", default="all",
                        help=f"Comma-separated: {','.join(ALL_SELECTORS)},all")
    parser.add_argument("--seeds", type=int, default=1, help="Number of random seeds (default 1)")
    parser.add_argument("--base-seed", type=int, default=42, help="Starting seed (default 42)")
    parser.add_argument("--output", default=None, help="Path to write summary CSV")

    # Combined objective params
    parser.add_argument("--lambda-cost", type=float, default=0.0,
                        help="Weight for cost penalty (default 0.0 = pure accuracy)")
    parser.add_argument("--lambda-latency", type=float, default=0.0,
                        help="Weight for latency penalty (default 0.0 = pure accuracy)")

    # Selector-specific params
    parser.add_argument("--rs-fraction", type=float, default=0.25)


    # Matrix UCB params
    parser.add_argument("--ucb-budget", type=float, default=0.2,
                        help="Matrix UCB / Gittins observation budget fraction (default 0.2)")
    parser.add_argument("--gittins-batch", type=int, default=20,
                        help="Gittins per-step example batch size (default 20)")
    parser.add_argument("--gittins-no-early-stop", action="store_true",
                        help="Disable Gittins natural early stop")

    args = parser.parse_args()

    lc = args.lambda_cost
    ll = args.lambda_latency

    if args.pickle:
        source = _require_data_path(args.pickle)
        print(f"Loading pickle: {source}")
        models, datapoints, table = load_pickle(source)
    elif args.jsonl:
        source = _require_data_path(args.jsonl)
        print(f"Loading JSONL: {source}")
        models, datapoints, table = load_jsonl(source)
    else:
        parser.error("One of --jsonl or --pickle is required")
    print(f"  Models: {len(models)}")
    print(f"  Samples: {len(datapoints)}")
    print(f"  Total entries: {sum(len(v) for v in table.values())}")
    print(f"  Lambda cost: {lc}, Lambda latency: {ll}")

    norm = set_norm_stats(table, datapoints)
    print(f"  Normalization: cost [{norm.cost_min:.4f}, {norm.cost_max:.4f}],"
          f" latency [{norm.latency_min:.2f}s, {norm.latency_max:.2f}s]")

    gt_name, gt_acc, gt_obj = compute_ground_truth(models, datapoints, table, lc, ll)
    print(f"\n  Ground truth best: {gt_name}  (acc={gt_acc:.4f}, obj={gt_obj:.4f})")

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
        if sel == "brute_force":
            fn = simulate_brute_force
            kwargs = {"lambda_cost": lc, "lambda_latency": ll}
        elif sel == "random_search":
            fn = simulate_random_search
            kwargs = {"sample_fraction": args.rs_fraction,
                      "lambda_cost": lc, "lambda_latency": ll}
        elif sel == "bayesian_optimization":
            if not _check_botorch():
                print(f"\n  Skipping {sel} — torch/botorch not installed")
                continue
            fn = simulate_bayesian_optimization
            kwargs = {"lambda_cost": lc, "lambda_latency": ll}
        elif sel == "matrix_ucb":
            fn = simulate_matrix_ucb
            kwargs = {"observation_budget_fraction": args.ucb_budget,
                      "lambda_cost": lc, "lambda_latency": ll}
        elif sel == "gittins":
            fn = simulate_gittins
            kwargs = {
                "observation_budget_fraction": args.ucb_budget,
                "batch_size": args.gittins_batch,
                "allow_early_stop": not args.gittins_no_early_stop,
                "lambda_cost": lc,
                "lambda_latency": ll,
            }
        else:
            print(f"\n  Unknown selector: {sel}")
            continue

        if args.seeds > 1:
            results = run_multi_seed(fn, models, datapoints, table,
                                     n_seeds=args.seeds, base_seed=args.base_seed, **kwargs)
            summary = summarize_multi_seed(results, gt_name, gt_acc, gt_obj,
                                           models, datapoints, table, lc, ll)
            print_summary(summary)
            summaries.append(summary)
        else:
            result = fn(models, datapoints, table, seed=args.base_seed, **kwargs)
            print_single_result(result, gt_name, gt_acc)
            summaries.append(summarize_multi_seed(
                [result], gt_name, gt_acc, gt_obj, models, datapoints, table, lc, ll))

    if args.output and summaries:
        import csv as csv_mod
        fields = ["selector", "n_seeds", "ground_truth_best",
                  "ground_truth_accuracy", "ground_truth_objective",
                  "mean_accuracy", "std_accuracy",
                  "mean_objective", "std_objective",
                  "found_true_best_pct",
                  "mean_evaluations", "mean_cost", "params"]
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
