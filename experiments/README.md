# Benchmark Experiments

Frozen results for 9 LLM configurations across 4 benchmarks. No API calls required.

Selector **algorithm implementations** live in `../src/agentopt/model_selection/`. The offline sims below re-implement the same decision logic against pickle lookup tables.

This repo is multi-objective. The main replay is `combined_objective/offline_radial_gittins.py` (accuracy vs deployment cost). Scalarized multi-objective also lives in `combined_objective/`. Accuracy-only baseline replay is under `single_objective/`.

## Data

Shared inputs live in `data/` (gitignored — keep them locally). Selector outputs live with the scripts that produce them.

Repo-root SCOPE benchmark directories can be passed directly to Radial
Gittins and both multi-objective random baselines:

```bash
python experiments/combined_objective/offline_radial_gittins.py \
    --scope data/scope/bird_mini_dev --budget-fraction 0.1

python experiments/combined_objective/offline_multiobjective_random_search.py \
    --scope data/scope/bird_mini_dev --seeds 1 \
    --output experiments/results/scope_bird_mini_dev_random.csv
```

The random command runs `random_configurations` (complete sampled rows) and
`random_questions` (shared sampled columns) across the budget sweep.

This is `experiments/data/`, not the repo-root `data/`. The latter holds the
committed accuracy / cost / token matrices extracted from these pickles; see
`../data/README.md`.

### Aggregated (per model combination)

`data/brute_force/{gpqa,bfcl,hotpotqa,mathqa}.csv`

| Column | Meaning |
|--------|---------|
| Accuracy | Task success rate |
| Server Latency (s) | Mean server-side latency |
| Wall Latency (s) | Mean wall-clock latency |
| Cost ($) | Total / mean cost for the combo |

Shared ablation: `data/gpqa_thinking_ablation.csv`.

Selector outputs:

- `combined_objective/results/radial_gittins_plots/`
- `combined_objective/results/multiobjective/`
- `single_objective/results/{gpqa,bfcl,hotpotqa,mathqa}/selector_results.csv`
- `single_objective/results/matrix_ucb_budget_sweep.csv`

### Per-sample lookup tables

`data/lookup/*_lookup.pkl`

| File | Combos × samples |
|------|------------------|
| `gpqa_lookup.pkl` | 9 × 198 |
| `bfcl_lookup.pkl` | 9 × 200 |
| `hotpotqa_lookup.pkl` | 81 × up to 200 (ragged) |
| `mathqa_lookup.pkl` | 81 × up to 200 (ragged) |

Schema: `model_names`, `datapoints`, `table[combo][dp_idx] → SampleResult(score, latency_seconds, cost, tokens…)`.

## Offline selector simulation

Pickles under `data/lookup/` are **gitignored** — keep them locally.

```bash
# Accuracy + deployment-cost Pareto search (empirical-Bayes warm start)
python combined_objective/offline_radial_gittins.py \
    --pickle data/lookup/hotpotqa_lookup.pkl \
    --batch-size 4 --budget-fraction 0.2 --eta 1.0

# Combined objective: J = acc − λ_cost·NormCost − λ_latency·NormLatency
python combined_objective/offline_selector_sim.py \
    --pickle data/lookup/gpqa_lookup.pkl \
    --selectors all --seeds 50 \
    --lambda-cost 0.1 --lambda-latency 0.1

cd combined_objective && python run_all_selectors.py

# Accuracy-only baseline (not the main protocol)
python single_objective/offline_selector_sim.py \
    --pickle data/lookup/gpqa_lookup.pkl \
    --selectors all --seeds 50

# Multi-objective random-search baselines (equal cell budget)
python combined_objective/offline_multiobjective_random_search.py \
    --pickle data/lookup/hotpotqa_lookup.pkl \
    --output combined_objective/results/multiobjective/hotpotqa_random_search.csv
```

Scripts auto-add `../src` to `PYTHONPATH` so `agentopt.model_selection` imports work.
If a pickle is missing, they exit with a pointer to the local data directory.

Available selectors: `brute_force` (combined objective), `random_search`, `matrix_ucb`, `gittins`, `bayesian_optimization`. Pareto-set identification baselines (not live `ModelSelector` methods): `ege_sh`, `ege_sr`, `ape_k`, `qnehvi`.

`combined_objective/offline_radial_gittins.py` has a separate set-valued result contract. It:

- uses one common seeded warm batch for every configuration;
- learns `C_ref` as the median configuration warm-batch mean cost;
- learns one common plug-in prior mean and fixes prior variance at `0.04` by default;
- applies reciprocal cost desirability per question before batch averaging;
- freezes an arm-specific expected pull cost from each arm's warm batch and
  bins those costs for practical boundary-table reuse;
- visits radial directions round-robin while sharing one posterior per configuration;
- returns direction winners filtered in online empirical raw space: maximize
  observed mean accuracy and minimize observed mean deployment cost in USD;
- restricts the deployable recommendation to completed direction winners,
  matching the required-completion Gittins stopping convention;
- records one archive per trajectory checkpoint, tagged with the eligibility
  scope that produced it: all-posterior `provisional` before the endogenous
  stop, when nothing is deployable yet, and completed-only `deployable` from
  the stop onward; unfinished provisional winners are candidates, not terminal
  recommendations;
- separately reports the legacy posterior-desirability archive and an offline
  oracle raw winner archive for diagnostics. The oracle archive uses the full
  cached matrix and never affects acquisition, stopping, or the deployable
  recommendation.

The incomplete HotpotQA and MathQA rows are truncated suffixes, so the main
protocol defaults to the complete question intersection (190 and 135 questions,
respectively). This gives every configuration the same benchmark universe and
horizon. `--ragged-diagnostic` enables the arm-specific-tail behavior only as a
diagnostic. Adaptive replay uses full batches, then a smaller final batch when
fewer than `batch_size` questions remain; Gittins tables still plan every stage
as a full batch. `--horizon-bin-width 1` uses exact per-configuration horizons; larger values
are an explicitly reported speed/accuracy approximation for larger sweeps.

`combined_objective/plot/plot_radial_gittins_trajectories.py` writes one raw-archive
comparison per benchmark, with each snapshot panel labelled by the scope its
checkpoint recorded. Its hypervolume, generational-distance, and inverted-generational-distance
series are each a single trajectory: a dashed
all-posterior diagnostic before the endogenous Gittins stop and the solid
completed-only recommendation from the stop onward, with the handover marked,
because the deployable archive is not a recommendation before the policy
stops. Online and oracle membership
are plotted at full-dataset raw coordinates against the global raw Pareto
front, so disagreements expose estimation error without feeding hidden
outcomes back into the selector. The replay after the marked endogenous stop
is forced only to show counterfactual later-budget diagnostics; it is not the
policy's terminal output.

`combined_objective/plot/plot_multiobjective_random_search.py` plots the random-search
budget sweep and `combined_objective/plot/plot_multiobjective_method_comparison.py`
plots radial-Gittins against it. `combined_objective/audit_multiobjective_results.py`
re-checks dominance, distance to the front, hypervolume, GD and IGD against the
brute-force frontier.

Pareto-set identification baselines that are *not* radial index rules live in
`combined_objective/offline_pareto_baselines.py`. They pull question batches
from the same lookup tables and run to the full cell budget (no endogenous
stop):

* `ege_sh` (default EGE variant) and `ege_sr` — Empirical Gap Elimination
  (Kone, Kaufmann, Richert, AISTATS 2024). Sequential Halving / Successive
  Rejects allocation; fixed budget, no stopping time.
* `ape_k` — Adaptive Pareto Exploration sampling (Kone et al., NeurIPS 2023).
  The paper's `|OPT_ε1| ≥ k` fixed-confidence stop is omitted; the LUCB-style
  sampler just exhausts the budget.
* `qnehvi` — BoTorch qNEHVI over categorical configuration features
  (`pip install -e ".[bayesian]"`). Sequential noisy hypervolume improvement
  until the budget is gone.

```bash
python combined_objective/offline_pareto_baselines.py \
    --pickle data/lookup/hotpotqa_lookup.pkl \
    --methods ege_sh ape_k qnehvi \
    --budget-fraction 1.0 --seeds 1

python combined_objective/plot_pareto_identification_baselines.py \
    --methods ege_sh ape_k qnehvi --seeds 1
```

`--budget-fraction` is a fraction of question cells, not dollars. A
`--max-search-cost` guard reserves the frozen warm-start expected batch cost and
can overshoot when the realized batch is more expensive; the result reports that
overshoot. Supplying a defensible `--guaranteed-batch-cost` makes the dollar cap
hard and validates the claimed bound against every replayed batch.

Scalar Gittins needs `jax` / `jaxtyping` / `torch`. Radial-Gittins uses NumPy
and SciPy. Its production path uses direction-aware 513-point grids and
separable FFT convolution with exact Gaussian integration of the grid's linear
basis; this prevents late, sub-grid posterior transitions from collapsing to
zero learning.

## Benchmarks

| Benchmark | Samples | Combos | Architecture |
|-----------|---------|--------|--------------|
| GPQA | 198 | 9 | Direct QA |
| BFCL | 200 | 9 | Multi-turn function calling |
| HotpotQA | 200 | 81 | Planner + Solver |
| MathQA | 200 | 81 | Answer + Critic |

## Models

Claude 3 Haiku, Claude Haiku 4.5, Claude Opus 4.6, gpt-oss-20b, gpt-oss-120b, Kimi K2.5, Ministral 3 8B, Qwen3 32B, Qwen3 Next 80B A3B

## Dependencies

Declared in `../pyproject.toml`; Python 3.10+. Install from the repo root:

```bash
pip install -e ".[dev]"        # core + Radial-Gittins + plots + pytest
pip install -e ".[gittins]"    # scalar Gittins only (jax, jaxtyping, torch)
pip install -e ".[bayesian]"   # Bayesian Optimization selector only
```

CSV inspection needs only Python / pandas. Pickle loading needs stdlib (+ the `SampleResult` class in the sim scripts).
