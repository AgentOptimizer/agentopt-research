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

Plotting scripts live in `combined_objective/plot/`. Generated figures go
with other selector outputs under `combined_objective/results/` (gitignored):

- `combined_objective/results/radial_gittins_plots/`
- `combined_objective/results/anytime_radial_gittins/`
- `combined_objective/results/qa_random_20seeds/`
- `combined_objective/results/scope_random/`
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

### Anytime radial Gittins

The anytime selector uses nine interior directions plus the exact accuracy
endpoint `(1, 0)` by default, so the most accurate combination can be recommended
even when it is expensive. This is the standard anytime configuration in both
the Python API and the plotting runner.

`--anytime` keeps the replay running when every direction triggers stopping:
it records the stop, halves lambda, and resumes acquisition with updated
indices. Defaults are `--lambda-initial 1.0 --lambda-decay 0.5`. The effective
pull penalty is lambda times the existing eta-scaled, quantized expected batch
cost. Quantization is frozen before applying lambda, so each halving also halves
every effective penalty exactly. Calibration, observations, posterior states,
question schedules, and cumulative spend are retained across stages.

Run these commands from the repository root:

```bash
# Replay one benchmark and record the complete trajectory.
.venv/bin/python experiments/combined_objective/offline_radial_gittins.py \
    --pickle experiments/data/lookup/hotpotqa_lookup.pkl \
    --anytime --lambda-initial 1.0 --lambda-decay 0.5 --record-trajectory \
    --output /tmp/hotpotqa_anytime.json

# Generate HV regret, GD, IGD, and every recommendation-addition frontier.
.venv/bin/python experiments/combined_objective/plot/plot_anytime_radial_gittins.py \
    --benchmarks hotpotqa mathqa --seed 42 --batch-size 4 --grid-size 129 \
    --boundary-z-padding-extra 2.0

# Redraw saved results without rerunning acquisition.
.venv/bin/python experiments/combined_objective/plot/plot_anytime_radial_gittins.py \
    --plot-only
```

The Python API is `simulate_radial_gittins(..., anytime=True,
lambda_initial=1.0, lambda_decay=0.5, record_recommendation_trajectory=True)`.
The general replay API retains fixed-lambda behavior and nine interior directions
unless `anytime=True` is set; explicitly supplied directions override either
mode's defaults. Anytime mode takes precedence over
`halt_on_gittins_stop`; it never force-pulls an arm that fails the current
stopping rule. It returns at the question or dollar budget, full completion,
or a recorded numerical lambda floor. The numerical floor prevents endless
halvings when the remaining cumulative penalty is below stopping precision;
it is not a proof of exact zero-cost optimality.

Anytime recommendations contain completed direction winners filtered for
raw accuracy/cost nondominance. They become available when the first arm
completes. Every addition to this set is retained even between ordinary
trajectory samples; replacements count as additions even when set size stays
constant. Completing a dominated combination need not add a recommendation.
The current recommendation may drop older members as better combinations
finish. A `lambda_stop` checkpoint separately records each global stop.
Warmup snapshots used to fit the prior never count as numbered key checkpoints,
even if a short benchmark completes a combination during calibration. Later
recommendations may still have lambda 1: lambda changes only at a global stop,
not at every recommendation addition.

Outputs are under `combined_objective/results/anytime_radial_gittins/`:

- `metrics.png` / `.svg` show relative HV regret, GD, and IGD (lower is better).
  Relative HV regret is `100 * (reference HV - HV) / reference HV`, with a
  logarithmic scale above 0.01% and a linear scale down to zero below it.
  The individual curves are saved as `hv_regret_curves`, `gd_curves`, `igd_curves`.
  `metrics_checkpoint_zoom` enlarges the interval containing recommendation
  changes, with matching `C` labels for the frontier snapshots.
- `lambda_schedule.png` / `.svg` show the recorded lambda halvings against
  cumulative cost; a stop at the numerical floor is not counted as a halving.
- `{benchmark}_frontiers_*.png` / `.svg` show all recommendation additions,
  paginated nine panels per image, plus the final frontier. Each panel labels
  cumulative cost percentage, current lambda, and recommendation count.
- `checkpoints.csv` gives the exact spend, added/removed combination IDs, full
  names of additions, and metrics at each key checkpoint. `combinations.csv`
  maps all `A` labels to full model combinations.
- `trajectory.csv`, `lambda_stops.json`, `summary.json`, and
  `{benchmark}_run.json` retain the data, parameters, and acquisition trace.
  Metric tables include raw HV, absolute HV regret, and relative HV regret.

The horizontal metric axis is actual cumulative search cost, including the
warm start, divided by exhaustive spend on the same complete question
intersection. `--budget-fraction` instead caps the number of question
evaluations; `--max-search-cost-usd` exposes the existing dollar cap.
HV/GD/IGD evaluate the recommendation against the full reference frontier in
normalized desirability space, using the frozen reciprocal cost transform
per question. Pareto snapshots show accuracy versus mean deployment cost in
USD. Full-data reference points are used only for evaluation, never acquisition.
Before any arm completes, HV is zero and GD/IGD are infinite (blank in CSV,
null in JSON, and omitted from the distance curves).
Relative regret is undefined if reference HV is zero. GD can start at zero
when the first recommendation is on the reference frontier, increase when
another recommendation is off that frontier, then decrease as it is replaced.
IGD also measures coverage of the reference frontier.

These defaults produce a reproducible single-seed illustration on a 129-point
plotting grid. Use `--grid-size 513` for the production resolution; changes in
grid resolution can change the acquisition trajectory. Anytime mode widens
the numerical boundary grid and retries at most four times if a root reaches
an edge as lambda decreases. These expansions are recorded in the run parameters;
a persistent grid failure still raises an error.

The default accuracy endpoint uses accuracy alone as its terminal
utility and a one-dimensional Gaussian retirement DP for acquisition. It
still pays the same lambda-scaled search penalty and shares observations,
posteriors, budget guards, and global stopping with the other directions.
The two-dimensional radial formula is used only for positive interior
directions, so no division by zero or near-axis approximation is needed.
If completed combinations tie on accuracy, the endpoint recommends the one
with lower observed mean deployment cost. `(0, 1)` is also supported for
cost desirability via `--extra-direction 0 1`. Repeating an already included
direction on the CLI has no effect.

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
plots radial-Gittins against it. QA random contact sheets and the seed-42 Gittins
versus 20-seed random comparison go to `combined_objective/results/qa_random_20seeds/`;
SCOPE random contact sheets go to `combined_objective/results/scope_random/`.
`combined_objective/audit_multiobjective_results.py`
re-checks dominance, distance to the front, hypervolume, GD and IGD against the
brute-force frontier. Raw CSV files, compressed caches, wall-time records, and
Slurm logs for those jobs remain under `analysis/vs/`.

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

Scalar Gittins needs `jax` / `jaxtyping` / `torch`. Radial-Gittins always
supports the NumPy/SciPy reference backend; install `.[radial-jax]` for its
optional batched backend. Its production path uses direction-aware 513-point
grids and separable FFT convolution with exact Gaussian integration of the
grid's linear basis; this prevents late, sub-grid posterior transitions from
collapsing to zero learning.

The radial replay keeps exact per-arm indices and terminal utilities lazily:
only the arm updated by the latest batch is invalidated. Boundary construction
is also lazy by direction. On a direction's first visit, duplicate arm
cost/horizon requests are removed and sufficiently large cold groups are
vectorized in one JAX call; small groups and automatic failure recovery use
SciPy. JAX carries only the current padded 2D value batch through
`lax.scan`, retaining boundary rows rather than the full 2D history. Use
`--boundary-build-backend {auto,jax,scipy}` and
`--boundary-jax-min-batch-size` to control routing. Auto mode also keeps
small one-off workloads on SciPy because CPU JIT compilation can dominate
them; explicit `jax` bypasses that workload heuristic. These choices control
only cold misses. A valid warm table is intentionally backend-neutral and is
reused because the JAX and SciPy implementations solve the same discretized
DP within the policy's numerical tolerance. For backend-isolated numerical
experiments, use separate cache directories (and a positive stop tolerance).

When the two objective variance/noise schedules are identical and the grids
are exact reflections, objective-swap symmetry further reduces the nine
defaults to five direction-specific batched solves for each shared
cost/horizon family (the default symmetric reference and grids satisfy this
condition). The offline CLI and trajectory plotting entry point persist the
resulting compact boundary schedules under
`combined_objective/results/cache_radial_gittins_boundaries/`, so later
processes reuse them without storing the large 2D work arrays. Use
`--no-boundary-disk-cache` for memory-only operation, `--boundary-cache-dir`
for another location, or set `AGENTOPT_RADIAL_GITTINS_CACHE_DIR`. Cache keys
include every explicit solver input plus schema and manually maintained solver
versions; invalid or version-mismatched entries are ignored and rebuilt.
The scope-comparison plot currently shares a memory cache within one process.
Trajectory plotting records every tenth adaptive pull by default because a
full Pareto/archive diagnostic at every pull can dominate warm-cache runtime;
`--trajectory-checkpoint-interval 1` restores every-pull curves. Warm-start,
the first Gittins stop, and final state are always recorded. Stopping and
deployable archives still follow the completed-arms required-completion
contract.

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
