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

The default two-direction replay reads the committed complete matrices from
`../data/hotpotqa/` and `../data/mathqa/`; no local pickle is required for these
two benchmarks. Run it without `--benchmarks` to select both datasets.

`data/lookup/*_lookup.pkl`

| File | Combos × samples |
|------|------------------|
| `gpqa_lookup.pkl` | 9 × 198 |
| `bfcl_lookup.pkl` | 9 × 200 |
| `hotpotqa_lookup.pkl` | legacy local source for the committed 100 × 200 matrices |
| `mathqa_lookup.pkl` | legacy local source for the committed 100 × 200 matrices |

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
- recommends the empirical Pareto frontier of all completed arms by default:
  maximize observed mean accuracy and minimize observed mean deployment cost
  in USD, without restricting recommendations to direction winners;
- records completed-only checkpoints as `deployable` throughout both anytime
  and fixed-budget runs, including the empty set before any arm completes;
  older fixed-budget results used a `provisional` all-posterior diagnostic
  prefix followed by `deployable` recommendations at the endogenous stop;
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
it records the stop, halves the search-cost weight η, and resumes acquisition
with updated indices. The legacy flags are `--lambda-initial 1.0
--lambda-decay 0.5`: their scalar `lambda` is a cost-weight multiplier, distinct
from the vector λ used for a scalarization direction in the design notes.
The effective pull penalty is that multiplier times the existing eta-scaled,
quantized expected batch cost. Quantization is frozen before applying the
multiplier, so each halving halves every effective penalty exactly. Calibration,
observations, posterior states, question schedules, and cumulative spend are
retained across stages.

The opt-in Python argument `eta_decay_schedule="direction_stop"` instead gives
each direction its own η multiplier. On a local stop, only that direction's
multiplier is halved; round-robin advances immediately and revisits it next
cycle. All directions still share the observations and completed-only
recommendations. This mode requires `anytime=True` and
`direction_scheduler="round_robin"`. It records per-direction η vectors and decay events rather than
treating these decisions as a common global stop. A direction at the numerical
cost floor remains eligible for reconsideration after shared observations.
The default `eta_decay_schedule="global_stop"` retains the global-stop rule.

#### Reproduce asynchronous η decay on BIRD dev

Run from the repository root in a Python 3.10+ environment with the project
dependencies installed. For a new environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

The SCOPE BIRD dev input matrices are tracked under `data/scope/bird_dev`.
Generated results are gitignored, so a fresh checkout needs the baseline first:

```bash
# 1. Generate the seed-42 global-decay baseline (skip if already saved).
python experiments/combined_objective/compare_direction_schedulers.py --benchmarks bird_dev

# 2. Run independent per-direction η decay and save the comparison (no figures).
python experiments/combined_objective/compare_eta_decay.py

# 3. Redraw all figures and tables from saved results, without replaying.
python experiments/combined_objective/compare_eta_decay.py --plot-only

# 4. Run the relevant regression tests.
python -m pytest \
    tests/test_direction_eta_decay.py tests/test_anytime_radial_gittins.py \
    tests/test_axis_radial_replay.py tests/test_offline_radial_gittins.py \
    tests/test_accuracy_last_scheduler.py tests/test_completed_recommendations.py \
    tests/test_deferred_recommendations.py -q
```

Step 1 runs both `round_robin` and the earlier `accuracy_last` ablation with
global decay. Step 2 takes only its `round_robin` result from
`experiments/combined_objective/results/direction_schedulers_bird_dev_completed_only_seed42/comparison.json`.
Both comparison runners fix seed 42, batch size 4, grid size 129, and
completed-only recommendations. In step 2, all ten directions, including
`(1, 0)`, participate in round-robin. Use `--baseline PATH` to read a different
baseline file and `--outdir DIR` to change the comparison output directory;
pass the same `--outdir DIR` when redrawing. `--plot-only` needs only that
directory's `comparison.json`. Running without it reruns the corresponding
experiment and replaces its saved outputs. The independent-decay runner saves
data without rendering by default; add `--plots` to also render at the end.
`--no-plots` remains an explicit alias for the default.

For a standalone replay through the general CLI, enable asynchronous decay
with `--anytime --direction-scheduler round_robin --eta-decay-schedule direction_stop`:

```bash
python experiments/combined_objective/offline_radial_gittins.py \
    --scope data/scope/bird_dev --anytime \
    --direction-scheduler round_robin --eta-decay-schedule direction_stop \
    --seeds 1 --base-seed 42 --batch-size 4 --grid-size 129 \
    --eta 1.0 --lambda-initial 1.0 --lambda-decay 0.5 \
    --record-trajectory --output /tmp/bird_dev_async_eta_seed42.json
```

The standalone CLI uses its default boundary padding; use the comparison
runner above to reproduce the reported experiment with extra padding 2.0.
The `--eta-decay-schedule` flag belongs to the general replay CLI; the comparison
runner selects `direction_stop` internally. Omitting the flag in the general
CLI keeps the existing `global_stop` default.

The runner verifies matching data, warm start, calibration, and HV reference;
saves comparison tables and a compressed trace. `--plots` or `--plot-only`
exports accuracy and HV curves, all key-checkpoint Pareto frontiers, and
per-direction η schedules.
Outputs are under `combined_objective/results/direction_eta_decay_bird_dev_completed_only_seed42/`.
Pareto panels use the existing `C1`, `C2`, ... convention: every recommendation
addition, removal, or replacement is a key checkpoint. Warmup and unchanged
stopping events are excluded; the actual final state is shown separately without
a new C number. Each method has its own numbered sequence and actual costs.
Filled circles mean completed and currently recommended; gold stars mark newly
recommended configurations. Gray points and the dotted full-data frontier are
references, not evaluation-status markers. `checkpoints.csv` preserves exact
costs and added/removed configuration IDs. `matched_budgets.csv` remains available
for comparisons at equal spend, separately from the checkpoint panels.
In this seed-42 run, independent decay completed the highest-accuracy configuration
at 24.99% of full-matrix search cost, versus 67.52% for global decay. At a 30%
cost budget, relative HV regret was 0.406% versus 1.800%; the final recommendation
set was identical. At the very early 1% budget, independent decay had slightly
higher HV regret (4.758% versus 3.739%). These are single-seed results.

#### Lightweight checkpoint recording with completed-only recommendations

The historical `asynchronous-eta-decay` branch kept its original recommendation
rule: anytime recommendations used completed direction winners. The current
`completed_only` rule instead returns the empirical Pareto frontier of all
completed arms, as described below. The checkpoint optimization itself
changes recording and postprocessing, not acquisition, per-direction stopping,
η decay, or the recommendation set. Historical fixed-budget results may retain
a provisional diagnostic prefix. Newly generated `completed_only` checkpoints
are always deployable and contain only the empirical completed frontier.

Previously, every inspected batch built a complete checkpoint, including
multiple diagnostic archives and HV/GD/IGD, before unchanged recommendations
were discarded. The new path first checks membership; only retained events
freeze winner IDs, posterior means, observed means, sample counts, completion
status, cost and η metadata. With `recommendation_changes_only=True`, the
trajectory retains only membership changes. Warm start and Final always keep
independent evidence, even when their recommendation sets are unchanged.
Events stay in memory during sampling and are serialized in bulk. No figures
are rendered by the selector.

Full checkpoint diagnostics are materialized after sampling by default for
compatibility with existing callers. For event-only results:

```python
from experiments.combined_objective.offline_radial_gittins import (
    materialize_recommendation_diagnostics,
    simulate_radial_gittins,
)

result = simulate_radial_gittins(
    models, questions, table,
    anytime=True, eta_decay_schedule="direction_stop", seed=42,
    record_recommendation_trajectory=True,
    recommendation_changes_only=True,
    recommendation_checkpoint_interval=None,
    defer_recommendation_diagnostics=True,
)
# Consume/save recommendation_events and recommendation_initial_event/final_event.
# Full checkpoint fields are populated only when explicitly requested:
materialize_recommendation_diagnostics(result)
```

For the standalone CLI command above, append
`--recommendation-changes-only --defer-recommendation-diagnostics` to save
lightweight events. Rebuild full diagnostics later, without replaying policy or
reloading the lookup matrices:

```python
import json
from pathlib import Path
from experiments.combined_objective.offline_radial_gittins import (
    materialize_saved_recommendation_diagnostics,
)

saved = json.loads(Path("/tmp/bird_dev_async_eta_seed42.json").read_text())
for run in saved["results"]:
    materialize_saved_recommendation_diagnostics(run)
```

Both materializers are idempotent and reconstruct metrics from frozen event
evidence, never from the final posterior. The saved result includes the fixed
truth vectors needed for offline metrics. `params["recommendation_recording"]`
reports membership checks, event captures and diagnostic materializations with
separate phase timings. The comparison runner explicitly defers diagnostics
until it builds its report and preserves all recommendation changes. Plot
selection does not change recorded events or online recommendations.

#### Recommend the empirical Pareto frontier of completed arms

The current `recommendation_rule="completed_only"` directly recommends every
nondominated completed arm, comparing its measured mean accuracy (maximize)
and measured mean USD cost (minimize). It uses observations from the declared
question universe, not posterior means, confidence penalties, or direction
winners. Equal objective pairs are both retained. Before any arm completes,
the recommendation is empty. An older member is removed only when another
completed arm empirically dominates it; partial arms never enter the set.

This changes recommendation membership only. The ten exploration directions,
round-robin schedule, required-completion Gittins comparator, and asynchronous
per-direction eta decay are unchanged. There is no recommendation beta or
additional completed raw guard. Older result files remain unchanged as frozen
artifacts; historical completed-only results used a direction-filtered set and
should not be labelled as the new empirical rule.

Use raw mean cost for the acquisition model explicitly, as in the preceding
raw-cost experiment, and change only the recommendation rule:

```bash
python experiments/combined_objective/run_lcb_benchmarks.py --question-order independent \
    --benchmarks restaurant_valid --cost-model raw_mean \
    --recommendation-rule completed_only \
    --outdir experiments/combined_objective/results/completed_raw_pareto_seed42

# Rebuild tables from the saved run without another replay.
python experiments/combined_objective/run_lcb_benchmarks.py --question-order independent \
    --benchmarks restaurant_valid --recommendation-rule completed_only \
    --outdir experiments/combined_objective/results/completed_raw_pareto_seed42 \
    --summarize-only

# Render every recommendation change from the saved run.
python experiments/combined_objective/plot/plot_lcb_recommendations.py \
    experiments/combined_objective/results/completed_raw_pareto_seed42/restaurant_valid/comparison.json
```

Both commands use seed 42. The result is stored as
`restaurant_valid/comparison.json` under `runs["completed_only"]`, alongside
`completed_only_trace.json.gz`; `summary.csv` and `matched_budgets.csv` use the
same method name. Membership checks and lightweight events remain separate
from deferred diagnostics and bulk output. Completed recommendation coordinates
are empirical; posterior values retained in events describe acquisition state.
For compatibility with saved-event readers, the checkpoint fields named
`direction_winner_*` carry the selected frontier arms and their evidence under
`completed_only`; those names do not imply a direction filter. The result-level
`direction_winners` still records the acquisition direction winners separately.

The benchmark runner now defaults to `finite_lcb` with `raw_mean` cost. With
`--recommendation-rule completed_only --cost-model raw_mean` and no explicit
output path, it writes to
`combined_objective/results/completed_only_benchmarks_seed42_raw_mean_shared_questions/`.
The core replay API and CLI retain `completed_only` as their conservative
general-purpose default.

Run the same raw-mean acquisition and empirical completed-frontier rule on
HotpotQA, MathQA, Stack Overflow and BIRD Dev (all seed 42):

```bash
python experiments/combined_objective/run_lcb_benchmarks.py --question-order independent \
    --benchmarks hotpotqa mathqa stackoverflow bird_dev \
    --cost-model raw_mean --recommendation-rule completed_only \
    --outdir experiments/combined_objective/results/completed_raw_pareto_seed42

for benchmark in hotpotqa mathqa stackoverflow bird_dev; do
    python experiments/combined_objective/plot/plot_lcb_recommendations.py \
        "experiments/combined_objective/results/completed_raw_pareto_seed42/$benchmark/comparison.json"
done
```

`bird_dev` is an explicit additional benchmark; the runner's existing default
benchmark list remains unchanged. Each folder contains its full checkpoint PDF,
comparison JSON and compressed physical sampling trace.

#### Full-test mean LCB recommendations

The opt-in `recommendation_rule="finite_lcb"` estimates each arm's mean over
the fixed benchmark question universe, including its already observed values.
For one objective, let `N` be the total question count, `n` the observed count,
`S_n` their sum, `mu` and `v` the posterior mean and variance of the latent
per-question mean, and `tau_squared` the frozen per-question observation noise.
The conditional full-test mean and variance are

```text
m = (S_n + (N - n) * mu) / N
s_squared = ((N - n)**2 * v + (N - n) * tau_squared) / N**2
```

These moments assume conditionally independent Gaussian unseen observations,
with shared uncertainty through the latent mean and fixed plug-in noise
calibrated from the warm batch. They describe the fixed test mean, not only
the latent population mean. Accuracy and cost use their respective moments.
The recommendation coordinates are `(m_accuracy - beta * s_accuracy,
m_cost_usd + beta * s_cost_usd)`: maximize the former and minimize the latter.
The selector returns the Pareto frontier of these conservative raw coordinates
over all arms, without a direction filter. Coordinates are not clipped.

When `n == N`, both unobserved terms vanish: `m` is the empirical full-test
mean and `s == 0`. Completion therefore requires no manual uncertainty
exemption. `beta=1` is a componentwise conservative score; plug-in calibration,
repeated adaptive recommendations, and comparisons across arms do not give
it a simultaneous or anytime confidence guarantee. The implementation
requires `cost_model="raw_mean"`. It is the benchmark runner's default
recommendation rule; the lower-level replay API still defaults to
`completed_only`.

This rule affects recommendation membership only. It introduces no change
to the sampling rule, exploration directions, Gaussian acquisition model,
Gittins DP, or asynchronous eta schedule. Light events are captured when
membership changes and diagnostics are computed after sampling. Frozen
`finite_target_mean_vectors`, `finite_target_std_vectors`, and
`recommendation_raw_vectors` provide the evidence for each returned arm;
they align with `direction_winner_arm_indices`, whose legacy name here means
the selected finite-test frontier. `estimated_raw_winner_vectors` still means
the observed sample averages, while posterior fields describe the latent
acquisition state. The affine `recommendation_desirability_vectors` export
exists for compatibility; raw conservative coordinates determine membership.

Run the five datasets with seed 42 and beta 1:

```bash
python experiments/combined_objective/run_lcb_benchmarks.py --question-order independent \
    --benchmarks hotpotqa mathqa stackoverflow bird_dev restaurant_valid \
    --cost-model raw_mean --recommendation-rule finite_lcb --beta 1 \
    --outdir experiments/combined_objective/results/finite_lcb_raw_mean_seed42_beta1

# Render every recommendation change and overlay the saved completed reference.
# Both methods' complete checkpoint PDFs are written to each finite_lcb folder.
for benchmark in hotpotqa mathqa stackoverflow bird_dev restaurant_valid
do
    python experiments/combined_objective/plot/plot_lcb_recommendations.py \
        "experiments/combined_objective/results/finite_lcb_raw_mean_seed42_beta1/$benchmark/comparison.json" \
        --reference "experiments/combined_objective/results/completed_raw_pareto_seed42/$benchmark/comparison.json"
done
```

If a completed reference is missing, generate it with the first command using
`--recommendation-rule completed_only` and the output directory
`experiments/combined_objective/results/completed_raw_pareto_seed42`.
The plot overlay checks lookup hashes, seed, model/question universe and metric
coordinates before combining saved results. It does not certify identical
acquisition traces. `comparison.png`/`.pdf` overlay both methods; each has its
own `*_pareto_key_checkpoints_all_pages.pdf`, and `pareto_snapshots.pdf` aliases
the finite-test method's full sequence. All plotted accuracy/cost positions
are offline full-data values; a hollow circle denotes an incomplete
recommendation, not an estimate of its accuracy or cost.

#### Full-test mean recommendations after 32 observations

The opt-in `recommendation_rule="finite_mean"` uses the same full-test mean
`m = (S_n + (N - n) * mu) / N` for accuracy and raw mean USD cost, without a
standard-deviation penalty. It returns the direct Pareto frontier of eligible
arms in `(m_accuracy, m_cost_usd)`. Set `--recommendation-min-samples 32` to make
an arm eligible only after **32 actual observed questions**, including warm-up
observations; the threshold counts observations, not batches. Until an arm
qualifies, the recommendation is empty. An arm with fewer than 32 total
questions never qualifies under this strict threshold. All five datasets below
have at least 32 common questions per arm.

`finite_mean` always records effective `recommendation_beta=0.0`, regardless
of `--beta`; it does not subtract or add posterior standard deviation. The
target moments and their Gaussian assumptions remain those described above.
`finite_target_std_vectors` are retained as diagnostic evidence, while
`recommendation_raw_vectors` equal `finite_target_mean_vectors`. Completed
arms naturally have their empirical full-test means and zero target variance.
The count threshold is a chosen evidence gate, not a confidence guarantee or
a stopping rule. It changes recommendation membership only; sampling,
directions, asynchronous eta decay, warm calibration and the raw cost model
remain the same.

`recommendation_min_samples` defaults to zero and may also be used with
`finite_lcb`; a nonzero threshold is rejected for other rules. Both finite
rules require `cost_model="raw_mean"`.
The new run exports `recommendation_min_samples`, the eligibility condition,
the `eligible_arms_finite_test_mean_raw_pareto` filter and frozen target evidence
alongside every lightweight membership event. Existing defaults and saved
results are unchanged.

Reproduce the five original independent-order datasets with seed 42, then
compare against both saved independent-order references:

```bash
python experiments/combined_objective/run_lcb_benchmarks.py --question-order independent \
    --benchmarks hotpotqa mathqa stackoverflow bird_dev restaurant_valid \
    --cost-model raw_mean --recommendation-rule finite_mean \
    --recommendation-min-samples 32 \
    --outdir experiments/combined_objective/results/finite_mean_min32_raw_mean_seed42

for benchmark in hotpotqa mathqa stackoverflow bird_dev restaurant_valid
do
    python experiments/combined_objective/plot/plot_lcb_recommendations.py \
        "experiments/combined_objective/results/finite_mean_min32_raw_mean_seed42/$benchmark/comparison.json" \
        --reference "experiments/combined_objective/results/completed_raw_pareto_seed42/$benchmark/comparison.json" \
        --reference "experiments/combined_objective/results/finite_lcb_raw_mean_seed42_beta1/$benchmark/comparison.json"
done
```

`--reference` is repeatable; the earlier single-reference command still works.
The same provenance and metric checks apply to every added reference. Each new
folder gets three-way `comparison.pdf`/`.png` curves and a full checkpoint PDF
for each method. `finite_mean_pareto_key_checkpoints_all_pages.pdf` is the new
method's canonical sequence; `pareto_snapshots.pdf` aliases that sequence.
Its label records `n≥32`, and hollow/filled markers still distinguish partial
and completed recommendations in offline full-data coordinates. No native
sampling is repeated by the plotting command.

#### Shared random question order

The combined radial simulator and benchmark runner now default to
`question_order="shared"`. The same seeded warm-up batch is retained, so this
switch leaves warm-up observations and their prior/noise calibration unchanged.
After warm-up, one random permutation orders the remaining questions for all
arms. Each arm keeps its **own cursor**: selecting one arm advances only that
arm. This does not force equal sample counts or evaluate every arm on the same
question simultaneously. On the runner's common question universe, two arms
with the same observed count have seen the same question prefix; an arm with
more observations is farther along that prefix. For noncommon universes, the
global order is restricted to each arm's available questions.

This change leaves the selected recommendation rule, `n>=32` gate, effective
beta zero, raw cost model, directions and asynchronous eta mechanism unchanged.
Post-warm observations differ, so posterior states, acquisition decisions and
eta events may differ from an independent-order run. Identical acquisition
traces are therefore not an expected property of a cross-order comparison.
The low-level `PerArmQuestionSchedule` constructor keeps its legacy
`independent` default for other callers; the radial simulator opts into shared
order explicitly.

Run the finite-mean variant on the same five datasets with seed 42:

```bash
python experiments/combined_objective/run_lcb_benchmarks.py \
    --benchmarks hotpotqa mathqa stackoverflow bird_dev restaurant_valid \
    --question-order shared \
    --cost-model raw_mean --recommendation-rule finite_mean \
    --recommendation-min-samples 32 \
    --outdir experiments/combined_objective/results/finite_mean_min32_raw_mean_seed42_shared_questions
```

To reproduce the earlier independent-order version, use the explicit old mode:

```bash
python experiments/combined_objective/run_lcb_benchmarks.py \
    --benchmarks hotpotqa mathqa stackoverflow bird_dev restaurant_valid \
    --question-order independent \
    --cost-model raw_mean --recommendation-rule finite_mean \
    --recommendation-min-samples 32 \
    --outdir experiments/combined_objective/results/finite_mean_min32_raw_mean_seed42
```

The historical recipes in this README specify `--question-order independent`
to preserve their meaning. Without `--outdir`, new shared runs receive a
`_shared_questions` suffix; explicit independent mode preserves the old output
names. An existing `comparison.json` with a different question order is
protected even when `--outdir` is explicit: choose a new directory. Repeating
a run with the same order remains allowed.

`question_order` is recorded in config, compact runs and summary tables;
`question_order_semantics` and `question_order_rng_scheme` are retained in the
compact evidence. Historical results without a question-order field mean
**independent**, including when overlaid with a new shared run. Plot reference
compatibility checks still require the same dataset, seed and metric space,
but permit different question orders and do not assert trace parity. Mixed
orders are identified in the comparison legend and retained in plot manifests
and checkpoint CSVs. The older BIRD exact-acquisition comparison CLI retains
its independent default and requires a baseline with the same order; use the
benchmark runner and saved plot references for cross-order comparisons.

#### Current recommendation rules

The executable recommendation choices are now `finite_lcb`, `finite_mean`,
and the `completed_only` comparison baseline. The only LCB rule is
`finite_lcb`; it uses finite-test predictive accuracy LCB and mean-cost UCB
coordinates and therefore requires `cost_model="raw_mean"`. The earlier
direction-winner LCB variants and their completed-raw guard have been removed
from the replay API, command-line interfaces, plotting entry points, and tests.

Historical result directories are frozen artifacts rather than runnable method
definitions. See `combined_objective/results/README.md` for provenance; do not
use their legacy method labels as current CLI arguments.

The default `--direction-scheduler round_robin` visits all ten directions in
one cycle. The experimental `--direction-scheduler accuracy_last` runs the
other directions round-robin until all stop, then runs `(1, 0)` until it stops.
An endpoint observation invalidates earlier stopping decisions, so the other
directions are checked again before lambda can decrease. Each new stage starts
with the other directions. Completed combinations reuse their results;
partially evaluated combinations continue on unseen questions. This changes
acquisition order, not the stopping, completion, or evaluation-deduplication
rules. The option is available in both replay and anytime plotting CLIs and
as `direction_scheduler="accuracy_last"` in the Python API. With no accuracy
endpoint, it preserves the ordinary round-robin behavior.

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

# Try deferred accuracy acquisition in a separate output directory.
.venv/bin/python experiments/combined_objective/plot/plot_anytime_radial_gittins.py \
    --benchmarks hotpotqa mathqa --direction-scheduler accuracy_last \
    --outdir experiments/combined_objective/results/anytime_accuracy_last

# Compare both schedulers on HotpotQA and MathQA, completed-only, seed 42.
.venv/bin/python experiments/combined_objective/compare_direction_schedulers.py

# Run the same Radial-Gittins-decay comparison on complete SCOPE BIRD dev.
.venv/bin/python experiments/combined_objective/compare_direction_schedulers.py \
    --benchmarks bird_dev \
    --outdir experiments/combined_objective/results/direction_schedulers_bird_dev_completed_only_seed42
```

The comparison writes `comparison.json`, CSV tables and a figure under
`results/direction_schedulers_completed_only_seed42/`. It uses the full common
question sets, batch size 4, a 129-point plotting grid and 2.0 extra grid
padding. Tables compare the current recommendation at matched realized search
costs and the spend needed to reach HV-regret thresholds. `first_hit` reports
the first crossing; `sustained_to_end` also requires every later checkpoint to
remain below the threshold. These are single-seed diagnostics, not an average
over seeds or a validation at the production 513-point resolution.

`--benchmarks bird_dev` loads `data/scope/bird_dev`, the complete 75-configuration,
1,534-question matrix. The comparison also exports Pareto snapshots at matched
search-cost fractions and the final recommendation, using observed accuracy
and mean deployment cost for completed configurations. Snapshot CSVs retain
configuration IDs and the exact preceding checkpoint cost. Grey reference
points are full-data diagnostics and never enter acquisition.

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

Anytime `completed_only` recommendations contain the empirical raw
accuracy/cost Pareto frontier of all completed arms, with no direction filter.
Older saved experiments in this section used completed direction winners;
rerunning the current code produces the expanded empirical rule. Recommendations
become available when the first arm
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
Current completed-only HV/GD/IGD evaluate the recommendation against the full
mean-USD Pareto frontier using the offline affine metric scale described above.
Historical completed-only results and reciprocal-model LCB results instead
used normalized desirability from the frozen per-question reciprocal transform;
their HV values require recomputation before comparison with the new metric.
Pareto snapshots show accuracy versus mean deployment cost in
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
Legacy direction-filtered recommendations broke completed accuracy ties using
lower observed mean deployment cost; current completed-only recommendations
are selected directly by empirical Pareto dominance. `(0, 1)` is also supported for
cost desirability via `--extra-direction 0 1`. Repeating an already included
direction on the CLI has no effect.

`combined_objective/plot/plot_radial_gittins_trajectories.py` writes one raw-archive
comparison per benchmark, with each snapshot panel labelled by the scope its
checkpoint recorded. Its hypervolume, generational-distance, and inverted-generational-distance
series preserve the recorded archive scopes. Historical fixed-budget results
show a dashed all-posterior diagnostic before the endogenous Gittins stop and
a solid completed-only recommendation afterwards, with the handover marked.
New `completed_only` runs instead record the deployable empirical frontier
throughout, starting with an empty set until the first arm completes; their
recommendations do not wait for an endogenous stop. Online and oracle membership
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
