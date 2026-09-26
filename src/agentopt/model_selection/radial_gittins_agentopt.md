> Historical implementation design. The current paper protocol is two-axis
> CC-Gittins, defined in [gittins_ablation_v2.py](../../../experiments/combined_objective/gittins_ablation_v2.py).
> Current run commands are in [experiments/README.md](../../../experiments/README.md).
> The defaults and proposed extensions below describe the original radial prototype.

# Cursor Implementation Spec: Cost-Aware Multi-Objective Radial-Gittins for AgentOpt

## Goal

Implement a new AgentOpt selector for **two-objective agent-configuration search** with:

- objective 1: task performance / accuracy, maximize;
- objective 2: deployment cost, minimize;
- cumulative **search cost** explicitly accounted for in the Gittins continuation cost;
- question-level partial evaluation in batches;
- a fixed set of radial/Chebyshev directions;
- **round-robin scheduling over directions**;
- a **single shared posterior per configuration**, reused across all directions;
- a direction-specific radial-Gittins index and stopping decision;
- global stopping when every direction currently says further evaluation is not worth its search cost;
- final output of **all Pareto-optimal (nondominated) direction winners**, expected to number **around 3** (can be fewer or more).

The initial target is AgentOpt-style 2-role workflows with **81 configurations** (e.g. answerer-critic or planner-solver).

This is a new selector. Do not replace or refactor existing selectors unnecessarily.

---

# 1. High-level algorithm

The key design is:

> Directions are different views of the same 81 configuration posteriors. They are **not** separate experiments.

There should be exactly **one posterior state per configuration**.

A physical question-batch evaluation is paid for once and updates that configuration's accuracy/cost posterior. Every scalarization direction reuses that posterior.

Use a default set of 9 fixed 2D directions:

```python
DEFAULT_DIRECTIONS = [
    (0.1, 0.9),
    (0.2, 0.8),
    (0.3, 0.7),
    (0.4, 0.6),
    (0.5, 0.5),
    (0.6, 0.4),
    (0.7, 0.3),
    (0.8, 0.2),
    (0.9, 0.1),
]
```

Make the direction list configurable.

Use **round-robin** over these directions.

For the current direction:

1. compute its radial-Gittins index for every unfinished configuration;
2. compute terminal radial utility for every completed configuration;
3. if the largest direction-specific index is attained by a completed configuration, that direction currently says **STOP**; skip it;
4. otherwise evaluate the unfinished configuration with largest radial-Gittins index on one new batch of questions;
5. update that configuration's single shared vector posterior;
6. move to the next direction.

Do **not** permanently eliminate a direction. A direction that says stop now may become active later because another direction can cause a shared posterior update.

Global stop:

> If one complete pass over all directions performs zero new physical evaluations, stop globally.

Since the posterior does not change during such a fully skipped cycle, this is equivalent to saying all directions simultaneously satisfy their stopping condition.

---

# 2. Important terminology and invariants

For configuration `c`, define its unknown two-objective latent mean

\[
\theta_c =
(\theta_{c,1}, \theta_{c,2}),
\]

where, after normalization,

- \(\theta_{c,1}\): mean task score / accuracy desirability;
- \(\theta_{c,2}\): mean deployment-cost desirability.

Both are represented so that **larger is better**.

Maintain one posterior:

\[
p(\theta_c \mid D_c).
\]

Critical invariant:

```text
number of posterior objects = number of configurations
```

NOT:

```text
number of configurations × number of directions
```

A direction only changes how the same posterior is scalarized.

---

# 3. Question-batch reuse

Suppose batch size is `B`.

For each configuration, maintain a fixed seeded permutation of question IDs and a cursor into that permutation.

If direction 1 selects configuration `c17`, and `c17` has not been evaluated before, evaluate its first `B` questions.

If direction 4 later also selects `c17`, evaluate the **next** `B` previously unobserved questions for `c17`.

Never pay twice for the same `(configuration, question)` entry.

Different configurations may evaluate the same question IDs; that is normal.

Suggested per-configuration state:

```python
@dataclass
class ConfigState:
    config_id: Hashable
    n_batches: int
    n_questions: int
    next_question_pos: int
    completed: bool

    # normalized objective posterior
    mean: np.ndarray      # shape (2,)
    var: np.ndarray       # shape (2,) for independent Gaussian MVP

    # measured costs for reporting
    cumulative_actual_search_cost_usd: float
```

The dataset permutation should be reproducible from the experiment seed.

For offline cached matrices, define the evaluation universe before sampling.
If missing cells are truncated data collection rather than structural benchmark
availability, the main protocol must use the complete question intersection
for every configuration. Otherwise truncated arms receive shorter horizons and
can become eligible winners prematurely. Arm-specific ragged tails are allowed
only as an explicitly labeled diagnostic mode unless the missingness mechanism
is genuinely part of the benchmark.

---

# 4. Objective normalization

The radial scalarization should operate on two objectives with comparable scales.

Use fixed maps into `[0, 1]` after the mandatory warm-start calibration. Do
not update the maps during adaptive search.

## Objective 1: accuracy/performance

If the benchmark score is already in `[0, 1]`, use:

\[
y_1 = \text{score}.
\]

Otherwise add a configurable fixed affine normalization.

## Objective 2: deployment cost

Deployment cost is minimized, so turn it into a desirability score:

\[
y_2 = \frac{C_{\mathrm{ref}}}{C_{\mathrm{ref}} + C}.
\]

Then larger `y2` means cheaper deployment.

`C_ref` is a scale midpoint, not a maximum: a question whose cost equals
`C_ref` has cost desirability `0.5`.

Default calibration protocol:

1. evaluate every configuration on the same seeded first batch of `B` questions;
2. compute each configuration's raw warm-batch mean cost;
3. set `C_ref` to the median of those configuration means (equal arm weighting);
4. freeze `C_ref` for the rest of the replay;
5. transform each question's cost before taking a batch mean.

An explicit positive `C_ref` may override calibration. Never derive it from
hidden full response-matrix outcomes in the default replay. If the calibrated
median is zero, fail clearly rather than inventing an epsilon.

Implement a reusable normalizer:

```python
class ObjectiveNormalizer:
    def normalize_score(self, score: float) -> float: ...
    def normalize_deployment_cost(self, cost_usd: float) -> float: ...
    def normalize_batch(
        self,
        scores: np.ndarray,
        costs_usd: np.ndarray,
    ) -> np.ndarray:
        """Return shape (B, 2)."""
```

Do not conflate:

- **deployment cost objective**, which is part of the Pareto vector;
- **search cost**, which is the money paid to obtain new observations.

They may originate from the same workflow execution, but they play different roles.

---

# 5. Bayesian posterior model

For the first implementation, use the same spirit as the scalar Gaussian Gittins design:

- Gaussian prior;
- Gaussian approximation to batch means;
- fixed observation-noise variance;
- deterministic posterior-variance schedule.

Use independent Gaussian objectives in the MVP.

For objective \(j\in\{1,2\}\):

\[
\theta_j \sim N(\mu_{0,j}, v_{0,j}),
\]

and for one batch mean:

\[
Y_{j,n} \mid \theta_j
\sim N(\theta_j, \tau_j^2).
\]

Default conservative noise for normalized bounded observations:

\[
\tau_j^2 = \frac{1}{4B}.
\]

This works as a worst-case bounded-variable batch-mean variance for values in `[0,1]`.

Posterior update:

\[
v_{j,n+1}
=
\left(
v_{j,n}^{-1} + \tau_j^{-2}
\right)^{-1},
\]

\[
\mu_{j,n+1}
=
\mu_{j,n}
+
\frac{v_{j,n}}
     {v_{j,n} + \tau_j^2}
\left(
Y_{j,n+1} - \mu_{j,n}
\right).
\]

Implement this as a small reusable vector-posterior object.

Use the same uniform warm batch to learn one common plug-in prior mean:

\[
\mu_0 = \frac{1}{K}\sum_{c=1}^{K}\bar y_c^{\mathrm{warm}},
\qquad
v_0=(0.04,0.04).
\]

Collect every arm's raw warm observations before fitting either `C_ref` or
`mu_0`. Initialize every arm from this same prior, then apply that arm's warm
batch exactly once as a likelihood update. Freeze the fitted hyperparameters
before adaptive selection.

The warm data therefore estimates the shared plug-in hypermean and also enters
each arm's likelihood. This is the chosen empirical-Bayes approximation; it
ignores hyperparameter uncertainty. “Once” means the arm batch is not applied
twice as a posterior likelihood.

The fixed prior variance remains configurable, with default:

```python
prior_var = np.array([0.04, 0.04])
```

### Future extension, not required for MVP

A fixed 2x2 observation covariance could model correlation between accuracy and deployment cost. Keep the code structure open to this, but do not block the MVP on it.

---

# 6. Radial / Chebyshev scalarization

For a direction

\[
\lambda = (\lambda_1,\lambda_2),
\qquad
\lambda_i > 0,
\qquad
\lambda_1 + \lambda_2 = 1,
\]

use a dominated reference point

\[
r = (r_1,r_2).
\]

With normalized objectives, default:

```python
reference_point = np.array([0.0, 0.0])
```

Do not use the positive-part clipping in the MVP; use standard radial/Chebyshev scalarization directly so that translational invariance is preserved.

Raw radial scalarization:

\[
\rho_\lambda(y)
=
\min_i
\frac{y_i-r_i}{\lambda_i}.
\]

Because the numeric scale differs across directions, normalize each direction so its maximum radial value over `[0,1]^2` is 1.

For `r=(0,0)` and simplex directions:

\[
M_\lambda
=
\min_i \frac{1}{\lambda_i}
=
\frac{1}{\max_i \lambda_i}.
\]

Therefore define

\[
\bar\rho_\lambda(y)
=
\frac{\rho_\lambda(y)}
     {M_\lambda}
=
a_\lambda
\min_i
\frac{y_i}{\lambda_i},
\]

with

\[
a_\lambda = \max(\lambda_1,\lambda_2).
\]

This preserves the maximizing configuration for the direction while putting all direction-specific utilities on approximately the same `[0,1]` scale.

Implement:

```python
def direction_scale(direction: np.ndarray) -> float:
    return float(np.max(direction))

def radial_utility(
    objective: np.ndarray,
    direction: np.ndarray,
    reference: np.ndarray,
) -> float:
    a = direction_scale(direction)
    x = a * (objective - reference) / direction
    return float(np.min(x))
```

---

# 7. Transform one configuration posterior for one direction

For one direction, define scaled latent coordinates

\[
X_1
=
a_\lambda
\frac{\theta_1-r_1}{\lambda_1},
\qquad
X_2
=
a_\lambda
\frac{\theta_2-r_2}{\lambda_2}.
\]

Then

\[
\bar\rho_\lambda(\theta)
=
\min(X_1,X_2).
\]

If the base posterior means are \(\mu_1,\mu_2\), define

\[
m_1
=
a_\lambda
\frac{\mu_1-r_1}{\lambda_1},
\qquad
m_2
=
a_\lambda
\frac{\mu_2-r_2}{\lambda_2}.
\]

Use the coordinate transform

\[
u = \frac{m_1+m_2}{2},
\qquad
\delta = m_1-m_2.
\]

Interpretation:

- `u`: overall radial level;
- `delta`: imbalance between the two direction-scaled objectives.

Also:

\[
\min(m_1,m_2)
=
u-\frac{|\delta|}{2}.
\]

The radial Gittins stopping boundary will be parameterized by `delta`.

---

# 8. Posterior-mean transition under another batch

For base objective `j`, the posterior-mean transition variance is:

\[
q_{j,n}
=
v_{j,n} - v_{j,n+1}.
\]

After direction scaling:

\[
\tilde q_{1,n}
=
\left(\frac{a_\lambda}{\lambda_1}\right)^2
q_{1,n},
\]

\[
\tilde q_{2,n}
=
\left(\frac{a_\lambda}{\lambda_2}\right)^2
q_{2,n}.
\]

For the independent-objective MVP, the increment

\[
(\Delta u,\Delta\delta)
\]

is zero-mean Gaussian with covariance

\[
\Sigma_{u,\delta,n}
=
\begin{pmatrix}
(\tilde q_{1,n}+\tilde q_{2,n})/4
&
(\tilde q_{1,n}-\tilde q_{2,n})/2
\\
(\tilde q_{1,n}-\tilde q_{2,n})/2
&
\tilde q_{1,n}+\tilde q_{2,n}
\end{pmatrix}.
\]

This covariance is deterministic given:

- direction;
- local pull count `n`;
- prior variance;
- observation noise;
- batch size.

That determinism is what makes offline boundary precomputation possible.

---

# 9. Terminal expected radial utility

For a completed configuration, do **not** simply use `min(m1, m2)` if posterior variance is nonzero.

The terminal reward for the Bayesian stopping problem is:

\[
E[\min(X_1,X_2)\mid D].
\]

For jointly Gaussian `X1`, `X2`, this has a closed form.

For independent `X1`, `X2`, let:

\[
s^2 = \operatorname{Var}(X_1)+\operatorname{Var}(X_2),
\]

\[
s = \sqrt{s^2},
\]

\[
d = \frac{m_1-m_2}{s}.
\]

Then:

\[
E[\min(X_1,X_2)]
=
m_1\Phi(-d)
+
m_2\Phi(d)
-
s\phi(d),
\]

where `Phi` is the standard normal CDF and `phi` is the standard normal PDF.

If `s` is numerically near zero, fall back to:

\[
\min(m_1,m_2).
\]

Because

\[
m_1=u+\delta/2,\qquad
m_2=u-\delta/2,
\]

write terminal expected radial utility as:

\[
R_H(u,\delta)
=
u+\psi_H(\delta),
\]

where `psi_H(delta)` is the non-linear imbalance term implied by the formula above.

Implement a tested function:

```python
def expected_min_of_two_normals(
    m1: float,
    v1: float,
    m2: float,
    v2: float,
    cov12: float = 0.0,
) -> float:
    ...
```

Use the general variance of the difference:

\[
s^2 = v_1+v_2-2\,cov_{12}.
\]

Even if the MVP posterior is independent, this makes the helper more general.

---

# 10. Radial-Gittins dynamic program

This is the main new implementation.

For one fixed:

- direction \(\lambda\);
- effective per-batch search cost \(c\);
- prior/noise schedule;
- horizon `H`;

define a centered value function with state:

\[
(z,\delta),
\]

where

\[
z = u-\alpha
\]

and `alpha` is the outside terminal option.

Translational invariance allows `alpha` to be removed from the DP.

## Terminal stage

At `n = H`:

\[
W_H(z,\delta)
=
\max
\left\{
0,
z+\psi_H(\delta)
\right\}.
\]

## Backward recursion

For `n < H`:

\[
q_n(z,\delta)
=
-c
+
E\left[
W_{n+1}
(
z+\Delta u,
\delta+\Delta\delta
)
\right],
\]

with

\[
(\Delta u,\Delta\delta)
\sim
N(0,\Sigma_{u,\delta,n}).
\]

Then:

\[
W_n(z,\delta)
=
\max\{0,q_n(z,\delta)\}.
\]

For each fixed `delta`, define the stopping boundary `b_n(delta)` by:

\[
q_n(b_n(\delta),\delta)=0.
\]

Continue if:

\[
z>b_n(\delta).
\]

The direction-specific Gittins index for an unfinished configuration is therefore:

\[
\boxed{
G_n(u,\delta)
=
u-b_n(\delta)
}
\]

This is the direct analogue of the scalar design:

\[
G_n(s)=s-r_n.
\]

The original scalar root `r_n` becomes a curve / stopping boundary `b_n(delta)`.

---

# 11. Numerical boundary precomputation

Do this deterministically. Do **not** use Monte Carlo.

Implement a `RadialGittinsBoundaryTable`.

Suggested interface:

```python
@dataclass(frozen=True)
class BoundaryKey:
    direction: tuple[float, float]
    effective_pull_cost: float
    prior_var: tuple[float, float]
    obs_noise_var: tuple[float, float]
    horizon: int

class RadialGittinsBoundaryTable:
    delta_grid: np.ndarray
    boundaries: np.ndarray
    # shape: (H + 1, len(delta_grid)); row H is the analytic terminal root
    # boundaries[n, k] = b_n(delta_grid[k])

    def boundary(self, n: int, delta: float) -> float:
        """1D interpolation in delta."""
```

For the backward DP, use the equivalent centered coordinates:

\[
x_1=z+\delta/2,
\qquad
x_2=z-\delta/2.
\]

Their increments are the two independent direction-scaled objective-mean
increments. This avoids rasterizing the highly correlated transition kernel
that appears directly in `(z, delta)` coordinates.

Preferred first implementation:

- use a regular `(x1, x2)` value grid with a sufficiently wide halo;
- integrate the Gaussian analytically against each piecewise-linear grid
  basis function, then apply the two resulting one-dimensional kernels with
  `scipy.signal.fftconvolve`; sampled Gaussian weights are not acceptable
  because late-stage transition standard deviations can be smaller than one
  grid cell and silently collapse to a point mass;
- derive separate per-axis halos from six times the cumulative scaled
  posterior-mean transition standard deviation;
- tighten the `delta` range to the direction-reachable normalized objective
  box and retain sufficiently wide padded state grids;
- enlarge the `z` root-search band for required completion by the possible
  cumulative pull penalty `H * effective_pull_cost`;
- after convolution subtract `effective_pull_cost`;
- apply `np.maximum(0, q)`;
- bilinearly interpolate `q` along
  `(x1, x2)=(z+delta/2, z-delta/2)`;
- find the zero crossing along the `z` line for every `delta` grid point;
- linearly interpolate the root.

Do not save every full `W_n` to disk unless useful for debugging. Only the final boundary schedule is required online.

### Numerical requirements

Expose grid settings in configuration.

The implementation must:

- warn/fail if a root is too close to a `z` grid boundary;
- include grid-refinement tests;
- compare the FFT result against brute-force Gaussian quadrature on small synthetic cases;
- include a sub-grid ReLU expectation test so the convolution cannot regress
  to a sampled point-mass kernel;
- verify monotonicity of `q_n(z, delta)` in `z` numerically;
- use interpolation for `b_n(delta)`.

Correctness is more important than premature optimization.

---

# 12. Search-cost handling

Keep **search cost** separate from the deployment-cost objective.

The Gittins immediate cost must be in the same normalized utility units as radial utility.

Use:

\[
c_{\text{eff},c}
=
\eta_{\text{search}}
\cdot
\widehat \kappa_c,
\]

where:

- `eta_search`: configurable conversion from dollars to normalized utility;
- `kappa_hat_c`: estimated dollar cost of evaluating one batch for configuration `c`.

For the first implementation, assume `kappa_hat_c` is a **fixed expected per-batch evaluation cost** supplied by an estimator or calibration layer.

Do not make the exact 2D DP depend on a continuously changing posterior search-cost estimate in the MVP.

Preferred API:

```python
class PullCostEstimator(Protocol):
    def expected_batch_cost_usd(
        self,
        config_id,
        batch_size: int,
    ) -> float:
        ...
```

Use existing AgentOpt pricing/cost infrastructure if available.

For offline replay, use the already observed warm batch as the no-leak default:

\[
\widehat\kappa_c
= B\,\overline C_c^{\mathrm{warm}}.
\]

Freeze this arm-specific estimate after calibration. Accept an explicit scalar,
vector, or configuration mapping as an override. The estimator is intentionally
simple and should be treated as a warm-start plug-in, not as known future cost.

Always separately record the **actual** realized API/search cost from each physical evaluation for plots and budget accounting.

### Cost-table practicality

The 2D boundary depends on effective pull cost.

Do not eagerly compute hundreds of huge boundary tables.

Implement caching keyed by a rounded/quantized effective cost.

The offline implementation uses fixed-anchor multiplicative bins so tables are
reused across configurations and seeds. For example:

```python
effective_cost_bin_anchor = 1e-4
effective_cost_bin_ratio = 2.0
```

or a configurable small set of cost bins.

Start with a small number of bins and validate sensitivity.

---

# 13. Online direction-specific index

For an unfinished configuration `c` at local pull count `n` under direction `lambda`:

1. get shared base posterior mean and variance;
2. transform to direction-scaled `m1`, `m2`;
3. compute:

\[
u=(m_1+m_2)/2,
\qquad
\delta=m_1-m_2;
\]

4. get effective pull cost for `c`;
5. fetch/interpolate the corresponding boundary table;
6. compute:

\[
G_{c,n}^{(\lambda)}
=
u-b_n(\delta).
\]

For a completed configuration:

\[
G_{c,H}^{(\lambda)}
=
E[\min(X_1,X_2)\mid D_c].
\]

This completed index is its terminal expected radial utility.

---

# 14. Direction-specific stopping rule

For a direction, calculate indices for all configurations.

A direction currently says **STOP** if an arm attaining the maximum current direction-specific index is completed.

Equivalent practical implementation:

```python
best_completed = max(terminal_index[c] for c if completed)
best_unfinished = max(gittins_index[c] for c if not completed)

direction_stops = (
    best_completed exists
    and best_completed >= best_unfinished - stop_tolerance
)
```

If no completed configuration exists yet, the direction cannot stop.

If the direction continues, select:

```python
selected_config = argmax_unfinished gittins_index[c]
```

This mirrors the required-completion convention of the scalar Gittins implementation.

---

# 15. Round-robin scheduler

Implement the scheduler exactly and simply.

Pseudo-code:

```python
direction_idx = 0
skipped_since_last_eval = 0

while search_budget_remaining:

    direction = directions[direction_idx]

    status = evaluate_direction_status(direction)

    if status.should_stop:
        skipped_since_last_eval += 1

        # If all directions were skipped with no posterior change,
        # all directions simultaneously say stop.
        if skipped_since_last_eval == len(directions):
            stop_reason = "all_directions_gittins_stop"
            break

        direction_idx = (direction_idx + 1) % len(directions)
        continue

    selected_config = status.best_unfinished_config

    batch = next_unobserved_batch(selected_config)

    result = evaluator.evaluate(
        config=selected_config,
        question_ids=batch,
    )

    update_shared_vector_posterior(
        selected_config,
        result,
    )

    record_actual_search_cost(result.cost_usd)

    skipped_since_last_eval = 0
    direction_idx = (direction_idx + 1) % len(directions)
```

Important:

- only **one physical configuration batch** is evaluated per active round;
- do not launch one batch per direction;
- direction-specific computations may be vectorized/parallelized, but physical evaluation remains sequential in this MVP;
- every later direction automatically sees the new shared posterior.

---

# 16. Hard budget cap

Keep a hard budget cap even with endogenous Gittins stopping.

Support at least:

```python
max_search_cost_usd: float | None
max_budget_fraction_of_bruteforce: float | None
max_total_question_evaluations: int | None
```

Stop at whichever criterion triggers first.

This lets the same selector be evaluated both:

1. under a fixed search budget;
2. at its own endogenous Gittins stopping point.

---

# 17. Final recommendation: return all Pareto-optimal pairs (around 3 expected)

With the default 9 directions, the final recommendation is **not** capped at 3.

At the end:

1. for each direction, select its best completed configuration by terminal expected radial utility;
2. deduplicate the direction winners;
3. compute each winner's estimated objective vector in the original interpretable scale:
   - performance/accuracy;
   - deployment cost;
4. remove dominated configurations;
5. **return all remaining Pareto-optimal (nondominated) pairs**.

The returned set size can be fewer than 3 or more than 3. With 9 fixed directions, the expected cardinality is **around 3**; do not force exactly 3 and do not subset-select by hypervolume for the recommendation itself.

Return:

```python
@dataclass
class MultiObjectiveSearchResult:
    selected_configs: list
    nondominated_archive: list
    direction_winners: dict
    cumulative_search_cost_usd: float
    total_question_evaluations: int
    stopped_by_gittins: bool
    stop_reason: str
    trace: ...
```

---

# 18. Hypervolume conventions

Use hypervolume only for experiment evaluation (e.g. regret curves), not for truncating the final recommendation.

Do not use hypervolume as the online acquisition rule in this implementation.

Be explicit about maximization/minimization convention.

Recommended internal convention:

- normalized accuracy desirability: maximize;
- normalized cost desirability: maximize;
- reference point: `(0, 0)`.

For user-facing reporting, convert back to:

- accuracy: higher better;
- deployment cost USD: lower better.

---

# 19. Experiment metrics

For replay experiments where the full 81-configuration matrix is available only to the evaluator, not the selector, compute:

## Primary

### Hypervolume regret of the returned Pareto set vs cumulative search cost

Let \(S_t\) be the selector's returned nondominated set at search time `t` (all Pareto-optimal direction winners; size expected around 3).

Let \(HV^\star\) be the hypervolume of the ground-truth nondominated front over the full configuration matrix.

Report:

\[
HVRegret(t)
=
HV^\star - HV(S_t).
\]

Optionally also report an around-3 diagnostic, e.g. regret vs the best ground-truth 3-set \(HV_3^\star = \max_{S\subseteq C,\ |S|\le3} HV(S)\), since returned cardinality is expected near 3—but the primary returned object remains the full nondominated direction-winner set, not a size-capped subset.

Plot against:

- actual cumulative search cost USD;
- percentage of brute-force search cost.

## Secondary

- returned-set cardinality (expect around 3);
- full returned archive hypervolume;
- generational distance (GD) and inverted generational distance (IGD) of the
  returned archive vs the ground-truth desirability front;
- number of distinct directional winners;
- number of nondominated returned configurations;
- fraction of directions currently stopped;
- total `(config, question)` entries evaluated;
- actual Gittins stopping cost;
- search-cost savings vs brute force.

---

# 20. Baseline/ablation hooks

Do not block the main implementation on these, but make the selector easy to compare against:

1. weighted-sum Gittins;
2. radial-Gittins with no search-cost penalty;
3. radial-Gittins with fixed budget only;
4. direction counts:
   - 5,
   - 9,
   - 17;
5. round-robin vs a future adaptive direction scheduler.

For now, **round-robin is the default and preferred scheduler**.

---

# 21. Logging

Log enough state to debug selection decisions.

At every direction visit:

```text
global_step
direction_idx
direction
direction_should_stop
best_completed_config
best_completed_terminal_index
best_unfinished_config
best_unfinished_gittins_index
selected_config_or_none
```

At every physical batch evaluation:

```text
config_id
question_ids
batch_score_mean
batch_deployment_cost_mean
actual_batch_search_cost_usd
posterior_mean_before
posterior_mean_after
posterior_var_before
posterior_var_after
cumulative_search_cost_usd
```

At global stop:

```text
stop_reason
direction_winners
nondominated_archive
selected_top_k
```

---

# 22. Tests

Add unit tests before running expensive workflow evaluations.

## Posterior tests

- posterior variance decreases monotonically;
- vector update equals two independent scalar Gaussian updates;
- repeated identical observations move the posterior mean appropriately.

## Radial scalarization tests

- balanced direction `(0.5, 0.5)` behaves as expected;
- scaling a direction before direction-normalization does not change the directional maximizer;
- a known unsupported Pareto point can be selected by radial scalarization even when weighted sum misses it.

Example:

```text
A = (1.0, 0.0)
B = (0.4, 0.4)
C = (0.0, 1.0)
```

Balanced radial scalarization should prefer `B`.

## Expected-min tests

Compare `expected_min_of_two_normals` against a large Monte Carlo sample in tests only.

Monte Carlo is allowed for validation tests, not for the online algorithm.

## DP tests

For a tiny horizon and coarse grid:

- compare FFT convolution against direct numerical integration;
- verify `q_n(z, delta)` is nondecreasing in `z`;
- verify extracted stopping boundary is stable under moderate grid refinement;
- verify larger effective pull cost makes continuation less attractive;
- verify as pull cost approaches zero, the continuation region expands.

## Scheduler tests

- one direction's batch updates the shared posterior seen by every direction;
- question entries are not reevaluated for the same configuration;
- skipped directions consume no search budget;
- a previously skipped direction can become active after another direction updates a shared posterior;
- one full skipped cycle triggers global stop;
- physical evaluations occur one batch at a time.

## Final-output tests

- dominated configurations are removed;
- all nondominated direction winners are returned (no hard cap at 3; size may be fewer or more than 3).

---

# 23. Repository integration instructions for Cursor

Before writing code:

1. inspect the AgentOpt repository structure;
2. identify the existing selector interface/base class;
3. identify how configurations are represented;
4. identify the current evaluator API for evaluating a configuration on a subset/batch of datapoints;
5. identify existing cost/latency accounting;
6. identify current experiment logging and result serialization;
7. identify whether any scalar Gittins implementation already exists in this repository or is imported from another local module.

Then implement this feature using the existing abstractions.

Do not invent a parallel framework if AgentOpt already has:

- selector base classes;
- response-matrix abstractions;
- batch evaluation utilities;
- pricing utilities;
- experiment trace structures.

Keep the diff minimal.

Suggested names, only if they fit the current repository:

```text
RadialGittinsSelector
RadialGittinsBoundaryTable
GaussianVectorPosterior
ObjectiveNormalizer
PullCostEstimator
```

Do not rename existing public APIs unless necessary.

---

# 24. Suggested implementation phases

## Phase 1: infrastructure + scalarization

Implement:

- objective normalization;
- vector posterior;
- fixed directions;
- shared per-configuration question state;
- round-robin scheduler skeleton;
- radial scalarization;
- terminal expected-min formula.

No expensive workflow experiment yet.

## Phase 2: deterministic radial-Gittins DP

Implement:

- an `(x1, x2)` value grid equivalent to the `(z, delta)` state;
- separable one-dimensional Gaussian convolutions for the independent scaled
  objective increments;
- backward DP;
- extraction of the `b_n(delta)` stopping boundary;
- interpolation;
- caching.

Validate thoroughly on synthetic problems.

## Phase 3: AgentOpt selector integration

Wire the selector into the existing evaluation loop.

Run a tiny synthetic/fixture benchmark first.

## Phase 4: 81-configuration replay

Run on one 81-config benchmark with a small number of seeds.

Check:

- no duplicate `(config, question)` evaluations;
- actual search cost accounting;
- number of direction winners;
- stopping behavior;
- returned Pareto-set HV regret (and cardinality around 3).

## Phase 5: full experiment

Run planner-solver and answerer-critic benchmarks.

Add fixed-budget curves plus automatic Gittins-stop markers.

---

# 25. Configuration surface

Expose a small config object / CLI surface similar to:

```python
@dataclass
class RadialGittinsConfig:
    directions: list[tuple[float, float]]
    batch_size: int
    prior_var: tuple[float, float]
    obs_noise_var: tuple[float, float] | None
    cost_reference_usd: float | None  # None => fit from uniform warm batch

    reference_point: tuple[float, float]

    search_cost_scale_eta: float
    cost_round_digits: int

    horizon_batches: int | None

    z_grid_min: float
    z_grid_max: float
    z_grid_size: int
    delta_grid_min: float
    delta_grid_max: float
    delta_grid_size: int

    stop_tolerance: float

    max_search_cost_usd: float | None = None
    max_total_question_evaluations: int | None = None

    seed: int = 0
```

If existing AgentOpt config conventions differ, adapt to them rather than forcing this exact dataclass.

---

# 26. Default behavior

Default conceptual setup:

```python
directions = [
    (0.1, 0.9),
    (0.2, 0.8),
    (0.3, 0.7),
    (0.4, 0.6),
    (0.5, 0.5),
    (0.6, 0.4),
    (0.7, 0.3),
    (0.8, 0.2),
    (0.9, 0.1),
]

scheduler = "round_robin"
reference_point = (0.0, 0.0)
# final recommendation: all nondominated direction winners (expect ~3)
```

Use the repository's existing batch size if one is already standard; otherwise make it configurable rather than hard-coding a new value.

---

# 27. Non-goals for this implementation

Do **not** add the following in the first version:

- planner/executor/verifier intermediate feedback;
- cross-configuration GP/categorical-kernel correlation;
- hypervolume improvement as the online acquisition rule;
- random direction resampling every iteration;
- adaptive direction scheduling;
- nested Monte Carlo Gittins;
- three-objective accuracy-cost-latency optimization;
- per-query routing at deployment time.

The goal is to get the clean two-objective question-level partial-evaluation algorithm correct first.

---

# 28. Algorithm summary

The final algorithm should be easy to describe as:

```text
Initialize one 2D Gaussian posterior for each of 81 configurations.
Fix 9 radial/Chebyshev directions.

Repeat directions in round-robin order:

    For the current direction:
        Compute terminal radial values for completed configs.
        Compute radial-Gittins indices for unfinished configs.

        If a completed config has the largest index:
            skip this direction for now.
        Else:
            choose the unfinished config with largest index;
            evaluate it on the next unseen batch of benchmark questions;
            observe score + deployment cost;
            pay and record actual search cost once;
            update that config's single shared 2D posterior.

    If an entire direction cycle performs no physical evaluation:
        stop globally.

Return each direction's best completed config.
Deduplicate and remove dominated configs.
Return all remaining Pareto-optimal pairs (expect around 3; can be fewer or more).
```

Core principle:

\[
\boxed{
\text{round-robin over directions}
+
\text{Gittins within a direction}
+
\text{shared observations across all directions}
}
\]

Do not implement 9 independent searches.

---

# 29. Acceptance criteria

The implementation is complete when:

- [ ] it runs through the existing AgentOpt experiment pipeline;
- [ ] only one shared vector posterior exists per configuration;
- [ ] directions are visited round-robin;
- [ ] one physical batch is evaluated at a time;
- [ ] every batch contains only previously unobserved questions for that configuration;
- [ ] the same posterior update is visible to every direction;
- [ ] radial/Chebyshev scalarization uses the configured directions;
- [ ] no Monte Carlo is used online for the radial Gittins index;
- [ ] a deterministic 2D stopping-boundary table is used;
- [ ] search cost enters the continuation value;
- [ ] each direction can independently skip/continue under the current posterior;
- [ ] a full no-evaluation cycle triggers global stopping;
- [ ] fixed-budget caps are also supported;
- [ ] final recommendations are nondominated;
- [ ] all Pareto-optimal direction winners are returned (no hard size cap; expect around 3);
- [ ] traces contain enough information to reproduce and explain every selection;
- [ ] unit tests cover posterior updates, radial utility, expected-min formula, DP boundary behavior, round-robin reuse, stopping, and final Pareto-set selection.

---

# 30. Important caution about theory

Do not claim in code comments or documentation that the full multi-direction round-robin policy is globally Bayes-optimal for hypervolume.

The intended interpretation is:

- for one fixed direction, radial scalarization defines a scalar terminal utility and supports a Gittins-style stopping problem;
- observations are shared across directions;
- round-robin is a simple practical outer scheduler;
- the global all-directions-stop rule is a practical extension, not yet a proved optimal policy for a joint hypervolume objective.

Keep this distinction explicit.
