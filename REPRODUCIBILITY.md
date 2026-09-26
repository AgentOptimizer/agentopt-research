# Reviewer instructions

This package supplies the CC-Gittins implementation, frozen replay matrices,
experiment and plotting code, and tests. Replays make no model API calls.
The current paper protocol is the two-objective G0 configuration; the
three-objective latency scripts are exploratory extensions.

## Environment

The portable NumPy/SciPy path was tested with Python 3.12.14 on Windows x86-64.
`constraints-tested.txt` records the installed versions used for the checks;
it is a tested constraints file, not a cross-platform dependency lock. A fresh
network installation and optional JAX/PyTorch/BoTorch backends were not tested.

Create and activate an environment, then install from the extracted root:

```sh
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell instead:
# .venv\Scripts\Activate.ps1
python -m pip install -c constraints-tested.txt -e ".[radial,plots]" pytest
python -m pytest -q
```

Python 3.10+ is the declared source requirement; the recorded constraints are
intended for the tested Python 3.12 environment. JAX-specific tests skip when
JAX is absent. Install `.[radial-jax]` to test that optional backend, and
`.[bayesian]` for qNEHVI. Legacy scalar-Gittins modules are not distributed in
this archive; they are not used by the CC-Gittins protocol.

Checks on the extracted archive passed: 396 tests and 219 subtests, with six
optional-backend tests skipped. The bounded replay below and one full native
G0 MathQA seed-42 task both completed. The full task stopped at its numerical
eta floor after approximately 96.856% of exhaustive USD cost; the bounded run
stopped at its observation cap at approximately 2.493% of exhaustive USD cost.
These checks used the existing tested environment, not a fresh installation.

## Small offline execution check

```sh
python -m experiments.combined_objective.run_two_direction_ablation --benchmarks mathqa --seed 42 --observation-budget-fraction 0.025 --skip-combined-export --outdir smoke_results
python -m experiments.combined_objective.run_gittins_ablation_v2_slurm_task --configuration g0_current_gittins --task-id 20 --dry-run
```

The first command writes `smoke_results/exact_axes/mathqa/result.json` and
related diagnostics. The 2.5% cap is a smoke test, not the full paper protocol.
The second command prints the canonical HotpotQA seed-42 task without running
it. Native execution works without Slurm; Unix-only resource counters are
unavailable on Windows. Exact runtime depends on hardware and caches. The
bounded replay takes seconds in the tested environment; full-grid runtime and
peak memory have not been measured here. No GPU is needed for this smoke test.

## Canonical method and ablations

`experiments/combined_objective/gittins_ablation_v2.py` fixes the eight benchmark
order, seeds 42–61, ten configurations G0–G9, and four comparison families.
Each configuration has 160 tasks; the full grid has 1,600 runs.

For one full native control run, remove `--dry-run` from the task command
above. Run all G0 tasks before G1–G9 so validation can use the control truth.
Use the supplied Slurm array script for each configuration on a cluster;
provide your own account and environment as described in
[`experiments/README.md`](experiments/README.md#retained-ablations).

```sh
python -m experiments.combined_objective.aggregate_gittins_ablation_v2_20seed
python -m experiments.combined_objective.plot_gittins_ablation_v2_20seed
python -m experiments.combined_objective.plot_gittins_direction_runtime_v2_20seed
```

These commands validate and summarize a completed grid and produce the
ablation and direction-runtime figures. The archive does not contain the
1,600 historical run outputs. Paper-wide baseline comparisons and final
manuscript figures are not certified by the small execution check above.
Baseline implementations are included, but some legacy entry points require
lookup pickles not supplied here; the matrix-based CC-Gittins commands above
are self-contained. The three-objective baseline runner reads the supplied QA
matrices; see the experiment guide for its commands.

## Input integrity and package scope

`CHECKSUMS.sha256` lists SHA-256 hashes for every supplied file except itself.
The QA matrices cover 100 configurations and 200 questions each. Six complete
SCOPE datasets are included. Their numeric cells are unchanged by packaging.
`data/README.md` explains formats and extraction provenance; source lookup
pickles and external SCOPE checkouts are unnecessary for replaying these matrices.

The archive omits sparse duplicate datasets, retired scalar-Gittins modules,
historical results, caches, Git history, and manuscript sources. The Python
import name `agentopt` is retained for compatibility. Packaging does not grant
new rights over third-party code or benchmark data; see
[`DATA_AND_CODE_NOTES.md`](DATA_AND_CODE_NOTES.md).

To rebuild this source selection after edits:

```sh
python tools/build_anonymous_supplement.py
```

The builder uses an explicit file manifest, neutral archive timestamps, and an
empty archive comment. It does not read Git metadata or include unlisted files.
