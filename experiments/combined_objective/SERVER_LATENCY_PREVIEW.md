# Three-objective server-latency preview

This experiment evaluates quality (maximize), cumulative server-side LLM
latency (minimize), and deployment cost in USD (minimize) on complete
HotpotQA and MathQA 100-configuration × 200-question matrices. Search cost
remains the recorded USD cost of evaluated questions. No model API is called
during the offline replay.

The latency input is `data/{hotpotqa,mathqa}/latency_matrix_seconds.csv`.
Each entry converts the lookup table's cumulative `server_latency_ms` to
seconds. The source tables are
`experiments/missing_cells/completed/{hotpotqa,mathqa}_lookup.10x10.csv`
from `AgentOptimizer/agentopt` commit `a4a6eb7`. The adjacent
`server_latency_manifest.json` files record source and output hashes,
row provenance, and exceptional cells. The build script checks that every
latency cell matches a configuration and question in the accuracy and cost
matrices, with the same recorded score and cost. Generate the latency matrices
locally with `build_server_latency_matrices.py`; the generated matrices and
manifests are excluded from Git.

`server_latency_ms` is the sum of successful invocation response metrics.
It excludes client processing, network transfer, tool work, and worker queue
time. Four MathQA context-overflow entries have only the successful-call
subtotal before a rejected request; their latency values are lower bounds.
The historical lookup tables do not provide full per-question call lists,
so a historical total cannot independently establish that every original
workflow call was included.

The replay uses seeds 42–61 and records recommendations at physical batch
boundaries. Each search runs to its natural stop or complete evaluation
budget; figures select checkpoints at 0–30% of exhaustive USD search cost.
The main method and all three-objective direction, acquisition-cost,
continuation, and Q/L/D prior controls use the same aligned matrices.
Random and Pareto baselines are recomputed on those inputs. The policy-runtime
comparison separately forces exactly 6,000 evaluated entries for each
direction configuration and seed.

The all-warm protocol estimates the quality, latency, and deployment-cost
prior variances from warm-start observations. Save its local figure set under
`figures/server_latency_all_warm_20261028`. A separate sensitivity analysis
uses quality prior variance 0.04 while estimating the other two variances
from warm start; save its local output under
`figures/server_latency_original_protocol_20261028`. The figure manifests
state which protocol each set uses. These generated outputs are excluded
from Git.

The renderer `render_server_latency_paper_figures.py` writes six figure
previews and source summaries to a requested output directory. The runners
and renderer default to the all-warm protocol. Set
`QUALITY_PRIOR_VARIANCE_MODE=fixed` for the Slurm array and
`--main-quality-prior-mode fixed` for the renderer to reproduce the
fixed-quality sensitivity analysis. The renderer does not write to the paper
repository.
