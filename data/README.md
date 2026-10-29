# Extracted response matrices

Each HotpotQA and MathQA directory contains six separate matrices:

- `accuracy_matrix.csv`: benchmark score for each cell;
- `cost_matrix_usd.csv`: one USD cost value per configuration-question cell;
- `latency_matrix_seconds.csv`: cumulative server-side LLM latency in seconds;
- `input_token_matrix.csv`: summed input tokens for the cell;
- `output_token_matrix.csv`: summed output tokens for the cell;
- `total_token_matrix.csv`: input plus output tokens.

Rows are the 100 model/workflow configurations from the complete 10x10 model
grid and columns are the 200 question IDs. Both committed matrices are complete:
100 configurations x 200 questions, with no blank cells. Each benchmark also
has `metadata.json` with completeness and the 20 highest-cost cells, plus
`server_latency_manifest.json` with latency provenance and exceptional cells.

`aws_bedrock_prices_10x10.json` records the USD-per-million-token AWS Bedrock
input and output prices used by these two datasets. The per-cell cost is the sum
of each recorded model call's input and output token costs; it is a token-derived
cost, not a direct AWS invoice export.

Accuracy, cost, and token matrices come from research lookup pickles under
`experiments/data/lookup/`. Latency comes from the separate per-question
server lookup CSVs described below. Both sources are required to regenerate
all six matrices; the local pickles are gitignored:

```bash
python data/extract_response_matrices.py \
  --server-lookup-dir /path/to/agentopt/experiments/missing_cells/completed \
  --source-revision a4a6eb7891bec384e2d2984a575c5f7de6ac9ec5
```

## Server latency

Each latency cell is the lookup row's cumulative `server_latency_ms` divided
by 1,000. This is the sum of successful LLM invocation response
`metrics.latencyMs` values for the workflow. It excludes client processing,
network transfer, tool execution, and worker-queue time; it is not an
end-to-end wall-clock measurement or a parallel workflow's critical-path time.

The source is
[`AgentOptimizer/agentopt` at `a4a6eb7`](https://github.com/AgentOptimizer/agentopt/tree/a4a6eb7891bec384e2d2984a575c5f7de6ac9ec5/experiments/missing_cells/completed):
`hotpotqa_lookup.10x10.csv` and `mathqa_lookup.10x10.csv`.
Each committed manifest records the source revision, source and output SHA-256
hashes, shape, record provenance counts, and exceptional cells. The builder
requires exactly one finite, nonnegative latency for every configuration and
question, and checks its recorded score and USD cost against the corresponding
accuracy and cost matrix cells before writing latency outputs.

Four MathQA cells with answer model Qwen3 32B and critic DeepSeek R1
(questions 65, 93, 108, and 121) ended in context overflow. Their latency
contains only successful calls before the rejected request and is therefore
a lower bound. HotpotQA has no such flagged cells. Historical per-question
call lists are unavailable, so the historical aggregates cannot independently
establish that every original workflow call was included.

To regenerate only latency and its metadata from that source checkout:

```bash
python data/build_server_latency_matrices.py \
  --lookup-dir /path/to/agentopt/experiments/missing_cells/completed \
  --source-revision a4a6eb7891bec384e2d2984a575c5f7de6ac9ec5
```

Both commands run offline. Use a fresh experiment output directory after
regenerating inputs. Three-objective results and figures must use matching
server-latency inputs across methods; wall-clock results are not interchangeable.

## SCOPE matrices

`data/scope/` contains five matrices per SCOPE benchmark: accuracy, USD cost,
input tokens, output tokens, and total tokens. These directories do not include
the QA latency matrix. Each also contains `metadata.json`, `configs.json`, and
compressed cell-level `records.csv.gz`. Regenerate them from
the adjacent SCOPE checkout with:

```bash
.venv/bin/python data/extract_scope_matrices.py \
  --scope-root /path/to/SCOPE-LLM-optimizer
```

The extractor reads prices already defined by SCOPE and supplements historical
or experiment-specific model aliases from `data/scope_supplemental_prices.json`.
It refuses to write a dataset if any workflow model remains unpriced.

Supplemental prices are standard input/output token rates in USD per million
tokens. Native CNY prices for Doubao are converted using the documented fixed
rate in the supplemental file so regenerated matrices remain reproducible.
Gemini names ending in `flash1` and `flash3` are experiment-replicate aliases
for `gemini-3-flash-preview`, not distinct model SKUs.

`data/scope_sparse/` has the identical directory and file layout, but also
includes configurations that evaluated only some benchmark questions. Empty
matrix cells are missing observations, never zero. Each `configs.json` entry
records `observed_count`, `coverage_fraction`, and `is_complete`. Regenerate it
with:

```bash
.venv/bin/python data/extract_scope_matrices.py \
  --scope-root /path/to/SCOPE-LLM-optimizer \
  --output-root data/scope_sparse \
  --include-partial
```
