# Extracted response matrices

Each HotpotQA and MathQA directory contains five separate matrices:

- `accuracy_matrix.csv`: benchmark score for each cell;
- `cost_matrix_usd.csv`: one USD cost value per configuration-question cell;
- `input_token_matrix.csv`: summed input tokens for the cell;
- `output_token_matrix.csv`: summed output tokens for the cell;
- `total_token_matrix.csv`: input plus output tokens.

Rows are the 100 model/workflow configurations from the complete 10x10 model
grid and columns are the 200 question IDs. Both committed matrices are complete:
100 configurations x 200 questions, with no blank cells. Each benchmark also
has `metadata.json` with completeness and the 20 highest-cost cells.

`aws_bedrock_prices_10x10.json` records the USD-per-million-token AWS Bedrock
input and output prices used by these two datasets. The per-cell cost is the sum
of each recorded model call's input and output token costs; it is a token-derived
cost, not a direct AWS invoice export.

The source is intentionally the current research lookup pickle under
`experiments/data/lookup/`. It is not silently mixed with the separate
aggregated JSONL run. That directory is gitignored, so regeneration needs the
pickles present locally:

```bash
PYTHONPATH=. .venv/bin/python data/extract_response_matrices.py
```

## SCOPE matrices

`data/scope/` contains the same five matrices for each SCOPE benchmark, plus
`configs.json` and compressed cell-level `records.csv.gz`. Regenerate them from
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
