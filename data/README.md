# Extracted response matrices

Each HotpotQA and MathQA directory contains five separate matrices:

- `accuracy_matrix.csv`: benchmark score for each cell;
- `cost_matrix_usd.csv`: one USD cost value per configuration-question cell;
- `input_token_matrix.csv`: summed input tokens for the cell;
- `output_token_matrix.csv`: summed output tokens for the cell;
- `total_token_matrix.csv`: input plus output tokens.

Rows are the 81 model/workflow configurations and columns are the 200 question
IDs. Blank cells mean the current lookup pickle has no observation; they do not
mean zero. Each benchmark also has `metadata.json` with missingness and the 20
highest-cost cells.

The source is intentionally the current research lookup pickle under
`experiments/results/cache_db_results/`. It is not silently mixed with the
separate aggregated JSONL run. Regenerate with:

```bash
PYTHONPATH=src:experiments .venv/bin/python data/extract_response_matrices.py
```
