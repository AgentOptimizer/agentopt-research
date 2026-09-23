# Canonical 10%/30% search-cost checkpoints

The current canonical dataset is:

`analysis/usd_cost_checkpoints_latest_under_20seed`

Its version is `usd-checkpoints-latest-under-v1` and its budget axis is realized
cumulative search USD divided by the exhaustive configuration-by-question
matrix USD.

For each target in `{0.10, 0.30}`, a freshly replayed baseline stores the
recommendation after the latest completed atomic policy update whose realized
cost fraction is at or below the target. The next atomic update may cross the
target, but its observations and recommendation are not included in that
checkpoint.

Radial-Gittins is not replayed for this version. Its existing 160 results retain
every recommendation membership change, which identifies the recommendation
interval active at each target. The dataset builder reads those runs from
`analysis/20seed_results/radial_gittins`.

Do not use `analysis/usd_cost_checkpoints_20seed` for new figures; that directory
contains the superseded first-crossing run family. Build the canonical combined
CSV only with:

```bash
PYTHONPATH=src:. .venv-plot/bin/python \
  experiments/combined_objective/build_usd_checkpoint_dataset.py \
  --require-complete
```

The builder rejects baseline outputs with checkpoint schema versions below 3.
