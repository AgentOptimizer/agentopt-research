# Code and data provenance

The package contains offline model-selection code and numeric benchmark
response matrices. Benchmark names, public model-provider names, published
method names, and pricing-source URLs are retained to describe the experiments.
Their inclusion does not imply ownership of those external works.

The HotpotQA and MathQA matrices contain per-configuration scores, costs,
latencies, and token counts. The SCOPE matrices contain scores, token-derived
costs, token counts, configuration metadata, and compressed numeric records.
The underlying question text and model response text are not part of these
matrix exports. Metadata describes extraction and pricing inputs using
portable paths. Original external inputs are unnecessary for replay.

This archive omits the legacy scalar-Gittins implementation, which is not
needed by the current CC-Gittins experiment path. Existing copyright and
license notices on those omitted sources remain intact in the development
repository; this package does not replace or relicense them.

No new repository-wide license or data-redistribution permission is asserted
by these notes. External benchmarks, sources, and dependencies remain subject
to their applicable terms. The submitted manuscript provides scientific
citations. A public release license requires a separate decision by the rights
holders.
