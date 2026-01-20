# Architecture

This page describes Barsmith’s `comb` pipeline at a high level.

## Crates

- `barsmith_cli`: CLI entrypoint (`comb`)
- `barsmith_builtin`: minimal built-in feature engineering + target labeling used by the default CLI
- `barsmith_rs`: core library (dataset loading, combination enumeration, evaluation, storage)
- `custom_rs`: example/advanced engine (not required by the default CLI)

## High-level flow

1. CLI parses flags and builds a `Config`.
2. The builtin engine:
   - reads the raw OHLCV CSV,
   - writes `output-dir/barsmith_prepared.csv`,
   - builds a feature catalog (`FeatureDescriptor`s) plus optional comparison predicates.
3. The core pipeline (`barsmith_rs`) runs:
   - loads the prepared dataset columnar (and applies optional date filters),
   - prunes to required columns,
   - builds caches/bitsets for fast evaluation,
   - enumerates combinations in a deterministic order (supports resume offsets),
   - evaluates each combination and persists qualifying results to Parquet + DuckDB,
   - optionally uploads batches to S3 (via AWS CLI).

The README contains a detailed flowchart: see `README.md` (“How `comb` works”).

## Durability model

Barsmith writes incremental Parquet parts under `output-dir/results_parquet/` and maintains a DuckDB catalog (`output-dir/cumulative.duckdb`) that provides a stable `results` view across all parts.

Resume is index-based and protected by a CSV fingerprint. This avoids silently “resuming on the wrong dataset”.

