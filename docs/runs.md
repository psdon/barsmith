# Running experiments

This page focuses on long-running `comb` runs: organizing output dirs, resuming safely, and choosing batch sizes.

## Output directory layout

Treat `--output-dir` as the run folder. Typical contents:

- `barsmith_prepared.csv`
- `results_parquet/part-*.parquet`
- `cumulative.duckdb`
- `barsmith.log` (unless `--no-file-log`)

## Resuming

Barsmith resumes by extending the combination enumeration stream, scoped to a CSV fingerprint.

Practical rules:

- Reuse the same `--output-dir` to continue a run.
- If the input CSV changes, Barsmith will refuse to reuse the output dir unless you pass `--force`.
- If you want to override the stored resume offset (start from a specific point), pass `--resume-from`.

## Prepared dataset overwrite (`--ack-new-df`)

The default CLI always writes `output-dir/barsmith_prepared.csv`.

- If it already exists, you must pass `--ack-new-df` to overwrite it.
- `--force` clears Parquet/DuckDB outputs but does not implicitly “bless” overwriting `barsmith_prepared.csv`.

## Choosing batch sizes

Batch size controls evaluation granularity:

- too small: overhead dominates (more writer churn, more scheduler overhead)
- too large: memory spikes and long tail latency (slow batches, reduced responsiveness)

Options:

- Start with a moderate `--batch-size` (e.g. 50k–500k) and scale up.
- Use `--auto-batch` for adaptive tuning on long runs.

## Sampling and dry runs

- Use `--dry-run` to validate that the catalog loads and the theoretical combination count looks sane.
- Use `--max-combos` for a short “smoke run” that still produces real outputs.

## Date filtering

`--date-start` and `--date-end` filter the prepared dataset at load time, so evaluation and reporting see the same time window.

See `docs/data-contract.md` for timestamp requirements.

