# Outputs and querying

Barsmith writes incremental Parquet batches plus a DuckDB catalog for querying “top results”.

## Files

In `--output-dir`:

- `barsmith_prepared.csv`: engineered dataset used for the run
- `results_parquet/part-*.parquet`: stored result rows (only combinations that pass storage filters)
- `cumulative.duckdb`: DuckDB database that exposes a `results` view over all Parquet parts
- `barsmith.log`: file log (unless disabled)

## What gets stored

Barsmith evaluates every enumerated combination, but only persists combinations that meet storage thresholds:

- `--min-samples` (minimum trade/sample count)
- `--max-drawdown` (max drawdown ceiling)

This keeps run folders smaller and reporting faster.

## Querying with DuckDB

You can query `cumulative.duckdb` with DuckDB’s CLI:

```bash
duckdb ./tmp/run_01/cumulative.duckdb "SELECT combination, total_bars, calmar_ratio, max_drawdown FROM results ORDER BY calmar_ratio DESC LIMIT 20"
```

Useful queries:

```bash
# Count stored combinations
duckdb ./tmp/run_01/cumulative.duckdb "SELECT COUNT(*) AS n FROM results"

# Best combos with minimum sample size
duckdb ./tmp/run_01/cumulative.duckdb "SELECT combination, total_bars, calmar_ratio FROM results WHERE total_bars >= 1000 ORDER BY calmar_ratio DESC LIMIT 50"
```

Note: the exact schema is versioned by code and may evolve (this repo is unstable). Prefer inspecting columns via:

```bash
duckdb ./tmp/run_01/cumulative.duckdb "DESCRIBE results"
```

## Resume metadata

The DuckDB database stores resume metadata used to continue enumeration without restarting from zero.

If you delete Parquet parts manually but keep the metadata, Barsmith may warn that resume offsets exist without corresponding stored parts.

