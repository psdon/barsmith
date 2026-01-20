# Data contract

This document describes the minimum data contract Barsmith expects for `comb` runs.

## Input CSV (raw OHLCV)

The default CLI (`barsmith_cli`) reads a raw OHLCV CSV and writes an engineered dataset into `output-dir/barsmith_prepared.csv`.

Required columns:

- `timestamp` (string; RFC3339/ISO-8601 recommended; UTC recommended)
- `open`, `high`, `low`, `close` (numeric)
- `volume` (numeric)

Notes:

- Date filtering (`--date-start`, `--date-end`) is applied after the engineered dataset is loaded and uses the calendar date derived from `timestamp`.
- `timestamp` is typically a string (RFC3339/ISO-8601). Internally Barsmith prefers a column named `timestamp` when present; for some prepared datasets it can also fall back to the first datetime-typed column.
- The prepared dataset may contain additional columns; the pipeline prunes to only the columns it needs.

## Prepared dataset (engineered)

The core pipeline (`barsmith_rs`) evaluates combinations against a prepared dataset. If you’re using the default CLI, this is generated automatically and stored at `output-dir/barsmith_prepared.csv`.

Required/recognized columns:

- `target` (boolean column named exactly as `--target`)
- `rr_<target>` (optional float): per-row reward in R units
- `<target>_eligible` (optional boolean): eligibility gate for counting a row as a trade candidate
- `<target>_exit_i` (required when `--stacking-mode no-stacking`): integer “next index” used to skip overlapping trades

If you want to supply your own prepared dataset (custom feature engineering / targets), ensure the above contract is satisfied for the chosen `--target`.
