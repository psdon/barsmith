# Engines and targets

Barsmith separates “core search + storage” from “feature engineering + target labeling”.

## Default (builtin) engine

The default CLI uses `barsmith_builtin` to:

- load the raw OHLCV CSV,
- add a minimal set of engineered columns,
- emit a boolean target column and supporting columns required by the evaluator,
- build a small feature catalog for `comb`.

Supported targets (builtin engine):

- `next_bar_up`
- `next_bar_down`
- `next_bar_color_and_wicks` (compatibility alias for `next_bar_up`)

## Custom engines / prepared datasets

If you want richer targets (e.g. ATR-based exits) or richer feature catalogs, you have two main options:

1. Use `barsmith_rs` as a library and provide your own “prepared dataset” that satisfies the contract in `docs/data-contract.md`.
2. Treat `custom_rs` as an example/advanced engine and adapt it to your needs.

## Feature catalog types

Barsmith’s feature catalog is a list of `FeatureDescriptor`s that can represent:

- boolean predicates (pre-computed boolean columns)
- feature-vs-constant thresholds (numeric columns compared to a constant)
- feature-vs-feature comparisons (optional pairwise comparisons when enabled)

The evaluator combines these predicates with AND/OR logic up to `--max-depth`.

