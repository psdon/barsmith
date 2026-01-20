# Quickstart

Barsmith is currently marked **unstable**. Expect breaking changes.

## Install (local)

From the repo root:

```bash
cargo install --path barsmith_cli
```

This installs the `barsmith_cli` binary into Cargo’s bin directory (typically `~/.cargo/bin/`).

## Run a tiny dry-run (no external data)

```bash
barsmith_cli comb \
  --csv tests/data/ohlcv_tiny.csv \
  --direction long \
  --target next_bar_color_and_wicks \
  --output-dir ./tmp/run_01 \
  --max-depth 3 \
  --min-samples 100 \
  --workers 1 \
  --max-combos 1000 \
  --dry-run
```

## Run a small exploration

```bash
barsmith_cli comb \
  --csv tests/data/ohlcv_tiny.csv \
  --direction long \
  --target next_bar_color_and_wicks \
  --output-dir ./tmp/run_01 \
  --max-depth 3 \
  --min-samples 100 \
  --workers 1 \
  --max-combos 10000
```

## Data contract

See `docs/data-contract.md`.

## Next steps

- CLI guide: `docs/cli.md`
- Running experiments: `docs/runs.md`
- Outputs and querying: `docs/outputs.md`
