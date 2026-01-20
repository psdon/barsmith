# CLI

Barsmith’s default CLI is `barsmith_cli` and currently exposes the `comb` subcommand.

## Help

```bash
barsmith_cli --help
barsmith_cli comb --help
```

## Common flags (practical guide)

### Inputs / outputs

- `--csv <FILE>`: raw OHLCV CSV input
- `--output-dir <DIR>`: run folder (durable outputs live here)
- `--target <NAME>`: target identifier (builtin engine supports: `next_bar_up`, `next_bar_down`, `next_bar_color_and_wicks`)
- `--direction long|short|both`: filter which side is evaluated

### Enumeration

- `--max-depth <N>`: maximum number of predicates per combination
- `--logic and|or|both`: AND/OR combination logic
- `--resume-from <OFFSET>`: resume offset in the global enumeration stream
- `--max-combos <N>`: stop after evaluating up to N combinations (useful for sampling / smoke runs)
- `--batch-size <N>`: combinations per batch (evaluation is parallel within a batch)
- `--auto-batch`: adapt batch size based on recent timings
- `--subset-pruning`: prune higher-depth combinations using under-min depth-2 “dead pairs”
- `--require-any-features <comma,list>`: only evaluate combinations that include at least one named feature (enumeration still proceeds)

### Evaluation / storage filters

- `--min-samples <N>`: combos below this sample threshold are evaluated but not persisted
- `--max-drawdown <R>`: combos with drawdown above this are not persisted
- `--stacking-mode stacking|no-stacking`:
  - `stacking`: every mask hit is treated as an independent sample
  - `no-stacking`: enforces one open trade at a time using `<target>_exit_i`

### Reporting

- `--report full|formula|top10|top100|off`
- `--top-k <N>`: size of the final report table (when reporting is enabled)
- `--max-drawdown-report <R>` / `--min-calmar-report <X>`: reporting-only query filters

### Performance

- `--workers <N>`: number of worker threads (omit to use all cores)
- `--stats-detail core|full`: compute cheaper “core” metrics vs full metrics
- `--profile-eval off|coarse|fine`: enable timing instrumentation

### Resume / overwrite knobs

- `--force`: clears existing cumulative outputs under `--output-dir` (DuckDB + Parquet batches) and starts fresh
- `--ack-new-df`: overwrite an existing `output-dir/barsmith_prepared.csv` (the builtin CLI always writes this file)

### S3 upload

- `--s3-output s3://bucket/prefix`
- `--s3-upload-each-batch`

This uses `aws s3 cp` (AWS CLI) and does not embed AWS credential logic inside Barsmith.

### Costs / sizing (optional)

Barsmith can model costs and contract sizing when you provide `--asset` and choose a sizing mode.

Start with:

- `--position-sizing fractional` (default)
- `--asset <CODE>` (e.g. `ES`, `MES`) to load tick/point value defaults

See `barsmith_cli comb --help` for all sizing/cost knobs.

