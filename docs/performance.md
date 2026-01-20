# Performance

Barsmith is designed for high-throughput batch exploration, but performance depends heavily on configuration and your machine.

## Build mode

Prefer release builds for real runs:

```bash
cargo run --release -p barsmith_cli -- comb --help
```

## Key knobs

- Catalog size and `--max-depth`: the search space grows combinatorially.
- `--stats-detail core|full`: `core` is much cheaper; `full` computes more metrics.
- `--workers`: scales evaluation across CPU cores (watch memory bandwidth).
- `--batch-size` and `--auto-batch`: impacts scheduling overhead vs per-batch latency.
- `--subset-pruning`: can drastically reduce work for deep searches when many depth-2 pairs are dead.
- `--feature-pairs`: increases catalog size (more predicates).

## CPU portability vs speed

This repo’s `.cargo/config.toml` sets `target-cpu=native` for local performance. This is great for on-machine runs, but not ideal for distributing binaries across heterogeneous CPUs.

## Benchmark note

Internal benchmark (not a guarantee): ~120B combination candidates over ~5 days on a MacBook Pro (Apple M4).

