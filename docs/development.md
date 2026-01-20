# Development

## Toolchain

This repo pins a Rust toolchain for consistent formatting/linting in CI. See `rust-toolchain.toml`.

## Common commands

```bash
cargo fmt
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-targets
```

## Dependency security (RustSec)

Run locally:

```bash
cargo audit
```

If you must temporarily ignore an advisory, record it in `audit.toml` with a short rationale in the PR and open a tracking issue to remove the ignore.

## Fixtures

Fixtures live under `tests/data/`:

- `ohlcv_tiny.csv`: small default fixture for smoke tests and docs
- `es_30m_sample.csv`: larger golden fixture (kept for deeper tests)

Avoid adding large datasets unless there is a clear test value.

