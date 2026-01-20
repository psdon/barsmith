# Contributing

Thanks for your interest in contributing to Barsmith.

## Development setup

Prerequisites:
- Rust (stable)

Common commands:
- Format: `cargo fmt`
- Lint: `cargo clippy --all-targets --all-features -- -D warnings`
- Test: `cargo test --all-targets`

## Test fixtures

This repo includes committed CSV fixtures under `tests/data/` to keep the test suite self-contained.

- `tests/data/ohlcv_tiny.csv` is the default smoke-test fixture (small, fast, and used by docs/examples).
- `tests/data/es_30m_sample.csv` is a larger “golden” fixture kept for deeper parity-style tests and may be replaced later with a smaller generated fixture if repo size becomes a concern.

## Pull requests

- Keep PRs focused and small when possible.
- Add/adjust tests for behavior changes.
- Prefer clear error messages over panics in runtime paths.
- Avoid committing large datasets, logs, or generated outputs.

## Code style

- `cargo fmt` must pass.
- Avoid `unsafe` unless it’s behind a clearly justified, well-tested, performance-critical boundary.
