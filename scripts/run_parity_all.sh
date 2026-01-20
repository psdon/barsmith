#!/usr/bin/env bash
set -euo pipefail

# Simple orchestrator to exercise the main Python↔Rust parity surfaces.
# Run from the barsmith workspace root:
#   ./tmp/run_parity_all.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "== Rust: core tests (no Python env required) =="
cargo test -p barsmith_rs --tests
cargo test -p custom_rs --tests -- --skip python_parity_regression

echo
echo "== Desktop indicator parity (~/Desktop/es_30m_pre2025.csv, via uv) =="
uv run python tmp/compare_indicator_flags_desktop.py
uv run python tmp/compare_continuous_values.py

echo
echo "== Rust: full parity regression harness (ignored by default) =="
cargo test -p custom_rs --test parity_regression -- --ignored

echo
echo "All parity checks completed."
