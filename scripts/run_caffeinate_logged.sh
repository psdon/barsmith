#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir/.."

cmd=(caffeinate -dimsu cargo run --release -p barsmith_cli -- "$@")
export BARSMITH_LAUNCHER_COMMAND="${cmd[*]}"
exec "${cmd[@]}"
