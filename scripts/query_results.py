#!/usr/bin/env python3
"""
Ad-hoc query helper for Barsmith results.

Given an output directory that contains a `cumulative.duckdb` plus Parquet batches
under `results_parquet/`, this script will query the `results` view and print the
top combinations subject to simple filters (win rate, max drawdown, equity Calmar).

Usage (from the repo root or barsmith/):

  uv run python barsmith/tmp/query_results.py \\
    --output-dir barsmith/tmp/barsmith_run \\
    --direction long \\
    --target next_bar_color_and_wicks \\
    --min-sample-size 500 \\
    --min-win-rate 30 \\
    --max-drawdown 30 \\
    --min-sortino 0.0 \\
    --limit 10
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect Barsmith cumulative.duckdb results with simple filters.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Barsmith output directory (must contain cumulative.duckdb and results_parquet/).",
    )
    parser.add_argument(
        "--direction",
        choices=["long", "short", "both"],
        default="long",
        help="Direction to inspect (matches the `direction` column in results).",
    )
    parser.add_argument(
        "--target",
        default="next_bar_color_and_wicks",
        help="Target name to filter on (matches the `target` column).",
    )
    parser.add_argument(
        "--min-sample-size",
        type=int,
        default=500,
        help="Minimum `total_bars` required for a combination to be shown.",
    )
    parser.add_argument(
        "--min-win-rate",
        type=float,
        default=0.0,
        help="Minimum win rate in percent.",
    )
    parser.add_argument(
        "--max-drawdown",
        type=float,
        default=10_000.0,
        help="Maximum allowed max_drawdown in R units.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum number of rows to print.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    db_path = args.output_dir / "cumulative.duckdb"
    if not db_path.exists():
        raise SystemExit(f"cumulative.duckdb not found at {db_path}")

    conn = duckdb.connect(str(db_path))

    # The Rust side maintains a `results` view over results_parquet/*.parquet.
    # We interpret it similarly to `CumulativeStore::top_results`, but add simple filters.
    sql = """
    SELECT
      direction,
      target,
      combination,
      depth,
      total_bars,
      profitable_bars,
      win_rate,
      max_drawdown,
      calmar_ratio
    FROM results
    WHERE total_bars >= ?
      AND win_rate >= ?
      AND max_drawdown <= ?
      AND direction = ?
      AND target = ?
    ORDER BY calmar_ratio DESC
    LIMIT ?
    """

    params = (
        args.min_sample_size,
        args.min_win_rate,
        args.max_drawdown,
        args.direction,
        args.target,
        args.limit,
    )

    rows = conn.execute(sql, params).fetchall()
    if not rows:
        print("No results matched the given filters.")
        return

    print(
        f"Top {len(rows)} combinations for direction={args.direction}, target={args.target} "
        f"(min_sample={args.min_sample_size}, min_win_rate={args.min_win_rate}%, "
        f"max_dd={args.max_drawdown})"
    )
    print("=" * 72)
    for idx, row in enumerate(rows, start=1):
        (
            direction,
            target,
            combo,
            depth,
            total_bars,
            profitable_bars,
            win_rate,
            max_dd,
            calmar,
        ) = row
        print(f"\nRank {idx}: {combo}")
        print(f"  Depth: {depth} | Bars: {profitable_bars}/{total_bars} ({win_rate:.2f}%)")
        print(f"  Max DD: {max_dd:.1f}R | Calmar (equity): {calmar:.2f}")


if __name__ == "__main__":
    main()
