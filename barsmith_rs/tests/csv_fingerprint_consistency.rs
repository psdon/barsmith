use std::fs;

use anyhow::Result;
use barsmith_rs::stats::StatSummary;
use barsmith_rs::storage::CumulativeStore;
use barsmith_rs::{
    Config, Direction, LogicMode, ReportMetricsMode,
    config::{EvalProfileMode, PositionSizingMode, StackingMode, StatsDetail, StopDistanceUnit},
};
use tempfile::tempdir;

fn base_config(csv_path: &std::path::Path, output_dir: &std::path::Path) -> Config {
    Config {
        input_csv: csv_path.to_path_buf(),
        source_csv: Some(csv_path.to_path_buf()),
        direction: Direction::Long,
        target: "is_green".to_string(),
        output_dir: output_dir.to_path_buf(),
        max_depth: 1,
        min_sample_size: 1,
        min_sample_size_report: 1,
        logic_mode: LogicMode::And,
        include_date_start: None,
        include_date_end: None,
        batch_size: 10,
        n_workers: 1,
        auto_batch: false,
        early_exit_when_reused: false,
        resume_offset: 0,
        explicit_resume_offset: false,
        max_combos: None,
        dry_run: false,
        quiet: true,
        report_metrics: ReportMetricsMode::Off,
        report_top: 5,
        force_recompute: false,
        max_drawdown: 50.0,
        max_drawdown_report: None,
        min_calmar_report: None,
        strict_min_pruning: true,
        enable_feature_pairs: false,
        feature_pairs_limit: None,
        enable_subset_pruning: false,
        catalog_hash: None,
        stats_detail: StatsDetail::Full,
        eval_profile: EvalProfileMode::Off,
        eval_profile_sample_rate: 1,
        s3_output: None,
        s3_upload_each_batch: false,
        capital_dollar: None,
        risk_pct_per_trade: None,
        equity_time_years: None,
        asset: None,
        risk_per_trade_dollar: None,
        cost_per_trade_dollar: None,
        cost_per_trade_r: None,
        dollars_per_r: None,
        tick_size: None,
        stacking_mode: StackingMode::Stacking,
        position_sizing: PositionSizingMode::Fractional,
        stop_distance_column: None,
        stop_distance_unit: StopDistanceUnit::Points,
        min_contracts: 1,
        max_contracts: None,
        point_value: None,
        tick_value: None,
        margin_per_contract_dollar: None,
        require_any_features: Vec::new(),
    }
}

fn make_combo(name: &str) -> String {
    name.to_string()
}

fn summary(total: usize) -> StatSummary {
    let wins = total / 2;
    let losses = total - wins;
    let win_rate = wins as f64 / total as f64;
    StatSummary {
        depth: 1,
        mask_hits: total,
        total_bars: total,
        profitable_bars: wins,
        unprofitable_bars: losses,
        win_rate,
        label_hit_rate: win_rate,
        label_hits: wins,
        label_misses: losses,
        expectancy: 0.0,
        profit_factor: 1.0,
        avg_winning_rr: 0.0,
        calmar_ratio: 0.0,
        max_drawdown: 0.0,
        win_loss_ratio: 1.0,
        ulcer_index: 0.0,
        pain_ratio: 0.0,
        max_consecutive_wins: 1,
        max_consecutive_losses: 1,
        avg_win_streak: 1.0,
        avg_loss_streak: 1.0,
        median_rr: 0.0,
        avg_losing_rr: 0.0,
        p05_rr: 0.0,
        p95_rr: 0.0,
        largest_win: 1.0,
        largest_loss: -1.0,
        sample_quality: "ok",
        total_return: 0.0,
        cost_per_trade_r: 0.0,
        dollars_per_r: 0.0,
        total_return_dollar: 0.0,
        max_drawdown_dollar: 0.0,
        expectancy_dollar: 0.0,
        final_capital: 0.0,
        total_return_pct: 0.0,
        cagr_pct: 0.0,
        max_drawdown_pct_equity: 0.0,
        calmar_equity: 0.0,
        sharpe_equity: 0.0,
        sortino_equity: 0.0,
    }
}

#[test]
fn same_csv_allows_reuse_without_force() -> Result<()> {
    let temp_dir = tempdir()?;
    let csv_path = temp_dir.path().join("data.csv");
    let output_dir = temp_dir.path().join("out");
    fs::write(
        &csv_path,
        "timestamp,open\n2024-01-01T00:00:00Z,1\n2024-01-01T00:30:00Z,2\n",
    )?;

    let cfg = base_config(&csv_path, &output_dir);

    let (_store1, resume1) = CumulativeStore::new(&cfg)?;
    assert_eq!(resume1, 0);

    // Re-opening with the same CSV and output directory should succeed and
    // not complain about CSV mismatch.
    let (_store2, resume2) = CumulativeStore::new(&cfg)?;
    assert_eq!(resume2, 0);

    Ok(())
}

#[test]
fn different_csv_requires_force_to_reuse_output_dir() -> Result<()> {
    let temp_dir = tempdir()?;
    let csv1 = temp_dir.path().join("data1.csv");
    let csv2 = temp_dir.path().join("data2.csv");
    let output_dir = temp_dir.path().join("out");

    fs::write(&csv1, "timestamp,open\n2024-01-01T00:00:00Z,1\n")?;
    fs::write(&csv2, "timestamp,open\n2024-01-01T00:00:00Z,2\n")?;

    let cfg1 = base_config(&csv1, &output_dir);
    let (mut store1, _) = CumulativeStore::new(&cfg1)?;
    // Ingest a small batch so that a metadata row is written for the first CSV.
    let combos = vec![make_combo("alpha")];
    let stats = vec![summary(100)];
    store1.ingest_with_enumerated(&combos, &stats, combos.len(), 0)?;
    store1.flush()?;

    // Using a different CSV against the same output directory without force
    // should fail with a clear "different CSV" error.
    let cfg2 = base_config(&csv2, &output_dir);
    let err = match CumulativeStore::new(&cfg2) {
        Ok(_) => {
            panic!(
                "reusing an output directory with a different CSV should error without force_recompute"
            )
        }
        Err(e) => e,
    };
    let msg = format!("{:#}", err);
    assert!(
        msg.contains("different CSV"),
        "error message should mention a different CSV fingerprint"
    );

    // Enabling force_recompute should allow reuse of the output directory and
    // reset resume_offset to zero.
    let mut cfg3 = base_config(&csv2, &output_dir);
    cfg3.force_recompute = true;
    let (_store3, resume3) = CumulativeStore::new(&cfg3)?;
    assert_eq!(resume3, 0);

    Ok(())
}
