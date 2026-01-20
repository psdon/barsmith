use std::fs;

use anyhow::Result;
use barsmith_rs::stats::StatSummary;
use barsmith_rs::storage::CumulativeStore;
use barsmith_rs::{
    Config, Direction, LogicMode, ReportMetricsMode,
    config::{EvalProfileMode, PositionSizingMode, StackingMode, StatsDetail, StopDistanceUnit},
};
use tempfile::tempdir;

fn make_combo(name: &str) -> String {
    name.to_string()
}

fn summary(wins: usize, total: usize, profit_factor: f64, calmar: f64) -> StatSummary {
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
        expectancy: 0.1,
        profit_factor,
        avg_winning_rr: 0.5,
        calmar_ratio: calmar,
        max_drawdown: 5.0,
        win_loss_ratio: 1.2,
        ulcer_index: 0.0,
        pain_ratio: 0.0,
        max_consecutive_wins: 2,
        max_consecutive_losses: 2,
        avg_win_streak: 1.0,
        avg_loss_streak: 1.0,
        median_rr: 0.0,
        avg_losing_rr: 0.0,
        p05_rr: 0.0,
        p95_rr: 0.0,
        largest_win: 1.0,
        largest_loss: -1.0,
        sample_quality: "excellent",
        total_return: 10.0,
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
fn top_results_are_sorted_by_calmar_only() -> Result<()> {
    let temp_dir = tempdir()?;
    let csv_path = temp_dir.path().join("data.csv");
    let output_dir = temp_dir.path().join("out");
    fs::write(&csv_path, "timestamp,open\n2024-01-01T00:00:00Z,1\n")?;

    let config = Config {
        input_csv: csv_path.clone(),
        source_csv: Some(csv_path),
        direction: Direction::Long,
        target: "is_green".to_string(),
        output_dir: output_dir.clone(),
        max_depth: 1,
        min_sample_size: 1,
        min_sample_size_report: 1,
        logic_mode: LogicMode::And,
        include_date_start: None,
        include_date_end: None,
        batch_size: 1,
        n_workers: 1,
        auto_batch: false,
        early_exit_when_reused: false,
        resume_offset: 0,
        explicit_resume_offset: false,
        max_combos: None,
        dry_run: false,
        quiet: false,
        report_metrics: ReportMetricsMode::Full,
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
    };

    let (mut store, _) = CumulativeStore::new(&config)?;

    let combos = vec![make_combo("alpha"), make_combo("beta"), make_combo("gamma")];
    let stats = vec![
        summary(55, 100, 1.5, 1.0),
        summary(60, 100, 1.1, 2.0),
        summary(65, 100, 1.3, 3.0),
    ];

    store.ingest(&combos, &stats)?;
    // Ensure the results view is up to date before querying top_results.
    store.refresh_view()?;
    store.flush()?;

    let rows = store.top_results(2, 1, 50.0, None)?;
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0].combination, "gamma");
    assert_eq!(rows[1].combination, "beta");
    Ok(())
}

#[test]
fn force_recompute_clears_cumulative_state_and_batches() -> Result<()> {
    let temp_dir = tempdir()?;
    let csv_path = temp_dir.path().join("data.csv");
    let output_dir = temp_dir.path().join("out");
    fs::write(&csv_path, "timestamp,open\n2024-01-01T00:00:00Z,1\n")?;

    let base_config = Config {
        input_csv: csv_path.clone(),
        source_csv: Some(csv_path),
        direction: Direction::Long,
        target: "is_green".to_string(),
        output_dir: output_dir.clone(),
        max_depth: 1,
        min_sample_size: 1,
        min_sample_size_report: 1,
        logic_mode: LogicMode::And,
        include_date_start: None,
        include_date_end: None,
        batch_size: 1,
        n_workers: 1,
        auto_batch: false,
        early_exit_when_reused: false,
        resume_offset: 0,
        explicit_resume_offset: false,
        max_combos: None,
        dry_run: false,
        quiet: false,
        report_metrics: ReportMetricsMode::Full,
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
    };

    // First run: no force, write one batch to cumulative storage.
    let mut config1 = base_config.clone();
    config1.force_recompute = false;
    let (mut store1, resume1) = CumulativeStore::new(&config1)?;
    assert_eq!(resume1, 0);

    let combos = vec![make_combo("alpha")];
    let stats = vec![summary(55, 100, 1.5, 1.0)];
    store1.ingest(&combos, &stats)?;
    store1.flush()?;

    let duckdb_path = output_dir.join("cumulative.duckdb");
    assert!(
        duckdb_path.exists(),
        "DuckDB catalog should exist after first run"
    );
    let results_dir = output_dir.join("results_parquet");
    let parts_before: Vec<_> = std::fs::read_dir(&results_dir)?
        .filter_map(Result::ok)
        .filter(|entry| entry.file_name().to_string_lossy().starts_with("part-"))
        .collect();
    assert!(
        !parts_before.is_empty(),
        "Expected at least one Parquet batch before force_recompute"
    );

    drop(store1);

    // Second run: with force_recompute, cumulative state should be cleared and
    // resume offset reset to zero.
    let mut config2 = base_config;
    config2.force_recompute = true;
    let (_store2, resume2) = CumulativeStore::new(&config2)?;
    assert_eq!(
        resume2, 0,
        "force_recompute should reset resume offset to zero"
    );

    // The DuckDB file should have been recreated, and Parquet batches cleared.
    assert!(
        duckdb_path.exists(),
        "DuckDB catalog should exist after force_recompute run"
    );
    let parts_after: Vec<_> = std::fs::read_dir(&results_dir)?
        .filter_map(Result::ok)
        .filter(|entry| entry.file_name().to_string_lossy().starts_with("part-"))
        .collect();
    assert!(
        parts_after.is_empty(),
        "force_recompute should remove existing Parquet batch files"
    );

    Ok(())
}
