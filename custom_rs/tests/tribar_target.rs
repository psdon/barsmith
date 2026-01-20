use std::fs;
use std::path::Path;

use anyhow::Result;
use barsmith_rs::{Config, Direction, LogicMode, ReportMetricsMode};
use custom_rs::{PrepareDatasetOptions, prepare_dataset_with_options};
use polars::prelude::*;
use tempfile::tempdir;

fn write_simple_4h_csv(path: &Path) -> Result<()> {
    // Minimal 4h-esque sample: timestamps are 4h apart and remain within a
    // single week so the filename/timeframe checks and basic wiring can be
    // exercised.
    const DATA: &str = "\
timestamp,open,high,low,close,volume
2024-01-01T00:00:00Z,100.0,102.0,99.0,101.0,1000
2024-01-01T04:00:00Z,101.0,103.0,100.0,102.0,1000
2024-01-01T08:00:00Z,102.0,104.0,101.0,103.0,1000
2024-01-01T12:00:00Z,103.0,105.0,102.0,104.0,1000
2024-01-01T16:00:00Z,104.0,106.0,103.0,105.0,1000
";
    fs::write(path, DATA)?;
    Ok(())
}

#[test]
fn tribar_4h_target_enforces_4h_filename_suffix() -> Result<()> {
    let temp_dir = tempdir()?;
    let csv_path = temp_dir.path().join("es_sample.csv"); // no _4h suffix
    write_simple_4h_csv(&csv_path)?;
    let output_dir = temp_dir.path().join("out");

    let config = Config {
        input_csv: csv_path.clone(),
        source_csv: Some(csv_path),
        direction: Direction::Long,
        target: "tribar_4h_2atr".to_string(),
        output_dir,
        max_depth: 1,
        min_sample_size: 1,
        min_sample_size_report: 1,
        logic_mode: LogicMode::And,
        include_date_start: None,
        include_date_end: None,
        batch_size: 8,
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
        stats_detail: barsmith_rs::config::StatsDetail::Full,
        eval_profile: barsmith_rs::config::EvalProfileMode::Off,
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
        stacking_mode: barsmith_rs::config::StackingMode::Stacking,
        position_sizing: barsmith_rs::config::PositionSizingMode::Fractional,
        stop_distance_column: None,
        stop_distance_unit: barsmith_rs::config::StopDistanceUnit::Points,
        min_contracts: 1,
        max_contracts: None,
        point_value: None,
        tick_value: None,
        margin_per_contract_dollar: None,
        require_any_features: Vec::new(),
    };

    let err = prepare_dataset_with_options(
        &config,
        PrepareDatasetOptions {
            drop_nan_rows_in_core: false,
            ..Default::default()
        },
    )
    .expect_err("expected tribar_4h_2atr to reject non-_4h filenames");
    let msg = format!("{err}");
    assert!(
        msg.contains("_4h"),
        "error message should mention _4h requirement (got: {msg})"
    );
    Ok(())
}

#[test]
fn tribar_4h_target_writes_label_and_rr_columns() -> Result<()> {
    let temp_dir = tempdir()?;
    let csv_path = temp_dir.path().join("es_sample_4h.csv");
    write_simple_4h_csv(&csv_path)?;
    let output_dir = temp_dir.path().join("out");

    let config = Config {
        input_csv: csv_path.clone(),
        source_csv: Some(csv_path),
        direction: Direction::Long,
        target: "tribar_4h_2atr".to_string(),
        output_dir: output_dir.clone(),
        max_depth: 1,
        min_sample_size: 1,
        min_sample_size_report: 1,
        logic_mode: LogicMode::And,
        include_date_start: None,
        include_date_end: None,
        batch_size: 8,
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
        stats_detail: barsmith_rs::config::StatsDetail::Full,
        eval_profile: barsmith_rs::config::EvalProfileMode::Off,
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
        stacking_mode: barsmith_rs::config::StackingMode::Stacking,
        position_sizing: barsmith_rs::config::PositionSizingMode::Fractional,
        stop_distance_column: None,
        stop_distance_unit: barsmith_rs::config::StopDistanceUnit::Points,
        min_contracts: 1,
        max_contracts: None,
        point_value: None,
        tick_value: None,
        margin_per_contract_dollar: None,
        require_any_features: Vec::new(),
    };

    let prepared_csv = prepare_dataset_with_options(
        &config,
        PrepareDatasetOptions {
            drop_nan_rows_in_core: false,
            ..Default::default()
        },
    )?;

    let df = CsvReader::from_path(&prepared_csv)?
        .has_header(true)
        .finish()?;

    for name in &["tribar_4h_2atr", "rr_tribar_4h_2atr", "rr_long"] {
        assert!(
            df.column(name).is_ok(),
            "expected column {name} to be present in engineered dataset"
        );
    }

    Ok(())
}
