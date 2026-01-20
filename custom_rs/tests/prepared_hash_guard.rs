use std::fs;
use std::path::Path;

use barsmith_rs::{Config, Direction, LogicMode, ReportMetricsMode};
use custom_rs::{PrepareDatasetOptions, prepare_dataset_with_options};
use tempfile::tempdir;

const SAMPLE_DATA: &str = "\
timestamp,open,high,low,close,volume
2024-01-01T00:00:00Z,100.0,101.5,99.5,101.0,1000
2024-01-01T00:30:00Z,101.0,102.4,100.8,102.2,1100
2024-01-01T01:00:00Z,102.2,103.0,101.9,102.0,900
2024-01-01T01:30:00Z,102.0,103.5,101.5,103.2,950
2024-01-01T02:00:00Z,103.2,103.8,102.0,102.5,1050
2024-01-01T02:30:00Z,102.5,103.0,101.0,101.2,990
2024-01-01T03:00:00Z,101.2,101.6,100.2,100.5,1010
2024-01-01T03:30:00Z,100.5,101.2,99.8,100.9,1005
";

fn base_config(csv_path: &Path, output_dir: &Path, target: &str) -> Config {
    Config {
        input_csv: csv_path.to_path_buf(),
        source_csv: Some(csv_path.to_path_buf()),
        direction: Direction::Long,
        target: target.to_string(),
        output_dir: output_dir.to_path_buf(),
        max_depth: 1,
        min_sample_size: 1,
        min_sample_size_report: 1,
        logic_mode: LogicMode::And,
        include_date_start: None,
        include_date_end: None,
        batch_size: 5,
        n_workers: 1,
        auto_batch: false,
        early_exit_when_reused: false,
        resume_offset: 0,
        explicit_resume_offset: false,
        max_combos: Some(10),
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
    }
}

#[test]
fn prepared_dataset_hash_mismatch_requires_ack() -> anyhow::Result<()> {
    let temp_dir = tempdir()?;
    let csv_path = temp_dir.path().join("sample.csv");
    fs::write(&csv_path, SAMPLE_DATA)?;
    let output_dir = temp_dir.path().join("output");

    let config1 = base_config(&csv_path, &output_dir, "is_green");
    prepare_dataset_with_options(
        &config1,
        PrepareDatasetOptions {
            drop_nan_rows_in_core: false,
            ..Default::default()
        },
    )?;

    let config2 = base_config(&csv_path, &output_dir, "next_bar_color_and_wicks");
    let err = prepare_dataset_with_options(
        &config2,
        PrepareDatasetOptions {
            drop_nan_rows_in_core: false,
            ..Default::default()
        },
    )
    .expect_err("expected hash mismatch error");
    let msg = err.to_string();
    assert!(
        msg.contains("ack-new-df"),
        "error should mention --ack-new-df, got: {msg}"
    );

    prepare_dataset_with_options(
        &config2,
        PrepareDatasetOptions {
            ack_new_df: true,
            drop_nan_rows_in_core: false,
            ..Default::default()
        },
    )?;

    Ok(())
}
