use std::fs;

use anyhow::Result;
use barsmith_rs::{Config, Direction, LogicMode, ReportMetricsMode};
use chrono::NaiveDate;
use custom_rs::{PrepareDatasetOptions, prepare_dataset_with_options};
use polars::prelude::*;
use tempfile::tempdir;

const SAMPLE_DATA: &str = "\
timestamp,open,high,low,close,volume
2023-12-31T23:30:00Z,100.0,101.0,99.5,100.5,1000
2024-01-01T00:00:00Z,101.0,102.0,100.5,101.5,1100
2024-06-01T00:00:00Z,102.0,103.0,101.5,102.5,1200
2025-01-01T00:00:00Z,103.0,104.0,102.5,103.5,1300
";

fn load_dates(path: &std::path::Path) -> Result<Vec<NaiveDate>> {
    let df = CsvReader::from_path(path)?.has_header(true).finish()?;
    let ts = df.column("timestamp")?;
    let mut out = Vec::with_capacity(ts.len());
    for v in ts.iter() {
        use polars::prelude::AnyValue;
        let raw = match v {
            AnyValue::String(s) => s,
            AnyValue::StringOwned(ref s) => s.as_str(),
            _ => continue,
        };
        let dt = chrono::DateTime::parse_from_rfc3339(raw)?;
        out.push(dt.date_naive());
    }
    Ok(out)
}

#[test]
fn prepare_dataset_does_not_apply_date_start_and_end_filters() -> Result<()> {
    let temp_dir = tempdir()?;
    let csv_path = temp_dir.path().join("sample_dates.csv");
    fs::write(&csv_path, SAMPLE_DATA)?;
    let output_dir = temp_dir.path().join("out_dates");

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
        include_date_start: Some(NaiveDate::from_ymd_opt(2024, 1, 1).unwrap()),
        include_date_end: Some(NaiveDate::from_ymd_opt(2024, 12, 31).unwrap()),
        batch_size: 16,
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

    let prepared = prepare_dataset_with_options(
        &config,
        PrepareDatasetOptions {
            drop_nan_rows_in_core: false,
            ..Default::default()
        },
    )?;
    let dates = load_dates(&prepared)?;
    assert_eq!(
        dates.len(),
        4,
        "prepare_dataset should engineer the full input range, independent of include_date_*"
    );
    assert_eq!(dates[0], NaiveDate::from_ymd_opt(2023, 12, 31).unwrap());
    assert_eq!(dates[1], NaiveDate::from_ymd_opt(2024, 1, 1).unwrap());
    assert_eq!(dates[2], NaiveDate::from_ymd_opt(2024, 6, 1).unwrap());
    assert_eq!(dates[3], NaiveDate::from_ymd_opt(2025, 1, 1).unwrap());

    Ok(())
}
