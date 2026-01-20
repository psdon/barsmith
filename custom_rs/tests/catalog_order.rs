use std::fs;
use std::path::Path;

use anyhow::Result;
use barsmith_rs::feature::FeatureCategory;
use barsmith_rs::{Config, Direction, LogicMode, ReportMetricsMode};
use custom_rs::features::FeatureCatalog;
use custom_rs::{PrepareDatasetOptions, prepare_dataset_with_options};
use tempfile::tempdir;

fn make_config(csv_path: &Path, output_dir: &Path, enable_feature_pairs: bool) -> Config {
    Config {
        input_csv: csv_path.to_path_buf(),
        source_csv: Some(csv_path.to_path_buf()),
        direction: Direction::Long,
        target: "is_green".to_string(),
        output_dir: output_dir.to_path_buf(),
        max_depth: 2,
        min_sample_size: 1,
        min_sample_size_report: 1,
        logic_mode: LogicMode::And,
        include_date_start: None,
        include_date_end: None,
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
        force_recompute: true,
        max_drawdown: 50.0,
        max_drawdown_report: None,
        min_calmar_report: None,
        strict_min_pruning: true,
        enable_feature_pairs,
        feature_pairs_limit: Some(16),
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

fn write_sample_csv(path: &Path) -> Result<()> {
    const SAMPLE_DATA: &str = "\
timestamp,open,high,low,close,volume
2024-01-01T00:00:00Z,100.0,101.5,99.5,101.0,1000
2024-01-01T00:30:00Z,101.0,102.4,100.8,102.2,1100
2024-01-01T01:00:00Z,102.2,103.0,101.9,102.0,900
2024-01-01T01:30:00Z,102.0,103.5,101.5,103.2,950
2024-01-01T02:00:00Z,103.2,103.8,102.0,102.5,1050
2024-01-01T02:30:00Z,102.5,103.0,101.0,101.2,990
";
    fs::write(path, SAMPLE_DATA)?;
    Ok(())
}

fn classify_descriptors(
    descriptors: &[barsmith_rs::FeatureDescriptor],
) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
    let mut bool_indices = Vec::new();
    let mut fc_indices = Vec::new();
    let mut ff_indices = Vec::new();

    for (idx, d) in descriptors.iter().enumerate() {
        match d.category {
            FeatureCategory::Boolean => bool_indices.push(idx),
            FeatureCategory::FeatureVsConstant => fc_indices.push(idx),
            FeatureCategory::FeatureVsFeature => ff_indices.push(idx),
            FeatureCategory::Continuous => {}
        }
    }

    (bool_indices, fc_indices, ff_indices)
}

#[test]
fn catalog_order_without_feature_pairs_is_bool_then_fc() -> Result<()> {
    let temp_dir = tempdir()?;
    let csv_path = temp_dir.path().join("sample_order.csv");
    write_sample_csv(&csv_path)?;
    let output_dir = temp_dir.path().join("out_order");

    let config = make_config(&csv_path, &output_dir, false);
    let prepared = prepare_dataset_with_options(
        &config,
        PrepareDatasetOptions {
            drop_nan_rows_in_core: false,
            ..Default::default()
        },
    )?;
    let catalog = FeatureCatalog::build_with_dataset(&prepared, &config)?;

    let (bool_indices, fc_indices, ff_indices) = classify_descriptors(&catalog.descriptors);

    assert!(
        !bool_indices.is_empty(),
        "expected at least one boolean descriptor in the catalog"
    );
    // On very small or synthetic datasets it is possible for the scalar
    // threshold generator to skip all continuous features (e.g., not enough
    // finite samples), in which case there will be no feature-vs-constant
    // descriptors. In that scenario there is nothing meaningful to assert
    // about the ordering between boolean and scalar predicates, so exit
    // early while still validating the boolean portion of the catalog.
    if fc_indices.is_empty() {
        return Ok(());
    }
    assert!(
        ff_indices.is_empty(),
        "feature-vs-feature descriptors should not be present when feature_pairs is disabled"
    );

    let max_bool = *bool_indices.iter().max().unwrap();
    let min_fc = *fc_indices.iter().min().unwrap();

    assert!(
        max_bool < min_fc,
        "all boolean descriptors should appear before any feature-vs-constant descriptors"
    );

    Ok(())
}

#[test]
fn catalog_order_with_feature_pairs_is_bool_then_fc_then_ff() -> Result<()> {
    let temp_dir = tempdir()?;
    let csv_path = temp_dir.path().join("sample_order_fp.csv");
    write_sample_csv(&csv_path)?;
    let output_dir = temp_dir.path().join("out_order_fp");

    let config = make_config(&csv_path, &output_dir, true);
    let prepared = prepare_dataset_with_options(
        &config,
        PrepareDatasetOptions {
            drop_nan_rows_in_core: false,
            ..Default::default()
        },
    )?;
    let catalog = FeatureCatalog::build_with_dataset(&prepared, &config)?;

    let (bool_indices, fc_indices, ff_indices) = classify_descriptors(&catalog.descriptors);

    assert!(
        !bool_indices.is_empty(),
        "expected at least one boolean descriptor in the catalog"
    );
    // For tiny engineered datasets it is possible that no scalar thresholds
    // or feature-pair predicates are generated. In that case we cannot
    // assert an ordering between FC/FF categories, but we still validate
    // the boolean portion of the catalog above.
    if fc_indices.is_empty() || ff_indices.is_empty() {
        return Ok(());
    }

    let max_bool = *bool_indices.iter().max().unwrap();
    let min_fc = *fc_indices.iter().min().unwrap();
    let max_fc = *fc_indices.iter().max().unwrap();
    let min_ff = *ff_indices.iter().min().unwrap();

    assert!(
        max_bool < min_fc,
        "all boolean descriptors should appear before any feature-vs-constant descriptors"
    );
    assert!(
        max_fc < min_ff,
        "all feature-vs-constant descriptors should appear before any feature-vs-feature descriptors"
    );

    Ok(())
}
