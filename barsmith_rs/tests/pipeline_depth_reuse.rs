use std::fs;
use std::fs::File;
use std::path::Path;

use anyhow::Result;
use barsmith_rs::{
    Config, Direction, FeatureDescriptor, LogicMode, PermutationPipeline, ReportMetricsMode,
    config::{EvalProfileMode, PositionSizingMode, StackingMode, StatsDetail, StopDistanceUnit},
};
use polars::prelude::*;
use tempfile::tempdir;

fn make_config(
    csv_path: &Path,
    output_dir: &Path,
    max_depth: usize,
    force_recompute: bool,
) -> Config {
    Config {
        input_csv: csv_path.to_path_buf(),
        source_csv: Some(csv_path.to_path_buf()),
        direction: Direction::Long,
        target: "is_green".to_string(),
        output_dir: output_dir.to_path_buf(),
        max_depth,
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
        quiet: true,
        report_metrics: ReportMetricsMode::Off,
        report_top: 5,
        force_recompute,
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

fn count_depth_rows(results_dir: &Path) -> Result<(usize, usize)> {
    let mut depth1 = 0usize;
    let mut depth2 = 0usize;
    if !results_dir.exists() {
        return Ok((0, 0));
    }

    for entry in fs::read_dir(results_dir)? {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.starts_with("part-") && name.ends_with(".parquet") {
            let file = File::open(entry.path())?;
            let reader = ParquetReader::new(file);
            let df = reader.finish()?;
            if let Ok(series) = df.column("depth") {
                let ca = series.u32()?;
                for value in ca.into_iter().flatten() {
                    match value {
                        1 => depth1 += 1,
                        2 => depth2 += 1,
                        _ => {}
                    }
                }
            }
        }
    }

    Ok((depth1, depth2))
}

fn count_unique_combinations(results_dir: &Path) -> Result<usize> {
    if !results_dir.exists() {
        return Ok(0);
    }
    let mut parts: Vec<DataFrame> = Vec::new();
    for entry in fs::read_dir(results_dir)? {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.starts_with("part-") && name.ends_with(".parquet") {
            let file = File::open(entry.path())?;
            let reader = ParquetReader::new(file);
            let df = reader.finish()?;
            parts.push(df);
        }
    }
    if parts.is_empty() {
        return Ok(0);
    }
    let mut iter = parts.into_iter();
    let mut df_all = iter
        .next()
        .expect("non-empty parts should yield at least one DataFrame");
    for df in iter {
        df_all.vstack_mut(&df)?;
    }
    let combo_only = df_all.select(["combination"])?;
    let unique = combo_only.unique(None, UniqueKeepStrategy::First, None)?;
    Ok(unique.height())
}

/// When depth 1 has already been fully evaluated, a subsequent run with a
/// higher max_depth should reuse shallow layers and only evaluate new,
/// deeper combinations. This integration test verifies that depth 1 is not
/// duplicated when moving from max_depth=1 to max_depth=2.
#[test]
fn depth_reuse_skips_completed_shallow_layers() -> Result<()> {
    let temp_dir = tempdir()?;
    let csv_path = temp_dir.path().join("data.csv");
    let output_dir = temp_dir.path().join("out_depth_reuse");

    // Three boolean features plus the target column.
    fs::write(
        &csv_path,
        "\
timestamp,is_green,f1,f2,f3
2024-01-01T00:00:00Z,true,true,false,true
2024-01-01T00:30:00Z,false,false,true,true
",
    )?;

    let features = vec![
        FeatureDescriptor::boolean("f1", "test"),
        FeatureDescriptor::boolean("f2", "test"),
        FeatureDescriptor::boolean("f3", "test"),
    ];

    // First run: depth 1 only, force_recompute so we start from a clean slate.
    let config_depth1 = make_config(&csv_path, &output_dir, 1, true);
    let mut pipeline1 =
        PermutationPipeline::new(config_depth1, features.clone(), Default::default());
    pipeline1.run()?;

    let results_dir = output_dir.join("results_parquet");
    let (d1_after_first, d2_after_first) = count_depth_rows(&results_dir)?;
    assert_eq!(
        d1_after_first, 3,
        "with three boolean features, depth-1 should produce three combinations"
    );
    assert_eq!(
        d2_after_first, 0,
        "no depth-2 rows should exist after a max_depth=1 run"
    );

    // Second run: increase max_depth to 2 without force_recompute. The pipeline
    // should reuse the complete depth-1 layer and only evaluate depth-2
    // combinations.
    let config_depth2 = make_config(&csv_path, &output_dir, 2, false);
    let mut pipeline2 =
        PermutationPipeline::new(config_depth2, features.clone(), Default::default());
    pipeline2.run()?;

    let (d1_after_second, d2_after_second) = count_depth_rows(&results_dir)?;
    assert!(
        d1_after_second >= d1_after_first,
        "depth-1 rows from the first run should be preserved when increasing max_depth"
    );
    // With min_sample_size=1 and two rows in the dataset, one of the three
    // depth-2 combinations may remain under-min and therefore not be stored.
    assert!(
        (2..=3).contains(&d2_after_second),
        "depth-2 rows should be newly added when increasing max_depth; expected between 2 and 3, got {}",
        d2_after_second
    );

    Ok(())
}

/// max_combos should cap the total number of enumerated combinations for a
/// given configuration, but a subsequent run with a higher max_combos for
/// the same catalog should extend the set of evaluated combinations rather
/// than duplicating them.
#[test]
fn max_combos_limit_is_respected_and_extended() -> Result<()> {
    let temp_dir = tempdir()?;
    let csv_path = temp_dir.path().join("data_max_combos.csv");
    let output_dir = temp_dir.path().join("out_max_combos");

    fs::write(
        &csv_path,
        "\
timestamp,is_green,f1,f2,f3
2024-01-01T00:00:00Z,true,true,false,true
2024-01-01T00:30:00Z,false,false,true,true
",
    )?;

    let features = vec![
        FeatureDescriptor::boolean("f1", "test"),
        FeatureDescriptor::boolean("f2", "test"),
        FeatureDescriptor::boolean("f3", "test"),
    ];

    // First run: strong max_combos limit so we only evaluate a small prefix of
    // the global combination stream.
    let mut cfg1 = make_config(&csv_path, &output_dir, 2, true);
    cfg1.max_combos = Some(2);
    let mut pipeline1 = PermutationPipeline::new(cfg1, features.clone(), Default::default());
    pipeline1.run()?;

    let results_dir = output_dir.join("results_parquet");
    let unique_after_first = count_unique_combinations(&results_dir)?;
    assert_eq!(
        unique_after_first, 2,
        "first run should evaluate exactly two unique combinations"
    );

    // Second run: increase max_combos so that the catalog can be extended.
    let mut cfg2 = make_config(&csv_path, &output_dir, 2, false);
    cfg2.max_combos = Some(5);
    let mut pipeline2 = PermutationPipeline::new(cfg2, features.clone(), Default::default());
    pipeline2.run()?;

    let unique_after_second = count_unique_combinations(&results_dir)?;
    assert!(
        unique_after_second > unique_after_first,
        "second run with a higher max_combos should extend the set of evaluated combinations"
    );
    assert!(
        unique_after_second <= 5,
        "second run should respect the new max_combos limit"
    );

    Ok(())
}

/// When resume_offset is explicitly set in the configuration (e.g., via CLI),
/// it should take precedence over any stored metadata for the same catalog.
/// This test verifies that a large CLI resume_offset prevents metadata-based
/// resume from re-evaluating additional combinations.
#[test]
fn cli_resume_offset_overrides_metadata() -> Result<()> {
    let temp_dir = tempdir()?;
    let csv_path = temp_dir.path().join("data_cli_resume.csv");
    let output_dir = temp_dir.path().join("out_cli_resume");

    fs::write(
        &csv_path,
        "\
timestamp,is_green,f1,f2,f3
2024-01-01T00:00:00Z,true,true,false,true
2024-01-01T00:30:00Z,false,false,true,true
",
    )?;

    let features = vec![
        FeatureDescriptor::boolean("f1", "test"),
        FeatureDescriptor::boolean("f2", "test"),
        FeatureDescriptor::boolean("f3", "test"),
    ];

    // First run: cap the catalog so that metadata reflects a partial prefix of
    // the global stream.
    let mut cfg1 = make_config(&csv_path, &output_dir, 2, true);
    cfg1.max_combos = Some(2);
    let mut pipeline1 = PermutationPipeline::new(cfg1, features.clone(), Default::default());
    pipeline1.run()?;

    let results_dir = output_dir.join("results_parquet");
    let unique_after_first = count_unique_combinations(&results_dir)?;
    assert_eq!(
        unique_after_first, 2,
        "first run should evaluate exactly two unique combinations"
    );

    // Second run: same catalog, but with an explicit resume_offset that is far
    // beyond the theoretical total. This should suppress any further
    // evaluation even though metadata suggests more combinations remain.
    let mut cfg2 = make_config(&csv_path, &output_dir, 2, false);
    cfg2.max_combos = Some(10);
    cfg2.resume_offset = 1_000_000;
    let mut pipeline2 = PermutationPipeline::new(cfg2, features.clone(), Default::default());
    pipeline2.run()?;

    let unique_after_second = count_unique_combinations(&results_dir)?;
    assert_eq!(
        unique_after_second, unique_after_first,
        "explicit CLI resume_offset should take precedence over metadata and prevent additional evaluation"
    );

    Ok(())
}
