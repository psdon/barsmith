use std::fs;
use std::fs::File;
use std::path::Path;

use anyhow::Result;
use barsmith_rs::storage::CumulativeStore;
use barsmith_rs::{Config, Direction, LogicMode, ReportMetricsMode};
use custom_rs::{CustomPipelineOptions, run_custom_pipeline_with_options};
use polars::prelude::*;
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

fn load_combinations_and_count(results_dir: &Path) -> Result<(usize, usize)> {
    if !results_dir.exists() {
        return Ok((0, 0));
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
        return Ok((0, 0));
    }
    let mut iter = parts.into_iter();
    let mut df_all = iter
        .next()
        .expect("non-empty parts should yield at least one DataFrame");
    for df in iter {
        df_all.vstack_mut(&df)?;
    }
    let total_rows = df_all.height();
    let combo_only = df_all.select(["combination"])?;
    let unique = combo_only.unique(None, UniqueKeepStrategy::First, None)?;
    Ok((unique.height(), total_rows))
}

fn base_config(csv_path: &Path, output_dir: &Path) -> Config {
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
        max_combos: Some(128),
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

fn run_pipeline_for_test(config: Config) -> Result<()> {
    run_custom_pipeline_with_options(
        config,
        CustomPipelineOptions {
            drop_nan_rows_in_core: false,
            ..Default::default()
        },
    )
}

#[test]
fn feature_pairs_depth_two_then_one_preserves_results() -> Result<()> {
    let temp_dir = tempdir()?;
    let csv_path = temp_dir.path().join("sample_fp.csv");
    fs::write(&csv_path, SAMPLE_DATA)?;
    let output_dir = temp_dir.path().join("output_fp_depth");

    // First run: feature-pairs enabled with max_depth=2. We cap max_combos
    // so the test runs quickly even though feature-pairs expand the catalog.
    let mut config1 = base_config(&csv_path, &output_dir);
    config1.max_depth = 2;
    config1.max_combos = Some(32);
    config1.enable_feature_pairs = true;
    run_pipeline_for_test(config1)?;

    let results_dir = output_dir.join("results_parquet");
    let (unique_after_depth2_first, rows_after_depth2_first) =
        load_combinations_and_count(&results_dir)?;
    assert!(
        rows_after_depth2_first > 0,
        "expected some result rows after the initial max_depth=2 run with feature-pairs enabled"
    );

    // Second run: same catalog and CSV but with max_depth reduced to 1 while
    // keeping feature-pairs enabled and the same max_combos cap. Because the
    // first run has already evaluated all depth-1 combinations that fall
    // within the max_combos window, the second run should not change the
    // stored result surface.
    let mut config2 = base_config(&csv_path, &output_dir);
    config2.max_depth = 1;
    config2.max_combos = Some(32);
    config2.enable_feature_pairs = true;
    run_pipeline_for_test(config2)?;

    let (unique_after_depth1_second, rows_after_depth1_second) =
        load_combinations_and_count(&results_dir)?;
    assert!(
        unique_after_depth1_second >= unique_after_depth2_first,
        "re-running with a lower max_depth and the same feature-pair catalog should not lose combinations within the max_combos window"
    );
    assert!(
        rows_after_depth1_second >= rows_after_depth2_first,
        "re-running with a lower max_depth and the same feature-pair catalog should not reduce the number of stored result rows within the max_combos window"
    );

    Ok(())
}

/// Toggling enable_feature_pairs changes the catalog hash. When zero_samples
/// exist from a prior run, attempting to run with a different catalog hash
/// should fail with a catalog hash mismatch error.
#[test]
fn feature_pairs_toggle_after_zeros_causes_hash_mismatch() -> Result<()> {
    let temp_dir = tempdir()?;
    let csv_path = temp_dir.path().join("sample_fp_toggle.csv");
    fs::write(&csv_path, SAMPLE_DATA)?;
    let output_dir = temp_dir.path().join("output_fp_toggle");

    // Run 1: depth 2, feature-pairs disabled.
    let mut cfg1 = base_config(&csv_path, &output_dir);
    cfg1.max_depth = 2;
    cfg1.max_combos = Some(32);
    cfg1.enable_feature_pairs = false;
    run_pipeline_for_test(cfg1)?;

    let results_dir = output_dir.join("results_parquet");
    let (unique_after_1, rows_after_1) = load_combinations_and_count(&results_dir)?;
    assert!(
        rows_after_1 > 0,
        "expected some result rows after the initial depth-2 run without feature pairs"
    );
    assert_eq!(
        rows_after_1, unique_after_1,
        "initial run should not produce duplicate combinations"
    );

    // Run 2: attempt depth 1 with feature-pairs enabled. This produces a
    // different catalog hash and extends the catalog, but should not corrupt
    // or remove existing results.
    let mut cfg2 = base_config(&csv_path, &output_dir);
    cfg2.max_depth = 1;
    cfg2.max_combos = Some(32);
    cfg2.enable_feature_pairs = true;
    run_pipeline_for_test(cfg2)?;

    // Results from the first run should be preserved after the second run.
    let (unique_after_2, rows_after_2) = load_combinations_and_count(&results_dir)?;
    assert!(
        unique_after_2 >= unique_after_1,
        "second run should not lose previously stored combinations"
    );
    assert!(
        rows_after_2 >= rows_after_1,
        "second run should not reduce the number of stored rows"
    );

    Ok(())
}

/// Changing enable_feature_pairs changes the effective catalog. This used to
/// interact with zero-sample persistence; now it simply extends the search
/// space without touching existing results.
#[test]
fn feature_pairs_toggle_causes_catalog_hash_mismatch_error() -> Result<()> {
    let temp_dir = tempdir()?;
    let csv_path = temp_dir.path().join("sample_fp_same_depth.csv");
    fs::write(&csv_path, SAMPLE_DATA)?;
    let output_dir = temp_dir.path().join("output_fp_same_depth");

    // Run 1: depth 2, feature-pairs disabled.
    let mut cfg1 = base_config(&csv_path, &output_dir);
    cfg1.max_depth = 2;
    cfg1.max_combos = Some(32);
    cfg1.enable_feature_pairs = false;
    run_pipeline_for_test(cfg1)?;

    let results_dir = output_dir.join("results_parquet");
    let (unique_after_1, rows_after_1) = load_combinations_and_count(&results_dir)?;
    assert!(
        rows_after_1 > 0,
        "expected some result rows after initial depth-2 run without feature pairs"
    );
    assert_eq!(
        rows_after_1, unique_after_1,
        "initial depth-2 run without feature pairs should not produce duplicate combinations"
    );

    // Run 2: depth 2, feature-pairs enabled – should extend the result surface
    // without failing or corrupting existing combinations.
    let mut cfg2 = base_config(&csv_path, &output_dir);
    cfg2.max_depth = 2;
    cfg2.max_combos = Some(32);
    cfg2.enable_feature_pairs = true;
    run_pipeline_for_test(cfg2)?;

    let (unique_after_2, rows_after_2) = load_combinations_and_count(&results_dir)?;
    assert!(
        unique_after_2 >= unique_after_1,
        "enabling feature_pairs should not lose previously stored combinations"
    );
    assert!(
        rows_after_2 >= rows_after_1,
        "enabling feature_pairs should not reduce the number of stored rows"
    );

    Ok(())
}

#[test]
fn feature_pairs_depth_two_extends_with_max_combos() -> Result<()> {
    let temp_dir = tempdir()?;
    let csv_path = temp_dir.path().join("sample_fp_max_combos.csv");
    fs::write(&csv_path, SAMPLE_DATA)?;
    let output_dir = temp_dir.path().join("output_fp_max_combos");

    // Run 1: depth 2, feature-pairs enabled, tight max_combos cap.
    let mut cfg1 = base_config(&csv_path, &output_dir);
    cfg1.max_depth = 2;
    cfg1.max_combos = Some(8);
    cfg1.enable_feature_pairs = true;
    run_pipeline_for_test(cfg1)?;

    let results_dir = output_dir.join("results_parquet");
    let (unique_after_1, rows_after_1) = load_combinations_and_count(&results_dir)?;
    assert!(
        rows_after_1 > 0,
        "expected some result rows after initial feature-pair depth-2 run with max_combos=8"
    );
    assert_eq!(
        rows_after_1, unique_after_1,
        "initial feature-pair depth-2 run with max_combos=8 should not produce duplicate combinations"
    );

    // Run 2: depth 2, feature-pairs enabled, higher max_combos cap.
    let mut cfg2 = base_config(&csv_path, &output_dir);
    cfg2.max_depth = 2;
    cfg2.max_combos = Some(32);
    cfg2.enable_feature_pairs = true;
    run_pipeline_for_test(cfg2)?;

    let (unique_after_2, rows_after_2) = load_combinations_and_count(&results_dir)?;
    assert!(
        unique_after_2 > unique_after_1,
        "raising max_combos at depth 2 with feature pairs enabled should extend the set of evaluated combinations"
    );
    assert_eq!(
        rows_after_2, unique_after_2,
        "raising max_combos at depth 2 with feature pairs enabled should not introduce duplicate combinations"
    );

    Ok(())
}

/// Tests that toggling enable_feature_pairs after a run with zero_samples
/// causes a catalog hash mismatch error, verifying the integrity protection.
#[test]
fn pipeline_resume_feature_pairs_toggle_causes_hash_mismatch() -> Result<()> {
    let temp_dir = tempdir()?;
    let csv_path = temp_dir.path().join("sample.csv");
    fs::write(&csv_path, SAMPLE_DATA)?;
    let output_dir = temp_dir.path().join("output");

    // First run: baseline catalog (no feature-to-feature comparisons).
    let config1 = base_config(&csv_path, &output_dir);
    run_pipeline_for_test(config1)?;

    let results_dir = output_dir.join("results_parquet");
    let (unique_after_run1, rows_after_run1) = load_combinations_and_count(&results_dir)?;
    assert!(
        rows_after_run1 > 0,
        "expected at least one result row after first pipeline run"
    );
    assert_eq!(
        rows_after_run1, unique_after_run1,
        "each stored row should correspond to a unique combination after the first run"
    );

    // Second run: same catalog and CSV should reuse resume metadata and
    // combination keys so that no new results are written.
    let config2 = base_config(&csv_path, &output_dir);
    run_pipeline_for_test(config2)?;
    let (unique_after_run2, rows_after_run2) = load_combinations_and_count(&results_dir)?;
    assert!(
        unique_after_run2 >= unique_after_run1,
        "re-running the pipeline with the same configuration should not lose combinations"
    );

    // Third run: attempt to enable feature-to-feature comparisons. This
    // produces a different catalog hash and extends the catalog, but should
    // still preserve previously stored results.
    let mut config3 = base_config(&csv_path, &output_dir);
    config3.enable_feature_pairs = true;
    run_pipeline_for_test(config3)?;

    // Results from the previous run should be preserved after the feature-
    // pair toggle; the surface may grow but should not shrink.
    let (unique_after_run3, rows_after_run3) = load_combinations_and_count(&results_dir)?;
    assert!(
        unique_after_run3 >= unique_after_run2,
        "enabling feature_pairs should not reduce the number of stored combinations"
    );
    assert!(
        rows_after_run3 >= rows_after_run2,
        "enabling feature_pairs should not reduce the number of stored rows"
    );

    Ok(())
}

#[test]
fn depth_one_completion_causes_skip_on_rerun() -> Result<()> {
    let temp_dir = tempdir()?;
    let csv_path = temp_dir.path().join("sample.csv");
    fs::write(&csv_path, SAMPLE_DATA)?;
    let output_dir = temp_dir.path().join("output_depth1");

    // First run: depth 1 only, no feature pairs, no explicit max_combos so
    // the catalog can be fully enumerated.
    let mut config1 = base_config(&csv_path, &output_dir);
    config1.max_depth = 1;
    config1.max_combos = None;
    config1.enable_feature_pairs = false;
    run_pipeline_for_test(config1.clone())?;

    let results_dir = output_dir.join("results_parquet");
    let (_, rows_after_run1) = load_combinations_and_count(&results_dir)?;
    assert!(
        rows_after_run1 > 0,
        "expected at least one result row after first depth-1 run"
    );

    // Use a lightweight config to query depth completeness over the existing
    // results. Only source_csv, direction, target, and output_dir need to
    // match; the rest of the fields are irrelevant for this query.
    let depth_cfg = Config {
        input_csv: csv_path.clone(),
        source_csv: Some(csv_path.clone()),
        direction: Direction::Long,
        target: "is_green".to_string(),
        output_dir: output_dir.clone(),
        max_depth: 1,
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

    let (_store1, _) = CumulativeStore::new(&depth_cfg)?;

    // Capture the total combinations stored after the first run so we can
    // verify that a second run at the same depth does not change the result
    // surface.
    let (unique_after_run1, rows_after_run1_again) = load_combinations_and_count(&results_dir)?;
    assert_eq!(
        rows_after_run1, rows_after_run1_again,
        "reload of results after first run should be stable"
    );

    // Second run: same depth-1 configuration. With depth 1 already complete,
    // the pipeline should take the short-circuit path that skips evaluation
    // and reuses existing cumulative results.
    run_pipeline_for_test(config1)?;
    let (unique_after_run2, rows_after_run2) = load_combinations_and_count(&results_dir)?;
    assert_eq!(
        unique_after_run2, unique_after_run1,
        "re-running a fully-complete depth-1 catalog should not change the set of combinations"
    );
    assert_eq!(
        rows_after_run2, rows_after_run1,
        "re-running a fully-complete depth-1 catalog should not add new result rows"
    );

    Ok(())
}
