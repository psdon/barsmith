use std::collections::HashMap;
use std::fs;
use std::fs::File;
use std::path::Path;

use anyhow::Result;
use barsmith_rs::feature::{ComparisonOperator, ComparisonSpec};
use barsmith_rs::{
    Config, Direction, FeatureDescriptor, LogicMode, PermutationPipeline, ReportMetricsMode,
    config::{EvalProfileMode, PositionSizingMode, StackingMode, StatsDetail, StopDistanceUnit},
};
use polars::prelude::*;
use tempfile::tempdir;

fn make_config(csv_path: &Path, output_dir: &Path, max_depth: usize) -> Config {
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
        force_recompute: true,
        max_drawdown: 50.0,
        max_drawdown_report: None,
        min_calmar_report: None,
        strict_min_pruning: true,
        enable_subset_pruning: false,
        enable_feature_pairs: false,
        feature_pairs_limit: None,
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

fn load_combinations(results_dir: &Path) -> Result<Vec<String>> {
    let mut frames: Vec<DataFrame> = Vec::new();
    for entry in fs::read_dir(results_dir)? {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.starts_with("part-") && name.ends_with(".parquet") {
            let file = File::open(entry.path())?;
            let reader = ParquetReader::new(file);
            let df = reader.finish()?;
            frames.push(df);
        }
    }
    if frames.is_empty() {
        return Ok(Vec::new());
    }
    let mut iter = frames.into_iter();
    let mut df_all = iter
        .next()
        .expect("non-empty frames should yield at least one DataFrame");
    for df in iter {
        df_all.vstack_mut(&df)?;
    }
    let combo_col = df_all.column("combination")?.str()?;
    let mut out = Vec::with_capacity(df_all.height());
    for i in 0..df_all.height() {
        out.push(combo_col.get(i).unwrap().to_string());
    }
    Ok(out)
}

fn has_combo(combos: &[String], needle: &str) -> bool {
    combos.iter().any(|c| c.contains(needle))
}

/// Combinations that form a proper bracket on a single base feature
/// (e.g. adx>20 && adx<50) should be allowed by the scalar-bound
/// structural constraint.
#[test]
fn scalar_bounds_allow_bracket_for_single_base() -> Result<()> {
    let temp_dir = tempdir()?;
    let csv_path = temp_dir.path().join("data_scalar_bracket.csv");
    let output_dir = temp_dir.path().join("out_scalar_bracket");

    fs::write(
        &csv_path,
        "\
timestamp,is_green,adx
2024-01-01T00:00:00Z,true,10.0
2024-01-01T00:30:00Z,true,30.0
2024-01-01T01:00:00Z,false,40.0
2024-01-01T01:30:00Z,true,60.0
",
    )?;

    let features = vec![
        FeatureDescriptor::comparison("adx>20", "test"),
        FeatureDescriptor::comparison("adx>40", "test"),
        FeatureDescriptor::comparison("adx<50", "test"),
    ];
    let mut specs: HashMap<String, ComparisonSpec> = HashMap::new();
    specs.insert(
        "adx>20".to_string(),
        ComparisonSpec::threshold("adx", ComparisonOperator::GreaterThan, 20.0),
    );
    specs.insert(
        "adx>40".to_string(),
        ComparisonSpec::threshold("adx", ComparisonOperator::GreaterThan, 40.0),
    );
    specs.insert(
        "adx<50".to_string(),
        ComparisonSpec::threshold("adx", ComparisonOperator::LessThan, 50.0),
    );

    let config = make_config(&csv_path, &output_dir, 2);
    let mut pipeline = PermutationPipeline::new(config, features, specs);
    pipeline.run()?;

    let results_dir = output_dir.join("results_parquet");
    let combos = load_combinations(&results_dir)?;

    assert!(
        has_combo(&combos, "adx>20 && adx<50"),
        "scalar bracket adx>20 && adx<50 should be allowed"
    );

    Ok(())
}

/// Combinations that stack multiple bounds of the same direction on a single
/// base feature (e.g. adx>20 && adx>40) should be rejected by the scalar-bound
/// structural constraint.
#[test]
fn scalar_bounds_reject_same_direction_ladder_for_single_base() -> Result<()> {
    let temp_dir = tempdir()?;
    let csv_path = temp_dir.path().join("data_scalar_ladder.csv");
    let output_dir = temp_dir.path().join("out_scalar_ladder");

    fs::write(
        &csv_path,
        "\
timestamp,is_green,adx
2024-01-01T00:00:00Z,true,10.0
2024-01-01T00:30:00Z,true,30.0
2024-01-01T01:00:00Z,false,40.0
2024-01-01T01:30:00Z,true,60.0
",
    )?;

    let features = vec![
        FeatureDescriptor::comparison("adx>20", "test"),
        FeatureDescriptor::comparison("adx>40", "test"),
    ];
    let mut specs: HashMap<String, ComparisonSpec> = HashMap::new();
    specs.insert(
        "adx>20".to_string(),
        ComparisonSpec::threshold("adx", ComparisonOperator::GreaterThan, 20.0),
    );
    specs.insert(
        "adx>40".to_string(),
        ComparisonSpec::threshold("adx", ComparisonOperator::GreaterThan, 40.0),
    );

    let config = make_config(&csv_path, &output_dir, 2);
    let mut pipeline = PermutationPipeline::new(config, features, specs);
    pipeline.run()?;

    let results_dir = output_dir.join("results_parquet");
    let combos = load_combinations(&results_dir)?;

    assert!(
        !has_combo(&combos, "adx>20 && adx>40"),
        "same-direction scalar ladder adx>20 && adx>40 should be rejected"
    );

    Ok(())
}
