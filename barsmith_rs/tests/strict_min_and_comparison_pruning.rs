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

fn make_config(csv_path: &Path, output_dir: &Path, max_depth: usize, min_samples: usize) -> Config {
    Config {
        input_csv: csv_path.to_path_buf(),
        source_csv: Some(csv_path.to_path_buf()),
        direction: Direction::Long,
        target: "is_green".to_string(),
        output_dir: output_dir.to_path_buf(),
        max_depth,
        min_sample_size: min_samples,
        min_sample_size_report: min_samples,
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

#[test]
fn constant_scalar_thresholds_are_pruned_from_catalog() -> Result<()> {
    let temp_dir = tempdir()?;
    let csv_path = temp_dir.path().join("data_const_thresholds.csv");
    let output_dir = temp_dir.path().join("out_const_thresholds");

    fs::write(
        &csv_path,
        "\
timestamp,is_green,x,flag
2024-01-01T00:00:00Z,true,1.0,true
2024-01-01T00:30:00Z,false,1.0,false
",
    )?;

    let features = vec![
        FeatureDescriptor::boolean("flag", "test"),
        FeatureDescriptor::comparison("x>0", "test"),
        FeatureDescriptor::comparison("x>2", "test"),
    ];
    let mut specs = std::collections::HashMap::new();
    specs.insert(
        "x>0".to_string(),
        ComparisonSpec::threshold("x", ComparisonOperator::GreaterThan, 0.0),
    );
    specs.insert(
        "x>2".to_string(),
        ComparisonSpec::threshold("x", ComparisonOperator::GreaterThan, 2.0),
    );

    let config = make_config(&csv_path, &output_dir, 1, 1);
    let mut pipeline = PermutationPipeline::new(config, features, specs);
    pipeline.run()?;

    let results_dir = output_dir.join("results_parquet");
    let combos = load_combinations(&results_dir)?;
    assert!(
        !has_combo(&combos, "x>0"),
        "constant-true scalar predicate x>0 should be pruned from the catalog"
    );
    assert!(
        !has_combo(&combos, "x>2"),
        "constant-false scalar predicate x>2 should be pruned from the catalog"
    );
    assert!(
        has_combo(&combos, "flag"),
        "non-constant boolean feature should still participate in combinations"
    );

    Ok(())
}

#[test]
fn non_constant_thresholds_are_kept_when_not_strict() -> Result<()> {
    let temp_dir = tempdir()?;
    let csv_path = temp_dir.path().join("data_non_const_thresholds.csv");
    let output_dir = temp_dir.path().join("out_non_const_thresholds");

    fs::write(
        &csv_path,
        "\
timestamp,is_green,x
2024-01-01T00:00:00Z,true,1.0
2024-01-01T00:30:00Z,false,2.0
2024-01-01T01:00:00Z,true,3.0
",
    )?;

    let features = vec![
        FeatureDescriptor::comparison("x>1", "test"), // [false,true,true]
        FeatureDescriptor::comparison("x>2", "test"), // [false,false,true]
    ];
    let mut specs = std::collections::HashMap::new();
    specs.insert(
        "x>1".to_string(),
        ComparisonSpec::threshold("x", ComparisonOperator::GreaterThan, 1.0),
    );
    specs.insert(
        "x>2".to_string(),
        ComparisonSpec::threshold("x", ComparisonOperator::GreaterThan, 2.0),
    );

    let config = make_config(&csv_path, &output_dir, 1, 1);
    let mut pipeline = PermutationPipeline::new(config, features, specs);
    pipeline.run()?;

    let results_dir = output_dir.join("results_parquet");
    let combos = load_combinations(&results_dir)?;
    assert!(
        has_combo(&combos, "x>1"),
        "non-constant threshold x>1 should be kept when it meets or exceeds min_samples"
    );
    assert!(
        has_combo(&combos, "x>2"),
        "non-constant threshold x>2 should be kept when it meets or exceeds min_samples"
    );

    Ok(())
}

#[test]
fn strict_min_drops_under_min_thresholds() -> Result<()> {
    let temp_dir = tempdir()?;
    let csv_path = temp_dir.path().join("data_strict_min_thresholds.csv");
    let output_dir = temp_dir.path().join("out_strict_min_thresholds");

    fs::write(
        &csv_path,
        "\
timestamp,is_green,x
2024-01-01T00:00:00Z,true,1.0
2024-01-01T00:30:00Z,false,3.0
2024-01-01T01:00:00Z,true,5.0
",
    )?;

    let features = vec![
        // x>1: [false,true,true] -> 2 samples (>=2)
        // x>4: [false,false,true] -> 1 sample (<2)
        FeatureDescriptor::comparison("x>1", "test"),
        FeatureDescriptor::comparison("x>4_under_min", "test"),
    ];
    let mut specs = std::collections::HashMap::new();
    specs.insert(
        "x>1".to_string(),
        ComparisonSpec::threshold("x", ComparisonOperator::GreaterThan, 1.0),
    );
    specs.insert(
        "x>4_under_min".to_string(),
        ComparisonSpec::threshold("x", ComparisonOperator::GreaterThan, 4.0),
    );

    let config = make_config(&csv_path, &output_dir, 1, 2);
    let mut pipeline = PermutationPipeline::new(config, features, specs);
    pipeline.run()?;

    let results_dir = output_dir.join("results_parquet");
    let combos = load_combinations(&results_dir)?;
    assert!(
        has_combo(&combos, "x>1"),
        "x>1 should be kept when it meets min_samples"
    );
    // x>1 is above the min_samples threshold and should be kept.
    assert!(
        !has_combo(&combos, "x>4_under_min"),
        "x>4_under_min should be dropped as an under-min threshold"
    );

    Ok(())
}

#[test]
fn duplicate_scalar_thresholds_collapse_to_one_predicate() -> Result<()> {
    let temp_dir = tempdir()?;
    let csv_path = temp_dir.path().join("data_duplicate_thresholds.csv");
    let output_dir = temp_dir.path().join("out_duplicate_thresholds");

    fs::write(
        &csv_path,
        "\
timestamp,is_green,x
2024-01-01T00:00:00Z,true,1.0
2024-01-01T00:30:00Z,false,3.0
",
    )?;

    let features = vec![
        FeatureDescriptor::comparison("x>1_alias1", "test"),
        FeatureDescriptor::comparison("x>1_alias2", "test"),
    ];
    let mut specs = std::collections::HashMap::new();
    specs.insert(
        "x>1_alias1".to_string(),
        ComparisonSpec::threshold("x", ComparisonOperator::GreaterThan, 1.0),
    );
    specs.insert(
        "x>1_alias2".to_string(),
        ComparisonSpec::threshold("x", ComparisonOperator::GreaterThan, 1.0),
    );

    let config = make_config(&csv_path, &output_dir, 1, 1);
    let mut pipeline = PermutationPipeline::new(config, features, specs);
    pipeline.run()?;

    let results_dir = output_dir.join("results_parquet");
    let combos = load_combinations(&results_dir)?;
    let alias1 = has_combo(&combos, "x>1_alias1");
    let alias2 = has_combo(&combos, "x>1_alias2");
    assert!(
        alias1 || alias2,
        "one of the duplicate scalar predicates should be kept"
    );
    assert!(
        !(alias1 && alias2),
        "both duplicate scalar predicates should not be simultaneously kept"
    );

    Ok(())
}

#[test]
fn feature_pair_constant_predicate_is_pruned() -> Result<()> {
    let temp_dir = tempdir()?;
    let csv_path = temp_dir.path().join("data_const_pair.csv");
    let output_dir = temp_dir.path().join("out_const_pair");

    fs::write(
        &csv_path,
        "\
timestamp,is_green,x,y
2024-01-01T00:00:00Z,true,1.0,1.0
2024-01-01T00:30:00Z,false,2.0,2.0
",
    )?;

    let features = vec![FeatureDescriptor::comparison("x>=y", "pair")];
    let mut specs = std::collections::HashMap::new();
    specs.insert(
        "x>=y".to_string(),
        ComparisonSpec::pair("x", ComparisonOperator::GreaterEqual, "y"),
    );

    let config = make_config(&csv_path, &output_dir, 1, 1);
    let mut pipeline = PermutationPipeline::new(config, features, specs);
    pipeline.run()?;

    let results_dir = output_dir.join("results_parquet");
    let combos = load_combinations(&results_dir)?;
    assert!(
        !has_combo(&combos, "x>=y"),
        "constant feature-pair predicate x>=y should be pruned from the catalog"
    );

    Ok(())
}

#[test]
fn strict_min_does_not_prune_supersets_of_under_min_subset() -> Result<()> {
    let temp_dir = tempdir()?;
    let csv_path = temp_dir.path().join("data_strict_branch.csv");
    let output_dir = temp_dir.path().join("out_strict_branch");

    fs::write(
        &csv_path,
        "\
timestamp,is_green,f1,f2
2024-01-01T00:00:00Z,true,true,true
2024-01-01T00:30:00Z,false,false,true
2024-01-01T01:00:00Z,true,false,true
",
    )?;

    let features = vec![
        FeatureDescriptor::boolean("f1", "test"),
        FeatureDescriptor::boolean("f2", "test"),
    ];

    // f1 has sample size 1; f2 has sample size 3. With min_samples=2,
    // f1 is under-min but still evaluated, and its supersets should not
    // be pruned by branch pruning; only true zero subsets participate
    // in pruning. Using batch_size=1 ensures that evaluation order does
    // not hide any branch behavior.
    let mut config = make_config(&csv_path, &output_dir, 2, 1);
    config.batch_size = 1;
    let mut pipeline = PermutationPipeline::new(config, features, Default::default());
    pipeline.run()?;

    let results_dir = output_dir.join("results_parquet");
    let combos = load_combinations(&results_dir)?;
    assert!(
        has_combo(&combos, "f1"),
        "depth-1 combination f1 should be evaluated even if under min_samples"
    );
    assert!(
        has_combo(&combos, "f2"),
        "depth-1 combination f2 should be evaluated"
    );
    assert!(
        has_combo(&combos, "f1 && f2"),
        "branch pruning should not drop depth-2 supersets that include an under-min subset; only zero-sample subsets participate in pruning"
    );

    Ok(())
}

#[test]
fn non_strict_min_does_not_prune_supersets_of_under_min_subset() -> Result<()> {
    let temp_dir = tempdir()?;
    let csv_path = temp_dir.path().join("data_non_strict_branch.csv");
    let output_dir = temp_dir.path().join("out_non_strict_branch");

    fs::write(
        &csv_path,
        "\
timestamp,is_green,f1,f2
2024-01-01T00:00:00Z,true,true,true
2024-01-01T00:30:00Z,false,false,true
2024-01-01T01:00:00Z,true,false,true
",
    )?;

    let features = vec![
        FeatureDescriptor::boolean("f1", "test"),
        FeatureDescriptor::boolean("f2", "test"),
    ];

    let mut config = make_config(&csv_path, &output_dir, 2, 1);
    config.batch_size = 1;
    let mut pipeline = PermutationPipeline::new(config, features, Default::default());
    pipeline.run()?;

    let results_dir = output_dir.join("results_parquet");
    let combos = load_combinations(&results_dir)?;
    assert!(
        has_combo(&combos, "f1 && f2"),
        "depth-2 combo f1 && f2 should still be evaluated even if f1 is under-min; branch pruning is not driven by min-sample thresholds"
    );

    Ok(())
}

#[test]
fn strict_min_and_zero_sample_pruning_coexist() -> Result<()> {
    let temp_dir = tempdir()?;
    let csv_path = temp_dir.path().join("data_strict_and_zero.csv");
    let output_dir = temp_dir.path().join("out_strict_and_zero");

    // f1 and f2 never co-occur; f3 is always true.
    fs::write(
        &csv_path,
        "\
timestamp,is_green,f1,f2,f3
2024-01-01T00:00:00Z,true,true,false,true
2024-01-01T00:30:00Z,false,false,true,true
2024-01-01T01:00:00Z,true,false,true,true
",
    )?;

    let features = vec![
        FeatureDescriptor::boolean("f1", "test"),
        FeatureDescriptor::boolean("f2", "test"),
        FeatureDescriptor::boolean("f3", "test"),
    ];

    let mut config = make_config(&csv_path, &output_dir, 3, 2);
    config.batch_size = 1;
    let mut pipeline = PermutationPipeline::new(config, features, Default::default());
    pipeline.run()?;

    let results_dir = output_dir.join("results_parquet");
    let combos = load_combinations(&results_dir)?;

    // Depth-3 supersets such as f1 && f2 && f3 should not appear when
    // either zero-sample or strict-min subsets allow us to prune branches.
    assert!(
        !has_combo(&combos, "f1 && f2 && f3"),
        "supersets of zero-sample subset f1 && f2 should be pruned"
    );

    Ok(())
}

#[test]
fn strict_min_pruning_affects_only_scalar_predicates_not_booleans() -> Result<()> {
    let temp_dir = tempdir()?;
    let csv_path = temp_dir.path().join("data_strict_scalar_only.csv");
    let output_dir = temp_dir.path().join("out_strict_scalar_only");

    fs::write(
        &csv_path,
        "\
timestamp,is_green,flag,x
2024-01-01T00:00:00Z,true,true,1.0
2024-01-01T00:30:00Z,false,false,3.0
",
    )?;

    let features = vec![
        FeatureDescriptor::boolean("flag", "test"),
        FeatureDescriptor::comparison("x>2_under_min", "test"),
    ];
    let mut specs = std::collections::HashMap::new();
    specs.insert(
        "x>2_under_min".to_string(),
        ComparisonSpec::threshold("x", ComparisonOperator::GreaterThan, 2.0),
    );

    let config = make_config(&csv_path, &output_dir, 2, 1);
    let mut pipeline = PermutationPipeline::new(config, features, specs);
    pipeline.run()?;

    let results_dir = output_dir.join("results_parquet");
    let combos = load_combinations(&results_dir)?;

    assert!(
        has_combo(&combos, "flag"),
        "boolean feature flag should still be eligible even when under-min scalar predicates are pruned from the catalog"
    );

    Ok(())
}

#[test]
fn boolean_under_min_is_pruned_from_catalog() -> Result<()> {
    let temp_dir = tempdir()?;
    let csv_path = temp_dir.path().join("data_boolean_under_min.csv");
    let output_dir = temp_dir.path().join("out_boolean_under_min");

    // b1 is true on exactly one is_green bar (sample=1).
    // b2 is true on all three is_green bars (sample=3).
    // With min_samples=2, b1 should be pruned and b2 kept.
    fs::write(
        &csv_path,
        "\
timestamp,is_green,b1,b2
2024-01-01T00:00:00Z,true,true,true
2024-01-01T00:30:00Z,false,false,true
2024-01-01T01:00:00Z,false,false,true
",
    )?;

    let features = vec![
        FeatureDescriptor::boolean("b1", "test"),
        FeatureDescriptor::boolean("b2", "test"),
    ];

    let config = make_config(&csv_path, &output_dir, 2, 2);
    let mut pipeline = PermutationPipeline::new(config, features, Default::default());
    pipeline.run()?;

    let results_dir = output_dir.join("results_parquet");
    let combos = load_combinations(&results_dir)?;
    assert!(
        !has_combo(&combos, "b1"),
        "under-min boolean feature b1 should be pruned from the catalog"
    );
    assert!(
        has_combo(&combos, "b2"),
        "boolean feature b2 meeting min_samples should be kept in the catalog"
    );

    Ok(())
}

#[test]
fn boolean_zero_sample_is_pruned_from_catalog() -> Result<()> {
    let temp_dir = tempdir()?;
    let csv_path = temp_dir.path().join("data_boolean_zero_sample.csv");
    let output_dir = temp_dir.path().join("out_boolean_zero_sample");

    // b_zero is never true (sample=0).
    // b_full is always true on is_green bars (sample=2).
    // With min_samples=1, b_zero should be pruned and b_full kept.
    fs::write(
        &csv_path,
        "\
timestamp,is_green,b_zero,b_full
2024-01-01T00:00:00Z,true,false,true
2024-01-01T00:30:00Z,true,false,true
",
    )?;

    let features = vec![
        FeatureDescriptor::boolean("b_zero", "test"),
        FeatureDescriptor::boolean("b_full", "test"),
    ];

    let config = make_config(&csv_path, &output_dir, 2, 1);
    let mut pipeline = PermutationPipeline::new(config, features, Default::default());
    pipeline.run()?;

    let results_dir = output_dir.join("results_parquet");
    let combos = load_combinations(&results_dir)?;
    assert!(
        !has_combo(&combos, "b_zero"),
        "zero-sample boolean feature b_zero should be pruned from the catalog"
    );
    assert!(
        has_combo(&combos, "b_full"),
        "boolean feature b_full with non-zero sample should be kept"
    );

    Ok(())
}

#[test]
fn feature_pair_under_min_is_pruned_from_catalog() -> Result<()> {
    let temp_dir = tempdir()?;
    let csv_path = temp_dir.path().join("data_pair_under_min.csv");
    let output_dir = temp_dir.path().join("out_pair_under_min");

    // x>y is true on exactly one bar (sample=1) and false on the others.
    // With min_samples=2, the non-constant pair predicate should be pruned.
    fs::write(
        &csv_path,
        "\
timestamp,is_green,x,y
2024-01-01T00:00:00Z,true,2.0,1.0
2024-01-01T00:30:00Z,false,1.0,2.0
2024-01-01T01:00:00Z,true,1.0,2.0
",
    )?;

    let features = vec![FeatureDescriptor::comparison("x>y_pair", "pair")];
    let mut specs = std::collections::HashMap::new();
    specs.insert(
        "x>y_pair".to_string(),
        ComparisonSpec::pair("x", ComparisonOperator::GreaterThan, "y"),
    );

    let config = make_config(&csv_path, &output_dir, 1, 2);
    let mut pipeline = PermutationPipeline::new(config, features, specs);
    pipeline.run()?;

    let results_dir = output_dir.join("results_parquet");
    let combos = load_combinations(&results_dir)?;
    assert!(
        !has_combo(&combos, "x>y_pair"),
        "feature-pair predicate x>y_pair with sample below min_samples should be pruned from the catalog"
    );

    Ok(())
}
