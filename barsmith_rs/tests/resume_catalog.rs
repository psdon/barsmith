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

fn summary(total: usize) -> StatSummary {
    summary_with_depth(1, total)
}

fn summary_with_depth(depth: usize, total: usize) -> StatSummary {
    let wins = total / 2;
    let losses = total - wins;
    let win_rate = wins as f64 / total as f64;
    StatSummary {
        depth,
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

#[test]
fn resume_offset_persists_when_catalog_hash_differs_and_combinations_are_reused() -> Result<()> {
    let temp_dir = tempdir()?;
    let csv_path = temp_dir.path().join("data.csv");
    let output_dir = temp_dir.path().join("out");
    fs::write(&csv_path, "timestamp,open\n2024-01-01T00:00:00Z,1\n")?;

    // First run with catalog hash A: ingest a small batch and record resume offset.
    let mut cfg_a = base_config(&csv_path, &output_dir);
    cfg_a.catalog_hash = Some("A".to_string());
    let (mut store_a, resume_a_1) = CumulativeStore::new(&cfg_a)?;
    assert_eq!(resume_a_1, 0, "fresh config should start at offset 0");

    let combos = vec![make_combo("alpha"), make_combo("beta")];
    let stats = vec![summary(100), summary(100)];
    store_a.ingest_with_enumerated(&combos, &stats, combos.len(), 0)?;
    store_a.flush()?;

    let (_store_a2, resume_a_2) = CumulativeStore::new(&cfg_a)?;
    assert_eq!(
        resume_a_2,
        combos.len() as u64,
        "same catalog should reuse resume offset"
    );

    // Second config with different catalog hash against same CSV/output_dir:
    // resume offset should still be reused, and the combination cache should
    // see the previously ingested combinations.
    let mut cfg_b = base_config(&csv_path, &output_dir);
    cfg_b.catalog_hash = Some("B".to_string());
    let (store_b, resume_b) = CumulativeStore::new(&cfg_b)?;
    assert_eq!(
        resume_b,
        combos.len() as u64,
        "different catalog hash should still reuse index-based resume offset when CSV is unchanged"
    );

    let existing = store_b.existing_combinations()?;
    assert!(
        existing.contains("alpha") && existing.contains("beta"),
        "combination-key reuse should still see previously ingested combinations"
    );

    Ok(())
}

#[test]
fn resume_same_config_reuses_processed_index() -> Result<()> {
    let temp_dir = tempdir()?;
    let csv_path = temp_dir.path().join("data.csv");
    let output_dir = temp_dir.path().join("out");
    fs::write(&csv_path, "timestamp,open\n2024-01-01T00:00:00Z,1\n")?;

    let cfg = base_config(&csv_path, &output_dir);

    // First run: no metadata yet, resume_offset should be zero.
    let (mut store1, resume1) = CumulativeStore::new(&cfg)?;
    assert_eq!(resume1, 0, "fresh config should start at offset 0");

    let combos = vec![make_combo("alpha"), make_combo("beta"), make_combo("gamma")];
    let stats = vec![summary(100), summary(100), summary(100)];
    store1.ingest_with_enumerated(&combos, &stats, combos.len(), 0)?;
    store1.flush()?;
    drop(store1);

    // Second run: same config and CSV should reuse the processed index.
    let (_store2, resume2) = CumulativeStore::new(&cfg)?;
    assert_eq!(
        resume2,
        combos.len() as u64,
        "same config and CSV should reuse resume offset from metadata"
    );

    Ok(())
}

/// Changing runtime knobs like `batch_size` should not affect the stable
/// config hash used for resume metadata. This test verifies that a second
/// run with a different batch size still picks up the previously processed
/// index rather than resetting to zero.
#[test]
fn resume_ignores_batch_size_changes() -> Result<()> {
    let temp_dir = tempdir()?;
    let csv_path = temp_dir.path().join("data.csv");
    let output_dir = temp_dir.path().join("out_batch_size_resume");
    fs::write(&csv_path, "timestamp,open\n2024-01-01T00:00:00Z,1\n")?;

    // First run: small batch size.
    let mut cfg1 = base_config(&csv_path, &output_dir);
    cfg1.batch_size = 10;

    let (mut store1, resume1) = CumulativeStore::new(&cfg1)?;
    assert_eq!(resume1, 0, "fresh config should start at offset 0");

    let combos = vec![make_combo("alpha"), make_combo("beta"), make_combo("gamma")];
    let stats = vec![summary(100), summary(100), summary(100)];
    store1.ingest_with_enumerated(&combos, &stats, combos.len(), 0)?;
    store1.flush()?;
    drop(store1);

    // Second run: same CSV/output_dir/catalog but a much larger batch size.
    // Resume offset should still reflect all previously processed combinations.
    let mut cfg2 = base_config(&csv_path, &output_dir);
    cfg2.batch_size = 1000;

    let (_store2, resume2) = CumulativeStore::new(&cfg2)?;
    assert_eq!(
        resume2,
        combos.len() as u64,
        "changing batch_size between runs must not reset resume offset"
    );

    Ok(())
}

/// Toggling auto_batch should not affect the stable config hash or resume
/// offset; it only changes how batch sizes are tuned within a run.
#[test]
fn resume_ignores_auto_batch_changes() -> Result<()> {
    let temp_dir = tempdir()?;
    let csv_path = temp_dir.path().join("data.csv");
    let output_dir = temp_dir.path().join("out_auto_batch_resume");
    fs::write(&csv_path, "timestamp,open\n2024-01-01T00:00:00Z,1\n")?;

    // First run: auto_batch disabled.
    let mut cfg1 = base_config(&csv_path, &output_dir);
    cfg1.auto_batch = false;

    let (mut store1, resume1) = CumulativeStore::new(&cfg1)?;
    assert_eq!(resume1, 0, "fresh config should start at offset 0");

    let combos = vec![make_combo("alpha"), make_combo("beta")];
    let stats = vec![summary(100), summary(100)];
    store1.ingest_with_enumerated(&combos, &stats, combos.len(), 0)?;
    store1.flush()?;
    drop(store1);

    // Second run: same catalog and CSV, but auto_batch enabled.
    let mut cfg2 = base_config(&csv_path, &output_dir);
    cfg2.auto_batch = true;

    let (_store2, resume2) = CumulativeStore::new(&cfg2)?;
    assert_eq!(
        resume2,
        combos.len() as u64,
        "changing auto_batch between runs must not reset resume offset"
    );

    Ok(())
}

/// Toggling early_exit_when_reused should not change the resume namespace;
/// it only changes whether a run can stop early based on reuse heuristics.
#[test]
fn resume_ignores_early_exit_changes() -> Result<()> {
    let temp_dir = tempdir()?;
    let csv_path = temp_dir.path().join("data.csv");
    let output_dir = temp_dir.path().join("out_early_exit_resume");
    fs::write(&csv_path, "timestamp,open\n2024-01-01T00:00:00Z,1\n")?;

    // First run: early_exit_when_reused disabled.
    let mut cfg1 = base_config(&csv_path, &output_dir);
    cfg1.early_exit_when_reused = false;

    let (mut store1, resume1) = CumulativeStore::new(&cfg1)?;
    assert_eq!(resume1, 0, "fresh config should start at offset 0");

    let combos = vec![make_combo("alpha"), make_combo("beta")];
    let stats = vec![summary(100), summary(100)];
    store1.ingest_with_enumerated(&combos, &stats, combos.len(), 0)?;
    store1.flush()?;
    drop(store1);

    // Second run: same catalog and CSV, but early_exit_when_reused enabled.
    let mut cfg2 = base_config(&csv_path, &output_dir);
    cfg2.early_exit_when_reused = true;

    let (_store2, resume2) = CumulativeStore::new(&cfg2)?;
    assert_eq!(
        resume2,
        combos.len() as u64,
        "changing early_exit_when_reused between runs must not reset resume offset"
    );

    Ok(())
}

/// Changing max_combos should not affect the resume namespace; it only
/// changes how far into the global combination stream a particular run
/// chooses to walk before stopping.
#[test]
fn resume_ignores_max_combos_changes() -> Result<()> {
    let temp_dir = tempdir()?;
    let csv_path = temp_dir.path().join("data.csv");
    let output_dir = temp_dir.path().join("out_max_combos_resume");
    fs::write(&csv_path, "timestamp,open\n2024-01-01T00:00:00Z,1\n")?;

    // First run: with a small max_combos limit.
    let mut cfg1 = base_config(&csv_path, &output_dir);
    cfg1.max_combos = Some(2);

    let (mut store1, resume1) = CumulativeStore::new(&cfg1)?;
    assert_eq!(resume1, 0, "fresh config should start at offset 0");

    let combos = vec![make_combo("alpha"), make_combo("beta")];
    let stats = vec![summary(100), summary(100)];
    store1.ingest_with_enumerated(&combos, &stats, combos.len(), 0)?;
    store1.flush()?;
    drop(store1);

    // Second run: same catalog and CSV, but with a larger max_combos.
    let mut cfg2 = base_config(&csv_path, &output_dir);
    cfg2.max_combos = Some(10);

    let (_store2, resume2) = CumulativeStore::new(&cfg2)?;
    assert_eq!(
        resume2,
        combos.len() as u64,
        "changing max_combos between runs must not reset resume offset"
    );

    Ok(())
}

#[test]
fn resume_with_metadata_but_no_parquet_is_safe() -> Result<()> {
    let temp_dir = tempdir()?;
    let csv_path = temp_dir.path().join("data.csv");
    let output_dir = temp_dir.path().join("out");
    fs::write(&csv_path, "timestamp,open\n2024-01-01T00:00:00Z,1\n")?;

    let cfg = base_config(&csv_path, &output_dir);

    // First run: ingest a small batch so metadata and Parquet are written.
    let (mut store1, resume1) = CumulativeStore::new(&cfg)?;
    assert_eq!(resume1, 0, "fresh config should start at offset 0");

    let combos = vec![make_combo("alpha"), make_combo("beta")];
    let stats = vec![summary(100), summary(100)];
    store1.ingest_with_enumerated(&combos, &stats, combos.len(), 0)?;
    store1.flush()?;
    drop(store1);

    // Simulate external deletion of Parquet batches while leaving DuckDB metadata.
    let results_dir = output_dir.join("results_parquet");
    if results_dir.exists() {
        for entry in fs::read_dir(&results_dir)? {
            let entry = entry?;
            let name = entry.file_name().to_string_lossy().into_owned();
            if name.starts_with("part-") {
                std::fs::remove_file(entry.path())?;
            }
        }
    }

    // Second run: CumulativeStore::new should not panic; it should still see
    // a non-zero resume offset from metadata but no existing combinations.
    let (store2, resume2) = CumulativeStore::new(&cfg)?;
    assert!(
        resume2 > 0,
        "resume offset should remain non-zero even if Parquet batches were removed"
    );
    let existing = store2.existing_combinations()?;
    assert!(
        existing.is_empty(),
        "existing_combinations should be empty when result batches were removed"
    );

    Ok(())
}

// Coverage helpers that reasoned about "complete depths" via DuckDB have
// been removed from the engine. Depth and resume behaviour is now exercised
// via the pipeline_integration tests rather than CumulativeStore helpers.
