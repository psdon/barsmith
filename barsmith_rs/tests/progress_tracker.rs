use std::path::PathBuf;

use barsmith_rs::progress::ProgressTracker;
use barsmith_rs::{
    Config, Direction, LogicMode, ReportMetricsMode,
    config::{EvalProfileMode, PositionSizingMode, StackingMode, StatsDetail, StopDistanceUnit},
};

fn tracker_config(resume_offset: u64, max_combos: Option<usize>) -> Config {
    Config {
        input_csv: PathBuf::from("dummy.csv"),
        source_csv: None,
        direction: Direction::Long,
        target: "is_green".to_string(),
        output_dir: PathBuf::from("dummy_out"),
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
        resume_offset,
        explicit_resume_offset: false,
        max_combos,
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
fn tracker_without_limit_never_stops() {
    let config = tracker_config(0, None);
    let mut tracker = ProgressTracker::new(&config);

    // With no max_combos limit, record_batch should always return true.
    for _ in 0..10 {
        assert!(tracker.record_batch(100));
    }
}

#[test]
fn tracker_respects_max_combos_and_resume_offset() {
    // Start at a non-zero resume offset and enforce a max_combos ceiling.
    let config = tracker_config(100, Some(150));
    let mut tracker = ProgressTracker::new(&config);

    assert_eq!(tracker.start_offset(), 100);
    assert_eq!(
        tracker.processed_since_start(),
        0,
        "processed_since_start should ignore the resume offset"
    );

    // After 40 enumerated combinations, we should still be below the limit.
    assert!(tracker.record_batch(40));
    assert_eq!(tracker.processed(), 140);
    assert_eq!(tracker.processed_since_start(), 40);

    // Another 5 keeps us below the limit.
    assert!(tracker.record_batch(5));
    assert_eq!(tracker.processed(), 145);
    assert_eq!(tracker.processed_since_start(), 45);

    // A final batch of 10 pushes us past the limit; record_batch should
    // signal that the caller should stop.
    let keep_going = tracker.record_batch(10);
    assert!(
        !keep_going,
        "tracker should stop once max_combos is reached"
    );
    assert_eq!(tracker.processed(), 155);
}
