use std::cmp::Ordering;
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

use crate::features::{
    CONTINUOUS_FEATURES, PAIRWISE_BASE_NUMERIC_FEATURES, PAIRWISE_EXTRA_NUMERIC_FEATURES,
};
use anyhow::{Context, Result, anyhow};
use barsmith_rs::backtest::{BacktestInputs, BacktestOutputs, TradeDirection, run_backtest};
use barsmith_rs::{Config, Direction};
use chrono::{DateTime, Datelike, NaiveDate, NaiveTime, Utc};
use polars::prelude::*;
use sha2::{Digest, Sha256};
use tracing::{info, warn};

const SMALL_DIVISOR: f64 = 1e-9;
const BODY_SIZE_EPS: f64 = 1e-9;
const NON_ZERO_RANGE_EPS: f64 = f64::EPSILON;
const FLOAT_TOLERANCE: f64 = 1e-10;
const NEXT_BAR_SL_MULTIPLIER: f64 = 1.5;

#[allow(dead_code)]
#[derive(Clone, Copy, Debug)]
enum TickRoundMode {
    Nearest,
    Floor,
    Ceil,
}

#[derive(Clone, Copy, Debug)]
pub enum BacktestTargetKind {
    TribarWeekly,
    TribarMonthly,
}

pub struct BacktestConfig {
    pub csv_path: PathBuf,
    pub output_dir: PathBuf,
    pub direction: Direction,
    pub target_kind: BacktestTargetKind,
    pub features_expr: String,
    pub tp_multiple: f64,
    pub max_trades_per_period: u8,
}

pub struct PrepareDatasetOptions {
    pub drop_nan_rows_in_core: bool,
    pub ack_new_df: bool,
}

impl Default for PrepareDatasetOptions {
    fn default() -> Self {
        Self {
            drop_nan_rows_in_core: true,
            ack_new_df: false,
        }
    }
}

pub fn run_backtest_with_target(config: &BacktestConfig) -> Result<PathBuf> {
    fs::create_dir_all(&config.output_dir)
        .with_context(|| format!("Unable to create {}", config.output_dir.display()))?;
    let output_path = config.output_dir.join("barsmith_backtest.csv");

    let mut engineer = FeatureEngineer::from_csv(&config.csv_path)?;
    engineer.compute_features()?;

    let frame_len = engineer.frame.height();
    if frame_len == 0 {
        let mut file = File::create(&output_path)
            .with_context(|| format!("Unable to create {}", output_path.display()))?;
        CsvWriter::new(&mut file)
            .include_header(true)
            .finish(engineer.data_frame_mut())
            .with_context(|| "Failed to persist empty backtest dataset")?;
        return Ok(output_path);
    }

    let high = column_with_nans(&engineer.frame, "high")?;
    let low = column_with_nans(&engineer.frame, "low")?;
    let close = column_with_nans(&engineer.frame, "close")?;
    let timestamps = timestamp_column(&engineer.frame)?;

    let len = close.len();
    if high.len() != len || low.len() != len || timestamps.len() != len {
        return Err(anyhow!(
            "Inconsistent series lengths when preparing backtest inputs"
        ));
    }

    // Compute 14-period ATR once; TP multiple is applied on top.
    let atr_values = atr(&high, &low, &close, 14);

    // Build entry mask from the feature expression.
    let base_entry = build_entry_mask(&engineer.frame, &config.features_expr)?;
    if base_entry.len() != len {
        return Err(anyhow!(
            "Entry mask length {} does not match price series length {}",
            base_entry.len(),
            len
        ));
    }

    // Optional trend filter: for longs, close should be above 200sma;
    // for shorts, close should be below 200sma. If the columns are
    // missing, we fall back to not filtering.
    let above_200 = match bool_column(&engineer.frame, "is_close_above_200sma") {
        Ok(mask) if mask.len() == len => mask,
        _ => vec![true; len],
    };
    let below_200 = match bool_column(&engineer.frame, "is_close_below_200sma") {
        Ok(mask) if mask.len() == len => mask,
        _ => above_200.iter().map(|flag| !*flag).collect(),
    };

    let (mut entry_long, mut entry_short) = match config.direction {
        Direction::Long => (
            base_entry
                .iter()
                .zip(above_200.iter())
                .map(|(b, a)| *b && *a)
                .collect(),
            vec![false; len],
        ),
        Direction::Short => (
            vec![false; len],
            base_entry
                .iter()
                .zip(below_200.iter())
                .map(|(b, d)| *b && *d)
                .collect(),
        ),
        Direction::Both => (
            base_entry
                .iter()
                .zip(above_200.iter())
                .map(|(b, a)| *b && *a)
                .collect(),
            base_entry
                .iter()
                .zip(below_200.iter())
                .map(|(b, d)| *b && *d)
                .collect(),
        ),
    };

    // Period indices and per-period caps from the target kind.
    let (period_index, period_end_index, default_tp_multiple, default_max_trades) =
        period_indices_for_target(config.target_kind, &timestamps);

    if period_index.len() != len || period_end_index.len() != len {
        return Err(anyhow!(
            "Period index lengths do not match price series length when preparing backtest inputs"
        ));
    }

    let tp_multiple = if config.tp_multiple > 0.0 {
        config.tp_multiple
    } else {
        default_tp_multiple
    };
    let max_trades = if config.max_trades_per_period > 0 {
        config.max_trades_per_period
    } else {
        default_max_trades
    };

    let (stop_long, tp_long) = build_long_levels(&close, &low, &atr_values, tp_multiple);
    let (stop_short, tp_short) = build_short_levels(&close, &high, &atr_values, tp_multiple);

    // Disable entries and caps per direction when not requested.
    let (max_trades_long, max_trades_short) = match config.direction {
        Direction::Long => (max_trades, 0),
        Direction::Short => (0, max_trades),
        Direction::Both => (max_trades, max_trades),
    };
    if max_trades_long == 0 {
        entry_long.fill(false);
    }
    if max_trades_short == 0 {
        entry_short.fill(false);
    }

    let inputs = BacktestInputs {
        high,
        low,
        close: close.clone(),
        entry_long,
        entry_short,
        stop_long,
        tp_long,
        stop_short,
        tp_short,
        period_index,
        period_end_index,
        max_trades_per_period_long: max_trades_long,
        max_trades_per_period_short: max_trades_short,
        stop_after_first_winning_trade_long: true,
        stop_after_first_winning_trade_short: true,
    };

    let outputs: BacktestOutputs = run_backtest(&inputs);

    // Attach per-bar targets and RR back onto the engineered frame.

    let prefix = match config.target_kind {
        BacktestTargetKind::TribarWeekly => "tribar_weekly",
        BacktestTargetKind::TribarMonthly => "tribar_monthly",
    };

    if max_trades_long > 0 {
        let target_name = format!("backtest_{}_long", prefix);
        let rr_name = format!("rr_backtest_{}_long", prefix);
        engineer.replace_bool_column(&target_name, outputs.target_long.clone())?;
        engineer.replace_float_column(&rr_name, outputs.rr_long.clone())?;
    }

    if max_trades_short > 0 {
        let target_name = format!("backtest_{}_short", prefix);
        let rr_name = format!("rr_backtest_{}_short", prefix);
        engineer.replace_bool_column(&target_name, outputs.target_short.clone())?;
        engineer.replace_float_column(&rr_name, outputs.rr_short.clone())?;
    }

    // Summarise and print trade-level information so CLI users can
    // inspect the most recent executions without opening the CSV.
    if !outputs.trades.is_empty() {
        let mut total_trades = 0usize;
        let mut wins = 0usize;
        let mut losses = 0usize;
        let mut rr_sum = 0.0_f64;
        let mut rr_win_sum = 0.0_f64;
        let mut rr_loss_sum = 0.0_f64;
        let mut long_trades = 0usize;
        let mut short_trades = 0usize;
        let mut best_rr = f64::NEG_INFINITY;
        let mut worst_rr = f64::INFINITY;

        for trade in &outputs.trades {
            if !trade.rr.is_finite() {
                continue;
            }
            total_trades += 1;
            rr_sum += trade.rr;
            if trade.rr > 0.0 {
                wins += 1;
                rr_win_sum += trade.rr;
            } else {
                losses += 1;
                rr_loss_sum += trade.rr;
            }
            match trade.direction {
                TradeDirection::Long => long_trades += 1,
                TradeDirection::Short => short_trades += 1,
            }
            if trade.rr > best_rr {
                best_rr = trade.rr;
            }
            if trade.rr < worst_rr {
                worst_rr = trade.rr;
            }
        }

        if total_trades > 0 {
            let win_rate = wins as f64 / total_trades as f64 * 100.0;
            let avg_rr = rr_sum / total_trades as f64;
            let avg_rr_win = if wins > 0 {
                rr_win_sum / wins as f64
            } else {
                f64::NAN
            };
            let avg_rr_loss = if losses > 0 {
                rr_loss_sum / losses as f64
            } else {
                f64::NAN
            };

            let (period_start_label, period_end_label) = match config.target_kind {
                BacktestTargetKind::TribarWeekly => ("Week start", "Week end"),
                BacktestTargetKind::TribarMonthly => ("Month start", "Month end"),
            };

            println!();
            println!(
                "Backtest summary for {} ({:?}, {} trades):",
                match config.target_kind {
                    BacktestTargetKind::TribarWeekly => "tribar-weekly",
                    BacktestTargetKind::TribarMonthly => "tribar-monthly",
                },
                config.direction,
                total_trades,
            );
            println!(
                "  Long trades: {}, Short trades: {}",
                long_trades, short_trades
            );
            println!(
                "  Wins: {}  Losses: {}  Win rate: {:.2}%",
                wins, losses, win_rate
            );
            println!(
                "  Avg RR: {:.2}  Avg RR (wins): {:.2}  Avg RR (losses): {:.2}",
                avg_rr, avg_rr_win, avg_rr_loss,
            );
            if best_rr.is_finite() && worst_rr.is_finite() {
                println!("  Best RR: {:.2}  Worst RR: {:.2}", best_rr, worst_rr);
            }

            // Print the last 20 trades in chronological order.
            let count_to_show = outputs.trades.len().min(20);
            println!();
            println!("Last {} trades:", count_to_show);
            println!(
                "  {:>4}  {:>5}  {:>25}  {:>25}  {:>25}  {:>25}  {:>10}  {:>10}  {:>10}  {:>10}  {:>6}",
                "Idx",
                "Side",
                "Entry time",
                "Exit time",
                period_start_label,
                period_end_label,
                "Entry",
                "Exit",
                "Stop",
                "Target",
                "RR",
            );

            let start = outputs.trades.len() - count_to_show;
            for (idx, trade) in outputs.trades[start..].iter().enumerate() {
                let entry_ts = timestamps
                    .get(trade.entry_index)
                    .map(|ts| ts.to_rfc3339())
                    .unwrap_or_else(|| "-".to_string());
                let exit_ts = timestamps
                    .get(trade.exit_index)
                    .map(|ts| ts.to_rfc3339())
                    .unwrap_or_else(|| "-".to_string());

                // Derive period start/end from the period index and end-index
                // masks used by the backtest engine.
                let period_id = trade.period;
                let mut period_start_idx = trade.entry_index;
                while period_start_idx > 0 && inputs.period_index[period_start_idx - 1] == period_id
                {
                    period_start_idx -= 1;
                }
                let period_end_idx = inputs.period_end_index[trade.entry_index];
                let period_start_ts = timestamps
                    .get(period_start_idx)
                    .map(|ts| ts.to_rfc3339())
                    .unwrap_or_else(|| "-".to_string());
                let period_end_ts = timestamps
                    .get(period_end_idx)
                    .map(|ts| ts.to_rfc3339())
                    .unwrap_or_else(|| "-".to_string());

                let side = match trade.direction {
                    TradeDirection::Long => "LONG",
                    TradeDirection::Short => "SHORT",
                };
                println!(
                    "  {:>4}  {:>5}  {:>25}  {:>25}  {:>25}  {:>25}  {:>10.2}  {:>10.2}  {:>10.2}  {:>10.2}  {:>6.2}",
                    start + idx + 1,
                    side,
                    entry_ts,
                    exit_ts,
                    period_start_ts,
                    period_end_ts,
                    trade.entry_price,
                    trade.exit_price,
                    trade.stop_price,
                    trade.tp_price,
                    trade.rr,
                );
            }
            println!();
        }
    }

    let mut file = File::create(&output_path)
        .with_context(|| format!("Unable to create {}", output_path.display()))?;
    CsvWriter::new(&mut file)
        .include_header(true)
        .finish(engineer.data_frame_mut())
        .with_context(|| "Failed to persist backtest dataset")?;
    Ok(output_path)
}

pub fn prepare_dataset(config: &Config) -> Result<PathBuf> {
    prepare_dataset_with_options(config, PrepareDatasetOptions::default())
}

pub fn prepare_dataset_with_options(
    config: &Config,
    options: PrepareDatasetOptions,
) -> Result<PathBuf> {
    fs::create_dir_all(&config.output_dir)
        .with_context(|| format!("Unable to create {}", config.output_dir.display()))?;
    let output_path = config.output_dir.join("barsmith_prepared.csv");
    let mut engineer = FeatureEngineer::from_csv(&config.input_csv)?;
    engineer.compute_features_with_options(options.drop_nan_rows_in_core)?;
    engineer.attach_targets(config)?;

    if output_path.exists() {
        let old_hash = sha256_file(&output_path)?;
        let new_hash = sha256_dataframe_as_csv(engineer.data_frame_mut())?;
        if old_hash != new_hash {
            if !options.ack_new_df {
                return Err(anyhow!(
                    "Existing barsmith_prepared.csv differs from newly engineered dataframe.\n\
                     path: {}\n\
                     existing sha256: {}\n\
                     new sha256: {}\n\
                     Rerun with --ack-new-df to overwrite and continue, or choose a fresh --output-dir to preserve prior results.",
                    output_path.display(),
                    old_hash,
                    new_hash
                ));
            }
            warn!(
                existing_hash = %old_hash,
                new_hash = %new_hash,
                path = %output_path.display(),
                "Prepared dataset hash mismatch; overwriting because ack_new_df=true"
            );
        } else {
            let row_count = engineer.data_frame_mut().height();
            info!(
                rows = row_count,
                path = %output_path.display(),
                "Prepared dataset unchanged; reusing existing barsmith_prepared.csv"
            );
            return Ok(output_path);
        }
    }

    let row_count = engineer.data_frame_mut().height();
    let mut file = File::create(&output_path)
        .with_context(|| format!("Unable to create {}", output_path.display()))?;
    CsvWriter::new(&mut file)
        .include_header(true)
        .finish(engineer.data_frame_mut())
        .with_context(|| "Failed to persist engineered dataset")?;
    info!(
        rows = row_count,
        path = %output_path.display(),
        "Prepared engineered dataset written for combination run"
    );
    Ok(output_path)
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut file = File::open(path)
        .with_context(|| format!("Unable to open {} for hashing", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 8192];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hex::encode(hasher.finalize()))
}

fn sha256_dataframe_as_csv(df: &mut DataFrame) -> Result<String> {
    let sink = io::sink();
    let mut writer = HashingWriter::new(sink);
    CsvWriter::new(&mut writer)
        .include_header(true)
        .finish(df)
        .with_context(|| "Failed to hash engineered dataset")?;
    Ok(writer.finalize_hex())
}

struct HashingWriter<W: Write> {
    inner: W,
    hasher: Sha256,
}

impl<W: Write> HashingWriter<W> {
    fn new(inner: W) -> Self {
        Self {
            inner,
            hasher: Sha256::new(),
        }
    }

    fn finalize_hex(self) -> String {
        hex::encode(self.hasher.finalize())
    }
}

impl<W: Write> Write for HashingWriter<W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.hasher.update(buf);
        self.inner.write(buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}

struct FeatureEngineer {
    frame: DataFrame,
}

impl FeatureEngineer {
    fn from_csv(path: &Path) -> Result<Self> {
        let mut df = CsvReader::from_path(path)
            .with_context(|| format!("Failed to load {}", path.display()))?
            .infer_schema(Some(1024))
            .has_header(true)
            .finish()
            .with_context(|| "Unable to read CSV into DataFrame")?;
        // Drop legacy indicator columns that duplicate metrics we recompute in
        // Rust (using canonical snake_case names). This keeps the engineered
        // dataframe lean and avoids confusion between legacy labels like
        // \"EMA 9\" / \"SMA 200\" and the Rust-native 9ema/200sma series.
        //
        // These columns are optional – if they are not present in the input
        // CSV, we simply leave the frame unchanged.
        const LEGACY_INDICATOR_COLUMNS: &[&str] = &["EMA 9", "SMA 200", "ADX", "ATR"];
        for name in LEGACY_INDICATOR_COLUMNS {
            if df.column(name).is_ok() {
                df = df
                    .drop(name)
                    .with_context(|| format!("Failed to drop legacy column {name}"))?;
            }
        }
        Ok(Self { frame: df })
    }

    fn data_frame_mut(&mut self) -> &mut DataFrame {
        &mut self.frame
    }

    fn compute_features(&mut self) -> Result<()> {
        self.compute_features_with_options(true)
    }

    fn compute_features_with_options(&mut self, drop_nan_rows_in_core: bool) -> Result<()> {
        let prices = PriceSeries::from_frame(&self.frame)?;
        let derived = DerivedMetrics::new(&prices);
        let mut bool_features = HashMap::<&'static str, Vec<bool>>::new();
        let mut float_features = HashMap::<&'static str, Vec<f64>>::new();

        candle_features(&prices, &derived, &mut bool_features, &mut float_features);
        ema_price_features(&prices, &derived, &mut bool_features, &mut float_features);
        volatility_features(&prices, &derived, &mut bool_features, &mut float_features);
        oscillator_features(&prices, &derived, &mut bool_features, &mut float_features);
        macd_features(&derived, &mut bool_features, &mut float_features);
        bollinger_features(&prices, &derived, &mut bool_features, &mut float_features);
        trend_state_features(&derived, &mut bool_features);
        kalman_features(&prices, &derived, &mut bool_features);

        // Indicator constructors already emit NaNs for their natural warmup
        // regions (e.g., 200-period averages, MACD, RSI). We no longer apply
        // additional ad-hoc warmup masking here; instead we rely on a single
        // pass later that drops any rows where the core numeric feature set
        // still contains NaNs.
        apply_indicator_warmups(&mut bool_features, &mut float_features);

        let mut float_names: Vec<&'static str> = float_features.keys().copied().collect();
        float_names.sort_unstable();
        for name in float_names {
            let values = float_features
                .remove(name)
                .expect("float feature value should exist");
            let series = Series::new(name, values);
            self.frame
                .with_column(series)
                .with_context(|| format!("Failed to insert column {name}"))?;
        }

        let mut bool_names: Vec<&'static str> = bool_features.keys().copied().collect();
        bool_names.sort_unstable();
        for name in bool_names {
            let values = bool_features
                .remove(name)
                .expect("boolean feature value should exist");
            let series = Series::new(name, values);
            self.frame
                .with_column(series)
                .with_context(|| format!("Failed to insert boolean column {name}"))?;
        }

        // Compute any dependent features (streaks, alignment flags, momentum
        // scores) once, on the full engineered frame. These will be filtered
        // alongside the core numerics in the subsequent NaN-drop step.
        self.recompute_consecutive_columns()?;
        self.recompute_kf_alignment()?;
        self.recompute_high_low()?;
        self.recompute_momentum_scores()?;

        // Optionally drop any rows where the core numeric feature set still
        // contains NaNs. This leaves a clean dataset where all continuous
        // indicators are fully defined, and all boolean/derived features are
        // aligned to that trimmed history.
        if drop_nan_rows_in_core {
            self.drop_rows_with_nan_in_core()?;
        }

        Ok(())
    }

    fn drop_rows_with_nan_in_core(&mut self) -> Result<()> {
        let height = self.frame.height();
        if height == 0 {
            return Ok(());
        }

        // Core numeric columns that define the engineered feature space.
        let mut core_cols: Vec<&str> = Vec::new();
        core_cols.extend_from_slice(CONTINUOUS_FEATURES);
        core_cols.extend_from_slice(PAIRWISE_BASE_NUMERIC_FEATURES);
        core_cols.extend_from_slice(PAIRWISE_EXTRA_NUMERIC_FEATURES);
        core_cols.sort();
        core_cols.dedup();

        let mut mask_opt: Option<BooleanChunked> = None;
        let mut skipped_all_nan: Vec<&str> = Vec::new();

        for name in core_cols {
            let series = match self.frame.column(name) {
                Ok(s) => s,
                // Some configured features may not be present in a given
                // engineered dataset; skip them when building the mask.
                Err(_) => continue,
            };
            let col = match series.f64() {
                Ok(c) => c,
                // We only consider float-like series here; boolean columns
                // are handled separately in the boolean catalog.
                Err(_) => continue,
            };
            let col_mask = col.is_not_nan();
            // Skip core columns that are entirely NaN on this slice (typically
            // long-warmup indicators like 200sma on short histories). We warn
            // so this never hides data-quality regressions.
            if col_mask.sum().unwrap_or(0) == 0 {
                skipped_all_nan.push(name);
                continue;
            }
            mask_opt = Some(match mask_opt {
                None => col_mask,
                Some(prev) => prev & col_mask,
            });
        }

        if !skipped_all_nan.is_empty() {
            warn!(
                skipped = ?skipped_all_nan,
                "Skipping core numeric columns that are all NaN during NaN-drop"
            );
        }

        if let Some(mask) = mask_opt {
            if mask.sum().unwrap_or(0) == 0 {
                return Err(anyhow!(
                    "Dropping rows with NaNs in core indicator set would remove all rows. \
                    Dataset may be too short for overlapping warmups; \
                    for tests or diagnostics, call prepare_dataset_with_options with drop_nan_rows_in_core=false."
                ));
            }
            self.frame = self
                .frame
                .filter(&mask)
                .with_context(|| "Failed to drop rows with NaNs in core indicator set")?;
        }

        Ok(())
    }

    fn recompute_consecutive_columns(&mut self) -> Result<()> {
        fn series_to_bool_vec(series: &Series) -> Result<Vec<bool>> {
            Ok(series
                .bool()
                .context("Expected boolean series")?
                .into_iter()
                .map(|value| value.unwrap_or(false))
                .collect())
        }

        let is_green = series_to_bool_vec(self.frame.column("is_green")?)?;
        let is_red = if let Ok(series) = self.frame.column("is_red") {
            series_to_bool_vec(series)?
        } else {
            vec![false; self.frame.height()]
        };
        let is_tribar = if let Ok(series) = self.frame.column("is_tribar") {
            series_to_bool_vec(series)?
        } else {
            vec![false; self.frame.height()]
        };
        let prev_green = shift_bool(&is_green, 1);
        let prev_tribar = shift_bool(&is_tribar, 1);

        let updates = [
            ("consecutive_green_2", streak(&is_green, 2)),
            ("consecutive_green_3", streak(&is_green, 3)),
            ("consecutive_red_2", streak(&is_red, 2)),
            ("consecutive_red_3", streak(&is_red, 3)),
            ("prev_green", prev_green),
            ("prev_tribar", prev_tribar),
        ];

        for (name, values) in updates {
            if self.frame.column(name).is_ok() {
                self.frame = self
                    .frame
                    .drop(name)
                    .with_context(|| format!("Failed to remove existing column {name}"))?;
            }
            let series = Series::new(name, values);
            self.frame
                .with_column(series)
                .with_context(|| format!("Failed to update column {name}"))?;
        }

        let consecutive_green_counts = rolling_bool_sum(&is_green, 3);
        if self.frame.column("consecutive_green").is_ok() {
            self.frame = self
                .frame
                .drop("consecutive_green")
                .with_context(|| "Failed to remove existing column consecutive_green")?;
        }
        let count_series = Series::new("consecutive_green", consecutive_green_counts);
        self.frame
            .with_column(count_series)
            .with_context(|| "Failed to update column consecutive_green")?;
        Ok(())
    }

    fn attach_targets(&mut self, config: &Config) -> Result<()> {
        match config.target.as_str() {
            "next_bar_color_and_wicks" => self.attach_next_bar_targets(config),
            "wicks_kf" => self.attach_wicks_kf_targets(config),
            "highlow_or_atr" => self.attach_highlow_or_atr_targets(config),
            "highlow_1r" => self.attach_highlow_1r_targets(config),
            "2x_atr_tp_atr_stop" => self.attach_2x_atr_tp_atr_stop_targets(config),
            "3x_atr_tp_atr_stop" => self.attach_3x_atr_tp_atr_stop_targets(config),
            "atr_stop" => self.attach_2x_atr_tp_atr_stop_targets(config),
            "atr_tp_atr_stop" => self.attach_atr_tp_atr_stop_targets(config),
            "highlow_sl_2x_atr_tp_rr_gt_1" => {
                self.attach_highlow_sl_2x_atr_tp_rr_gt_1_targets(config)
            }
            "highlow_sl_1x_atr_tp_rr_gt_1" => {
                self.attach_highlow_sl_1x_atr_tp_rr_gt_1_targets(config)
            }
            "highlow_or_atr_tightest_stop" | "highlow_or_atr_tighest_stop" => {
                self.attach_highlow_or_atr_tightest_stop_targets(config)
            }
            "tribar_4h_2atr" => self.attach_tribar_4h_targets(config),
            _ => Ok(()),
        }
    }

    fn attach_next_bar_targets(&mut self, config: &Config) -> Result<()> {
        const TARGET_NAME: &str = "next_bar_color_and_wicks";

        let open = column_with_nans(&self.frame, "open")?;
        let high = column_with_nans(&self.frame, "high")?;
        let low = column_with_nans(&self.frame, "low")?;
        let close = column_with_nans(&self.frame, "close")?;
        let wicks = column_with_nans(&self.frame, "wicks_diff_sma14")?;

        let tick_size = config.tick_size;

        let (long, short, rr_long, rr_short, exit_i_long, exit_i_short) =
            compute_next_bar_targets_and_rr(
                &open,
                &high,
                &low,
                &close,
                &wicks,
                NEXT_BAR_SL_MULTIPLIER,
                tick_size,
                config.direction,
            );

        self.replace_bool_column("next_bar_color_and_wicks_long", long)?;
        self.replace_bool_column("next_bar_color_and_wicks_short", short)?;
        self.replace_float_column("rr_long", rr_long)?;
        self.replace_float_column("rr_short", rr_short)?;
        let exit_i_long_i64: Vec<Option<i64>> = exit_i_long
            .into_iter()
            .map(|v| v.map(|idx| idx as i64))
            .collect();
        let exit_i_short_i64: Vec<Option<i64>> = exit_i_short
            .into_iter()
            .map(|v| v.map(|idx| idx as i64))
            .collect();
        self.replace_i64_column("next_bar_color_and_wicks_exit_i_long", exit_i_long_i64)?;
        self.replace_i64_column("next_bar_color_and_wicks_exit_i_short", exit_i_short_i64)?;

        let (target_source, rr_source, exit_source) = match config.direction {
            Direction::Short => (
                "next_bar_color_and_wicks_short",
                "rr_short",
                "next_bar_color_and_wicks_exit_i_short",
            ),
            _ => (
                "next_bar_color_and_wicks_long",
                "rr_long",
                "next_bar_color_and_wicks_exit_i_long",
            ),
        };

        let mut target_series = self.frame.column(target_source)?.clone();
        target_series.rename(TARGET_NAME);
        self.replace_series(target_series)?;

        let mut rr_series = self.frame.column(rr_source)?.clone();
        rr_series.rename("rr_next_bar_color_and_wicks");
        self.replace_series(rr_series)?;

        let mut exit_series = self.frame.column(exit_source)?.clone();
        exit_series.rename("next_bar_color_and_wicks_exit_i");
        self.replace_series(exit_series)?;
        Ok(())
    }

    fn attach_wicks_kf_targets(&mut self, config: &Config) -> Result<()> {
        const TARGET_NAME: &str = "wicks_kf";

        let open = column_with_nans(&self.frame, "open")?;
        let high = column_with_nans(&self.frame, "high")?;
        let low = column_with_nans(&self.frame, "low")?;
        let close = column_with_nans(&self.frame, "close")?;
        let kf_wicks = column_with_nans(&self.frame, "kf_wicks_smooth")?;

        let tick_size = config.tick_size;

        let (long, short, rr_long, rr_short, exit_i_long, exit_i_short) =
            compute_wicks_kf_targets_and_rr(
                &open,
                &high,
                &low,
                &close,
                &kf_wicks,
                NEXT_BAR_SL_MULTIPLIER,
                tick_size,
                config.direction,
            );

        self.replace_bool_column("wicks_kf_long", long)?;
        self.replace_bool_column("wicks_kf_short", short)?;
        self.replace_float_column("rr_long", rr_long)?;
        self.replace_float_column("rr_short", rr_short)?;
        let exit_i_long_i64: Vec<Option<i64>> = exit_i_long
            .into_iter()
            .map(|v| v.map(|idx| idx as i64))
            .collect();
        let exit_i_short_i64: Vec<Option<i64>> = exit_i_short
            .into_iter()
            .map(|v| v.map(|idx| idx as i64))
            .collect();
        self.replace_i64_column("wicks_kf_exit_i_long", exit_i_long_i64)?;
        self.replace_i64_column("wicks_kf_exit_i_short", exit_i_short_i64)?;

        let (target_source, rr_source, exit_source) = match config.direction {
            Direction::Short => ("wicks_kf_short", "rr_short", "wicks_kf_exit_i_short"),
            _ => ("wicks_kf_long", "rr_long", "wicks_kf_exit_i_long"),
        };

        let mut target_series = self.frame.column(target_source)?.clone();
        target_series.rename(TARGET_NAME);
        self.replace_series(target_series)?;

        let mut rr_series = self.frame.column(rr_source)?.clone();
        rr_series.rename("rr_wicks_kf");
        self.replace_series(rr_series)?;

        let mut exit_series = self.frame.column(exit_source)?.clone();
        exit_series.rename("wicks_kf_exit_i");
        self.replace_series(exit_series)?;
        Ok(())
    }

    fn attach_highlow_or_atr_targets(&mut self, config: &Config) -> Result<()> {
        const TARGET_NAME: &str = "highlow_or_atr";

        let open = column_with_nans(&self.frame, "open")?;
        let high = column_with_nans(&self.frame, "high")?;
        let low = column_with_nans(&self.frame, "low")?;
        let close = column_with_nans(&self.frame, "close")?;
        let atr_values = column_with_nans(&self.frame, "atr").with_context(|| {
            "Missing required 'atr' column for highlow_or_atr target. Re-generate the engineered dataset \
            (e.g., rerun Barsmith with --ack-new-df or choose a fresh --output-dir) so 'atr' is present."
        })?;

        // Prevent conceptual leakage across `--date-end` for this multi-bar target:
        // when a date cutoff is provided, cap TP/SL resolution for entry bars
        // up to the last in-sample bar (and force-exit remaining open trades
        // at that bar's close). Post-cutoff bars still resolve normally so
        // the prepared CSV contains RR values for the full dataset.
        let resolve_end_idx = if let Some(date_end) = config.include_date_end {
            let timestamps = timestamp_column(&self.frame)?;
            timestamps
                .iter()
                .rposition(|ts| ts.date_naive() <= date_end)
        } else {
            None
        };

        let (long, short, rr_long, rr_short, exit_i_long, exit_i_short) =
            compute_highlow_or_atr_targets_and_rr(
                &open,
                &high,
                &low,
                &close,
                &atr_values,
                config.tick_size,
                resolve_end_idx,
                config.direction,
            );

        self.replace_bool_column("highlow_or_atr_long", long)?;
        self.replace_bool_column("highlow_or_atr_short", short)?;
        self.replace_float_column("rr_long", rr_long)?;
        self.replace_float_column("rr_short", rr_short)?;
        let exit_i_long_i64: Vec<Option<i64>> = exit_i_long
            .into_iter()
            .map(|v| v.map(|idx| idx as i64))
            .collect();
        let exit_i_short_i64: Vec<Option<i64>> = exit_i_short
            .into_iter()
            .map(|v| v.map(|idx| idx as i64))
            .collect();
        self.replace_i64_column("highlow_or_atr_exit_i_long", exit_i_long_i64)?;
        self.replace_i64_column("highlow_or_atr_exit_i_short", exit_i_short_i64)?;

        let eligible_long: Vec<bool> = open
            .iter()
            .zip(close.iter())
            .map(|(o, c)| o.is_finite() && c.is_finite() && c > o)
            .collect();
        let eligible_short: Vec<bool> = open
            .iter()
            .zip(close.iter())
            .map(|(o, c)| o.is_finite() && c.is_finite() && c < o)
            .collect();
        self.replace_bool_column("highlow_or_atr_eligible_long", eligible_long)?;
        self.replace_bool_column("highlow_or_atr_eligible_short", eligible_short)?;

        let (target_source, rr_source, eligible_source, exit_source) = match config.direction {
            Direction::Short => (
                "highlow_or_atr_short",
                "rr_short",
                "highlow_or_atr_eligible_short",
                "highlow_or_atr_exit_i_short",
            ),
            _ => (
                "highlow_or_atr_long",
                "rr_long",
                "highlow_or_atr_eligible_long",
                "highlow_or_atr_exit_i_long",
            ),
        };

        let mut target_series = self.frame.column(target_source)?.clone();
        target_series.rename(TARGET_NAME);
        self.replace_series(target_series)?;

        let mut rr_series = self.frame.column(rr_source)?.clone();
        rr_series.rename("rr_highlow_or_atr");
        self.replace_series(rr_series)?;

        let mut eligible_series = self.frame.column(eligible_source)?.clone();
        eligible_series.rename("highlow_or_atr_eligible");
        self.replace_series(eligible_series)?;

        let mut exit_series = self.frame.column(exit_source)?.clone();
        exit_series.rename("highlow_or_atr_exit_i");
        self.replace_series(exit_series)?;
        Ok(())
    }

    fn attach_highlow_1r_targets(&mut self, config: &Config) -> Result<()> {
        const TARGET_NAME: &str = "highlow_1r";

        let open = column_with_nans(&self.frame, "open")?;
        let high = column_with_nans(&self.frame, "high")?;
        let low = column_with_nans(&self.frame, "low")?;
        let close = column_with_nans(&self.frame, "close")?;

        // Prevent conceptual leakage across `--date-end` for this multi-bar target:
        // when a date cutoff is provided, cap TP/SL resolution for entry bars
        // up to the last in-sample bar (and force-exit remaining open trades
        // at that bar's close). Post-cutoff bars still resolve normally so
        // the prepared CSV contains RR values for the full dataset.
        let resolve_end_idx = if let Some(date_end) = config.include_date_end {
            let timestamps = timestamp_column(&self.frame)?;
            timestamps
                .iter()
                .rposition(|ts| ts.date_naive() <= date_end)
        } else {
            None
        };

        let (long, short, rr_long, rr_short, exit_i_long, exit_i_short) =
            compute_highlow_1r_targets_and_rr(
                &open,
                &high,
                &low,
                &close,
                config.tick_size,
                resolve_end_idx,
                config.direction,
            );

        self.replace_bool_column("highlow_1r_long", long)?;
        self.replace_bool_column("highlow_1r_short", short)?;
        self.replace_float_column("rr_long", rr_long)?;
        self.replace_float_column("rr_short", rr_short)?;
        let exit_i_long_i64: Vec<Option<i64>> = exit_i_long
            .into_iter()
            .map(|v| v.map(|idx| idx as i64))
            .collect();
        let exit_i_short_i64: Vec<Option<i64>> = exit_i_short
            .into_iter()
            .map(|v| v.map(|idx| idx as i64))
            .collect();
        self.replace_i64_column("highlow_1r_exit_i_long", exit_i_long_i64)?;
        self.replace_i64_column("highlow_1r_exit_i_short", exit_i_short_i64)?;

        let eligible_long: Vec<bool> = open
            .iter()
            .zip(close.iter())
            .map(|(o, c)| o.is_finite() && c.is_finite() && c > o)
            .collect();
        let eligible_short: Vec<bool> = open
            .iter()
            .zip(close.iter())
            .map(|(o, c)| o.is_finite() && c.is_finite() && c < o)
            .collect();
        self.replace_bool_column("highlow_1r_eligible_long", eligible_long)?;
        self.replace_bool_column("highlow_1r_eligible_short", eligible_short)?;

        let (target_source, rr_source, eligible_source, exit_source) = match config.direction {
            Direction::Short => (
                "highlow_1r_short",
                "rr_short",
                "highlow_1r_eligible_short",
                "highlow_1r_exit_i_short",
            ),
            _ => (
                "highlow_1r_long",
                "rr_long",
                "highlow_1r_eligible_long",
                "highlow_1r_exit_i_long",
            ),
        };

        let mut target_series = self.frame.column(target_source)?.clone();
        target_series.rename(TARGET_NAME);
        self.replace_series(target_series)?;

        let mut rr_series = self.frame.column(rr_source)?.clone();
        rr_series.rename("rr_highlow_1r");
        self.replace_series(rr_series)?;

        let mut eligible_series = self.frame.column(eligible_source)?.clone();
        eligible_series.rename("highlow_1r_eligible");
        self.replace_series(eligible_series)?;

        let mut exit_series = self.frame.column(exit_source)?.clone();
        exit_series.rename("highlow_1r_exit_i");
        self.replace_series(exit_series)?;
        Ok(())
    }

    fn attach_2x_atr_tp_atr_stop_targets(&mut self, config: &Config) -> Result<()> {
        const TARGET_NAME: &str = "2x_atr_tp_atr_stop";

        let open = column_with_nans(&self.frame, "open")?;
        let high = column_with_nans(&self.frame, "high")?;
        let low = column_with_nans(&self.frame, "low")?;
        let close = column_with_nans(&self.frame, "close")?;
        let atr_values = column_with_nans(&self.frame, "atr").with_context(|| {
	            "Missing required 'atr' column for 2x_atr_tp_atr_stop target. Re-generate the engineered dataset \
	            (e.g., rerun Barsmith with --ack-new-df or choose a fresh --output-dir) so 'atr' is present."
	        })?;

        // Prevent conceptual leakage across `--date-end` for this multi-bar target:
        // when a date cutoff is provided, cap TP/SL resolution for entry bars
        // up to the last in-sample bar (and force-exit remaining open trades
        // at that bar's close). Post-cutoff bars still resolve normally so
        // the prepared CSV contains RR values for the full dataset.
        let resolve_end_idx = if let Some(date_end) = config.include_date_end {
            let timestamps = timestamp_column(&self.frame)?;
            timestamps
                .iter()
                .rposition(|ts| ts.date_naive() <= date_end)
        } else {
            None
        };

        let (long, short, rr_long, rr_short, exit_i_long, exit_i_short) =
            compute_2x_atr_tp_atr_stop_targets_and_rr(
                &open,
                &high,
                &low,
                &close,
                &atr_values,
                config.tick_size,
                resolve_end_idx,
                config.direction,
            );

        self.replace_bool_column("2x_atr_tp_atr_stop_long", long)?;
        self.replace_bool_column("2x_atr_tp_atr_stop_short", short)?;
        self.replace_float_column("rr_long", rr_long)?;
        self.replace_float_column("rr_short", rr_short)?;
        let exit_i_long_i64: Vec<Option<i64>> = exit_i_long
            .into_iter()
            .map(|v| v.map(|idx| idx as i64))
            .collect();
        let exit_i_short_i64: Vec<Option<i64>> = exit_i_short
            .into_iter()
            .map(|v| v.map(|idx| idx as i64))
            .collect();
        self.replace_i64_column("2x_atr_tp_atr_stop_exit_i_long", exit_i_long_i64)?;
        self.replace_i64_column("2x_atr_tp_atr_stop_exit_i_short", exit_i_short_i64)?;

        let eligible_long: Vec<bool> = open
            .iter()
            .zip(close.iter())
            .map(|(o, c)| o.is_finite() && c.is_finite() && c > o)
            .collect();
        let eligible_short: Vec<bool> = open
            .iter()
            .zip(close.iter())
            .map(|(o, c)| o.is_finite() && c.is_finite() && c < o)
            .collect();
        self.replace_bool_column("2x_atr_tp_atr_stop_eligible_long", eligible_long)?;
        self.replace_bool_column("2x_atr_tp_atr_stop_eligible_short", eligible_short)?;

        let (target_source, rr_source, eligible_source, exit_source) = match config.direction {
            Direction::Short => (
                "2x_atr_tp_atr_stop_short",
                "rr_short",
                "2x_atr_tp_atr_stop_eligible_short",
                "2x_atr_tp_atr_stop_exit_i_short",
            ),
            _ => (
                "2x_atr_tp_atr_stop_long",
                "rr_long",
                "2x_atr_tp_atr_stop_eligible_long",
                "2x_atr_tp_atr_stop_exit_i_long",
            ),
        };

        let mut target_series = self.frame.column(target_source)?.clone();
        target_series.rename(TARGET_NAME);
        self.replace_series(target_series)?;

        let mut rr_series = self.frame.column(rr_source)?.clone();
        rr_series.rename("rr_2x_atr_tp_atr_stop");
        self.replace_series(rr_series)?;

        let mut eligible_series = self.frame.column(eligible_source)?.clone();
        eligible_series.rename("2x_atr_tp_atr_stop_eligible");
        self.replace_series(eligible_series)?;

        let mut exit_series = self.frame.column(exit_source)?.clone();
        exit_series.rename("2x_atr_tp_atr_stop_exit_i");
        self.replace_series(exit_series)?;
        Ok(())
    }

    fn attach_3x_atr_tp_atr_stop_targets(&mut self, config: &Config) -> Result<()> {
        const TARGET_NAME: &str = "3x_atr_tp_atr_stop";

        let open = column_with_nans(&self.frame, "open")?;
        let high = column_with_nans(&self.frame, "high")?;
        let low = column_with_nans(&self.frame, "low")?;
        let close = column_with_nans(&self.frame, "close")?;
        let atr_values = column_with_nans(&self.frame, "atr").with_context(|| {
	            "Missing required 'atr' column for 3x_atr_tp_atr_stop target. Re-generate the engineered dataset \
	            (e.g., rerun Barsmith with --ack-new-df or choose a fresh --output-dir) so 'atr' is present."
	        })?;

        // Prevent conceptual leakage across `--date-end` for this multi-bar target:
        // when a date cutoff is provided, cap TP/SL resolution for entry bars
        // up to the last in-sample bar (and force-exit remaining open trades
        // at that bar's close). Post-cutoff bars still resolve normally so
        // the prepared CSV contains RR values for the full dataset.
        let resolve_end_idx = if let Some(date_end) = config.include_date_end {
            let timestamps = timestamp_column(&self.frame)?;
            timestamps
                .iter()
                .rposition(|ts| ts.date_naive() <= date_end)
        } else {
            None
        };

        let (long, short, rr_long, rr_short, exit_i_long, exit_i_short) =
            compute_3x_atr_tp_atr_stop_targets_and_rr(
                &open,
                &high,
                &low,
                &close,
                &atr_values,
                config.tick_size,
                resolve_end_idx,
                config.direction,
            );

        self.replace_bool_column("3x_atr_tp_atr_stop_long", long)?;
        self.replace_bool_column("3x_atr_tp_atr_stop_short", short)?;
        self.replace_float_column("rr_long", rr_long)?;
        self.replace_float_column("rr_short", rr_short)?;
        let exit_i_long_i64: Vec<Option<i64>> = exit_i_long
            .into_iter()
            .map(|v| v.map(|idx| idx as i64))
            .collect();
        let exit_i_short_i64: Vec<Option<i64>> = exit_i_short
            .into_iter()
            .map(|v| v.map(|idx| idx as i64))
            .collect();
        self.replace_i64_column("3x_atr_tp_atr_stop_exit_i_long", exit_i_long_i64)?;
        self.replace_i64_column("3x_atr_tp_atr_stop_exit_i_short", exit_i_short_i64)?;

        let eligible_long: Vec<bool> = open
            .iter()
            .zip(close.iter())
            .map(|(o, c)| o.is_finite() && c.is_finite() && c > o)
            .collect();
        let eligible_short: Vec<bool> = open
            .iter()
            .zip(close.iter())
            .map(|(o, c)| o.is_finite() && c.is_finite() && c < o)
            .collect();
        self.replace_bool_column("3x_atr_tp_atr_stop_eligible_long", eligible_long)?;
        self.replace_bool_column("3x_atr_tp_atr_stop_eligible_short", eligible_short)?;

        let (target_source, rr_source, eligible_source, exit_source) = match config.direction {
            Direction::Short => (
                "3x_atr_tp_atr_stop_short",
                "rr_short",
                "3x_atr_tp_atr_stop_eligible_short",
                "3x_atr_tp_atr_stop_exit_i_short",
            ),
            _ => (
                "3x_atr_tp_atr_stop_long",
                "rr_long",
                "3x_atr_tp_atr_stop_eligible_long",
                "3x_atr_tp_atr_stop_exit_i_long",
            ),
        };

        let mut target_series = self.frame.column(target_source)?.clone();
        target_series.rename(TARGET_NAME);
        self.replace_series(target_series)?;

        let mut rr_series = self.frame.column(rr_source)?.clone();
        rr_series.rename("rr_3x_atr_tp_atr_stop");
        self.replace_series(rr_series)?;

        let mut eligible_series = self.frame.column(eligible_source)?.clone();
        eligible_series.rename("3x_atr_tp_atr_stop_eligible");
        self.replace_series(eligible_series)?;

        let mut exit_series = self.frame.column(exit_source)?.clone();
        exit_series.rename("3x_atr_tp_atr_stop_exit_i");
        self.replace_series(exit_series)?;
        Ok(())
    }

    fn attach_atr_tp_atr_stop_targets(&mut self, config: &Config) -> Result<()> {
        const TARGET_NAME: &str = "atr_tp_atr_stop";

        let open = column_with_nans(&self.frame, "open")?;
        let high = column_with_nans(&self.frame, "high")?;
        let low = column_with_nans(&self.frame, "low")?;
        let close = column_with_nans(&self.frame, "close")?;
        let atr_values = column_with_nans(&self.frame, "atr").with_context(|| {
	            "Missing required 'atr' column for atr_tp_atr_stop target. Re-generate the engineered dataset \
	            (e.g., rerun Barsmith with --ack-new-df or choose a fresh --output-dir) so 'atr' is present."
	        })?;

        // Prevent conceptual leakage across `--date-end` for this multi-bar target:
        // when a date cutoff is provided, cap TP/SL resolution for entry bars
        // up to the last in-sample bar (and force-exit remaining open trades
        // at that bar's close). Post-cutoff bars still resolve normally so
        // the prepared CSV contains RR values for the full dataset.
        let resolve_end_idx = if let Some(date_end) = config.include_date_end {
            let timestamps = timestamp_column(&self.frame)?;
            timestamps
                .iter()
                .rposition(|ts| ts.date_naive() <= date_end)
        } else {
            None
        };

        let (long, short, rr_long, rr_short, exit_i_long, exit_i_short) =
            compute_atr_tp_atr_stop_targets_and_rr(
                &open,
                &high,
                &low,
                &close,
                &atr_values,
                config.tick_size,
                resolve_end_idx,
                config.direction,
            );

        self.replace_bool_column("atr_tp_atr_stop_long", long)?;
        self.replace_bool_column("atr_tp_atr_stop_short", short)?;
        self.replace_float_column("rr_long", rr_long)?;
        self.replace_float_column("rr_short", rr_short)?;
        let exit_i_long_i64: Vec<Option<i64>> = exit_i_long
            .into_iter()
            .map(|v| v.map(|idx| idx as i64))
            .collect();
        let exit_i_short_i64: Vec<Option<i64>> = exit_i_short
            .into_iter()
            .map(|v| v.map(|idx| idx as i64))
            .collect();
        self.replace_i64_column("atr_tp_atr_stop_exit_i_long", exit_i_long_i64)?;
        self.replace_i64_column("atr_tp_atr_stop_exit_i_short", exit_i_short_i64)?;

        let eligible_long: Vec<bool> = open
            .iter()
            .zip(close.iter())
            .map(|(o, c)| o.is_finite() && c.is_finite() && c > o)
            .collect();
        let eligible_short: Vec<bool> = open
            .iter()
            .zip(close.iter())
            .map(|(o, c)| o.is_finite() && c.is_finite() && c < o)
            .collect();
        self.replace_bool_column("atr_tp_atr_stop_eligible_long", eligible_long)?;
        self.replace_bool_column("atr_tp_atr_stop_eligible_short", eligible_short)?;

        let (target_source, rr_source, eligible_source, exit_source) = match config.direction {
            Direction::Short => (
                "atr_tp_atr_stop_short",
                "rr_short",
                "atr_tp_atr_stop_eligible_short",
                "atr_tp_atr_stop_exit_i_short",
            ),
            _ => (
                "atr_tp_atr_stop_long",
                "rr_long",
                "atr_tp_atr_stop_eligible_long",
                "atr_tp_atr_stop_exit_i_long",
            ),
        };

        let mut target_series = self.frame.column(target_source)?.clone();
        target_series.rename(TARGET_NAME);
        self.replace_series(target_series)?;

        let mut rr_series = self.frame.column(rr_source)?.clone();
        rr_series.rename("rr_atr_tp_atr_stop");
        self.replace_series(rr_series)?;

        let mut eligible_series = self.frame.column(eligible_source)?.clone();
        eligible_series.rename("atr_tp_atr_stop_eligible");
        self.replace_series(eligible_series)?;

        let mut exit_series = self.frame.column(exit_source)?.clone();
        exit_series.rename("atr_tp_atr_stop_exit_i");
        self.replace_series(exit_series)?;
        Ok(())
    }

    fn attach_highlow_sl_2x_atr_tp_rr_gt_1_targets(&mut self, config: &Config) -> Result<()> {
        const TARGET_NAME: &str = "highlow_sl_2x_atr_tp_rr_gt_1";

        let open = column_with_nans(&self.frame, "open")?;
        let high = column_with_nans(&self.frame, "high")?;
        let low = column_with_nans(&self.frame, "low")?;
        let close = column_with_nans(&self.frame, "close")?;
        let atr_values = column_with_nans(&self.frame, "atr").with_context(|| {
	            "Missing required 'atr' column for highlow_sl_2x_atr_tp_rr_gt_1 target. Re-generate the engineered dataset \
	            (e.g., rerun Barsmith with --ack-new-df or choose a fresh --output-dir) so 'atr' is present."
	        })?;

        // Prevent conceptual leakage across `--date-end` for this multi-bar target:
        // when a date cutoff is provided, cap TP/SL resolution for entry bars
        // up to the last in-sample bar (and force-exit remaining open trades
        // at that bar's close). Post-cutoff bars still resolve normally so
        // the prepared CSV contains RR values for the full dataset.
        let resolve_end_idx = if let Some(date_end) = config.include_date_end {
            let timestamps = timestamp_column(&self.frame)?;
            timestamps
                .iter()
                .rposition(|ts| ts.date_naive() <= date_end)
        } else {
            None
        };

        let (long, short, rr_long, rr_short, exit_i_long, exit_i_short) =
            compute_highlow_sl_2x_atr_tp_rr_gt_1_targets_and_rr(
                &open,
                &high,
                &low,
                &close,
                &atr_values,
                config.tick_size,
                resolve_end_idx,
                config.direction,
            );

        self.replace_bool_column("highlow_sl_2x_atr_tp_rr_gt_1_long", long)?;
        self.replace_bool_column("highlow_sl_2x_atr_tp_rr_gt_1_short", short)?;
        self.replace_float_column("rr_long", rr_long)?;
        self.replace_float_column("rr_short", rr_short)?;
        let exit_i_long_i64: Vec<Option<i64>> = exit_i_long
            .into_iter()
            .map(|v| v.map(|idx| idx as i64))
            .collect();
        let exit_i_short_i64: Vec<Option<i64>> = exit_i_short
            .into_iter()
            .map(|v| v.map(|idx| idx as i64))
            .collect();
        self.replace_i64_column("highlow_sl_2x_atr_tp_rr_gt_1_exit_i_long", exit_i_long_i64)?;
        self.replace_i64_column(
            "highlow_sl_2x_atr_tp_rr_gt_1_exit_i_short",
            exit_i_short_i64,
        )?;

        let eligible_long: Vec<bool> = open
            .iter()
            .zip(close.iter())
            .map(|(o, c)| o.is_finite() && c.is_finite() && c > o)
            .collect();
        let eligible_short: Vec<bool> = open
            .iter()
            .zip(close.iter())
            .map(|(o, c)| o.is_finite() && c.is_finite() && c < o)
            .collect();
        self.replace_bool_column("highlow_sl_2x_atr_tp_rr_gt_1_eligible_long", eligible_long)?;
        self.replace_bool_column(
            "highlow_sl_2x_atr_tp_rr_gt_1_eligible_short",
            eligible_short,
        )?;

        let (target_source, rr_source, eligible_source, exit_source) = match config.direction {
            Direction::Short => (
                "highlow_sl_2x_atr_tp_rr_gt_1_short",
                "rr_short",
                "highlow_sl_2x_atr_tp_rr_gt_1_eligible_short",
                "highlow_sl_2x_atr_tp_rr_gt_1_exit_i_short",
            ),
            _ => (
                "highlow_sl_2x_atr_tp_rr_gt_1_long",
                "rr_long",
                "highlow_sl_2x_atr_tp_rr_gt_1_eligible_long",
                "highlow_sl_2x_atr_tp_rr_gt_1_exit_i_long",
            ),
        };

        let mut target_series = self.frame.column(target_source)?.clone();
        target_series.rename(TARGET_NAME);
        self.replace_series(target_series)?;

        let mut rr_series = self.frame.column(rr_source)?.clone();
        rr_series.rename("rr_highlow_sl_2x_atr_tp_rr_gt_1");
        self.replace_series(rr_series)?;

        let mut eligible_series = self.frame.column(eligible_source)?.clone();
        eligible_series.rename("highlow_sl_2x_atr_tp_rr_gt_1_eligible");
        self.replace_series(eligible_series)?;

        let mut exit_series = self.frame.column(exit_source)?.clone();
        exit_series.rename("highlow_sl_2x_atr_tp_rr_gt_1_exit_i");
        self.replace_series(exit_series)?;
        Ok(())
    }

    fn attach_highlow_sl_1x_atr_tp_rr_gt_1_targets(&mut self, config: &Config) -> Result<()> {
        const TARGET_NAME: &str = "highlow_sl_1x_atr_tp_rr_gt_1";

        let open = column_with_nans(&self.frame, "open")?;
        let high = column_with_nans(&self.frame, "high")?;
        let low = column_with_nans(&self.frame, "low")?;
        let close = column_with_nans(&self.frame, "close")?;
        let atr_values = column_with_nans(&self.frame, "atr").with_context(|| {
	            "Missing required 'atr' column for highlow_sl_1x_atr_tp_rr_gt_1 target. Re-generate the engineered dataset \
	            (e.g., rerun Barsmith with --ack-new-df or choose a fresh --output-dir) so 'atr' is present."
	        })?;

        // Prevent conceptual leakage across `--date-end` for this multi-bar target:
        // when a date cutoff is provided, cap TP/SL resolution for entry bars
        // up to the last in-sample bar (and force-exit remaining open trades
        // at that bar's close). Post-cutoff bars still resolve normally so
        // the prepared CSV contains RR values for the full dataset.
        let resolve_end_idx = if let Some(date_end) = config.include_date_end {
            let timestamps = timestamp_column(&self.frame)?;
            timestamps
                .iter()
                .rposition(|ts| ts.date_naive() <= date_end)
        } else {
            None
        };

        let (long, short, rr_long, rr_short, exit_i_long, exit_i_short) =
            compute_highlow_sl_1x_atr_tp_rr_gt_1_targets_and_rr(
                &open,
                &high,
                &low,
                &close,
                &atr_values,
                config.tick_size,
                resolve_end_idx,
                config.direction,
            );

        self.replace_bool_column("highlow_sl_1x_atr_tp_rr_gt_1_long", long)?;
        self.replace_bool_column("highlow_sl_1x_atr_tp_rr_gt_1_short", short)?;
        self.replace_float_column("rr_long", rr_long)?;
        self.replace_float_column("rr_short", rr_short)?;
        let exit_i_long_i64: Vec<Option<i64>> = exit_i_long
            .into_iter()
            .map(|v| v.map(|idx| idx as i64))
            .collect();
        let exit_i_short_i64: Vec<Option<i64>> = exit_i_short
            .into_iter()
            .map(|v| v.map(|idx| idx as i64))
            .collect();
        self.replace_i64_column("highlow_sl_1x_atr_tp_rr_gt_1_exit_i_long", exit_i_long_i64)?;
        self.replace_i64_column(
            "highlow_sl_1x_atr_tp_rr_gt_1_exit_i_short",
            exit_i_short_i64,
        )?;

        let eligible_long: Vec<bool> = open
            .iter()
            .zip(close.iter())
            .map(|(o, c)| o.is_finite() && c.is_finite() && c > o)
            .collect();
        let eligible_short: Vec<bool> = open
            .iter()
            .zip(close.iter())
            .map(|(o, c)| o.is_finite() && c.is_finite() && c < o)
            .collect();
        self.replace_bool_column("highlow_sl_1x_atr_tp_rr_gt_1_eligible_long", eligible_long)?;
        self.replace_bool_column(
            "highlow_sl_1x_atr_tp_rr_gt_1_eligible_short",
            eligible_short,
        )?;

        let (target_source, rr_source, eligible_source, exit_source) = match config.direction {
            Direction::Short => (
                "highlow_sl_1x_atr_tp_rr_gt_1_short",
                "rr_short",
                "highlow_sl_1x_atr_tp_rr_gt_1_eligible_short",
                "highlow_sl_1x_atr_tp_rr_gt_1_exit_i_short",
            ),
            _ => (
                "highlow_sl_1x_atr_tp_rr_gt_1_long",
                "rr_long",
                "highlow_sl_1x_atr_tp_rr_gt_1_eligible_long",
                "highlow_sl_1x_atr_tp_rr_gt_1_exit_i_long",
            ),
        };

        let mut target_series = self.frame.column(target_source)?.clone();
        target_series.rename(TARGET_NAME);
        self.replace_series(target_series)?;

        let mut rr_series = self.frame.column(rr_source)?.clone();
        rr_series.rename("rr_highlow_sl_1x_atr_tp_rr_gt_1");
        self.replace_series(rr_series)?;

        let mut eligible_series = self.frame.column(eligible_source)?.clone();
        eligible_series.rename("highlow_sl_1x_atr_tp_rr_gt_1_eligible");
        self.replace_series(eligible_series)?;

        let mut exit_series = self.frame.column(exit_source)?.clone();
        exit_series.rename("highlow_sl_1x_atr_tp_rr_gt_1_exit_i");
        self.replace_series(exit_series)?;
        Ok(())
    }

    fn attach_highlow_or_atr_tightest_stop_targets(&mut self, config: &Config) -> Result<()> {
        const TARGET_NAME: &str = "highlow_or_atr_tightest_stop";

        let open = column_with_nans(&self.frame, "open")?;
        let high = column_with_nans(&self.frame, "high")?;
        let low = column_with_nans(&self.frame, "low")?;
        let close = column_with_nans(&self.frame, "close")?;
        let atr_values = column_with_nans(&self.frame, "atr").with_context(|| {
            "Missing required 'atr' column for highlow_or_atr_tightest_stop target. Re-generate the engineered dataset \
            (e.g., rerun Barsmith with --ack-new-df or choose a fresh --output-dir) so 'atr' is present."
        })?;

        // Prevent conceptual leakage across `--date-end` for this multi-bar target:
        // when a date cutoff is provided, cap TP/SL resolution for entry bars
        // up to the last in-sample bar (and force-exit remaining open trades
        // at that bar's close). Post-cutoff bars still resolve normally so
        // the prepared CSV contains RR values for the full dataset.
        let resolve_end_idx = if let Some(date_end) = config.include_date_end {
            let timestamps = timestamp_column(&self.frame)?;
            timestamps
                .iter()
                .rposition(|ts| ts.date_naive() <= date_end)
        } else {
            None
        };

        let (long, short, rr_long, rr_short, exit_i_long, exit_i_short) =
            compute_highlow_or_atr_tightest_stop_targets_and_rr(
                &open,
                &high,
                &low,
                &close,
                &atr_values,
                config.tick_size,
                resolve_end_idx,
                config.direction,
            );

        self.replace_bool_column("highlow_or_atr_tightest_stop_long", long)?;
        self.replace_bool_column("highlow_or_atr_tightest_stop_short", short)?;
        self.replace_float_column("rr_long", rr_long)?;
        self.replace_float_column("rr_short", rr_short)?;
        let exit_i_long_i64: Vec<Option<i64>> = exit_i_long
            .into_iter()
            .map(|v| v.map(|idx| idx as i64))
            .collect();
        let exit_i_short_i64: Vec<Option<i64>> = exit_i_short
            .into_iter()
            .map(|v| v.map(|idx| idx as i64))
            .collect();
        self.replace_i64_column("highlow_or_atr_tightest_stop_exit_i_long", exit_i_long_i64)?;
        self.replace_i64_column(
            "highlow_or_atr_tightest_stop_exit_i_short",
            exit_i_short_i64,
        )?;

        let eligible_long: Vec<bool> = open
            .iter()
            .zip(close.iter())
            .map(|(o, c)| o.is_finite() && c.is_finite() && c > o)
            .collect();
        let eligible_short: Vec<bool> = open
            .iter()
            .zip(close.iter())
            .map(|(o, c)| o.is_finite() && c.is_finite() && c < o)
            .collect();
        self.replace_bool_column("highlow_or_atr_tightest_stop_eligible_long", eligible_long)?;
        self.replace_bool_column(
            "highlow_or_atr_tightest_stop_eligible_short",
            eligible_short,
        )?;

        let (target_source, rr_source, eligible_source, exit_source) = match config.direction {
            Direction::Short => (
                "highlow_or_atr_tightest_stop_short",
                "rr_short",
                "highlow_or_atr_tightest_stop_eligible_short",
                "highlow_or_atr_tightest_stop_exit_i_short",
            ),
            _ => (
                "highlow_or_atr_tightest_stop_long",
                "rr_long",
                "highlow_or_atr_tightest_stop_eligible_long",
                "highlow_or_atr_tightest_stop_exit_i_long",
            ),
        };

        let mut target_series = self.frame.column(target_source)?.clone();
        target_series.rename(TARGET_NAME);
        self.replace_series(target_series)?;

        let mut rr_series = self.frame.column(rr_source)?.clone();
        rr_series.rename("rr_highlow_or_atr_tightest_stop");
        self.replace_series(rr_series)?;

        let mut eligible_series = self.frame.column(eligible_source)?.clone();
        eligible_series.rename("highlow_or_atr_tightest_stop_eligible");
        self.replace_series(eligible_series)?;

        let mut exit_series = self.frame.column(exit_source)?.clone();
        exit_series.rename("highlow_or_atr_tightest_stop_exit_i");
        self.replace_series(exit_series)?;
        Ok(())
    }

    fn attach_tribar_4h_targets(&mut self, config: &Config) -> Result<()> {
        // Sanity check: enforce 4h timeframe via filename convention.
        let file_name = config
            .input_csv
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("");
        if !file_name.contains("_4h") {
            return Err(anyhow!(
                "tribar_4h_2atr target expects input CSV filename to contain '_4h' (got '{}')",
                file_name
            ));
        }

        // This Tribar target is long-only.
        if !matches!(config.direction, Direction::Long) {
            return Err(anyhow!(
                "tribar_4h_2atr target currently supports only Direction::Long (got {:?})",
                config.direction
            ));
        }

        let open = column_with_nans(&self.frame, "open")?;
        let high = column_with_nans(&self.frame, "high")?;
        let low = column_with_nans(&self.frame, "low")?;
        let close = column_with_nans(&self.frame, "close")?;
        let is_tribar = bool_column(&self.frame, "is_tribar")?;
        let is_close_above_kf_ma = bool_column(&self.frame, "is_close_above_kf_ma")?;
        let timestamps = timestamp_column(&self.frame)?;

        let len = open.len();
        if high.len() != len
            || low.len() != len
            || close.len() != len
            || is_tribar.len() != len
            || is_close_above_kf_ma.len() != len
            || timestamps.len() != len
        {
            return Err(anyhow!(
                "Inconsistent series lengths when building tribar_4h_2atr target"
            ));
        }

        // Compute 14-period ATR on 4h bars.
        let atr_values = atr(&high, &low, &close, 14);

        // Precompute week indices and the last index for each week to enforce
        // per-week trade caps and end-of-week exits.
        let (week_index, week_end_index) = compute_week_indices(&timestamps);

        let mut label = vec![false; len];
        let mut rr = vec![f64::NAN; len];
        let mut exit_i_long: Vec<Option<i64>> = vec![None; len];
        let exit_i_short: Vec<Option<i64>> = vec![None; len];

        let mut trades_per_week: HashMap<i64, u8> = HashMap::new();
        let mut idx = 0usize;
        const ATR_MULTIPLE: f64 = 2.0;

        while idx < len {
            let mut advanced = false;

            if is_tribar[idx] && is_close_above_kf_ma[idx] {
                let week = week_index[idx];
                let used = trades_per_week.get(&week).copied().unwrap_or(0);
                if used < 2 {
                    let entry = close[idx];
                    let atr = atr_values[idx];
                    let bar_low = low[idx];

                    if entry.is_finite() && atr.is_finite() && bar_low.is_finite() {
                        // Stop-loss: below the low or 1x ATR below entry, whichever is lower.
                        let atr_stop = entry - atr;
                        let stop = bar_low.min(atr_stop);
                        if stop.is_finite() && stop < entry {
                            let risk = entry - stop;
                            if risk.abs() > SMALL_DIVISOR {
                                let tp = entry + ATR_MULTIPLE * atr;
                                let last_idx = week_end_index[idx];

                                let mut exit_price = close[idx];
                                let mut exit_idx = idx;

                                for j in idx..=last_idx {
                                    let h = high[j];
                                    let l = low[j];
                                    let c = close[j];
                                    if !h.is_finite() || !l.is_finite() || !c.is_finite() {
                                        continue;
                                    }

                                    // Conservative ordering: treat SL hit as dominant when both
                                    // SL and TP would be touched within the same bar.
                                    if l <= stop {
                                        exit_price = stop;
                                        exit_idx = j;
                                        break;
                                    }
                                    if h >= tp {
                                        exit_price = tp;
                                        exit_idx = j;
                                        break;
                                    }

                                    if j == last_idx {
                                        exit_price = c;
                                        exit_idx = j;
                                    }
                                }

                                let trade_rr = (exit_price - entry) / risk;
                                rr[idx] = trade_rr;
                                label[idx] = trade_rr.is_finite() && trade_rr > 0.0;
                                exit_i_long[idx] = Some(exit_idx as i64);

                                trades_per_week.insert(week, used + 1);
                                idx = exit_idx.saturating_add(1);
                                advanced = true;
                            }
                        }
                    }
                }
            }

            if !advanced {
                idx += 1;
            }
        }

        // Main Tribar target and reward column.
        self.replace_bool_column("tribar_4h_2atr", label)?;
        self.replace_float_column("rr_long", rr)?;
        let mut rr_series = self.frame.column("rr_long")?.clone();
        rr_series.rename("rr_tribar_4h_2atr");
        self.replace_series(rr_series)?;

        self.replace_i64_column("tribar_4h_2atr_exit_i_long", exit_i_long)?;
        self.replace_i64_column("tribar_4h_2atr_exit_i_short", exit_i_short)?;
        let mut exit_series = self.frame.column("tribar_4h_2atr_exit_i_long")?.clone();
        exit_series.rename("tribar_4h_2atr_exit_i");
        self.replace_series(exit_series)?;

        Ok(())
    }

    fn recompute_kf_alignment(&mut self) -> Result<()> {
        let smooth = match column_with_nans(&self.frame, "kf_smooth") {
            Ok(values) => values,
            Err(_) => return Ok(()),
        };
        let ema9 = match column_with_nans(&self.frame, "9ema") {
            Ok(values) => values,
            Err(_) => return Ok(()),
        };
        let ema200 = match column_with_nans(&self.frame, "200sma") {
            Ok(values) => values,
            Err(_) => return Ok(()),
        };

        let mut values = Vec::with_capacity(smooth.len());
        for ((smooth_val, ema9_val), ema200_val) in
            smooth.iter().zip(ema9.iter()).zip(ema200.iter())
        {
            let aligned = smooth_val.is_finite()
                && ema9_val.is_finite()
                && ema200_val.is_finite()
                && smooth_val > ema9_val
                && ema9_val > ema200_val;
            values.push(aligned);
        }

        self.replace_bool_column("kf_ema_aligned", values)?;

        Ok(())
    }

    fn recompute_high_low(&mut self) -> Result<()> {
        let highs = match column_with_nans(&self.frame, "high") {
            Ok(values) => values,
            Err(_) => return Ok(()),
        };
        let lows = match column_with_nans(&self.frame, "low") {
            Ok(values) => values,
            Err(_) => return Ok(()),
        };
        let len = highs.len().min(lows.len());
        if len == 0 {
            return Ok(());
        }

        let mut higher_high = vec![false; len];
        let mut higher_low = vec![false; len];
        let mut lower_high = vec![false; len];
        let mut lower_low = vec![false; len];

        for i in 1..len {
            let high = highs[i];
            let prev_high = highs[i - 1];
            if high.is_finite() && prev_high.is_finite() {
                higher_high[i] = high > prev_high;
                lower_high[i] = high < prev_high;
            }
            let low = lows[i];
            let prev_low = lows[i - 1];
            if low.is_finite() && prev_low.is_finite() {
                higher_low[i] = low > prev_low;
                lower_low[i] = low < prev_low;
            }
        }

        self.replace_bool_column("higher_high", higher_high.clone())?;
        self.replace_bool_column("higher_low", higher_low.clone())?;
        self.replace_bool_column("lower_high", lower_high.clone())?;
        self.replace_bool_column("lower_low", lower_low.clone())?;

        let bullish: Vec<bool> = higher_high
            .iter()
            .zip(higher_low.iter())
            .map(|(hh, hl)| *hh && *hl)
            .collect();
        let bearish: Vec<bool> = lower_high
            .iter()
            .zip(lower_low.iter())
            .map(|(lh, ll)| *lh && *ll)
            .collect();
        self.replace_bool_column("bullish_bar_sequence", bullish)?;
        self.replace_bool_column("bearish_bar_sequence", bearish)?;

        Ok(())
    }

    fn replace_bool_column(&mut self, name: &str, values: Vec<bool>) -> Result<()> {
        if self.frame.column(name).is_ok() {
            self.frame = self.frame.drop(name)?;
        }
        let series = Series::new(name, values);
        self.frame
            .with_column(series)
            .with_context(|| format!("Failed to update column {name}"))?;
        Ok(())
    }

    fn replace_float_column(&mut self, name: &str, values: Vec<f64>) -> Result<()> {
        if self.frame.column(name).is_ok() {
            self.frame = self.frame.drop(name)?;
        }
        let series = Series::new(name, values);
        self.frame
            .with_column(series)
            .with_context(|| format!("Failed to update column {name}"))?;
        Ok(())
    }

    fn replace_i64_column(&mut self, name: &str, values: Vec<Option<i64>>) -> Result<()> {
        if self.frame.column(name).is_ok() {
            self.frame = self.frame.drop(name)?;
        }
        let series = Series::new(name, values);
        self.frame
            .with_column(series)
            .with_context(|| format!("Failed to update column {name}"))?;
        Ok(())
    }

    fn replace_series(&mut self, series: Series) -> Result<()> {
        let name = series.name().to_string();
        if self.frame.column(&name).is_ok() {
            self.frame = self.frame.drop(&name)?;
        }
        self.frame
            .with_column(series)
            .with_context(|| format!("Failed to update column {name}"))?;
        Ok(())
    }

    fn recompute_momentum_scores(&mut self) -> Result<()> {
        let rsi = match column_with_nans(&self.frame, "rsi_14") {
            Ok(values) => values,
            Err(_) => return Ok(()),
        };
        let roc5 = match column_with_nans(&self.frame, "roc_5") {
            Ok(values) => values,
            Err(_) => return Ok(()),
        };
        let roc10 = match column_with_nans(&self.frame, "roc_10") {
            Ok(values) => values,
            Err(_) => return Ok(()),
        };
        let score = momentum_score(&rsi, &roc5, &roc10);
        self.replace_float_column("momentum_score", score.clone())?;

        let strong: Vec<bool> = score
            .iter()
            .map(|val| val.is_finite() && *val > 0.75)
            .collect();
        let weak: Vec<bool> = score
            .iter()
            .map(|val| val.is_finite() && *val < 0.25)
            .collect();
        self.replace_bool_column("is_strong_momentum_score", strong)?;
        self.replace_bool_column("is_weak_momentum_score", weak)?;

        Ok(())
    }
}
fn apply_indicator_warmups(
    bools: &mut HashMap<&'static str, Vec<bool>>,
    floats: &mut HashMap<&'static str, Vec<f64>>,
) {
    // No-op: indicator constructors are responsible for emitting NaNs during
    // their own warmup periods. We rely on a single NaN-drop pass over core
    // numeric columns instead of masking additional warmup regions here.
    let _ = (bools, floats);
}

fn quantize_distance_to_tick(distance: f64, tick_size: f64, mode: TickRoundMode) -> f64 {
    if !distance.is_finite() || tick_size <= 0.0 {
        return distance;
    }
    if distance.abs() < f64::EPSILON {
        return 0.0;
    }

    let ticks = distance / tick_size;
    let raw_rounded = match mode {
        TickRoundMode::Nearest => ticks.round(),
        TickRoundMode::Floor => ticks.floor(),
        TickRoundMode::Ceil => ticks.ceil(),
    };
    // Enforce a minimum of one tick for non-zero distances so that we never
    // end up with a zero-risk trade when a stop is requested.
    let ticks_final = raw_rounded.max(1.0);
    ticks_final * tick_size
}

fn quantize_price_to_tick(price: f64, tick_size: f64, mode: TickRoundMode) -> f64 {
    if !price.is_finite() || tick_size <= 0.0 {
        return price;
    }
    let ticks = price / tick_size;
    let rounded = match mode {
        TickRoundMode::Nearest => ticks.round(),
        TickRoundMode::Floor => ticks.floor(),
        TickRoundMode::Ceil => ticks.ceil(),
    };
    rounded * tick_size
}

fn compute_next_bar_targets_and_rr(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    wicks_diff_sma14: &[f64],
    sl_multiplier: f64,
    tick_size: Option<f64>,
    direction: Direction,
) -> (
    Vec<bool>,
    Vec<bool>,
    Vec<f64>,
    Vec<f64>,
    Vec<Option<usize>>,
    Vec<Option<usize>>,
) {
    let len = open
        .len()
        .min(high.len())
        .min(low.len())
        .min(close.len())
        .min(wicks_diff_sma14.len());
    let mut long = vec![false; len];
    let mut short = vec![false; len];
    let mut long_rr = vec![f64::NAN; len];
    let mut short_rr = vec![f64::NAN; len];
    let mut exit_i_long = vec![None; len];
    let mut exit_i_short = vec![None; len];
    if len < 2 {
        return (long, short, long_rr, short_rr, exit_i_long, exit_i_short);
    }

    let want_long = matches!(direction, Direction::Long | Direction::Both);
    let want_short = matches!(direction, Direction::Short | Direction::Both);

    for idx in 0..(len - 1) {
        let next = idx + 1;
        let entry = open[next];
        let high_next = high[next];
        let low_next = low[next];
        let close_next = close[next];
        let wick = wicks_diff_sma14[idx];
        if !entry.is_finite()
            || !high_next.is_finite()
            || !low_next.is_finite()
            || !close_next.is_finite()
            || !wick.is_finite()
        {
            continue;
        }
        let sl_distance_raw = (wick * sl_multiplier).abs();
        let sl_distance = if let Some(ts) = tick_size {
            quantize_distance_to_tick(sl_distance_raw, ts, TickRoundMode::Ceil)
        } else {
            sl_distance_raw
        };
        if sl_distance <= SMALL_DIVISOR {
            continue;
        }

        if want_long {
            let long_sl = entry - sl_distance;
            let long_sl_hit = low_next <= long_sl;
            long[idx] = close_next > entry && !long_sl_hit;
            let long_exit = if long_sl_hit { long_sl } else { close_next };
            long_rr[idx] = (long_exit - entry) / sl_distance;
            exit_i_long[idx] = Some(next);
        }

        if want_short {
            let short_sl = entry + sl_distance;
            let short_sl_hit = high_next >= short_sl;
            short[idx] = close_next < entry && !short_sl_hit;
            let short_exit = if short_sl_hit { short_sl } else { close_next };
            short_rr[idx] = (entry - short_exit) / sl_distance;
            exit_i_short[idx] = Some(next);
        }
    }

    (long, short, long_rr, short_rr, exit_i_long, exit_i_short)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HighlowOrAtrStopMode {
    /// Wider (higher-risk) stop:
    /// - long: min(low, entry - 1x ATR)
    /// - short: max(high, entry + 1x ATR)
    Wide,
    /// High/low only stop:
    /// - long: low
    /// - short: high
    HighlowOnly,
    /// ATR only stop:
    /// - long: entry - 1x ATR
    /// - short: entry + 1x ATR
    AtrOnly,
    /// Tighter (lower-risk) stop:
    /// - long: highest stop < entry from {low, entry - 1x ATR}
    /// - short: lowest stop > entry from {high, entry + 1x ATR}
    Tightest,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum HighlowOrAtrTpMode {
    AtrMultiple(f64),
    RiskMultiple(f64),
}

fn highlow_or_atr_stop_long(entry: f64, low: f64, atr: f64, mode: HighlowOrAtrStopMode) -> f64 {
    match mode {
        HighlowOrAtrStopMode::Wide => low.min(entry - atr),
        HighlowOrAtrStopMode::HighlowOnly => low,
        HighlowOrAtrStopMode::AtrOnly => entry - atr,
        HighlowOrAtrStopMode::Tightest => {
            let mut stop_raw = f64::NAN;
            if low < entry {
                stop_raw = low;
            }
            let atr_stop = entry - atr;
            if atr_stop < entry && (!stop_raw.is_finite() || atr_stop > stop_raw) {
                stop_raw = atr_stop;
            }
            stop_raw
        }
    }
}

fn highlow_or_atr_stop_short(entry: f64, high: f64, atr: f64, mode: HighlowOrAtrStopMode) -> f64 {
    match mode {
        HighlowOrAtrStopMode::Wide => high.max(entry + atr),
        HighlowOrAtrStopMode::HighlowOnly => high,
        HighlowOrAtrStopMode::AtrOnly => entry + atr,
        HighlowOrAtrStopMode::Tightest => {
            let mut stop_raw = f64::NAN;
            if high > entry {
                stop_raw = high;
            }
            let atr_stop = entry + atr;
            if atr_stop > entry && (!stop_raw.is_finite() || atr_stop < stop_raw) {
                stop_raw = atr_stop;
            }
            stop_raw
        }
    }
}

fn compute_highlow_or_atr_targets_and_rr_with_stop_mode(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    atr: &[f64],
    tick_size: Option<f64>,
    resolve_end_idx: Option<usize>,
    direction: Direction,
    stop_mode: HighlowOrAtrStopMode,
    tp_mode: HighlowOrAtrTpMode,
    min_tp_rr: Option<f64>,
) -> (
    Vec<bool>,
    Vec<bool>,
    Vec<f64>,
    Vec<f64>,
    Vec<Option<usize>>,
    Vec<Option<usize>>,
) {
    let requires_atr = matches!(
        stop_mode,
        HighlowOrAtrStopMode::Wide | HighlowOrAtrStopMode::AtrOnly | HighlowOrAtrStopMode::Tightest
    ) || matches!(tp_mode, HighlowOrAtrTpMode::AtrMultiple(_));

    let mut len = open.len().min(high.len()).min(low.len()).min(close.len());
    if requires_atr {
        len = len.min(atr.len());
    }
    let mut long = vec![false; len];
    let mut short = vec![false; len];
    let mut long_rr = vec![f64::NAN; len];
    let mut short_rr = vec![f64::NAN; len];
    let mut exit_i_long = vec![None; len];
    let mut exit_i_short = vec![None; len];
    if len < 2 {
        return (long, short, long_rr, short_rr, exit_i_long, exit_i_short);
    }

    let cutoff_horizon = resolve_end_idx.unwrap_or(len - 1).min(len - 1);

    let want_long = matches!(direction, Direction::Long | Direction::Both);
    let want_short = matches!(direction, Direction::Short | Direction::Both);

    for idx in 0..(len - 1) {
        let cap_to_cutoff = resolve_end_idx.is_some() && idx <= cutoff_horizon;
        let local_horizon = if cap_to_cutoff {
            cutoff_horizon
        } else {
            len - 1
        };
        if idx >= local_horizon {
            // Do not open trades that have no future bars available for TP/SL resolution.
            continue;
        }

        let open_idx = open[idx];
        let close_idx = close[idx];
        let high_idx = high[idx];
        let low_idx = low[idx];
        let atr_idx = if requires_atr { atr[idx] } else { f64::NAN };

        if !open_idx.is_finite()
            || !close_idx.is_finite()
            || !high_idx.is_finite()
            || !low_idx.is_finite()
            || (requires_atr && !atr_idx.is_finite())
        {
            continue;
        }

        let body = close_idx - open_idx;
        if body.abs() <= f64::EPSILON {
            continue;
        }

        // Entry at signal bar close.
        let entry = close_idx;

        if body > 0.0 {
            if !want_long {
                continue;
            }
            let stop_raw = highlow_or_atr_stop_long(entry, low_idx, atr_idx, stop_mode);
            let stop = if let Some(ts) = tick_size {
                quantize_price_to_tick(stop_raw, ts, TickRoundMode::Floor)
            } else {
                stop_raw
            };

            if !stop.is_finite() || stop >= entry {
                continue;
            }
            let risk = entry - stop;
            if risk <= SMALL_DIVISOR {
                continue;
            }

            let tp_raw = match tp_mode {
                HighlowOrAtrTpMode::AtrMultiple(m) => entry + m * atr_idx,
                HighlowOrAtrTpMode::RiskMultiple(m) => entry + m * risk,
            };
            let tp = if let Some(ts) = tick_size {
                quantize_price_to_tick(tp_raw, ts, TickRoundMode::Ceil)
            } else {
                tp_raw
            };
            if !tp.is_finite() {
                continue;
            }
            if let Some(min_rr) = min_tp_rr {
                let rr_at_tp = (tp - entry) / risk;
                if !rr_at_tp.is_finite() || rr_at_tp <= min_rr {
                    continue;
                }
            }

            let mut rr = f64::NAN;
            let mut hit_tp = false;
            let mut exit_idx: Option<usize> = None;
            for j in (idx + 1)..=local_horizon {
                let o = open[j];
                let h = high[j];
                let l = low[j];
                if !h.is_finite() || !l.is_finite() {
                    continue;
                }

                // Gap-aware fills: if the next bar opens beyond our stop/TP,
                // assume the fill happens at the open price (RR can be < -1 or > 2).
                if o.is_finite() {
                    if o <= stop {
                        rr = (o - entry) / risk;
                        hit_tp = false;
                        exit_idx = Some(j);
                        break;
                    }
                    if o >= tp {
                        rr = (o - entry) / risk;
                        hit_tp = true;
                        exit_idx = Some(j);
                        break;
                    }
                }

                // Conservative ordering: SL dominates if both touched.
                if l <= stop {
                    rr = -1.0;
                    hit_tp = false;
                    exit_idx = Some(j);
                    break;
                }
                if h >= tp {
                    rr = (tp - entry) / risk;
                    hit_tp = true;
                    exit_idx = Some(j);
                    break;
                }
            }

            if !rr.is_finite() && cap_to_cutoff {
                let exit = close[local_horizon];
                if exit.is_finite() {
                    rr = (exit - entry) / risk;
                    hit_tp = false;
                    exit_idx = Some(local_horizon);
                }
            }

            if rr.is_finite() {
                long_rr[idx] = rr;
                long[idx] = hit_tp;
                exit_i_long[idx] = exit_idx;
            }
        } else {
            if !want_short {
                continue;
            }
            let stop_raw = highlow_or_atr_stop_short(entry, high_idx, atr_idx, stop_mode);
            let stop = if let Some(ts) = tick_size {
                quantize_price_to_tick(stop_raw, ts, TickRoundMode::Ceil)
            } else {
                stop_raw
            };

            if !stop.is_finite() || stop <= entry {
                continue;
            }
            let risk = stop - entry;
            if risk <= SMALL_DIVISOR {
                continue;
            }

            let tp_raw = match tp_mode {
                HighlowOrAtrTpMode::AtrMultiple(m) => entry - m * atr_idx,
                HighlowOrAtrTpMode::RiskMultiple(m) => entry - m * risk,
            };
            let tp = if let Some(ts) = tick_size {
                quantize_price_to_tick(tp_raw, ts, TickRoundMode::Floor)
            } else {
                tp_raw
            };
            if !tp.is_finite() {
                continue;
            }
            if let Some(min_rr) = min_tp_rr {
                let rr_at_tp = (entry - tp) / risk;
                if !rr_at_tp.is_finite() || rr_at_tp <= min_rr {
                    continue;
                }
            }

            let mut rr = f64::NAN;
            let mut hit_tp = false;
            let mut exit_idx: Option<usize> = None;
            for j in (idx + 1)..=local_horizon {
                let o = open[j];
                let h = high[j];
                let l = low[j];
                if !h.is_finite() || !l.is_finite() {
                    continue;
                }

                // Gap-aware fills: if the next bar opens beyond our stop/TP,
                // assume the fill happens at the open price (RR can be < -1 or > 2).
                if o.is_finite() {
                    if o >= stop {
                        rr = (entry - o) / risk;
                        hit_tp = false;
                        exit_idx = Some(j);
                        break;
                    }
                    if o <= tp {
                        rr = (entry - o) / risk;
                        hit_tp = true;
                        exit_idx = Some(j);
                        break;
                    }
                }

                if h >= stop {
                    rr = -1.0;
                    hit_tp = false;
                    exit_idx = Some(j);
                    break;
                }
                if l <= tp {
                    rr = (entry - tp) / risk;
                    hit_tp = true;
                    exit_idx = Some(j);
                    break;
                }
            }

            if !rr.is_finite() && cap_to_cutoff {
                let exit = close[local_horizon];
                if exit.is_finite() {
                    rr = (entry - exit) / risk;
                    hit_tp = false;
                    exit_idx = Some(local_horizon);
                }
            }

            if rr.is_finite() {
                short_rr[idx] = rr;
                short[idx] = hit_tp;
                exit_i_short[idx] = exit_idx;
            }
        }
    }

    (long, short, long_rr, short_rr, exit_i_long, exit_i_short)
}

fn compute_highlow_or_atr_targets_and_rr(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    atr: &[f64],
    tick_size: Option<f64>,
    resolve_end_idx: Option<usize>,
    direction: Direction,
) -> (
    Vec<bool>,
    Vec<bool>,
    Vec<f64>,
    Vec<f64>,
    Vec<Option<usize>>,
    Vec<Option<usize>>,
) {
    compute_highlow_or_atr_targets_and_rr_with_stop_mode(
        open,
        high,
        low,
        close,
        atr,
        tick_size,
        resolve_end_idx,
        direction,
        HighlowOrAtrStopMode::Wide,
        HighlowOrAtrTpMode::AtrMultiple(2.0),
        None,
    )
}

fn compute_highlow_1r_targets_and_rr(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    tick_size: Option<f64>,
    resolve_end_idx: Option<usize>,
    direction: Direction,
) -> (
    Vec<bool>,
    Vec<bool>,
    Vec<f64>,
    Vec<f64>,
    Vec<Option<usize>>,
    Vec<Option<usize>>,
) {
    compute_highlow_or_atr_targets_and_rr_with_stop_mode(
        open,
        high,
        low,
        close,
        &[],
        tick_size,
        resolve_end_idx,
        direction,
        HighlowOrAtrStopMode::HighlowOnly,
        HighlowOrAtrTpMode::RiskMultiple(1.0),
        None,
    )
}

fn compute_2x_atr_tp_atr_stop_targets_and_rr(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    atr: &[f64],
    tick_size: Option<f64>,
    resolve_end_idx: Option<usize>,
    direction: Direction,
) -> (
    Vec<bool>,
    Vec<bool>,
    Vec<f64>,
    Vec<f64>,
    Vec<Option<usize>>,
    Vec<Option<usize>>,
) {
    compute_highlow_or_atr_targets_and_rr_with_stop_mode(
        open,
        high,
        low,
        close,
        atr,
        tick_size,
        resolve_end_idx,
        direction,
        HighlowOrAtrStopMode::AtrOnly,
        HighlowOrAtrTpMode::AtrMultiple(2.0),
        None,
    )
}

fn compute_3x_atr_tp_atr_stop_targets_and_rr(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    atr: &[f64],
    tick_size: Option<f64>,
    resolve_end_idx: Option<usize>,
    direction: Direction,
) -> (
    Vec<bool>,
    Vec<bool>,
    Vec<f64>,
    Vec<f64>,
    Vec<Option<usize>>,
    Vec<Option<usize>>,
) {
    compute_highlow_or_atr_targets_and_rr_with_stop_mode(
        open,
        high,
        low,
        close,
        atr,
        tick_size,
        resolve_end_idx,
        direction,
        HighlowOrAtrStopMode::AtrOnly,
        HighlowOrAtrTpMode::AtrMultiple(3.0),
        None,
    )
}

fn compute_atr_tp_atr_stop_targets_and_rr(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    atr: &[f64],
    tick_size: Option<f64>,
    resolve_end_idx: Option<usize>,
    direction: Direction,
) -> (
    Vec<bool>,
    Vec<bool>,
    Vec<f64>,
    Vec<f64>,
    Vec<Option<usize>>,
    Vec<Option<usize>>,
) {
    compute_highlow_or_atr_targets_and_rr_with_stop_mode(
        open,
        high,
        low,
        close,
        atr,
        tick_size,
        resolve_end_idx,
        direction,
        HighlowOrAtrStopMode::AtrOnly,
        HighlowOrAtrTpMode::AtrMultiple(1.0),
        None,
    )
}

fn compute_highlow_sl_2x_atr_tp_rr_gt_1_targets_and_rr(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    atr: &[f64],
    tick_size: Option<f64>,
    resolve_end_idx: Option<usize>,
    direction: Direction,
) -> (
    Vec<bool>,
    Vec<bool>,
    Vec<f64>,
    Vec<f64>,
    Vec<Option<usize>>,
    Vec<Option<usize>>,
) {
    compute_highlow_or_atr_targets_and_rr_with_stop_mode(
        open,
        high,
        low,
        close,
        atr,
        tick_size,
        resolve_end_idx,
        direction,
        HighlowOrAtrStopMode::HighlowOnly,
        HighlowOrAtrTpMode::AtrMultiple(2.0),
        Some(1.0),
    )
}

fn compute_highlow_sl_1x_atr_tp_rr_gt_1_targets_and_rr(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    atr: &[f64],
    tick_size: Option<f64>,
    resolve_end_idx: Option<usize>,
    direction: Direction,
) -> (
    Vec<bool>,
    Vec<bool>,
    Vec<f64>,
    Vec<f64>,
    Vec<Option<usize>>,
    Vec<Option<usize>>,
) {
    compute_highlow_or_atr_targets_and_rr_with_stop_mode(
        open,
        high,
        low,
        close,
        atr,
        tick_size,
        resolve_end_idx,
        direction,
        HighlowOrAtrStopMode::HighlowOnly,
        HighlowOrAtrTpMode::AtrMultiple(1.0),
        Some(1.0),
    )
}

fn compute_highlow_or_atr_tightest_stop_targets_and_rr(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    atr: &[f64],
    tick_size: Option<f64>,
    resolve_end_idx: Option<usize>,
    direction: Direction,
) -> (
    Vec<bool>,
    Vec<bool>,
    Vec<f64>,
    Vec<f64>,
    Vec<Option<usize>>,
    Vec<Option<usize>>,
) {
    compute_highlow_or_atr_targets_and_rr_with_stop_mode(
        open,
        high,
        low,
        close,
        atr,
        tick_size,
        resolve_end_idx,
        direction,
        HighlowOrAtrStopMode::Tightest,
        HighlowOrAtrTpMode::AtrMultiple(2.0),
        None,
    )
}

fn compute_wicks_kf_targets_and_rr(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    kf_wicks_smooth: &[f64],
    sl_multiplier: f64,
    tick_size: Option<f64>,
    direction: Direction,
) -> (
    Vec<bool>,
    Vec<bool>,
    Vec<f64>,
    Vec<f64>,
    Vec<Option<usize>>,
    Vec<Option<usize>>,
) {
    let len = open
        .len()
        .min(high.len())
        .min(low.len())
        .min(close.len())
        .min(kf_wicks_smooth.len());
    let mut long = vec![false; len];
    let mut short = vec![false; len];
    let mut long_rr = vec![f64::NAN; len];
    let mut short_rr = vec![f64::NAN; len];
    let mut exit_i_long = vec![None; len];
    let mut exit_i_short = vec![None; len];
    if len < 2 {
        return (long, short, long_rr, short_rr, exit_i_long, exit_i_short);
    }

    let want_long = matches!(direction, Direction::Long | Direction::Both);
    let want_short = matches!(direction, Direction::Short | Direction::Both);

    for idx in 0..(len - 1) {
        let next = idx + 1;
        let entry = open[next];
        let high_next = high[next];
        let low_next = low[next];
        let close_next = close[next];
        let wick = kf_wicks_smooth[idx];
        if !entry.is_finite()
            || !high_next.is_finite()
            || !low_next.is_finite()
            || !close_next.is_finite()
            || !wick.is_finite()
        {
            continue;
        }

        let sl_distance_raw = (wick * sl_multiplier).abs();
        let sl_distance = if let Some(ts) = tick_size {
            quantize_distance_to_tick(sl_distance_raw, ts, TickRoundMode::Ceil)
        } else {
            sl_distance_raw
        };
        if sl_distance <= SMALL_DIVISOR {
            continue;
        }

        if want_long {
            let long_sl = entry - sl_distance;
            let long_sl_hit = low_next <= long_sl;
            long[idx] = close_next > entry && !long_sl_hit;
            let long_exit = if long_sl_hit { long_sl } else { close_next };
            long_rr[idx] = (long_exit - entry) / sl_distance;
            exit_i_long[idx] = Some(next);
        }

        if want_short {
            let short_sl = entry + sl_distance;
            let short_sl_hit = high_next >= short_sl;
            short[idx] = close_next < entry && !short_sl_hit;
            let short_exit = if short_sl_hit { short_sl } else { close_next };
            short_rr[idx] = (entry - short_exit) / sl_distance;
            exit_i_short[idx] = Some(next);
        }
    }

    (long, short, long_rr, short_rr, exit_i_long, exit_i_short)
}

struct PriceSeries {
    open: Vec<f64>,
    high: Vec<f64>,
    low: Vec<f64>,
    close: Vec<f64>,
}

impl PriceSeries {
    fn from_frame(frame: &DataFrame) -> Result<Self> {
        Ok(Self {
            open: column_to_vec(frame, "open")?,
            high: column_to_vec(frame, "high")?,
            low: column_to_vec(frame, "low")?,
            close: column_to_vec(frame, "close")?,
        })
    }

    fn len(&self) -> usize {
        self.close.len()
    }
}

struct DerivedMetrics {
    abs_body: Vec<f64>,
    upper_wick: Vec<f64>,
    lower_wick: Vec<f64>,
    max_wick: Vec<f64>,
    body_to_total_wick: Vec<f64>,
    body_atr_ratio: Vec<f64>,
    ema9: Vec<f64>,
    ema20: Vec<f64>,
    ema50: Vec<f64>,
    sma200: Vec<f64>,
    kf_ma: Vec<f64>,
    momentum_14: Vec<f64>,
    momentum_score: Vec<f64>,
    roc5: Vec<f64>,
    roc10: Vec<f64>,
    adx: Vec<f64>,
    atr: Vec<f64>,
    atr_c2c: Vec<f64>,
    atr_pct: Vec<f64>,
    atr_c2c_pct: Vec<f64>,
    bar_range_pct: Vec<f64>,
    volatility_20_cv: Vec<f64>,
    body_size_pct: Vec<f64>,
    upper_shadow_ratio: Vec<f64>,
    lower_shadow_ratio: Vec<f64>,
    wicks_diff: Vec<f64>,
    wicks_diff_sma14: Vec<f64>,
    kf_wicks_smooth: Vec<f64>,
    price_vs_200sma_dev: Vec<f64>,
    price_vs_9ema_dev: Vec<f64>,
    nine_to_two_hundred: Vec<f64>,
    rsi14: Vec<f64>,
    rsi7: Vec<f64>,
    rsi21: Vec<f64>,
    atr_mean50: Vec<f64>,
    atr_c2c_mean50: Vec<f64>,
    macd: Vec<f64>,
    macd_signal: Vec<f64>,
    macd_hist: Vec<f64>,
    stoch_k: Vec<f64>,
    stoch_d: Vec<f64>,
    bb_mid: Vec<f64>,
    bb_upper: Vec<f64>,
    bb_lower: Vec<f64>,
    bb_std: Vec<f64>,
    ext: Vec<f64>,
    ext_sma14: Vec<f64>,
    kf_smooth: Vec<f64>,
    kf_innovation: Vec<f64>,
    kf_close_momentum: Vec<f64>,
    kf_slope_5: Vec<f64>,
    kf_trend: Vec<f64>,
    kf_adx: Vec<f64>,
    kf_atr: Vec<f64>,
    kf_atr_c2c: Vec<f64>,
    kf_adx_slope: Vec<f64>,
    kf_trend_momentum: Vec<f64>,
    kf_trend_volatility_ratio: Vec<f64>,
    kf_price_deviation: Vec<f64>,
    kf_vs_9ema: Vec<f64>,
    kf_vs_200sma: Vec<f64>,
    kf_innovation_abs: Vec<f64>,
    kf_adx_deviation: Vec<f64>,
    kf_adx_innovation_abs: Vec<f64>,
    kf_adx_momentum_5: Vec<f64>,
    kf_atr_pct: Vec<f64>,
    kf_atr_c2c_pct: Vec<f64>,
    kf_atr_vs_c2c: Vec<f64>,
    kf_atr_deviation: Vec<f64>,
    kf_atr_momentum_5: Vec<f64>,
    kf_atr_c2c_momentum_5: Vec<f64>,
    kf_atr_innovation: Vec<f64>,
    kf_atr_c2c_innovation: Vec<f64>,
}

impl DerivedMetrics {
    fn new(prices: &PriceSeries) -> Self {
        let _len = prices.len();
        let body = diff(&prices.close, &prices.open);
        let abs_body = body.iter().map(|v| v.abs()).collect::<Vec<_>>();
        let upper_wick = upper_wick(&prices.open, &prices.close, &prices.high);
        let lower_wick = lower_wick(&prices.open, &prices.close, &prices.low);
        let max_wick = elementwise_max(&upper_wick, &lower_wick);

        let ema9 = ema(&prices.close, 9);
        let ema20 = ema(&prices.close, 20);
        let ema50 = ema(&prices.close, 50);
        let ema200 = ema(&prices.close, 200);
        let sma200 = sma(&prices.close, 200);
        let (kf_smooth, kf_innovation) = kalman_filter(&prices.close, 0.01, 0.1);
        let (kf_trend, _) = kalman_filter(&prices.close, 0.001, 0.5);
        let kf_ma = kf_smooth.clone();
        let kf_close_momentum = derivative(&kf_smooth, 1);
        let kf_slope_5 = derivative(&kf_smooth, 5)
            .into_iter()
            .map(|slope| slope / 5.0)
            .collect::<Vec<_>>();

        let momentum_14 = momentum(&prices.close, 14);
        let roc5 = roc(&prices.close, 5);
        let roc10 = roc(&prices.close, 10);

        let adx = adx(&prices.high, &prices.low, &prices.close, 14);

        let atr = atr(&prices.high, &prices.low, &prices.close, 14);
        let atr_c2c = atr_close_to_close(&prices.close, 14);
        let atr_pct = ratio(&atr, &prices.close);
        let atr_c2c_pct = ratio(&atr_c2c, &prices.close);
        let bar_range = range(&prices.high, &prices.low);
        let bar_range_pct = ratio(&bar_range, &prices.close);
        let volatility_20_cv = rolling_coeff_var(&prices.close, 20);
        let body_size_pct = ratio(&abs_body, &prices.close);
        let upper_shadow_ratio = ratio(&upper_wick, &bar_range);
        let lower_shadow_ratio = ratio(&lower_wick, &bar_range);
        let wicks_diff = prices
            .open
            .iter()
            .zip(prices.close.iter())
            .zip(prices.low.iter())
            .zip(prices.high.iter())
            .map(|(((open, close), low), high)| {
                if close > open {
                    open - low
                } else {
                    high - open
                }
            })
            .collect::<Vec<_>>();
        let wicks_diff_sma14 = sma(&wicks_diff, 14);
        let (kf_wicks_smooth, _) = kalman_filter(&wicks_diff, 0.01, 0.1);
        let total_wick: Vec<f64> = upper_wick
            .iter()
            .zip(lower_wick.iter())
            .map(|(u, l)| u + l)
            .collect();
        let body_to_total_wick = ratio_with_eps(&abs_body, &total_wick, SMALL_DIVISOR);

        let price_vs_200sma_dev = deviation(&prices.close, &sma200);
        let price_vs_9ema_dev = deviation(&prices.close, &ema9);
        let nine_to_two_hundred = deviation(&ema9, &ema200);

        let rsi14 = rsi(&prices.close, 14, 0);
        let rsi7 = rsi(&prices.close, 7, 0);
        let rsi21 = rsi(&prices.close, 21, 0);
        let momentum_score = momentum_score(&rsi14, &roc5, &roc10);
        let atr_mean50 = sma(&atr, 50);
        let atr_c2c_mean50 = sma(&atr_c2c, 50);

        let (macd, macd_signal, macd_hist) = macd(&prices.close, 0);
        let (stoch_k, stoch_d) = stochastic(&prices.close, &prices.high, &prices.low, 14, 3, 0);

        let (bb_mid, bb_upper, bb_lower, bb_std) = bollinger(&prices.close, 20, 2.0, 0);
        let ext = extension(&prices.high, &prices.low, 20);
        let ext_sma14 = sma(&ext, 14);

        let kf_close_minus = diff(&prices.close, &kf_smooth);
        let kf_price_deviation = ratio(&kf_close_minus, &prices.close);
        let kf_vs_9ema = ratio(&diff(&kf_smooth, &ema9), &ema9);
        let kf_vs_200sma = ratio(&diff(&kf_smooth, &sma200), &sma200);
        let kf_innovation_abs = vector_abs(&kf_innovation);

        let (kf_adx, kf_adx_innovation) = kalman_filter(&adx, 0.005, 0.2);
        let kf_adx_slope = derivative(&kf_adx, 1);
        let kf_adx_deviation = ratio_with_eps(
            &diff(&adx, &kf_adx),
            &add_scalar(&kf_adx, SMALL_DIVISOR),
            SMALL_DIVISOR,
        );
        let kf_adx_innovation_abs = vector_abs(&kf_adx_innovation);
        let kf_adx_momentum_5 = derivative(&kf_adx, 5);

        let (kf_atr, kf_atr_innovation) = kalman_filter(&atr, 0.01, 0.15);
        let (kf_atr_c2c, kf_atr_c2c_innovation) = kalman_filter(&atr_c2c, 0.01, 0.15);
        let kf_atr_pct = ratio(&kf_atr, &prices.close);
        let kf_atr_c2c_pct = ratio(&kf_atr_c2c, &prices.close);
        let kf_atr_vs_c2c = ratio_with_eps(
            &kf_atr,
            &add_scalar(&kf_atr_c2c, SMALL_DIVISOR),
            SMALL_DIVISOR,
        );
        let kf_atr_deviation = ratio_with_eps(
            &diff(&atr, &kf_atr),
            &add_scalar(&kf_atr, SMALL_DIVISOR),
            SMALL_DIVISOR,
        );
        let kf_atr_momentum_5 = derivative(&kf_atr, 5);
        let kf_atr_c2c_momentum_5 = derivative(&kf_atr_c2c, 5);
        let kf_trend_momentum = derivative(&kf_trend, 5);
        let denom_vol: Vec<f64> = kf_atr_pct
            .iter()
            .map(|v| v * 100.0 + SMALL_DIVISOR)
            .collect();
        let kf_trend_volatility_ratio = ratio_with_eps(&kf_adx, &denom_vol, SMALL_DIVISOR);
        let body_atr_ratio = ratio_with_eps(&abs_body, &atr, SMALL_DIVISOR);

        Self {
            abs_body,
            upper_wick,
            lower_wick,
            max_wick,
            body_to_total_wick,
            body_atr_ratio,
            ema9,
            ema20,
            ema50,
            sma200,
            kf_ma,
            momentum_14,
            momentum_score,
            roc5,
            roc10,
            adx,
            atr,
            atr_c2c,
            atr_pct,
            atr_c2c_pct,
            bar_range_pct,
            volatility_20_cv,
            body_size_pct,
            upper_shadow_ratio,
            lower_shadow_ratio,
            wicks_diff,
            wicks_diff_sma14,
            kf_wicks_smooth,
            price_vs_200sma_dev,
            price_vs_9ema_dev,
            nine_to_two_hundred,
            rsi14,
            rsi7,
            rsi21,
            atr_mean50,
            atr_c2c_mean50,
            macd,
            macd_signal,
            macd_hist,
            stoch_k,
            stoch_d,
            bb_mid,
            bb_upper,
            bb_lower,
            bb_std,
            ext,
            ext_sma14,
            kf_smooth,
            kf_innovation,
            kf_close_momentum,
            kf_slope_5,
            kf_trend,
            kf_adx,
            kf_atr,
            kf_atr_c2c,
            kf_adx_slope,
            kf_trend_momentum,
            kf_trend_volatility_ratio,
            kf_price_deviation,
            kf_vs_9ema,
            kf_vs_200sma,
            kf_innovation_abs,
            kf_adx_deviation,
            kf_adx_innovation_abs,
            kf_adx_momentum_5,
            kf_atr_pct,
            kf_atr_c2c_pct,
            kf_atr_vs_c2c,
            kf_atr_deviation,
            kf_atr_momentum_5,
            kf_atr_c2c_momentum_5,
            kf_atr_innovation,
            kf_atr_c2c_innovation,
        }
    }
}

fn candle_features(
    prices: &PriceSeries,
    derived: &DerivedMetrics,
    bools: &mut HashMap<&'static str, Vec<bool>>,
    floats: &mut HashMap<&'static str, Vec<f64>>,
) {
    let len = prices.len();
    let mut is_green = vec![false; len];
    let mut is_red = vec![false; len];
    for i in 0..len {
        let body = prices.close[i] - prices.open[i];
        is_green[i] = body > 0.0;
        is_red[i] = body < 0.0;
    }

    bools.insert("is_green", is_green.clone());
    bools.insert("is_red", is_red.clone());

    let custom_high: Vec<f64> = prices
        .open
        .iter()
        .zip(prices.close.iter())
        .map(|(o, c)| o.max(*c))
        .collect();
    let custom_low: Vec<f64> = prices
        .open
        .iter()
        .zip(prices.close.iter())
        .map(|(o, c)| o.min(*c))
        .collect();

    floats.insert("upper_shadow_ratio", derived.upper_shadow_ratio.clone());
    floats.insert("lower_shadow_ratio", derived.lower_shadow_ratio.clone());
    floats.insert("wicks_diff", derived.wicks_diff.clone());
    floats.insert("wicks_diff_sma14", derived.wicks_diff_sma14.clone());
    floats.insert("kf_wicks_smooth", derived.kf_wicks_smooth.clone());
    floats.insert("body_to_total_wick", derived.body_to_total_wick.clone());
    floats.insert("body_atr_ratio", derived.body_atr_ratio.clone());
    let body_vs_max_wick_ratio =
        ratio_with_eps(&derived.abs_body, &derived.max_wick, SMALL_DIVISOR);
    floats.insert("body_vs_max_wick_ratio", body_vs_max_wick_ratio);

    let wick_max = &derived.max_wick;
    let multipliers = [0.5, 1.0, 1.5, 2.0, 2.5];
    for mult in multipliers {
        let key = match mult {
            0.5 => "body_dominant_0_5x",
            1.0 => "body_dominant_1_0x",
            1.5 => "body_dominant_1_5x",
            2.0 => "body_dominant_2_0x",
            _ => "body_dominant_2_5x",
        };
        let mut col = vec![false; len];
        for i in 0..len {
            col[i] = derived.abs_body[i] > wick_max[i] * mult;
        }
        bools.insert(key, col);
    }

    let mut tribar = vec![false; len];
    let mut tribar_green = vec![false; len];
    let mut tribar_red = vec![false; len];
    let mut tribar_hl = vec![false; len];
    let mut tribar_hl_green = vec![false; len];
    let mut tribar_hl_red = vec![false; len];
    for i in 2..len {
        let prev_high_1 = custom_high[i - 1];
        let prev_high_2 = custom_high[i - 2];
        let prev_low_1 = custom_low[i - 1];
        let prev_low_2 = custom_low[i - 2];

        let is_bullish =
            is_green[i] && prices.close[i] > prev_high_1 && prices.close[i] > prev_high_2;
        let is_bearish = is_red[i] && prices.close[i] < prev_low_1 && prices.close[i] < prev_low_2;
        tribar[i] = is_bullish || is_bearish;
        tribar_green[i] = is_bullish;
        tribar_red[i] = is_bearish;

        let bull_hl = is_green[i]
            && prices.close[i] > prices.high[i - 1]
            && prices.close[i] > prices.high[i - 2];
        let bear_hl =
            is_red[i] && prices.close[i] < prices.low[i - 1] && prices.close[i] < prices.low[i - 2];
        tribar_hl[i] = bull_hl || bear_hl;
        tribar_hl_green[i] = bull_hl;
        tribar_hl_red[i] = bear_hl;
    }
    bools.insert("is_tribar", tribar.clone());
    bools.insert("is_tribar_green", tribar_green);
    bools.insert("is_tribar_red", tribar_red);
    bools.insert("is_tribar_hl", tribar_hl.clone());
    bools.insert("is_tribar_hl_green", tribar_hl_green);
    bools.insert("is_tribar_hl_red", tribar_hl_red);

    bools.insert("prev_tribar", shift_bool(&tribar, 1));
    bools.insert("prev_green", shift_bool(&is_green, 1));

    bools.insert("consecutive_green_2", streak(&is_green, 2));
    bools.insert("consecutive_green_3", streak(&is_green, 3));
    bools.insert("consecutive_red_2", streak(&is_red, 2));
    bools.insert("consecutive_red_3", streak(&is_red, 3));

    bools.insert(
        "higher_high",
        comparison(&prices.high, 1, Comparison::Greater),
    );
    bools.insert(
        "higher_low",
        comparison(&prices.low, 1, Comparison::Greater),
    );
    bools.insert("lower_high", comparison(&prices.high, 1, Comparison::Less));
    bools.insert("lower_low", comparison(&prices.low, 1, Comparison::Less));

    bools.insert("bullish_bar_sequence", streak(&is_green, 3));
    bools.insert("bearish_bar_sequence", streak(&is_red, 3));

    bools.insert(
        "is_hammer",
        hammer(&derived.abs_body, &derived.upper_wick, &derived.lower_wick),
    );
    bools.insert(
        "is_shooting_star",
        shooting_star(&derived.abs_body, &derived.upper_wick, &derived.lower_wick),
    );
    bools.insert(
        "bullish_engulfing",
        engulfing(&is_green, &is_red, &derived.abs_body, true),
    );
    bools.insert(
        "bearish_engulfing",
        engulfing(&is_green, &is_red, &derived.abs_body, false),
    );

    bools.insert(
        "is_very_large_green",
        large_colored_body(&is_green, &derived.abs_body, &derived.atr, 1.5),
    );
    bools.insert(
        "is_very_large_red",
        large_colored_body(&is_red, &derived.abs_body, &derived.atr, 1.5),
    );

    let body_pct_mean20 = sma(&derived.body_size_pct, 20);
    bools.insert(
        "is_large_body",
        large_body_ratio(&derived.body_size_pct, &body_pct_mean20, 1.5),
    );
    bools.insert(
        "is_very_large_body",
        large_body_ratio(&derived.body_size_pct, &body_pct_mean20, 2.0),
    );
}

fn ema_price_features(
    prices: &PriceSeries,
    derived: &DerivedMetrics,
    bools: &mut HashMap<&'static str, Vec<bool>>,
    floats: &mut HashMap<&'static str, Vec<f64>>,
) {
    floats.insert("9ema", derived.ema9.clone());
    floats.insert("20ema", derived.ema20.clone());
    floats.insert("50ema", derived.ema50.clone());
    floats.insert("200sma", derived.sma200.clone());
    floats.insert("rsi_14", derived.rsi14.clone());
    floats.insert("rsi_7", derived.rsi7.clone());
    floats.insert("rsi_21", derived.rsi21.clone());
    floats.insert("momentum_14", derived.momentum_14.clone());
    floats.insert("momentum_score", derived.momentum_score.clone());
    floats.insert("roc_5", derived.roc5.clone());
    floats.insert("roc_10", derived.roc10.clone());
    floats.insert("adx", derived.adx.clone());
    let trend_strength: Vec<f64> = prices
        .close
        .iter()
        .zip(derived.sma200.iter().zip(derived.adx.iter()))
        .map(|(price, (sma200, adx))| {
            if !price.is_finite()
                || !sma200.is_finite()
                || !adx.is_finite()
                || sma200.abs() < f64::EPSILON
            {
                f64::NAN
            } else {
                let adx_term = (adx / 100.0) * 0.4;
                let deviation_term = ((price / sma200) - 1.0).abs() * 10.0 * 0.6;
                adx_term + deviation_term
            }
        })
        .collect();
    floats.insert("trend_strength", trend_strength.clone());
    if let Some(threshold) = quantile(&trend_strength, 0.8) {
        let strong_trend_flags: Vec<bool> = trend_strength
            .iter()
            .map(|value| value.is_finite() && *value > threshold)
            .collect();
        bools.insert("is_very_strong_trend", strong_trend_flags);
    } else {
        bools.insert("is_very_strong_trend", vec![false; trend_strength.len()]);
    }
    floats.insert("adx_sma", sma(&derived.adx, 14));

    bools.insert(
        "all_emas_aligned",
        ema_alignment(&prices.close, &derived.ema9, &derived.ema20, &derived.ema50),
    );
    bools.insert(
        "all_emas_dealigned",
        ema_alignment(
            &prices.close.iter().map(|v| -v).collect::<Vec<_>>(),
            &derived.ema9.iter().map(|v| -v).collect::<Vec<_>>(),
            &derived.ema20.iter().map(|v| -v).collect::<Vec<_>>(),
            &derived.ema50.iter().map(|v| -v).collect::<Vec<_>>(),
        ),
    );
    bools.insert(
        "ema_ribbon_aligned",
        ribbon_alignment(
            &derived.ema9,
            &derived.ema20,
            &derived.ema50,
            &derived.sma200,
            true,
        ),
    );
    bools.insert(
        "ema_ribbon_dealigned",
        ribbon_alignment(
            &derived.ema9,
            &derived.ema20,
            &derived.ema50,
            &derived.sma200,
            false,
        ),
    );

    bools.insert(
        "is_close_above_200sma",
        compare_series(&prices.close, &derived.sma200, Comparison::Greater),
    );
    bools.insert(
        "is_close_below_200sma",
        compare_series(&prices.close, &derived.sma200, Comparison::Less),
    );
    bools.insert(
        "is_close_above_9ema",
        compare_series(&prices.close, &derived.ema9, Comparison::Greater),
    );
    bools.insert(
        "is_close_below_9ema",
        compare_series(&prices.close, &derived.ema9, Comparison::Less),
    );
    bools.insert(
        "is_close_above_kf_ma",
        compare_series(&prices.close, &derived.kf_ma, Comparison::Greater),
    );
    bools.insert(
        "is_close_below_kf_ma",
        compare_series(&prices.close, &derived.kf_ma, Comparison::Less),
    );

    floats.insert("kf_smooth", derived.kf_smooth.clone());
    floats.insert("kf_vs_9ema", derived.kf_vs_9ema.clone());
    floats.insert("kf_vs_200sma", derived.kf_vs_200sma.clone());
    floats.insert("kf_price_deviation", derived.kf_price_deviation.clone());
    floats.insert("kf_innovation_abs", derived.kf_innovation_abs.clone());
    floats.insert("kf_innovation", derived.kf_innovation.clone());
    floats.insert(
        "kf_adx_innovation_abs",
        derived.kf_adx_innovation_abs.clone(),
    );
    // Export the remaining Kalman/ADX continuous features so every entry
    // listed in CONTINUOUS_FEATURES is present in the engineered dataset.
    floats.insert("kf_adx", derived.kf_adx.clone());
    floats.insert("kf_trend_momentum", derived.kf_trend_momentum.clone());
    floats.insert(
        "kf_trend_volatility_ratio",
        derived.kf_trend_volatility_ratio.clone(),
    );
    floats.insert("kf_adx_deviation", derived.kf_adx_deviation.clone());
    floats.insert("kf_adx_momentum_5", derived.kf_adx_momentum_5.clone());

    floats.insert("price_vs_200sma_dev", derived.price_vs_200sma_dev.clone());
    floats.insert("price_vs_9ema_dev", derived.price_vs_9ema_dev.clone());
    floats.insert("9ema_to_200sma", derived.nine_to_two_hundred.clone());
}

fn volatility_features(
    _prices: &PriceSeries,
    derived: &DerivedMetrics,
    bools: &mut HashMap<&'static str, Vec<bool>>,
    floats: &mut HashMap<&'static str, Vec<f64>>,
) {
    // NOTE: `atr` is exported primarily for target generation (e.g., highlow_or_atr).
    // It is intentionally NOT part of the core NaN-drop indicator set so it doesn't
    // change the warmup trimming mask.
    floats.insert("atr", derived.atr.clone());
    floats.insert("kf_atr", derived.kf_atr.clone());
    floats.insert("kf_atr_c2c", derived.kf_atr_c2c.clone());
    floats.insert("kf_atr_pct", derived.kf_atr_pct.clone());
    floats.insert("kf_atr_c2c_pct", derived.kf_atr_c2c_pct.clone());
    floats.insert("atr_c2c", derived.atr_c2c.clone());
    floats.insert("kf_atr_vs_c2c", derived.kf_atr_vs_c2c.clone());
    floats.insert("kf_atr_deviation", derived.kf_atr_deviation.clone());
    floats.insert("kf_atr_momentum_5", derived.kf_atr_momentum_5.clone());
    floats.insert(
        "kf_atr_c2c_momentum_5",
        derived.kf_atr_c2c_momentum_5.clone(),
    );
    floats.insert("kf_atr_innovation", derived.kf_atr_innovation.clone());
    floats.insert(
        "kf_atr_c2c_innovation",
        derived.kf_atr_c2c_innovation.clone(),
    );
    floats.insert("atr_pct", derived.atr_pct.clone());
    floats.insert("atr_c2c_pct", derived.atr_c2c_pct.clone());
    floats.insert("bar_range_pct", derived.bar_range_pct.clone());
    floats.insert("volatility_20_cv", derived.volatility_20_cv.clone());
    floats.insert("body_size_pct", derived.body_size_pct.clone());

    let atr_mean20 = sma(&derived.atr, 20);
    let kf_atr_mean20 = sma(&derived.kf_atr, 20);
    let kf_atr_std20 = rolling_std(&derived.kf_atr, 20);
    let kf_atr_c2c_mean20 = sma(&derived.kf_atr_c2c, 20);
    let kf_atr_c2c_std20 = rolling_std(&derived.kf_atr_c2c, 20);

    bools.insert(
        "is_high_volatility",
        threshold_compare(&derived.atr, &atr_mean20, 1.2, Comparison::Greater),
    );
    bools.insert(
        "is_low_volatility",
        threshold_compare(&derived.atr, &atr_mean20, 0.8, Comparison::Less),
    );
    bools.insert(
        "expanding_atr",
        comparison(&derived.atr, 1, Comparison::Greater),
    );
    bools.insert(
        "is_kf_atr_high_volatility",
        threshold_compare(&derived.kf_atr, &kf_atr_mean20, 1.2, Comparison::Greater),
    );
    bools.insert(
        "is_kf_atr_low_volatility",
        threshold_compare(&derived.kf_atr, &kf_atr_mean20, 0.8, Comparison::Less),
    );
    bools.insert("is_kf_atr_squeeze", squeeze(&derived.kf_atr_pct, 50, 1.1));
    bools.insert(
        "is_kf_atr_very_high_volatility",
        zscore_compare(&derived.kf_atr, &kf_atr_mean20, &kf_atr_std20, 2.0),
    );
    let atr_innovation_std20 = rolling_std(&derived.kf_atr_innovation, 20);
    let atr_innovation_spike: Vec<f64> = atr_innovation_std20.iter().map(|std| std * 2.0).collect();
    let atr_innovation_drop: Vec<f64> = atr_innovation_spike.iter().map(|value| -*value).collect();
    bools.insert(
        "is_kf_atr_volatility_spike",
        compare_series(
            &derived.kf_atr_innovation,
            &atr_innovation_spike,
            Comparison::Greater,
        ),
    );
    bools.insert(
        "is_kf_atr_volatility_drop",
        compare_series(
            &derived.kf_atr_innovation,
            &atr_innovation_drop,
            Comparison::Less,
        ),
    );
    bools.insert(
        "is_kf_atr_c2c_high_volatility",
        threshold_compare(
            &derived.kf_atr_c2c,
            &kf_atr_c2c_mean20,
            1.2,
            Comparison::Greater,
        ),
    );
    bools.insert(
        "is_kf_atr_c2c_low_volatility",
        threshold_compare(
            &derived.kf_atr_c2c,
            &kf_atr_c2c_mean20,
            0.8,
            Comparison::Less,
        ),
    );
    bools.insert(
        "is_kf_atr_c2c_squeeze",
        squeeze(&derived.kf_atr_c2c, 50, 1.1),
    );
    bools.insert(
        "is_kf_atr_c2c_very_high",
        zscore_compare(
            &derived.kf_atr_c2c,
            &kf_atr_c2c_mean20,
            &kf_atr_c2c_std20,
            2.0,
        ),
    );
    let atr_c2c_innovation_std20 = rolling_std(&derived.kf_atr_c2c_innovation, 20);
    let atr_c2c_innovation_spike: Vec<f64> = atr_c2c_innovation_std20
        .iter()
        .map(|std| std * 2.0)
        .collect();
    let atr_c2c_innovation_drop: Vec<f64> = atr_c2c_innovation_spike
        .iter()
        .map(|value| -*value)
        .collect();
    bools.insert(
        "is_kf_atr_c2c_spike",
        compare_series(
            &derived.kf_atr_c2c_innovation,
            &atr_c2c_innovation_spike,
            Comparison::Greater,
        ),
    );
    bools.insert(
        "is_kf_atr_c2c_drop",
        compare_series(
            &derived.kf_atr_c2c_innovation,
            &atr_c2c_innovation_drop,
            Comparison::Less,
        ),
    );

    bools.insert(
        "kf_atr_c2c_contracting",
        derivative_threshold(&derived.kf_atr_c2c, 0.0, Comparison::Less),
    );
    bools.insert(
        "kf_atr_c2c_expanding",
        derivative_threshold(&derived.kf_atr_c2c, 0.0, Comparison::Greater),
    );
    bools.insert(
        "kf_atr_contracting",
        derivative_threshold(&derived.kf_atr, 0.0, Comparison::Less),
    );
    bools.insert(
        "kf_atr_expanding",
        derivative_threshold(&derived.kf_atr, 0.0, Comparison::Greater),
    );
    bools.insert(
        "is_kf_gap_volatility",
        threshold(&derived.kf_atr_vs_c2c, 1.5, Comparison::Greater),
    );
    bools.insert(
        "is_kf_continuous_volatility",
        threshold(&derived.kf_atr_vs_c2c, 1.2, Comparison::Less),
    );

    floats.insert("atr_pct_mean50", derived.atr_mean50.clone());
    floats.insert("atr_c2c_mean50", derived.atr_c2c_mean50.clone());
}

fn oscillator_features(
    _prices: &PriceSeries,
    derived: &DerivedMetrics,
    bools: &mut HashMap<&'static str, Vec<bool>>,
    floats: &mut HashMap<&'static str, Vec<f64>>,
) {
    const STOCH_PRECISION: u32 = 15;
    let stoch_k_raw = derived.stoch_k.clone();
    let stoch_d_raw = derived.stoch_d.clone();
    let stoch_k_export: Vec<f64> = stoch_k_raw
        .iter()
        .map(|value| round_to_decimals(*value, STOCH_PRECISION))
        .collect();
    let stoch_d_export: Vec<f64> = stoch_d_raw
        .iter()
        .map(|value| round_to_decimals(*value, STOCH_PRECISION))
        .collect();
    let stoch_k_logic = stoch_k_raw.clone();
    let stoch_d_logic = stoch_d_raw.clone();
    floats.insert("stoch_k", stoch_k_export);
    floats.insert("stoch_d", stoch_d_export);
    bools.insert(
        "is_rsi_oversold_recovery",
        recovery(&derived.rsi14, 30.0, Comparison::Greater),
    );
    bools.insert(
        "is_rsi_overbought_recovery",
        recovery(&derived.rsi14, 70.0, Comparison::Less),
    );
    bools.insert(
        "rsi_bullish",
        threshold(&derived.rsi14, 50.0, Comparison::Greater),
    );
    bools.insert(
        "rsi_bearish",
        threshold(&derived.rsi14, 50.0, Comparison::Less),
    );
    bools.insert(
        "rsi_very_bullish",
        threshold(&derived.rsi14, 60.0, Comparison::Greater),
    );
    bools.insert(
        "rsi_very_bearish",
        threshold(&derived.rsi14, 40.0, Comparison::Less),
    );

    bools.insert(
        "is_stoch_oversold",
        stoch_threshold(&stoch_k_logic, 20.0, Comparison::Less),
    );
    bools.insert(
        "is_stoch_overbought",
        stoch_threshold(&stoch_k_logic, 80.0, Comparison::Greater),
    );
    // NOTE: stoch_bullish_cross still drifts on a few bars relative to the pandas
    // reference (see README testing notes); keep this case in mind when asserting
    // parity until the Typer implementation is ported line-for-line.
    bools.insert(
        "stoch_bullish_cross",
        stoch_cross(
            &stoch_k_logic,
            &stoch_d_logic,
            &stoch_k_raw,
            &stoch_d_raw,
            true,
        ),
    );
    bools.insert(
        "stoch_bearish_cross",
        stoch_cross(
            &stoch_k_logic,
            &stoch_d_logic,
            &stoch_k_raw,
            &stoch_d_raw,
            false,
        ),
    );

    bools.insert(
        "is_strong_momentum_score",
        threshold(&derived.momentum_score, 0.75, Comparison::Greater),
    );
    bools.insert(
        "is_weak_momentum_score",
        threshold(&derived.momentum_score, 0.25, Comparison::Less),
    );
}

fn macd_features(
    derived: &DerivedMetrics,
    bools: &mut HashMap<&'static str, Vec<bool>>,
    floats: &mut HashMap<&'static str, Vec<f64>>,
) {
    floats.insert("macd", derived.macd.clone());
    floats.insert("macd_signal", derived.macd_signal.clone());
    floats.insert("macd_hist", derived.macd_hist.clone());
    let macd_hist_delta_1 = derivative(&derived.macd_hist, 1);
    floats.insert("macd_hist_delta_1", macd_hist_delta_1);

    bools.insert(
        "macd_bullish",
        compare_series(&derived.macd, &derived.macd_signal, Comparison::Greater),
    );
    bools.insert(
        "macd_bearish",
        compare_series(&derived.macd, &derived.macd_signal, Comparison::Less),
    );
    bools.insert(
        "macd_cross_up",
        cross(&derived.macd, &derived.macd_signal, true),
    );
    bools.insert(
        "macd_cross_down",
        cross(&derived.macd, &derived.macd_signal, false),
    );
    bools.insert(
        "macd_histogram_increasing",
        derivative_threshold(&derived.macd_hist, 0.0, Comparison::Greater),
    );
    bools.insert(
        "macd_histogram_decreasing",
        derivative_threshold(&derived.macd_hist, 0.0, Comparison::Less),
    );
}

fn bollinger_features(
    prices: &PriceSeries,
    derived: &DerivedMetrics,
    bools: &mut HashMap<&'static str, Vec<bool>>,
    floats: &mut HashMap<&'static str, Vec<f64>>,
) {
    floats.insert(
        "bb_position",
        bollinger_position(&prices.close, &derived.bb_lower, &derived.bb_upper),
    );
    floats.insert("bb_std", derived.bb_std.clone());

    let bb_std_sma20 = sma_strict(&derived.bb_std, 20);
    let squeeze_flags: Vec<bool> = derived
        .bb_std
        .iter()
        .zip(bb_std_sma20.iter())
        .map(|(std, mean)| {
            if !std.is_finite() || !mean.is_finite() {
                false
            } else {
                *std < (*mean * 0.8)
            }
        })
        .collect();
    bools.insert("is_bb_squeeze", squeeze_flags);
    bools.insert(
        "above_bb_middle",
        compare_series(&prices.close, &derived.bb_mid, Comparison::Greater),
    );
    bools.insert(
        "below_bb_middle",
        compare_series(&prices.close, &derived.bb_mid, Comparison::Less),
    );

    floats.insert("ext", derived.ext.clone());
    floats.insert("ext_sma14", derived.ext_sma14.clone());
}

fn trend_state_features(derived: &DerivedMetrics, bools: &mut HashMap<&'static str, Vec<bool>>) {
    bools.insert(
        "adx_rising",
        derivative_threshold(&derived.adx, 0.0, Comparison::Greater),
    );
    bools.insert("adx_accelerating", double_rising(&derived.adx));

    bools.insert(
        "higher_high",
        comparison_series(&derived.adx, 1, Comparison::Greater),
    );
}

fn kalman_features(
    prices: &PriceSeries,
    derived: &DerivedMetrics,
    bools: &mut HashMap<&'static str, Vec<bool>>,
) {
    let len = derived.kf_smooth.len();
    let kf_atr_mean20 = sma(&derived.kf_atr, 20);
    let kf_atr_std20 = rolling_std(&derived.kf_atr, 20);
    let kf_innovation_mean20 = sma(&derived.kf_innovation_abs, 20);
    let kf_innovation_std20 = rolling_std(&derived.kf_innovation, 20);
    let atr_expanding = bools
        .get("kf_atr_expanding")
        .cloned()
        .unwrap_or_else(|| vec![false; len]);
    let atr_contracting = bools
        .get("kf_atr_contracting")
        .cloned()
        .unwrap_or_else(|| vec![false; len]);
    let atr_c2c_expanding = bools
        .get("kf_atr_c2c_expanding")
        .cloned()
        .unwrap_or_else(|| vec![false; len]);
    let atr_c2c_contracting = bools
        .get("kf_atr_c2c_contracting")
        .cloned()
        .unwrap_or_else(|| vec![false; len]);
    let gap_volatility = bools
        .get("is_kf_gap_volatility")
        .cloned()
        .unwrap_or_else(|| vec![false; len]);
    let continuous_volatility = bools
        .get("is_kf_continuous_volatility")
        .cloned()
        .unwrap_or_else(|| vec![false; len]);
    let atr_c2c_high_flags = bools
        .get("is_kf_atr_c2c_high_volatility")
        .cloned()
        .unwrap_or_else(|| vec![false; len]);
    let atr_c2c_low_flags = bools
        .get("is_kf_atr_c2c_low_volatility")
        .cloned()
        .unwrap_or_else(|| vec![false; len]);

    bools.insert(
        "kf_above_smooth",
        threshold(&derived.kf_price_deviation, 0.0, Comparison::Greater),
    );
    bools.insert(
        "kf_above_trend",
        compare_series(&prices.close, &derived.kf_trend, Comparison::Greater),
    );
    bools.insert(
        "kf_below_smooth",
        threshold(&derived.kf_price_deviation, 0.0, Comparison::Less),
    );
    bools.insert(
        "kf_below_trend",
        compare_series(&prices.close, &derived.kf_trend, Comparison::Less),
    );

    let kf_adx_above_25 = threshold(&derived.kf_adx, 25.0, Comparison::Greater);
    let kf_adx_above_40 = threshold(&derived.kf_adx, 40.0, Comparison::Greater);
    let kf_adx_below_20 = threshold(&derived.kf_adx, 20.0, Comparison::Less);
    let kf_adx_below_25 = threshold(&derived.kf_adx, 25.0, Comparison::Less);
    let kf_adx_increasing = comparison_series(&derived.kf_adx, 1, Comparison::Greater);
    let kf_adx_decreasing = comparison_series(&derived.kf_adx, 1, Comparison::Less);

    bools.insert("kf_adx_above_25", kf_adx_above_25.clone());
    bools.insert("kf_adx_above_40", kf_adx_above_40.clone());
    bools.insert("kf_adx_below_20", kf_adx_below_20.clone());
    bools.insert("kf_adx_increasing", kf_adx_increasing.clone());
    bools.insert("kf_adx_decreasing", kf_adx_decreasing.clone());
    bools.insert(
        "kf_adx_accelerating",
        momentum_acceleration(&derived.kf_adx_slope, Comparison::Greater),
    );
    bools.insert(
        "kf_adx_decelerating",
        momentum_acceleration(&derived.kf_adx_slope, Comparison::Less),
    );
    let kf_adx_innovation_mean20 = sma(&derived.kf_adx_innovation_abs, 20);
    let kf_adx_surprise_threshold: Vec<f64> = kf_adx_innovation_mean20
        .iter()
        .map(|mean| mean * 1.5)
        .collect();
    bools.insert(
        "is_kf_adx_surprise",
        compare_series(
            &derived.kf_adx_innovation_abs,
            &kf_adx_surprise_threshold,
            Comparison::Greater,
        ),
    );
    let mut adx_trend_emerging = vec![false; len];
    let mut adx_trend_fading = vec![false; len];
    for i in 5..len {
        if derived.kf_adx[i].is_finite() && derived.kf_adx[i - 5].is_finite() {
            adx_trend_emerging[i] = derived.kf_adx[i] > 20.0 && derived.kf_adx[i - 5] < 20.0;
            adx_trend_fading[i] = derived.kf_adx[i] < 25.0 && derived.kf_adx[i - 5] > 30.0;
        }
    }
    bools.insert("kf_adx_trend_emerging", adx_trend_emerging);
    bools.insert("kf_adx_trend_fading", adx_trend_fading);

    let atr_high_flags =
        threshold_compare(&derived.kf_atr, &kf_atr_mean20, 1.2, Comparison::Greater);
    let atr_low_flags = threshold_compare(&derived.kf_atr, &kf_atr_mean20, 0.8, Comparison::Less);

    bools.insert("is_kf_atr_high_volatility", atr_high_flags.clone());
    bools.insert("is_kf_atr_low_volatility", atr_low_flags.clone());

    let atr_very_high_flags: Vec<bool> = derived
        .kf_atr
        .iter()
        .zip(kf_atr_mean20.iter().zip(kf_atr_std20.iter()))
        .map(|(atr, (mean, std))| {
            atr.is_finite() && mean.is_finite() && std.is_finite() && *atr > *mean + 2.0 * *std
        })
        .collect();
    bools.insert("is_kf_atr_very_high_volatility", atr_very_high_flags);

    let mut atr_squeeze_flags = vec![false; len];
    for i in 0..len {
        // Require a full 50-bar history for the squeeze window; rely on the
        // indicator's own NaN warmup rather than an external BASE_WARMUP_ROWS
        // offset.
        if i + 1 < 50 {
            continue;
        }
        let start = i + 1 - 50;
        let mut window_min = f64::INFINITY;
        for &value in &derived.kf_atr[start..=i] {
            if value.is_finite() {
                window_min = window_min.min(value);
            }
        }
        if window_min.is_finite()
            && derived.kf_atr[i].is_finite()
            && derived.kf_atr[i] < window_min * 1.1
        {
            atr_squeeze_flags[i] = true;
        }
    }
    bools.insert("is_kf_atr_squeeze", atr_squeeze_flags);

    bools.insert(
        "kf_c2c_dominance",
        compare_series(
            &derived.kf_atr_c2c,
            &derived
                .kf_atr
                .iter()
                .map(|value| value * 0.7)
                .collect::<Vec<_>>(),
            Comparison::Greater,
        ),
    );
    let divergence_flags: Vec<bool> = atr_expanding
        .iter()
        .zip(atr_c2c_contracting.iter())
        .zip(atr_contracting.iter().zip(atr_c2c_expanding.iter()))
        .map(|((atr_up, c2c_down), (atr_down, c2c_up))| {
            (*atr_up && *c2c_down) || (*atr_down && *c2c_up)
        })
        .collect();
    bools.insert("kf_volatility_divergence", divergence_flags);
    bools.insert(
        "kf_momentum_increasing",
        comparison(&derived.kf_close_momentum, 1, Comparison::Greater),
    );
    bools.insert(
        "kf_momentum_decreasing",
        comparison(&derived.kf_close_momentum, 1, Comparison::Less),
    );
    bools.insert(
        "kf_slope_increasing",
        comparison(&derived.kf_slope_5, 1, Comparison::Greater),
    );
    bools.insert(
        "kf_slope_decreasing",
        comparison(&derived.kf_slope_5, 1, Comparison::Less),
    );

    bools.insert(
        "kf_trending_volatile",
        dual_condition(&kf_adx_above_25, &atr_high_flags),
    );
    bools.insert(
        "kf_trending_quiet",
        dual_condition(&kf_adx_above_25, &atr_low_flags),
    );
    bools.insert(
        "kf_ranging_volatile",
        dual_condition(&kf_adx_below_20, &atr_high_flags),
    );
    bools.insert(
        "kf_ranging_quiet",
        dual_condition(&kf_adx_below_20, &atr_low_flags),
    );
    bools.insert(
        "is_kf_strong_trend_low_vol",
        dual_condition(
            &threshold(&derived.kf_adx, 35.0, Comparison::Greater),
            &threshold(&derived.kf_atr_pct, 0.012, Comparison::Less),
        ),
    );
    bools.insert(
        "is_kf_breakout_potential",
        dual_condition(&kf_adx_increasing, &atr_expanding),
    );
    bools.insert(
        "is_kf_consolidation",
        dual_condition(&kf_adx_decreasing, &atr_contracting),
    );
    bools.insert(
        "kf_trending_c2c_volatile",
        dual_condition(&kf_adx_above_25, &atr_c2c_high_flags),
    );
    bools.insert(
        "kf_trending_c2c_quiet",
        dual_condition(&kf_adx_above_25, &atr_c2c_low_flags),
    );
    let gap_opportunity = dual_condition(&gap_volatility, &kf_adx_below_25);
    bools.insert("is_kf_gap_trading_opportunity", gap_opportunity);
    bools.insert(
        "is_kf_smooth_trend",
        dual_condition(
            &continuous_volatility,
            &threshold(&derived.kf_adx, 30.0, Comparison::Greater),
        ),
    );
    let kf_close_above_ema9 =
        compare_series(&derived.kf_smooth, &derived.ema9, Comparison::Greater);
    let ema9_above_200 = compare_series(&derived.ema9, &derived.sma200, Comparison::Greater);
    bools.insert(
        "kf_ema_aligned",
        dual_condition(&kf_close_above_ema9, &ema9_above_200),
    );
    bools.insert(
        "kf_ema_divergence",
        dual_condition(
            &compare_series(&derived.kf_smooth, &derived.kf_trend, Comparison::Less),
            &compare_series(&derived.kf_trend, &derived.ema50, Comparison::Less),
        ),
    );
    bools.insert(
        "is_kf_positive_surprise",
        compare_series(
            &derived.kf_innovation,
            &kf_innovation_std20,
            Comparison::Greater,
        ),
    );
    let neg_innovation_std: Vec<f64> = kf_innovation_std20.iter().map(|std| -*std).collect();
    bools.insert(
        "is_kf_negative_surprise",
        compare_series(
            &derived.kf_innovation,
            &neg_innovation_std,
            Comparison::Less,
        ),
    );
    let innovation_large_threshold: Vec<f64> =
        kf_innovation_mean20.iter().map(|mean| mean * 1.5).collect();
    bools.insert(
        "is_kf_innovation_large",
        compare_series(
            &derived.kf_innovation_abs,
            &innovation_large_threshold,
            Comparison::Greater,
        ),
    );
}

fn column_to_vec(frame: &DataFrame, name: &str) -> Result<Vec<f64>> {
    let series = frame
        .column(name)
        .with_context(|| format!("Missing required column {name}"))?;
    let chunk = series
        .f64()
        .with_context(|| format!("Column {name} must be float"))?;
    chunk
        .into_iter()
        .map(|opt| opt.ok_or_else(|| anyhow!("Column {name} contains nulls")))
        .collect()
}

fn column_with_nans(frame: &DataFrame, name: &str) -> Result<Vec<f64>> {
    let series = frame
        .column(name)
        .with_context(|| format!("Missing required column {name}"))?;
    Ok(series
        .f64()
        .with_context(|| format!("Column {name} must be float"))?
        .into_iter()
        .map(|value| value.unwrap_or(f64::NAN))
        .collect())
}

fn bool_column(frame: &DataFrame, name: &str) -> Result<Vec<bool>> {
    let series = frame
        .column(name)
        .with_context(|| format!("Missing required column {name}"))?;
    Ok(series
        .bool()
        .with_context(|| format!("Column {name} should be boolean"))?
        .into_iter()
        .map(|value| value.unwrap_or(false))
        .collect())
}

fn timestamp_column(frame: &DataFrame) -> Result<Vec<DateTime<Utc>>> {
    // Prefer the canonical "timestamp" column; fall back to "datetime" or
    // "time" when present to support lean CSVs used for backtesting.
    let series = frame
        .column("timestamp")
        .or_else(|_| frame.column("datetime"))
        .or_else(|_| frame.column("time"))
        .with_context(|| {
            "Missing required timestamp/datetime column (expected 'timestamp', 'datetime', or 'time')"
        })?;

    let mut out = Vec::with_capacity(series.len());
    for value in series.iter() {
        use polars::prelude::AnyValue;

        let raw = match value {
            AnyValue::String(s) => s,
            AnyValue::StringOwned(ref s) => s.as_str(),
            AnyValue::Null => return Err(anyhow!("Timestamp column contains nulls")),
            other => {
                return Err(anyhow!(
                    "Timestamp column must be UTF-8 strings (got {:?})",
                    other.dtype()
                ));
            }
        };
        // Accept RFC3339-style strings like 2024-01-01T00:00:00Z.
        let parsed = DateTime::parse_from_rfc3339(raw)
            .with_context(|| format!("Failed to parse timestamp '{raw}' as RFC3339"))?;
        out.push(parsed.with_timezone(&Utc));
    }
    Ok(out)
}

fn compute_week_indices(timestamps: &[DateTime<Utc>]) -> (Vec<i64>, Vec<usize>) {
    let len = timestamps.len();
    if len == 0 {
        return (Vec::new(), Vec::new());
    }

    // Anchor Sunday 22:00 UTC, mirroring the weekly rolling helpers in Python.
    let anchor_date = NaiveDate::from_ymd_opt(1970, 1, 4).expect("valid date");
    let anchor_time = NaiveTime::from_hms_opt(22, 0, 0).expect("valid time");
    let anchor = DateTime::<Utc>::from_naive_utc_and_offset(anchor_date.and_time(anchor_time), Utc);

    let week_secs = 7 * 24 * 60 * 60;
    let mut week_index = Vec::with_capacity(len);
    for ts in timestamps {
        let diff = ts.signed_duration_since(anchor);
        let secs = diff.num_seconds().max(0);
        week_index.push(secs / week_secs);
    }

    // For each bar, record the last index in its week.
    let mut week_end_index = vec![0usize; len];
    let mut current_week = week_index[len - 1];
    let mut current_end = len - 1;
    let mut i = len;
    while i > 0 {
        i -= 1;
        if week_index[i] != current_week {
            current_week = week_index[i];
            current_end = i;
        }
        week_end_index[i] = current_end;
    }

    (week_index, week_end_index)
}

fn compute_month_indices(timestamps: &[DateTime<Utc>]) -> (Vec<i64>, Vec<usize>) {
    let len = timestamps.len();
    if len == 0 {
        return (Vec::new(), Vec::new());
    }

    let mut month_index = Vec::with_capacity(len);
    for ts in timestamps {
        let year = ts.year() as i64;
        let month = ts.month() as i64; // 1..=12
        month_index.push(year * 12 + (month - 1));
    }

    let mut month_end_index = vec![0usize; len];
    let mut current_month = month_index[len - 1];
    let mut current_end = len - 1;
    let mut i = len;
    while i > 0 {
        i -= 1;
        if month_index[i] != current_month {
            current_month = month_index[i];
            current_end = i;
        }
        month_end_index[i] = current_end;
    }

    (month_index, month_end_index)
}

fn period_indices_for_target(
    target: BacktestTargetKind,
    timestamps: &[DateTime<Utc>],
) -> (Vec<i64>, Vec<usize>, f64, u8) {
    match target {
        BacktestTargetKind::TribarWeekly => {
            let (idx, end) = compute_week_indices(timestamps);
            (idx, end, 2.0, 2)
        }
        BacktestTargetKind::TribarMonthly => {
            let (idx, end) = compute_month_indices(timestamps);
            (idx, end, 2.0, 2)
        }
    }
}

fn round_to_decimals(value: f64, decimals: u32) -> f64 {
    if !value.is_finite() {
        return value;
    }
    let factor = 10f64.powi(decimals as i32);
    (value * factor).round() / factor
}

fn diff(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(x, y)| x - y).collect()
}

fn elementwise_max(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(x, y)| x.max(*y)).collect()
}

fn upper_wick(open: &[f64], close: &[f64], high: &[f64]) -> Vec<f64> {
    high.iter()
        .zip(open.iter().zip(close.iter()))
        .map(|(h, (o, c))| h - o.max(*c))
        .collect()
}

fn lower_wick(open: &[f64], close: &[f64], low: &[f64]) -> Vec<f64> {
    open.iter()
        .zip(close.iter().zip(low.iter()))
        .map(|(o, (c, l))| o.min(*c) - l)
        .collect()
}

fn ema(values: &[f64], period: usize) -> Vec<f64> {
    if period == 0 {
        return vec![f64::NAN; values.len()];
    }
    if values.is_empty() {
        return Vec::new();
    }
    let len = values.len();
    let alpha = 2.0 / (period as f64 + 1.0);
    let mut result = vec![f64::NAN; len];
    if len < period {
        return result;
    }
    let seed = values[..period].iter().sum::<f64>() / period as f64;
    result[period - 1] = seed;
    let mut prev = seed;
    for i in period..len {
        let val = values[i];
        prev = alpha * val + (1.0 - alpha) * prev;
        result[i] = prev;
    }
    result
}

fn build_entry_mask(frame: &DataFrame, expr: &str) -> Result<Vec<bool>> {
    let len = frame.height();
    if len == 0 {
        return Ok(Vec::new());
    }

    // Minimal parser: support `&&` between terms and simple
    // `feature`, or `feature OP constant` comparisons.
    let mut entry = vec![true; len];
    for raw_term in expr.split("&&") {
        let term = raw_term.trim();
        if term.is_empty() {
            continue;
        }
        let parts: Vec<&str> = term.split_whitespace().collect();
        let mask = if parts.len() == 1 {
            // Boolean column
            bool_column(frame, parts[0])?
        } else if parts.len() == 3 {
            let col_name = parts[0];
            let op = parts[1];
            let value: f64 = parts[2].parse().with_context(|| {
                format!(
                    "Unable to parse numeric literal '{}' in features expression",
                    parts[2]
                )
            })?;
            let series = column_with_nans(frame, col_name)?;
            series
                .iter()
                .map(|v| match v {
                    x if !x.is_finite() => false,
                    x => match op {
                        ">" => *x > value,
                        "<" => *x < value,
                        ">=" => *x >= value,
                        "<=" => *x <= value,
                        "==" => (*x - value).abs() <= FLOAT_TOLERANCE,
                        "!=" => (*x - value).abs() > FLOAT_TOLERANCE,
                        _ => false,
                    },
                })
                .collect()
        } else {
            return Err(anyhow!(
                "Unsupported features expression term '{}'; expected 'flag' or 'feature OP value'",
                term
            ));
        };

        if mask.len() != len {
            return Err(anyhow!(
                "Features expression term '{}' produced mask of length {}, expected {}",
                term,
                mask.len(),
                len
            ));
        }
        for (idx, flag) in mask.into_iter().enumerate() {
            entry[idx] = entry[idx] && flag;
        }
    }
    Ok(entry)
}

fn rma(values: &[f64], period: usize) -> Vec<f64> {
    let len = values.len();
    if period == 0 || len == 0 {
        return vec![f64::NAN; len];
    }
    let alpha = 1.0 / period as f64;
    let mut result = vec![f64::NAN; len];
    let mut prev: Option<f64> = None;
    for (idx, &value) in values.iter().enumerate() {
        if !value.is_finite() {
            if let Some(prev_val) = prev {
                result[idx] = prev_val;
            }
            continue;
        }
        let next = match prev {
            Some(prev_val) => alpha * value + (1.0 - alpha) * prev_val,
            None => value,
        };
        result[idx] = next;
        prev = Some(next);
    }
    result
}

fn sma(values: &[f64], period: usize) -> Vec<f64> {
    let len = values.len();
    let mut result = vec![f64::NAN; len];
    if period == 0 || period > len {
        return result;
    }
    let weight = 1.0 / period as f64;
    let conv_len = len + period - 1;
    let mut conv = vec![0.0; conv_len];
    let mut conv_invalid = vec![false; conv_len];
    for k in 0..period {
        for j in 0..len {
            let idx = k + j;
            let value = values[j];
            if !value.is_finite() {
                conv_invalid[idx] = true;
            } else if !conv_invalid[idx] {
                conv[idx] += value * weight;
            }
        }
    }
    for (flag, entry) in conv_invalid.iter().zip(conv.iter_mut()) {
        if *flag {
            *entry = f64::NAN;
        }
    }
    for i in period - 1..len {
        result[i] = conv[i];
    }
    result
}

fn sma_strict(values: &[f64], period: usize) -> Vec<f64> {
    sma(values, period)
}

fn momentum(values: &[f64], period: usize) -> Vec<f64> {
    values
        .iter()
        .enumerate()
        .map(|(i, &current)| {
            if i < period {
                0.0
            } else {
                current - values[i - period]
            }
        })
        .collect()
}

fn roc(values: &[f64], period: usize) -> Vec<f64> {
    values
        .iter()
        .enumerate()
        .map(|(i, &current)| {
            if i < period || values[i - period].abs() < f64::EPSILON {
                f64::NAN
            } else {
                current / values[i - period] - 1.0
            }
        })
        .collect()
}

fn derivative(values: &[f64], lag: usize) -> Vec<f64> {
    values
        .iter()
        .enumerate()
        .map(
            |(i, &val)| {
                if i < lag { 0.0 } else { val - values[i - lag] }
            },
        )
        .collect()
}

fn atr(high: &[f64], low: &[f64], close: &[f64], period: usize) -> Vec<f64> {
    let len = close.len();
    let mut tr = vec![0.0; len];
    for i in 0..len {
        let high_low = high[i] - low[i];
        let high_close = if i == 0 {
            high_low
        } else {
            (high[i] - close[i - 1]).abs()
        };
        let low_close = if i == 0 {
            high_low
        } else {
            (low[i] - close[i - 1]).abs()
        };
        tr[i] = high_low.max(high_close).max(low_close);
    }
    rma(&tr, period)
}

fn build_long_levels(
    close: &[f64],
    low: &[f64],
    atr_values: &[f64],
    tp_multiple: f64,
) -> (Vec<f64>, Vec<f64>) {
    let len = close.len();
    let mut stop = vec![f64::NAN; len];
    let mut tp = vec![f64::NAN; len];
    for i in 0..len {
        let entry = close[i];
        let atr = atr_values[i];
        let bar_low = low[i];
        if !entry.is_finite() || !atr.is_finite() || !bar_low.is_finite() {
            continue;
        }
        let atr_stop = entry - atr;

        // Choose the stop that results in the lower risk (closer to
        // entry) while still being below the entry price.
        let mut sl = f64::NAN;
        if bar_low < entry {
            sl = bar_low;
        }
        if atr_stop < entry && (!sl.is_finite() || atr_stop > sl) {
            sl = atr_stop;
        }
        if sl.is_finite() {
            stop[i] = sl;
            tp[i] = entry + tp_multiple * atr;
        }
    }
    (stop, tp)
}

fn build_short_levels(
    close: &[f64],
    high: &[f64],
    atr_values: &[f64],
    tp_multiple: f64,
) -> (Vec<f64>, Vec<f64>) {
    let len = close.len();
    let mut stop = vec![f64::NAN; len];
    let mut tp = vec![f64::NAN; len];
    for i in 0..len {
        let entry = close[i];
        let atr = atr_values[i];
        let bar_high = high[i];
        if !entry.is_finite() || !atr.is_finite() || !bar_high.is_finite() {
            continue;
        }
        let atr_stop = entry + atr;

        // Mirror the "lower risk" logic for shorts: choose the stop
        // that is just above entry (smaller distance), while still
        // being protective.
        let mut sl = f64::NAN;
        if bar_high > entry {
            sl = bar_high;
        }
        if atr_stop > entry && (!sl.is_finite() || atr_stop < sl) {
            sl = atr_stop;
        }
        if sl.is_finite() {
            stop[i] = sl;
            tp[i] = entry - tp_multiple * atr;
        }
    }
    (stop, tp)
}

fn atr_close_to_close_core(close: &[f64], period: usize) -> Vec<f64> {
    let len = close.len();
    let mut result = vec![f64::NAN; len];
    if len == 0 || period == 0 {
        return result;
    }
    let mut ranges = vec![f64::NAN; len];
    for i in 1..len {
        let current = close[i];
        let previous = close[i - 1];
        if current.is_finite() && previous.is_finite() {
            ranges[i] = (current - previous).abs();
        }
    }
    for i in 0..len {
        let start = if i + 1 >= period { i + 1 - period } else { 0 };
        let mut sum = 0.0;
        let mut count = 0usize;
        for value in &ranges[start..=i] {
            if value.is_finite() {
                sum += *value;
                count += 1;
            }
        }
        if count > 0 {
            result[i] = sum / count as f64;
        }
    }
    if len > period {
        let alpha = 2.0 / (period as f64 + 1.0);
        for i in period..len {
            let value = ranges[i];
            let prev = result[i - 1];
            if value.is_finite() && prev.is_finite() {
                result[i] = alpha * value + (1.0 - alpha) * prev;
            }
        }
    }
    result
}

fn atr_close_to_close(close: &[f64], period: usize) -> Vec<f64> {
    // Use the core close-to-close ATR implementation directly. This emits
    // NaNs naturally during its own warmup period; any rows that still
    // contain NaNs in the engineered feature set will be removed in a
    // single pass by drop_rows_with_nan_in_core.
    atr_close_to_close_core(close, period)
}

fn range(high: &[f64], low: &[f64]) -> Vec<f64> {
    high.iter().zip(low.iter()).map(|(h, l)| h - l).collect()
}

fn ratio(num: &[f64], denom: &[f64]) -> Vec<f64> {
    num.iter()
        .zip(denom.iter())
        .map(|(n, d)| if d.abs() < f64::EPSILON { 0.0 } else { n / d })
        .collect()
}

fn ratio_with_eps(num: &[f64], denom: &[f64], eps: f64) -> Vec<f64> {
    num.iter()
        .zip(denom.iter())
        .map(|(n, d)| if d.abs() < eps { 0.0 } else { n / d })
        .collect()
}

fn deviation(values: &[f64], reference: &[f64]) -> Vec<f64> {
    values
        .iter()
        .zip(reference.iter())
        .map(|(v, r)| {
            if r.abs() < f64::EPSILON {
                0.0
            } else {
                (v - r) / r
            }
        })
        .collect()
}

fn rsi(close: &[f64], period: usize, start: usize) -> Vec<f64> {
    let len = close.len();
    let mut full = vec![f64::NAN; len];
    if start >= len {
        return full;
    }
    let partial = rsi_core(&close[start..], period);
    for (idx, value) in partial.into_iter().enumerate() {
        if start + idx < len {
            full[start + idx] = value;
        }
    }
    full
}

fn rsi_core(close: &[f64], period: usize) -> Vec<f64> {
    let len = close.len();
    let mut gains = vec![f64::NAN; len];
    let mut losses = vec![f64::NAN; len];
    for i in 1..len {
        let change = close[i] - close[i - 1];
        gains[i] = change.max(0.0);
        losses[i] = (-change).max(0.0);
    }

    let avg_gain = rma(&gains, period);
    let avg_loss = rma(&losses, period);
    avg_gain
        .iter()
        .zip(avg_loss.iter())
        .map(|(gain, loss)| {
            if *loss == 0.0 {
                100.0
            } else {
                100.0 - (100.0 / (1.0 + gain / loss))
            }
        })
        .collect()
}

fn macd(close: &[f64], start: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let len = close.len();
    let mut macd_full = vec![f64::NAN; len];
    let mut signal_full = vec![f64::NAN; len];
    let mut hist_full = vec![f64::NAN; len];
    if start >= len {
        return (macd_full, signal_full, hist_full);
    }
    let (macd_slice, signal_slice, hist_slice) = macd_core(&close[start..]);
    for (idx, value) in macd_slice.into_iter().enumerate() {
        if start + idx < macd_full.len() {
            macd_full[start + idx] = value;
        }
    }
    for (idx, value) in signal_slice.into_iter().enumerate() {
        if start + idx < signal_full.len() {
            signal_full[start + idx] = value;
        }
    }
    for (idx, value) in hist_slice.into_iter().enumerate() {
        if start + idx < hist_full.len() {
            hist_full[start + idx] = value;
        }
    }
    (macd_full, signal_full, hist_full)
}

fn kalman_filter(values: &[f64], process_var: f64, obs_var: f64) -> (Vec<f64>, Vec<f64>) {
    let len = values.len();
    let mut filtered = vec![f64::NAN; len];
    let mut innovations = vec![0.0; len];
    if len == 0 {
        return (filtered, innovations);
    }

    let mut x = values
        .iter()
        .copied()
        .find(|v| v.is_finite())
        .unwrap_or(0.0);
    let mut p = 1.0;

    for (i, &value) in values.iter().enumerate() {
        let x_pred = x;
        let p_pred = p + process_var;
        if value.is_finite() {
            let k = p_pred / (p_pred + obs_var);
            let innovation = value - x_pred;
            x = x_pred + k * innovation;
            p = (1.0 - k) * p_pred;
            filtered[i] = x;
            innovations[i] = innovation;
        } else {
            filtered[i] = x_pred;
            innovations[i] = 0.0;
            x = x_pred;
            p = p_pred;
        }
    }

    (filtered, innovations)
}

fn macd_core(close: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    const FAST: usize = 12;
    const SLOW: usize = 26;
    const SIGNAL: usize = 9;

    let ema_fast = ema(close, FAST);
    let ema_slow = ema(close, SLOW);
    let mut macd_line = diff(&ema_fast, &ema_slow);

    for value in macd_line.iter_mut().take(SLOW - 1) {
        *value = f64::NAN;
    }

    let mut signal = vec![f64::NAN; macd_line.len()];
    if let Some(first_valid) = macd_line.iter().position(|v| v.is_finite()) {
        let signal_slice = &macd_line[first_valid..];
        let ema_values = ema(signal_slice, SIGNAL);
        for (offset, value) in ema_values.into_iter().enumerate() {
            if first_valid + offset < signal.len() {
                signal[first_valid + offset] = value;
            }
        }
    }

    let hist = macd_line
        .iter()
        .zip(signal.iter())
        .map(|(line, sig)| {
            if line.is_finite() && sig.is_finite() {
                line - sig
            } else {
                f64::NAN
            }
        })
        .collect();

    (macd_line, signal, hist)
}

fn stochastic(
    close: &[f64],
    high: &[f64],
    low: &[f64],
    period: usize,
    signal: usize,
    start: usize,
) -> (Vec<f64>, Vec<f64>) {
    let len = close.len();
    let mut k_full = vec![f64::NAN; len];
    let mut d_full = vec![f64::NAN; len];
    if start >= len {
        return (k_full, d_full);
    }
    let (k_slice, d_slice) = stochastic_core(
        &close[start..],
        &high[start..],
        &low[start..],
        period,
        signal,
    );
    for (idx, value) in k_slice.into_iter().enumerate() {
        if start + idx < len {
            k_full[start + idx] = value;
        }
    }
    for (idx, value) in d_slice.into_iter().enumerate() {
        if start + idx < len {
            d_full[start + idx] = value;
        }
    }
    (k_full, d_full)
}

fn stochastic_core(
    close: &[f64],
    high: &[f64],
    low: &[f64],
    period: usize,
    signal: usize,
) -> (Vec<f64>, Vec<f64>) {
    let len = close.len();
    let mut highest = vec![f64::NAN; len];
    let mut lowest = vec![f64::NAN; len];
    let mut ranges = vec![f64::NAN; len];
    for i in 0..len {
        if i + 1 < period {
            continue;
        }
        let slice_high = &high[i + 1 - period..=i];
        let slice_low = &low[i + 1 - period..=i];
        let high_val = slice_high.iter().cloned().fold(f64::MIN, f64::max);
        let low_val = slice_low.iter().cloned().fold(f64::MAX, f64::min);
        highest[i] = high_val;
        lowest[i] = low_val;
        ranges[i] = high_val - low_val;
    }
    let needs_eps = ranges
        .iter()
        .any(|value| value.is_finite() && *value == 0.0);
    if needs_eps {
        for value in ranges.iter_mut() {
            if value.is_finite() {
                *value += NON_ZERO_RANGE_EPS;
            }
        }
    }
    let mut raw_k = vec![f64::NAN; len];
    for i in 0..len {
        let range = ranges[i];
        let low_val = lowest[i];
        if !range.is_finite() || !low_val.is_finite() {
            continue;
        }
        let mut denom = range;
        if denom.abs() < NON_ZERO_RANGE_EPS {
            denom += NON_ZERO_RANGE_EPS;
        }
        if denom.abs() < NON_ZERO_RANGE_EPS {
            continue;
        }
        raw_k[i] = ((close[i] - low_val) / denom * 100.0).clamp(0.0, 100.0);
    }
    let mut smooth_k = vec![f64::NAN; len];
    if let Some(first_valid) = raw_k.iter().position(|v| v.is_finite()) {
        let slice = &raw_k[first_valid..];
        let slice_sma = sma(slice, signal);
        for (offset, value) in slice_sma.into_iter().enumerate() {
            if first_valid + offset < len {
                smooth_k[first_valid + offset] = value;
            }
        }
    }
    let mut d = vec![f64::NAN; len];
    if let Some(first_valid) = smooth_k.iter().position(|v| v.is_finite()) {
        let slice = &smooth_k[first_valid..];
        let slice_sma = sma(slice, signal);
        for (offset, value) in slice_sma.into_iter().enumerate() {
            if first_valid + offset < len {
                d[first_valid + offset] = value;
            }
        }
    }
    (smooth_k, d)
}

fn bollinger(
    close: &[f64],
    period: usize,
    std_mult: f64,
    start: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let len = close.len();
    let mut mid_full = vec![f64::NAN; len];
    let mut upper_full = vec![f64::NAN; len];
    let mut lower_full = vec![f64::NAN; len];
    let mut std_full = vec![f64::NAN; len];
    if start >= len {
        return (mid_full, upper_full, lower_full, std_full);
    }
    let (mid_slice, upper_slice, lower_slice, std_slice) =
        bollinger_core(&close[start..], period, std_mult);
    for (idx, value) in mid_slice.into_iter().enumerate() {
        if start + idx < len {
            mid_full[start + idx] = value;
        }
    }
    for (idx, value) in upper_slice.into_iter().enumerate() {
        if start + idx < len {
            upper_full[start + idx] = value;
        }
    }
    for (idx, value) in lower_slice.into_iter().enumerate() {
        if start + idx < len {
            lower_full[start + idx] = value;
        }
    }
    for (idx, value) in std_slice.into_iter().enumerate() {
        if start + idx < len {
            std_full[start + idx] = value;
        }
    }
    (mid_full, upper_full, lower_full, std_full)
}

fn bollinger_core(
    close: &[f64],
    period: usize,
    std_mult: f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let mid = sma(close, period);
    let std = rolling_std(close, period);
    let upper = mid
        .iter()
        .zip(std.iter())
        .map(|(m, s)| m + s * std_mult)
        .collect();
    let lower = mid
        .iter()
        .zip(std.iter())
        .map(|(m, s)| m - s * std_mult)
        .collect();
    (mid, upper, lower, std)
}

fn rolling_std(values: &[f64], period: usize) -> Vec<f64> {
    if period == 0 {
        return vec![f64::NAN; values.len()];
    }
    if period == 1 {
        return vec![0.0; values.len()];
    }
    let mean = sma(values, period);
    values
        .iter()
        .enumerate()
        .map(|(i, _)| {
            if i + 1 < period {
                return f64::NAN;
            }
            let start = i + 1 - period;
            let slice = &values[start..=i];
            if slice.iter().any(|v| !v.is_finite()) {
                return f64::NAN;
            }
            let mean_val = mean[i];
            if !mean_val.is_finite() {
                return f64::NAN;
            }
            let variance_sum = slice.iter().map(|x| (x - mean_val).powi(2)).sum::<f64>();
            let denom = (period - 1) as f64;
            (variance_sum / denom).sqrt()
        })
        .collect()
}

fn bollinger_position(close: &[f64], lower: &[f64], upper: &[f64]) -> Vec<f64> {
    close
        .iter()
        .zip(lower.iter().zip(upper.iter()))
        .map(|(c, (l, u))| {
            if (u - l).abs() < f64::EPSILON {
                0.5
            } else {
                (c - l) / (u - l)
            }
        })
        .collect()
}

fn extension(high: &[f64], low: &[f64], period: usize) -> Vec<f64> {
    high.iter()
        .enumerate()
        .map(|(i, &h)| {
            if i + 1 < period {
                0.0
            } else {
                let start = i + 1 - period;
                h - low[start..=i].iter().cloned().fold(f64::MAX, f64::min)
            }
        })
        .collect()
}

fn momentum_score(rsi14: &[f64], roc5: &[f64], roc10: &[f64]) -> Vec<f64> {
    let rsi_rank = percentile_rank(rsi14);
    let roc5_rank = percentile_rank(roc5);
    let roc10_rank = percentile_rank(roc10);

    rsi_rank
        .iter()
        .zip(roc5_rank.iter().zip(roc10_rank.iter()))
        .map(|(rsi, (r5, r10))| {
            if !rsi.is_finite() || !r5.is_finite() || !r10.is_finite() {
                f64::NAN
            } else {
                (rsi + r5 + r10) / 3.0
            }
        })
        .collect()
}

fn percentile_rank(values: &[f64]) -> Vec<f64> {
    let mut finite = values
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, value)| value.is_finite())
        .collect::<Vec<_>>();
    let count = finite.len();
    if count == 0 {
        return vec![f64::NAN; values.len()];
    }
    finite.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
    let mut ranks = vec![f64::NAN; values.len()];
    let denom = count as f64;
    let mut i = 0;
    while i < count {
        let mut j = i + 1;
        while j < count && finite[j].1.partial_cmp(&finite[i].1) == Some(Ordering::Equal) {
            j += 1;
        }
        let avg_rank = ((i + j - 1) as f64 / 2.0 + 1.0) / denom;
        for k in i..j {
            ranks[finite[k].0] = avg_rank;
        }
        i = j;
    }
    ranks
}

fn quantile(values: &[f64], q: f64) -> Option<f64> {
    let mut finite = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    if finite.is_empty() {
        return None;
    }
    finite.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let clamped = q.clamp(0.0, 1.0);
    let pos = clamped * (finite.len() - 1) as f64;
    let lower = pos.floor() as usize;
    let upper = pos.ceil() as usize;
    if lower == upper {
        Some(finite[lower])
    } else {
        Some(finite[lower] + (finite[upper] - finite[lower]) * (pos - lower as f64))
    }
}

fn rolling_coeff_var(values: &[f64], period: usize) -> Vec<f64> {
    let mean = sma(values, period);
    let std = rolling_std(values, period);
    std.iter()
        .zip(mean.iter())
        .map(|(s, m)| if m.abs() < f64::EPSILON { 0.0 } else { s / m })
        .collect()
}

#[derive(Clone, Copy)]
enum Comparison {
    Greater,
    Less,
}

fn stoch_threshold(values: &[f64], target: f64, comparison: Comparison) -> Vec<bool> {
    values
        .iter()
        .map(|value| {
            if !value.is_finite() {
                return false;
            }
            let rounded = round_to_decimals(*value, 10);
            match comparison {
                Comparison::Greater => rounded > target,
                Comparison::Less => rounded < target,
            }
        })
        .collect()
}

fn threshold(values: &[f64], target: f64, comparison: Comparison) -> Vec<bool> {
    values
        .iter()
        .map(|v| match comparison {
            Comparison::Greater => *v > target,
            Comparison::Less => *v < target,
        })
        .collect()
}

fn threshold_compare(
    values: &[f64],
    baseline: &[f64],
    mult: f64,
    comparison: Comparison,
) -> Vec<bool> {
    values
        .iter()
        .zip(baseline.iter())
        .map(|(v, b)| match comparison {
            Comparison::Greater => *v > b * mult,
            Comparison::Less => *v < b * mult,
        })
        .collect()
}

fn zscore_compare(values: &[f64], mean: &[f64], std: &[f64], threshold: f64) -> Vec<bool> {
    values
        .iter()
        .zip(mean.iter().zip(std.iter()))
        .map(|(value, (m, s))| {
            if *s == 0.0 {
                false
            } else {
                ((*value - *m) / *s).abs() > threshold
            }
        })
        .collect()
}

fn squeeze(values: &[f64], period: usize, mult: f64) -> Vec<bool> {
    let rolling_minima = rolling_min(values, period);
    values
        .iter()
        .zip(rolling_minima.iter())
        .map(|(v, m)| {
            if !v.is_finite() || !m.is_finite() {
                false
            } else {
                *v < *m * mult
            }
        })
        .collect()
}

fn rolling_min(values: &[f64], period: usize) -> Vec<f64> {
    values
        .iter()
        .enumerate()
        .map(|(i, _)| {
            if i + 1 < period {
                f64::NAN
            } else {
                values[i + 1 - period..=i]
                    .iter()
                    .cloned()
                    .fold(f64::MAX, f64::min)
            }
        })
        .collect()
}

fn derivative_threshold(values: &[f64], target: f64, comparison: Comparison) -> Vec<bool> {
    derivative(values, 1)
        .iter()
        .map(|v| match comparison {
            Comparison::Greater => *v > target,
            Comparison::Less => *v < target,
        })
        .collect()
}

fn recovery(values: &[f64], level: f64, comparison: Comparison) -> Vec<bool> {
    values
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            if i == 0 {
                false
            } else {
                let prev = values[i - 1];
                match comparison {
                    Comparison::Greater => prev < level && v > level,
                    Comparison::Less => prev > level && v < level,
                }
            }
        })
        .collect()
}

fn rolling_bool_sum(values: &[bool], period: usize) -> Vec<f64> {
    values
        .iter()
        .enumerate()
        .map(|(i, _)| {
            if period == 0 || i + 1 < period {
                f64::NAN
            } else {
                values[i + 1 - period..=i]
                    .iter()
                    .fold(0.0, |acc, flag| acc + if *flag { 1.0 } else { 0.0 })
            }
        })
        .collect()
}

fn streak(values: &[bool], period: usize) -> Vec<bool> {
    values
        .iter()
        .enumerate()
        .map(|(i, _)| {
            if i + 1 < period {
                false
            } else {
                values[i + 1 - period..=i].iter().all(|flag| *flag)
            }
        })
        .collect()
}

fn shift_bool(values: &[bool], lag: usize) -> Vec<bool> {
    let mut result = vec![false; values.len()];
    for i in lag..values.len() {
        result[i] = values[i - lag];
    }
    result
}

fn comparison(values: &[f64], lag: usize, cmp: Comparison) -> Vec<bool> {
    values
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            if i < lag {
                false
            } else {
                match cmp {
                    Comparison::Greater => v > values[i - lag],
                    Comparison::Less => v < values[i - lag],
                }
            }
        })
        .collect()
}

fn double_rising(values: &[f64]) -> Vec<bool> {
    values
        .iter()
        .enumerate()
        .map(|(i, &current)| {
            if i < 2 {
                return false;
            }
            let prev1 = values[i - 1];
            let prev2 = values[i - 2];
            if !current.is_finite() || !prev1.is_finite() || !prev2.is_finite() {
                false
            } else {
                current > prev1 && prev1 > prev2
            }
        })
        .collect()
}

fn momentum_acceleration(values: &[f64], direction: Comparison) -> Vec<bool> {
    values
        .iter()
        .enumerate()
        .map(|(i, &current)| {
            if i == 0 {
                return false;
            }
            let prev = values[i - 1];
            if !current.is_finite() || !prev.is_finite() {
                false
            } else {
                match direction {
                    Comparison::Greater => current > 0.0 && current > prev,
                    Comparison::Less => current < 0.0 && current < prev,
                }
            }
        })
        .collect()
}

fn comparison_series(values: &[f64], lag: usize, cmp: Comparison) -> Vec<bool> {
    comparison(values, lag, cmp)
}

#[cfg(test)]
mod tests {
    use super::{
        DerivedMetrics, FeatureEngineer, NEXT_BAR_SL_MULTIPLIER, PriceSeries, TickRoundMode,
        atr_close_to_close, candle_features, column_with_nans,
        compute_2x_atr_tp_atr_stop_targets_and_rr, compute_3x_atr_tp_atr_stop_targets_and_rr,
        compute_atr_tp_atr_stop_targets_and_rr, compute_highlow_1r_targets_and_rr,
        compute_highlow_or_atr_targets_and_rr, compute_highlow_or_atr_tightest_stop_targets_and_rr,
        compute_highlow_sl_1x_atr_tp_rr_gt_1_targets_and_rr,
        compute_highlow_sl_2x_atr_tp_rr_gt_1_targets_and_rr, compute_next_bar_targets_and_rr,
        compute_wicks_kf_targets_and_rr, quantize_distance_to_tick, quantize_price_to_tick, streak,
    };
    use barsmith_rs::Direction;
    use polars::prelude::*;
    use std::collections::HashMap;
    use std::path::Path;

    #[test]
    fn streak_basic_behaves_like_shift_logic() {
        let values = vec![true, true, false, true];
        let result_two = streak(&values, 2);
        assert_eq!(result_two, vec![false, true, false, false]);
        let result_three = streak(&values, 3);
        assert_eq!(result_three, vec![false, false, false, false]);
    }

    #[test]
    fn quantize_distance_to_tick_basic_modes() {
        let tick = 0.25;
        let dist = 0.26;

        let nearest = quantize_distance_to_tick(dist, tick, TickRoundMode::Nearest);
        let floor = quantize_distance_to_tick(dist, tick, TickRoundMode::Floor);
        let ceil = quantize_distance_to_tick(dist, tick, TickRoundMode::Ceil);

        assert!(
            (nearest - 0.25).abs() < 1e-9,
            "nearest should round to 0.25"
        );
        assert!(
            (floor - 0.25).abs() < 1e-9,
            "floor should round down to 0.25"
        );
        assert!((ceil - 0.50).abs() < 1e-9, "ceil should round up to 0.50");

        // Very small non-zero distances still map to one tick so that risk
        // is never zero when a stop is requested.
        let tiny = 0.01;
        let nearest_tiny = quantize_distance_to_tick(tiny, tick, TickRoundMode::Nearest);
        let floor_tiny = quantize_distance_to_tick(tiny, tick, TickRoundMode::Floor);
        let ceil_tiny = quantize_distance_to_tick(tiny, tick, TickRoundMode::Ceil);

        assert!((nearest_tiny - 0.25).abs() < 1e-9);
        assert!((floor_tiny - 0.25).abs() < 1e-9);
        assert!((ceil_tiny - 0.25).abs() < 1e-9);
    }

    #[test]
    fn quantize_price_to_tick_basic_modes() {
        let tick = 0.25;
        let price = 100.26;

        let nearest = quantize_price_to_tick(price, tick, TickRoundMode::Nearest);
        let floor = quantize_price_to_tick(price, tick, TickRoundMode::Floor);
        let ceil = quantize_price_to_tick(price, tick, TickRoundMode::Ceil);

        assert!((nearest - 100.25).abs() < 1e-9);
        assert!((floor - 100.25).abs() < 1e-9);
        assert!((ceil - 100.50).abs() < 1e-9);
    }

    #[test]
    fn atr_c2c_matches_python_reference_for_sample() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("tests/data/es_30m_sample.csv");
        let df = CsvReader::from_path(path)
            .unwrap()
            .infer_schema(Some(1024))
            .has_header(true)
            .finish()
            .unwrap();
        let prices = PriceSeries::from_frame(&df).unwrap();
        let atr = atr_close_to_close(&prices.close, 14);
        let start = 195usize;
        let expected = [
            1.0042706809186688,
            0.9703679234628463,
            0.9076522003344668,
            0.786631906956538,
            0.8150809860289996,
            0.8397368545584663,
            0.894438607284004,
            0.8418467929794702,
            0.8629338872488741,
            1.0478760356156909,
        ];
        for (offset, &value) in expected.iter().enumerate() {
            let idx = start + offset;
            let actual = atr[idx];
            assert!(
                (actual - value).abs() < 1e-9,
                "idx {} expected {} got {}",
                idx,
                value,
                actual
            );
        }
    }

    #[test]
    fn next_bar_color_and_wicks_open_equals_close_has_no_target_and_zero_rr() {
        // Two-bar synthetic series: trade decision at idx 0, next bar at idx 1.
        // Next bar has open == close with a finite wick distance so that
        // RR is defined but the color-based target remains false.
        let open = vec![99.0, 100.0];
        let high = vec![100.0, 100.0];
        let low = vec![98.5, 100.0];
        let close = vec![99.5, 100.0];
        let wicks = vec![1.0, 1.0];

        let (long, short, rr_long, rr_short, _exit_i_long, _exit_i_short) =
            compute_next_bar_targets_and_rr(
                &open,
                &high,
                &low,
                &close,
                &wicks,
                NEXT_BAR_SL_MULTIPLIER,
                None,
                Direction::Both,
            );

        assert_eq!(long.len(), 2);
        assert_eq!(short.len(), 2);
        // At idx 0, next bar has open == close, so no directional target.
        assert!(
            !long[0],
            "long target should be false when next_close == next_open"
        );
        assert!(
            !short[0],
            "short target should be false when next_close == next_open"
        );

        // RR is still defined and exactly 0R for both directions.
        assert!(rr_long[0].is_finite());
        assert!(rr_short[0].is_finite());
        assert!((rr_long[0]).abs() < 1e-9);
        assert!((rr_short[0]).abs() < 1e-9);
    }

    #[test]
    fn next_bar_color_and_wicks_uses_tick_rounded_stop_distance_when_available() {
        // Two-bar synthetic series with a wick-based stop distance that is
        // not an integer multiple of the tick size. With tick rounding
        // enabled, the effective stop distance is snapped to the tick
        // grid (using a ceil mode in production), which changes the
        // R-multiple geometry.
        //
        // Setup:
        // - next_open = 100.0
        // - next_close = 100.5
        // - current wick * NEXT_BAR_SL_MULTIPLIER = 0.30 (raw distance)
        // - tick_size = 0.25
        //
        // Without rounding: RR = 0.5 / 0.30 ≈ 1.67R
        // With ceil rounding to ticks (2 ticks): RR = 0.5 / 0.50 = 1.0R
        let open = vec![0.0, 100.0];
        let high = vec![0.0, 100.6];
        let low = vec![0.0, 99.8];
        let close = vec![0.0, 100.5];
        // Choose wick so that wick * NEXT_BAR_SL_MULTIPLIER = 0.30.
        let wick_raw = 0.30 / NEXT_BAR_SL_MULTIPLIER;
        let wicks = vec![wick_raw, 0.0];

        let (_long, _short, rr_long, _rr_short, _exit_i_long, _exit_i_short) =
            compute_next_bar_targets_and_rr(
                &open,
                &high,
                &low,
                &close,
                &wicks,
                NEXT_BAR_SL_MULTIPLIER,
                Some(0.25),
                Direction::Long,
            );

        assert!(
            rr_long[0].is_finite(),
            "expected finite RR value with tick rounding applied"
        );
        assert!(
            (rr_long[0] - 1.0).abs() < 1e-6,
            "expected RR to reflect ceil tick-rounded stop distance (got {})",
            rr_long[0]
        );
    }

    #[test]
    fn next_bar_color_and_wicks_uses_current_bar_wicks_for_stop_distance() {
        // Ensure stop distance is based on the current bar's wicks_diff_sma14
        // rather than the next bar's value.
        //
        // Setup:
        // - current wick = 1.0 => sl_distance = 1.5
        // - next wick = 10.0 (should be ignored for stop sizing)
        // - next bar moves +1.0 without hitting SL
        // Expected RR = 1.0 / 1.5 = 0.666...
        let open = vec![0.0, 100.0];
        let high = vec![0.0, 101.0];
        let low = vec![0.0, 99.0];
        let close = vec![0.0, 101.0];
        let wicks = vec![1.0, 10.0];

        let (long, _short, rr_long, _rr_short, _exit_i_long, _exit_i_short) =
            compute_next_bar_targets_and_rr(
                &open,
                &high,
                &low,
                &close,
                &wicks,
                NEXT_BAR_SL_MULTIPLIER,
                None,
                Direction::Long,
            );

        assert!(long[0], "expected long target to be true");
        assert!(
            (rr_long[0] - (1.0 / 1.5)).abs() < 1e-9,
            "expected RR to reflect current-bar wick sizing (got {})",
            rr_long[0]
        );
    }

    #[test]
    fn wicks_kf_open_equals_close_has_no_target_and_zero_rr() {
        let open = vec![99.0, 100.0];
        let high = vec![100.0, 100.0];
        let low = vec![98.5, 100.0];
        let close = vec![99.5, 100.0];
        let kf_wicks = vec![1.0, 1.0];

        let (long, short, rr_long, rr_short, _exit_i_long, _exit_i_short) =
            compute_wicks_kf_targets_and_rr(
                &open,
                &high,
                &low,
                &close,
                &kf_wicks,
                NEXT_BAR_SL_MULTIPLIER,
                None,
                Direction::Both,
            );

        assert_eq!(long.len(), 2);
        assert_eq!(short.len(), 2);
        assert!(!long[0]);
        assert!(!short[0]);

        assert!(rr_long[0].is_finite());
        assert!(rr_short[0].is_finite());
        assert!(rr_long[0].abs() < 1e-9);
        assert!(rr_short[0].abs() < 1e-9);
    }

    #[test]
    fn wicks_kf_uses_tick_rounded_stop_distance_when_available() {
        let open = vec![0.0, 100.0];
        let high = vec![0.0, 100.6];
        let low = vec![0.0, 99.8];
        let close = vec![0.0, 100.5];
        let wick_raw = 0.30 / NEXT_BAR_SL_MULTIPLIER;
        let kf_wicks = vec![wick_raw, 0.0];

        let (_long, _short, rr_long, _rr_short, _exit_i_long, _exit_i_short) =
            compute_wicks_kf_targets_and_rr(
                &open,
                &high,
                &low,
                &close,
                &kf_wicks,
                NEXT_BAR_SL_MULTIPLIER,
                Some(0.25),
                Direction::Long,
            );

        assert!(rr_long[0].is_finite());
        assert!(
            (rr_long[0] - 1.0).abs() < 1e-6,
            "expected RR to reflect ceil tick-rounded stop distance (got {})",
            rr_long[0]
        );
    }

    #[test]
    fn wicks_kf_uses_current_bar_kf_wicks_for_stop_distance() {
        let open = vec![0.0, 100.0];
        let high = vec![0.0, 101.0];
        let low = vec![0.0, 99.0];
        let close = vec![0.0, 101.0];
        let kf_wicks = vec![1.0, 10.0];

        let (long, _short, rr_long, _rr_short, _exit_i_long, _exit_i_short) =
            compute_wicks_kf_targets_and_rr(
                &open,
                &high,
                &low,
                &close,
                &kf_wicks,
                NEXT_BAR_SL_MULTIPLIER,
                None,
                Direction::Long,
            );

        assert!(long[0], "expected long target to be true");
        assert!(
            (rr_long[0] - (1.0 / 1.5)).abs() < 1e-9,
            "expected RR to reflect current-bar kf wick sizing (got {})",
            rr_long[0]
        );
    }

    #[test]
    fn highlow_or_atr_long_hits_tp_before_sl_and_returns_rr() {
        // idx 0 is green => long. Entry at close[0]=100. ATR[0]=1.
        // Stop=min(low[0]=99.5, entry-atr=99.0)=99.0. TP=102.0.
        // Next bar hits TP without touching SL => RR=2.0, label true.
        let open = vec![99.0, 100.0, 100.0];
        let high = vec![100.5, 102.0, 100.0];
        let low = vec![99.5, 99.25, 100.0];
        let close = vec![100.0, 101.0, 100.0];
        let atr = vec![1.0, 1.0, 1.0];

        let (long, short, rr_long, rr_short, _exit_i_long, _exit_i_short) =
            compute_highlow_or_atr_targets_and_rr(
                &open,
                &high,
                &low,
                &close,
                &atr,
                None,
                None,
                Direction::Both,
            );

        assert!(long[0]);
        assert!(!short[0]);
        assert!((rr_long[0] - 2.0).abs() < 1e-9, "got {}", rr_long[0]);
        assert!(rr_short[0].is_nan());
    }

    #[test]
    fn highlow_or_atr_tightest_stop_long_hits_tp_before_sl_and_returns_rr() {
        // Same as above, but with the tighter-stop variant:
        // Stop=tighter of low[0]=99.5 vs entry-atr=99.0 => 99.5. TP=102.0.
        // Risk=0.5, reward=2.0 => RR=4.0.
        let open = vec![99.0, 100.0, 100.0];
        let high = vec![100.5, 102.0, 100.0];
        let low = vec![99.5, 99.75, 100.0];
        let close = vec![100.0, 101.0, 100.0];
        let atr = vec![1.0, 1.0, 1.0];

        let (long, short, rr_long, rr_short, _exit_i_long, _exit_i_short) =
            compute_highlow_or_atr_tightest_stop_targets_and_rr(
                &open,
                &high,
                &low,
                &close,
                &atr,
                None,
                None,
                Direction::Both,
            );

        assert!(long[0]);
        assert!(!short[0]);
        assert!((rr_long[0] - 4.0).abs() < 1e-9, "got {}", rr_long[0]);
        assert!(rr_short[0].is_nan());
    }

    #[test]
    fn highlow_1r_long_uses_signal_low_only_and_tp_is_1r() {
        // idx 0 is green => long. Entry at close[0]=100.
        // Stop=low[0]=99 => risk=1. TP=entry+risk=101.
        // Next bar opens above TP => gap-fill at open => RR=(111-100)/1=11.
        let open = vec![99.0, 111.0];
        let high = vec![101.0, 111.0];
        let low = vec![99.0, 110.0];
        let close = vec![100.0, 111.0];

        let (long, short, rr_long, rr_short, _exit_i_long, _exit_i_short) =
            compute_highlow_1r_targets_and_rr(
                &open,
                &high,
                &low,
                &close,
                None,
                None,
                Direction::Both,
            );

        assert!(long[0]);
        assert!(!short[0]);
        assert!((rr_long[0] - 11.0).abs() < 1e-9, "got {}", rr_long[0]);
        assert!(rr_short[0].is_nan());
    }

    #[test]
    fn two_x_atr_tp_atr_stop_long_uses_entry_minus_atr_only_for_stop() {
        // idx 0 is green => long. Entry at close[0]=100. ATR=2.
        // Stop=98. TP=104. Next bar hits TP => RR=2.0.
        let open = vec![99.0, 100.0, 100.0];
        let high = vec![100.5, 104.0, 100.0];
        let low = vec![80.0, 99.0, 100.0];
        let close = vec![100.0, 101.0, 100.0];
        let atr = vec![2.0, 2.0, 2.0];

        let (long, short, rr_long, rr_short, _exit_i_long, _exit_i_short) =
            compute_2x_atr_tp_atr_stop_targets_and_rr(
                &open,
                &high,
                &low,
                &close,
                &atr,
                None,
                None,
                Direction::Both,
            );

        assert!(long[0]);
        assert!(!short[0]);
        assert!((rr_long[0] - 2.0).abs() < 1e-9, "got {}", rr_long[0]);
        assert!(rr_short[0].is_nan());
    }

    #[test]
    fn highlow_1r_short_uses_signal_high_only_and_tp_is_1r() {
        // idx 0 is red => short. Entry at close[0]=100.
        // Stop=high[0]=101 => risk=1. TP=entry-risk=99.
        // Next bar opens below TP => gap-fill at open => RR=(100-89)/1=11.
        let open = vec![101.0, 89.0];
        let high = vec![101.0, 100.0];
        let low = vec![99.0, 89.0];
        let close = vec![100.0, 90.0];

        let (_long, short, _rr_long, rr_short, _exit_i_long, _exit_i_short) =
            compute_highlow_1r_targets_and_rr(
                &open,
                &high,
                &low,
                &close,
                None,
                None,
                Direction::Both,
            );

        assert!(short[0]);
        assert!((rr_short[0] - 11.0).abs() < 1e-9, "got {}", rr_short[0]);
    }

    #[test]
    fn two_x_atr_tp_atr_stop_short_uses_entry_plus_atr_only_for_stop() {
        // idx 0 is red => short. Entry at close[0]=100. ATR=2.
        // Stop=102. TP=96. Next bar hits TP => RR=2.0.
        let open = vec![101.0, 100.0, 100.0];
        let high = vec![101.0, 101.0, 100.0];
        let low = vec![99.0, 96.0, 100.0];
        let close = vec![100.0, 99.0, 100.0];
        let atr = vec![2.0, 2.0, 2.0];

        let (_long, short, _rr_long, rr_short, _exit_i_long, _exit_i_short) =
            compute_2x_atr_tp_atr_stop_targets_and_rr(
                &open,
                &high,
                &low,
                &close,
                &atr,
                None,
                None,
                Direction::Both,
            );

        assert!(short[0]);
        assert!((rr_short[0] - 2.0).abs() < 1e-9, "got {}", rr_short[0]);
    }

    #[test]
    fn three_x_atr_tp_atr_stop_long_tp_is_3x_atr() {
        // idx 0 is green => long. Entry at close[0]=100. ATR=2.
        // Stop=98. TP=106. Next bar hits TP => RR=3.0.
        let open = vec![99.0, 100.0, 100.0];
        let high = vec![100.5, 106.0, 100.0];
        let low = vec![80.0, 99.0, 100.0];
        let close = vec![100.0, 101.0, 100.0];
        let atr = vec![2.0, 2.0, 2.0];

        let (long, short, rr_long, rr_short, _exit_i_long, _exit_i_short) =
            compute_3x_atr_tp_atr_stop_targets_and_rr(
                &open,
                &high,
                &low,
                &close,
                &atr,
                None,
                None,
                Direction::Both,
            );

        assert!(long[0]);
        assert!(!short[0]);
        assert!((rr_long[0] - 3.0).abs() < 1e-9, "got {}", rr_long[0]);
        assert!(rr_short[0].is_nan());
    }

    #[test]
    fn three_x_atr_tp_atr_stop_short_tp_is_3x_atr() {
        // idx 0 is red => short. Entry at close[0]=100. ATR=2.
        // Stop=102. TP=94. Next bar hits TP => RR=3.0.
        let open = vec![101.0, 100.0, 100.0];
        let high = vec![101.0, 101.0, 100.0];
        let low = vec![99.0, 94.0, 100.0];
        let close = vec![100.0, 99.0, 100.0];
        let atr = vec![2.0, 2.0, 2.0];

        let (_long, short, _rr_long, rr_short, _exit_i_long, _exit_i_short) =
            compute_3x_atr_tp_atr_stop_targets_and_rr(
                &open,
                &high,
                &low,
                &close,
                &atr,
                None,
                None,
                Direction::Both,
            );

        assert!(short[0]);
        assert!((rr_short[0] - 3.0).abs() < 1e-9, "got {}", rr_short[0]);
    }

    #[test]
    fn atr_tp_atr_stop_long_tp_is_1x_atr() {
        // idx 0 is green => long. Entry at close[0]=100. ATR=2.
        // Stop=98. TP=102. Next bar hits TP => RR=1.0.
        let open = vec![99.0, 100.0, 100.0];
        let high = vec![100.5, 102.0, 100.0];
        let low = vec![80.0, 99.0, 100.0];
        let close = vec![100.0, 101.0, 100.0];
        let atr = vec![2.0, 2.0, 2.0];

        let (long, short, rr_long, rr_short, _exit_i_long, _exit_i_short) =
            compute_atr_tp_atr_stop_targets_and_rr(
                &open,
                &high,
                &low,
                &close,
                &atr,
                None,
                None,
                Direction::Both,
            );

        assert!(long[0]);
        assert!(!short[0]);
        assert!((rr_long[0] - 1.0).abs() < 1e-9, "got {}", rr_long[0]);
        assert!(rr_short[0].is_nan());
    }

    #[test]
    fn atr_tp_atr_stop_short_tp_is_1x_atr() {
        // idx 0 is red => short. Entry at close[0]=100. ATR=2.
        // Stop=102. TP=98. Next bar hits TP => RR=1.0.
        let open = vec![101.0, 100.0, 100.0];
        let high = vec![101.0, 101.0, 100.0];
        let low = vec![99.0, 98.0, 100.0];
        let close = vec![100.0, 99.0, 100.0];
        let atr = vec![2.0, 2.0, 2.0];

        let (_long, short, _rr_long, rr_short, _exit_i_long, _exit_i_short) =
            compute_atr_tp_atr_stop_targets_and_rr(
                &open,
                &high,
                &low,
                &close,
                &atr,
                None,
                None,
                Direction::Both,
            );

        assert!(short[0]);
        assert!((rr_short[0] - 1.0).abs() < 1e-9, "got {}", rr_short[0]);
    }

    #[test]
    fn highlow_sl_2x_atr_tp_rr_gt_1_long_hits_tp_when_rr_at_tp_gt_1() {
        // idx 0 is green => long. Entry=close[0]=100. Stop=low[0]=99 => risk=1.
        // ATR=1 => TP=102 => RR_at_tp=2 (>1) => trade allowed and TP hit.
        let open = vec![99.0, 100.0, 100.0];
        let high = vec![100.5, 102.0, 100.0];
        let low = vec![99.0, 99.25, 100.0];
        let close = vec![100.0, 101.0, 100.0];
        let atr = vec![1.0, 1.0, 1.0];

        let (long, short, rr_long, rr_short, _exit_i_long, _exit_i_short) =
            compute_highlow_sl_2x_atr_tp_rr_gt_1_targets_and_rr(
                &open,
                &high,
                &low,
                &close,
                &atr,
                None,
                None,
                Direction::Both,
            );

        assert!(long[0]);
        assert!(!short[0]);
        assert!((rr_long[0] - 2.0).abs() < 1e-9, "got {}", rr_long[0]);
        assert!(rr_short[0].is_nan());
    }

    #[test]
    fn highlow_sl_2x_atr_tp_rr_gt_1_long_is_rejected_when_rr_at_tp_is_1_after_tick_rounding() {
        // idx 0 is green => long. Entry=100. Stop=97.5 => risk=2.5.
        // ATR=1.25 => TP=102.5. With tick_size=0.25, stop and TP are already on-grid.
        // RR_at_tp=(102.5-100)/2.5 = 1.0 => strict gate (>1) rejects.
        let open = vec![99.0, 100.0, 100.0];
        let high = vec![100.5, 103.0, 100.0];
        let low = vec![97.5, 99.0, 100.0];
        let close = vec![100.0, 101.0, 100.0];
        let atr = vec![1.25, 1.25, 1.25];

        let (long, _short, rr_long, _rr_short, _exit_i_long, _exit_i_short) =
            compute_highlow_sl_2x_atr_tp_rr_gt_1_targets_and_rr(
                &open,
                &high,
                &low,
                &close,
                &atr,
                Some(0.25),
                None,
                Direction::Both,
            );

        assert!(!long[0]);
        assert!(rr_long[0].is_nan());
    }

    #[test]
    fn highlow_sl_1x_atr_tp_rr_gt_1_short_hits_tp_when_rr_at_tp_gt_1() {
        // idx 0 is red => short. Entry=100. Stop=high[0]=101 => risk=1.
        // ATR=2 => TP=98 => RR_at_tp=2 (>1) => trade allowed and TP hit.
        let open = vec![101.0, 100.0, 100.0];
        let high = vec![101.0, 100.75, 100.0];
        let low = vec![99.0, 98.0, 100.0];
        let close = vec![100.0, 99.0, 100.0];
        let atr = vec![2.0, 2.0, 2.0];

        let (_long, short, _rr_long, rr_short, _exit_i_long, _exit_i_short) =
            compute_highlow_sl_1x_atr_tp_rr_gt_1_targets_and_rr(
                &open,
                &high,
                &low,
                &close,
                &atr,
                None,
                None,
                Direction::Both,
            );

        assert!(short[0]);
        assert!((rr_short[0] - 2.0).abs() < 1e-9, "got {}", rr_short[0]);
    }

    #[test]
    fn highlow_sl_1x_atr_tp_rr_gt_1_short_is_rejected_when_rr_at_tp_is_1_after_tick_rounding() {
        // idx 0 is red => short. Entry=100. Stop=101 => risk=1.
        // ATR=1 => TP=99. With tick_size=0.25, stop and TP are already on-grid.
        // RR_at_tp=(100-99)/1 = 1.0 => strict gate (>1) rejects.
        let open = vec![101.0, 100.0, 100.0];
        let high = vec![101.0, 101.0, 100.0];
        let low = vec![99.0, 98.0, 100.0];
        let close = vec![100.0, 99.0, 100.0];
        let atr = vec![1.0, 1.0, 1.0];

        let (_long, short, _rr_long, rr_short, _exit_i_long, _exit_i_short) =
            compute_highlow_sl_1x_atr_tp_rr_gt_1_targets_and_rr(
                &open,
                &high,
                &low,
                &close,
                &atr,
                Some(0.25),
                None,
                Direction::Both,
            );

        assert!(!short[0]);
        assert!(rr_short[0].is_nan());
    }

    #[test]
    fn highlow_or_atr_long_sl_dominates_when_both_tp_and_sl_touch_same_bar() {
        // Same setup as above, but next bar touches both TP and SL.
        // Conservative ordering: SL dominates => RR=-1 and label false.
        let open = vec![99.0, 100.0, 100.0];
        let high = vec![100.5, 102.0, 100.0];
        let low = vec![99.5, 98.0, 100.0];
        let close = vec![100.0, 101.0, 100.0];
        let atr = vec![1.0, 1.0, 1.0];

        let (long, _short, rr_long, _rr_short, _exit_i_long, _exit_i_short) =
            compute_highlow_or_atr_targets_and_rr(
                &open,
                &high,
                &low,
                &close,
                &atr,
                None,
                None,
                Direction::Both,
            );

        assert!(!long[0]);
        assert!((rr_long[0] + 1.0).abs() < 1e-9);
    }

    #[test]
    fn highlow_or_atr_short_hits_tp_before_sl_and_tick_rounds_prices() {
        // idx 0 is red => short. Entry at close[0]=99.0. ATR=1.03.
        // Stop=max(high[0]=99.5, entry+atr=100.03)=100.03 => ceil tick(0.25)=100.25.
        // TP=entry-2*atr=96.94 => floor tick=96.75.
        // Risk=1.25, reward=2.25 => RR=1.8.
        let open = vec![100.0, 99.0, 99.0];
        let high = vec![99.5, 100.0, 99.0];
        let low = vec![98.5, 96.5, 99.0];
        let close = vec![99.0, 98.0, 99.0];
        let atr = vec![1.03, 1.03, 1.03];

        let (_long, short, _rr_long, rr_short, _exit_i_long, _exit_i_short) =
            compute_highlow_or_atr_targets_and_rr(
                &open,
                &high,
                &low,
                &close,
                &atr,
                Some(0.25),
                None,
                Direction::Both,
            );

        assert!(short[0]);
        assert!((rr_short[0] - 1.8).abs() < 1e-9, "got {}", rr_short[0]);
    }

    #[test]
    fn highlow_or_atr_tightest_stop_short_hits_tp_before_sl_and_tick_rounds_prices() {
        // Same as above, but with the tighter-stop variant:
        // Stop=tighter of high[0]=99.5 vs entry+atr=100.03 => 99.5 => ceil tick=99.50.
        // Risk=0.5, reward=2.25 => RR=4.5.
        let open = vec![100.0, 99.0, 99.0];
        let high = vec![99.5, 99.25, 99.0];
        let low = vec![98.5, 96.5, 99.0];
        let close = vec![99.0, 98.0, 99.0];
        let atr = vec![1.03, 1.03, 1.03];

        let (_long, short, _rr_long, rr_short, _exit_i_long, _exit_i_short) =
            compute_highlow_or_atr_tightest_stop_targets_and_rr(
                &open,
                &high,
                &low,
                &close,
                &atr,
                Some(0.25),
                None,
                Direction::Both,
            );

        assert!(short[0]);
        assert!((rr_short[0] - 4.5).abs() < 1e-9, "got {}", rr_short[0]);
    }

    #[test]
    fn highlow_or_atr_doji_signal_has_no_trade() {
        let open = vec![100.0, 100.0, 100.0];
        let high = vec![101.0, 101.0, 101.0];
        let low = vec![99.0, 99.0, 99.0];
        let close = vec![100.0, 100.5, 100.0];
        let atr = vec![1.0, 1.0, 1.0];

        let (long, short, rr_long, rr_short, _exit_i_long, _exit_i_short) =
            compute_highlow_or_atr_targets_and_rr(
                &open,
                &high,
                &low,
                &close,
                &atr,
                None,
                None,
                Direction::Both,
            );

        assert!(!long[0]);
        assert!(!short[0]);
        assert!(rr_long[0].is_nan());
        assert!(rr_short[0].is_nan());
    }

    #[test]
    fn highlow_or_atr_long_gap_below_stop_fills_at_open() {
        // idx 0 is green => long. Entry at close[0]=100. ATR[0]=1.
        // Stop=tighter of low[0]=99.0 vs entry-atr=99.0 => 99.0. TP=102.0.
        // Next bar opens below stop at 98.5 => fill at open => RR=-1.5, label false.
        let open = vec![99.0, 98.5];
        let high = vec![100.5, 99.0];
        let low = vec![99.0, 98.0];
        let close = vec![100.0, 98.75];
        let atr = vec![1.0, 1.0];

        let (long, _short, rr_long, _rr_short, _exit_i_long, _exit_i_short) =
            compute_highlow_or_atr_targets_and_rr(
                &open,
                &high,
                &low,
                &close,
                &atr,
                None,
                None,
                Direction::Both,
            );

        assert!(!long[0]);
        assert!((rr_long[0] + 1.5).abs() < 1e-9, "got {}", rr_long[0]);
    }

    #[test]
    fn highlow_or_atr_long_gap_above_tp_fills_at_open() {
        // idx 0 is green => long. Entry at close[0]=100. ATR[0]=1.
        // Stop=99.0. TP=102.0.
        // Next bar opens above TP at 103.0 => fill at open => RR=3.0, label true.
        let open = vec![99.0, 103.0];
        let high = vec![100.5, 104.0];
        let low = vec![99.0, 102.5];
        let close = vec![100.0, 103.5];
        let atr = vec![1.0, 1.0];

        let (long, _short, rr_long, _rr_short, _exit_i_long, _exit_i_short) =
            compute_highlow_or_atr_targets_and_rr(
                &open,
                &high,
                &low,
                &close,
                &atr,
                None,
                None,
                Direction::Both,
            );

        assert!(long[0]);
        assert!((rr_long[0] - 3.0).abs() < 1e-9, "got {}", rr_long[0]);
    }

    #[test]
    fn highlow_or_atr_short_gap_above_stop_fills_at_open() {
        // idx 0 is red => short. Entry at close[0]=100. ATR[0]=1.
        // Stop=tighter of high[0]=101.0 vs entry+atr=101.0 => 101.0. TP=98.0.
        // Next bar opens above stop at 102.0 => fill at open => RR=-2.0, label false.
        let open = vec![101.0, 102.0];
        let high = vec![101.0, 102.5];
        let low = vec![99.5, 99.0];
        let close = vec![100.0, 101.5];
        let atr = vec![1.0, 1.0];

        let (_long, short, _rr_long, rr_short, _exit_i_long, _exit_i_short) =
            compute_highlow_or_atr_targets_and_rr(
                &open,
                &high,
                &low,
                &close,
                &atr,
                None,
                None,
                Direction::Both,
            );

        assert!(!short[0]);
        assert!((rr_short[0] + 2.0).abs() < 1e-9, "got {}", rr_short[0]);
    }

    #[test]
    fn highlow_or_atr_short_gap_below_tp_fills_at_open() {
        // idx 0 is red => short. Entry at close[0]=100. ATR[0]=1.
        // Stop=101.0. TP=98.0.
        // Next bar opens below TP at 97.0 => fill at open => RR=3.0, label true.
        let open = vec![101.0, 97.0];
        let high = vec![101.0, 100.0];
        let low = vec![99.5, 96.5];
        let close = vec![100.0, 98.0];
        let atr = vec![1.0, 1.0];

        let (_long, short, _rr_long, rr_short, _exit_i_long, _exit_i_short) =
            compute_highlow_or_atr_targets_and_rr(
                &open,
                &high,
                &low,
                &close,
                &atr,
                None,
                None,
                Direction::Both,
            );

        assert!(short[0]);
        assert!((rr_short[0] - 3.0).abs() < 1e-9, "got {}", rr_short[0]);
    }

    #[test]
    fn highlow_or_atr_forces_exit_at_cutoff_close_when_tp_after_cutoff() {
        // idx 0 is green => long. Entry at close[0]=100. ATR[0]=1.
        // Stop=tighter of low[0]=99.0 vs entry-atr=99.0 => 99.0. TP=102.0.
        // TP is only reached on bar 2, but with a cutoff horizon at bar 1
        // we force-exit at close[1]=100.5 => RR=0.5 and label false.
        let open = vec![99.0, 100.0, 100.0];
        let high = vec![100.5, 101.0, 102.0];
        let low = vec![99.0, 99.5, 100.0];
        let close = vec![100.0, 100.5, 101.0];
        let atr = vec![1.0, 1.0, 1.0];

        let (long_full, _short_full, rr_long_full, _rr_short_full, _exit_i_long, _exit_i_short) =
            compute_highlow_or_atr_targets_and_rr(
                &open,
                &high,
                &low,
                &close,
                &atr,
                None,
                None,
                Direction::Both,
            );
        assert!(long_full[0]);
        assert!((rr_long_full[0] - 2.0).abs() < 1e-9);

        let (long_cut, _short_cut, rr_long_cut, _rr_short_cut, _exit_i_long, _exit_i_short) =
            compute_highlow_or_atr_targets_and_rr(
                &open,
                &high,
                &low,
                &close,
                &atr,
                None,
                Some(1),
                Direction::Both,
            );
        assert!(!long_cut[0]);
        assert!(
            (rr_long_cut[0] - 0.5).abs() < 1e-9,
            "got {}",
            rr_long_cut[0]
        );
    }

    #[test]
    fn highlow_or_atr_populates_rr_after_cutoff_horizon_for_post_period_entries() {
        // With a cutoff horizon at index 1, entries after that cutoff should
        // still resolve normally to the end of the dataset so the prepared
        // CSV remains usable for forward evaluation.
        //
        // idx 2 is green => long. Entry at close[2]=100. ATR[2]=1.
        // Stop=tighter of low[2]=99.0 vs entry-atr=99.0 => 99.0. TP=102.0.
        // Bar 3 hits TP => RR=2.0, label true.
        let open = vec![99.0, 100.0, 99.0, 100.0];
        let high = vec![100.5, 101.0, 100.5, 102.0];
        let low = vec![99.0, 99.5, 99.0, 99.5];
        let close = vec![100.0, 100.5, 100.0, 101.0];
        let atr = vec![1.0, 1.0, 1.0, 1.0];

        let (long_cut, _short_cut, rr_long_cut, _rr_short_cut, _exit_i_long, _exit_i_short) =
            compute_highlow_or_atr_targets_and_rr(
                &open,
                &high,
                &low,
                &close,
                &atr,
                None,
                Some(1),
                Direction::Both,
            );

        assert!(long_cut[2]);
        assert!(
            (rr_long_cut[2] - 2.0).abs() < 1e-9,
            "got {}",
            rr_long_cut[2]
        );
    }

    #[test]
    fn tribar_variations_split_directionally() {
        // Construct a 5-bar series:
        // - idx 2 triggers bullish tribar + tribar_hl (green close breaks prior two highs)
        // - idx 4 triggers bearish tribar + tribar_hl (red close breaks prior two lows)
        let prices = PriceSeries {
            open: vec![100.0, 101.0, 102.0, 104.0, 103.0],
            high: vec![102.0, 103.0, 105.0, 106.0, 104.0],
            low: vec![99.0, 100.0, 101.0, 102.0, 99.0],
            close: vec![101.0, 102.0, 104.0, 103.0, 100.0],
        };
        let derived = DerivedMetrics::new(&prices);
        let mut bools: HashMap<&'static str, Vec<bool>> = HashMap::new();
        let mut floats: HashMap<&'static str, Vec<f64>> = HashMap::new();

        candle_features(&prices, &derived, &mut bools, &mut floats);

        let tribar_green = bools.get("is_tribar_green").unwrap();
        let tribar_red = bools.get("is_tribar_red").unwrap();
        let tribar_hl_green = bools.get("is_tribar_hl_green").unwrap();
        let tribar_hl_red = bools.get("is_tribar_hl_red").unwrap();
        let tribar = bools.get("is_tribar").unwrap();
        let tribar_hl = bools.get("is_tribar_hl").unwrap();

        assert!(tribar_green[2]);
        assert!(!tribar_red[2]);
        assert!(tribar_hl_green[2]);
        assert!(!tribar_hl_red[2]);
        assert!(tribar[2]);
        assert!(tribar_hl[2]);

        assert!(!tribar_green[4]);
        assert!(tribar_red[4]);
        assert!(!tribar_hl_green[4]);
        assert!(tribar_hl_red[4]);
        assert!(tribar[4]);
        assert!(tribar_hl[4]);
    }

    #[test]
    fn atr_column_survives_nan_drop_and_matches_full_history() {
        // We want ATR values used by targets to be computed on the full
        // history and then filtered by the NaN-drop mask, rather than
        // being recomputed after trimming (which would reset the RMA state).
        let len = 210usize;
        let mut open = Vec::with_capacity(len);
        let mut high = Vec::with_capacity(len);
        let mut low = Vec::with_capacity(len);
        let mut close = Vec::with_capacity(len);
        for i in 0..len {
            let base = 100.0 + (i as f64) * 0.1;
            let wiggle = ((i % 7) as f64) * 0.05;
            let c = base + wiggle;
            let o = c - 0.2;
            let range_up = 0.3 + ((i % 5) as f64) * 0.02;
            let range_dn = 0.25 + ((i % 3) as f64) * 0.03;
            open.push(o);
            close.push(c);
            high.push(c + range_up);
            low.push(o - range_dn);
        }

        let prices = PriceSeries {
            open: open.clone(),
            high: high.clone(),
            low: low.clone(),
            close: close.clone(),
        };
        let derived_full = DerivedMetrics::new(&prices);
        let sma200_start = derived_full
            .sma200
            .iter()
            .position(|v| v.is_finite())
            .expect("sma200 should become finite for len >= 200");

        // Build a DataFrame and run the standard engineering pipeline with NaN-drop.
        let df = DataFrame::new(vec![
            Series::new("open", open),
            Series::new("high", high),
            Series::new("low", low),
            Series::new("close", close),
        ])
        .unwrap();
        let mut engineer = FeatureEngineer { frame: df.clone() };
        engineer.compute_features_with_options(true).unwrap();

        // After NaN-drop, the first remaining row should line up with the first finite SMA200,
        // and ATR should match the full-history ATR at those original indices.
        let engineered_atr = column_with_nans(&engineer.frame, "atr").unwrap();
        let actual_start = len - engineered_atr.len();
        assert!(
            actual_start >= sma200_start,
            "expected NaN-drop start ({}) to be >= SMA200 warmup start ({})",
            actual_start,
            sma200_start
        );
        for (offset, &atr_val) in engineered_atr.iter().enumerate() {
            let orig_idx = actual_start + offset;
            let expected = derived_full.atr[orig_idx];
            assert!(
                (atr_val - expected).abs() < 1e-12,
                "ATR mismatch at orig_idx={}: got {}, expected {}",
                orig_idx,
                atr_val,
                expected
            );
        }

        // Demonstrate why recomputing ATR after trimming would be wrong:
        // the first ATR value would reset to the first TR of the trimmed slice.
        let trimmed_prices = PriceSeries {
            open: df
                .column("open")
                .unwrap()
                .f64()
                .unwrap()
                .into_no_null_iter()
                .skip(actual_start)
                .collect(),
            high: df
                .column("high")
                .unwrap()
                .f64()
                .unwrap()
                .into_no_null_iter()
                .skip(actual_start)
                .collect(),
            low: df
                .column("low")
                .unwrap()
                .f64()
                .unwrap()
                .into_no_null_iter()
                .skip(actual_start)
                .collect(),
            close: df
                .column("close")
                .unwrap()
                .f64()
                .unwrap()
                .into_no_null_iter()
                .skip(actual_start)
                .collect(),
        };
        let derived_trim = DerivedMetrics::new(&trimmed_prices);
        let full_first = derived_full.atr[actual_start];
        let trim_first = derived_trim.atr[0];
        assert!(
            (full_first - trim_first).abs() > 1e-6,
            "expected trimmed ATR to differ at first kept bar (full={}, trim={})",
            full_first,
            trim_first
        );
    }
}

fn compare_series(left: &[f64], right: &[f64], comparison: Comparison) -> Vec<bool> {
    left.iter()
        .zip(right.iter())
        .map(|(l, r)| {
            if !l.is_finite() || !r.is_finite() {
                false
            } else {
                match comparison {
                    Comparison::Greater => *l > *r + FLOAT_TOLERANCE,
                    Comparison::Less => *l < *r - FLOAT_TOLERANCE,
                }
            }
        })
        .collect()
}

fn ema_alignment(close: &[f64], ema9: &[f64], ema20: &[f64], ema50: &[f64]) -> Vec<bool> {
    close
        .iter()
        .zip(ema9.iter().zip(ema20.iter().zip(ema50.iter())))
        .map(|(c, (e9, (e20, e50)))| c > e9 && e9 > e20 && e20 > e50)
        .collect()
}

fn ribbon_alignment(
    ema9: &[f64],
    ema20: &[f64],
    ema50: &[f64],
    sma200: &[f64],
    bullish: bool,
) -> Vec<bool> {
    ema9.iter()
        .zip(ema20.iter().zip(ema50.iter().zip(sma200.iter())))
        .map(|(e9, (e20, (e50, s200)))| {
            if !e9.is_finite() || !e20.is_finite() || !e50.is_finite() || !s200.is_finite() {
                false
            } else if bullish {
                e9 > e20 && e20 > e50 && e50 > s200
            } else {
                e9 < e20 && e20 < e50 && e50 < s200
            }
        })
        .collect()
}

fn hammer(body: &[f64], upper: &[f64], lower: &[f64]) -> Vec<bool> {
    body.iter()
        .zip(upper.iter().zip(lower.iter()))
        .map(|(b, (u, l))| b.abs() < *l * 0.5 && *l > *u * 2.0)
        .collect()
}

fn shooting_star(body: &[f64], upper: &[f64], lower: &[f64]) -> Vec<bool> {
    body.iter()
        .zip(upper.iter().zip(lower.iter()))
        .map(|(b, (u, l))| b.abs() < *u * 0.5 && *u > *l * 2.0)
        .collect()
}

fn engulfing(is_green: &[bool], is_red: &[bool], bodies: &[f64], bullish: bool) -> Vec<bool> {
    bodies
        .iter()
        .enumerate()
        .map(|(i, &body)| {
            if i == 0 {
                return false;
            }
            let prev_body = bodies[i - 1].abs();
            if bullish {
                is_green[i] && is_red[i - 1] && body.abs() > prev_body
            } else {
                is_red[i] && is_green[i - 1] && body.abs() > prev_body
            }
        })
        .collect()
}

fn large_colored_body(condition: &[bool], bodies: &[f64], atr: &[f64], mult: f64) -> Vec<bool> {
    bodies
        .iter()
        .zip(condition.iter().zip(atr.iter()))
        .map(|(body, (cond, atr_val))| *cond && body.abs() > atr_val * mult)
        .collect()
}

fn large_body_ratio(values: &[f64], baseline: &[f64], mult: f64) -> Vec<bool> {
    values
        .iter()
        .zip(baseline.iter())
        .map(|(value, mean)| {
            if !value.is_finite() || !mean.is_finite() {
                return false;
            }
            if *value <= BODY_SIZE_EPS {
                return false;
            }
            *value > *mean * mult
        })
        .collect()
}

fn cross(fast: &[f64], slow: &[f64], upward: bool) -> Vec<bool> {
    fast.iter()
        .enumerate()
        .map(|(i, &value)| {
            if i == 0 || !value.is_finite() {
                return false;
            }
            let prev_fast = fast[i - 1];
            let prev_slow = slow[i - 1];
            let current_slow = slow[i];
            if !prev_fast.is_finite() || !prev_slow.is_finite() || !current_slow.is_finite() {
                return false;
            }
            if upward {
                prev_fast <= prev_slow && value > current_slow
            } else {
                prev_fast >= prev_slow && value < current_slow
            }
        })
        .collect()
}

fn stoch_cross(
    _rounded_fast: &[f64],
    _rounded_slow: &[f64],
    raw_fast: &[f64],
    raw_slow: &[f64],
    upward: bool,
) -> Vec<bool> {
    raw_fast
        .iter()
        .enumerate()
        .map(|(i, &current_fast)| {
            if i == 0 || !current_fast.is_finite() {
                return false;
            }
            let current_slow = raw_slow[i];
            if !current_slow.is_finite() {
                return false;
            }
            let prev_fast = raw_fast[i - 1];
            let prev_slow = raw_slow[i - 1];
            if !prev_fast.is_finite() || !prev_slow.is_finite() {
                return false;
            }
            if upward {
                (current_fast > current_slow) && (prev_fast <= prev_slow)
            } else {
                (current_fast < current_slow) && (prev_fast >= prev_slow)
            }
        })
        .collect()
}

fn dual_condition(first: &[bool], second: &[bool]) -> Vec<bool> {
    first
        .iter()
        .zip(second.iter())
        .map(|(a, b)| *a && *b)
        .collect()
}

fn vector_abs(values: &[f64]) -> Vec<f64> {
    values.iter().map(|v| v.abs()).collect()
}

fn add_scalar(values: &[f64], scalar: f64) -> Vec<f64> {
    values.iter().map(|v| v + scalar).collect()
}

fn adx(high: &[f64], low: &[f64], close: &[f64], period: usize) -> Vec<f64> {
    let len = close.len();
    let mut plus_dm = vec![0.0; len];
    let mut minus_dm = vec![0.0; len];
    let mut tr = vec![0.0; len];

    for i in 1..len {
        let up_move = high[i] - high[i - 1];
        let down_move = low[i - 1] - low[i];
        plus_dm[i] = if up_move > down_move && up_move > 0.0 {
            up_move
        } else {
            0.0
        };
        minus_dm[i] = if down_move > up_move && down_move > 0.0 {
            down_move
        } else {
            0.0
        };
        let high_low = high[i] - low[i];
        let high_close = (high[i] - close[i - 1]).abs();
        let low_close = (low[i] - close[i - 1]).abs();
        tr[i] = high_low.max(high_close).max(low_close);
    }

    let atr_values = rma(&tr, period);
    let plus_smoothed = rma(&plus_dm, period);
    let minus_smoothed = rma(&minus_dm, period);
    let plus_di = plus_smoothed
        .iter()
        .zip(atr_values.iter())
        .map(|(p, atr)| {
            if atr.abs() < f64::EPSILON {
                0.0
            } else {
                (p / atr) * 100.0
            }
        })
        .collect::<Vec<_>>();
    let minus_di = minus_smoothed
        .iter()
        .zip(atr_values.iter())
        .map(|(m, atr)| {
            if atr.abs() < f64::EPSILON {
                0.0
            } else {
                (m / atr) * 100.0
            }
        })
        .collect::<Vec<_>>();
    let dx = plus_di
        .iter()
        .zip(minus_di.iter())
        .map(|(p, m)| {
            if (p + m).abs() < f64::EPSILON {
                0.0
            } else {
                ((p - m).abs() / (p + m)) * 100.0
            }
        })
        .collect::<Vec<_>>();
    let dx_clean = dx
        .into_iter()
        .map(|v| if v.is_finite() { v } else { 0.0 })
        .collect::<Vec<_>>();
    rma(&dx_clean, period)
}
