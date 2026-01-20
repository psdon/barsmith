use std::path::PathBuf;

use anyhow::{Context, Result};
use chrono::NaiveDate;
use clap::{Parser, Subcommand, ValueEnum};

use crate::stats_detail::StatsDetailValue;
use barsmith_rs::asset::find_asset;
use barsmith_rs::config::{
    Config, Direction, EvalProfileMode, LogicMode, PositionSizingMode, ReportMetricsMode,
    StackingMode, StopDistanceUnit,
};

const DEFAULT_CAPITAL_DOLLAR: f64 = 100_000.0;
const DEFAULT_RISK_PCT_PER_TRADE: f64 = 1.0;

#[derive(Parser, Debug)]
#[command(
    name = "barsmith",
    about = "High-performance feature permutation explorer"
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Run feature combination search over an engineered dataset
    #[command(name = "comb")]
    Comb(CombArgs),
}

#[derive(Parser, Debug)]
pub struct CombArgs {
    /// Path to the input CSV file with OHLCV data
    #[arg(long = "csv", value_name = "FILE", value_hint = clap::ValueHint::FilePath)]
    pub csv_path: PathBuf,

    /// Direction to analyze (long/short/both)
    #[arg(long, default_value = "long")]
    pub direction: DirectionValue,

    /// Target generator identifier
    #[arg(long, default_value = "next_bar_color_and_wicks")]
    pub target: String,

    /// Output directory for cumulative artefacts
    #[arg(long = "output-dir", value_hint = clap::ValueHint::DirPath)]
    pub output_dir: PathBuf,

    /// Optional S3 base URI to upload output artefacts to (e.g. s3://bucket/prefix).
    ///
    /// When combined with --s3-upload-each-batch, Barsmith will upload the
    /// newly produced Parquet part (and resume metadata) after every batch.
    #[arg(long = "s3-output", value_name = "S3_URI")]
    pub s3_output: Option<String>,

    /// Upload results to S3 after every batch is ingested.
    #[arg(long = "s3-upload-each-batch", default_value_t = false)]
    pub s3_upload_each_batch: bool,

    /// Maximum feature depth per combination
    #[arg(long = "max-depth", default_value_t = 3)]
    pub max_depth: usize,

    /// Minimum samples required for storing a combination in cumulative
    /// results. Combinations below this threshold are evaluated but not
    /// persisted.
    #[arg(long = "min-samples", alias = "min-sample-size", default_value_t = 100)]
    pub min_samples: usize,

    /// Minimum samples required for a combination to be considered
    /// "eligible" in reporting (top tables, overview). When omitted, this
    /// defaults to the value of --min-samples.
    #[arg(long = "min-samples-report")]
    pub min_samples_report: Option<usize>,

    /// Logical operator for feature combinations (and/or/both)
    #[arg(long = "logic", alias = "logic-mode", default_value = "and")]
    pub logic: LogicModeValue,

    /// Inclusive start date filter (YYYY-MM-DD)
    #[arg(long = "date-start")]
    pub date_start: Option<String>,

    /// Inclusive end date filter (YYYY-MM-DD)
    #[arg(long = "date-end")]
    pub date_end: Option<String>,

    /// Batch size for combination streaming
    #[arg(long = "batch-size", default_value_t = 20_000)]
    pub batch_size: usize,

    /// Enable adaptive batch sizing based on recent filter/eval timings
    #[arg(long = "auto-batch", default_value_t = false)]
    pub auto_batch: bool,

    /// Optionally stop early once a sustained window of batches is almost
    /// entirely reused (very high reuse, negligible pruning). This can save
    /// wall-clock time on long runs but may skip a small tail of remaining
    /// combinations.
    #[arg(long = "early-exit-when-reused", default_value_t = false)]
    pub early_exit_when_reused: bool,

    /// Number of worker threads (omit to use all logical cores)
    #[arg(long = "workers", alias = "n-jobs")]
    pub workers: Option<usize>,

    /// Optional resume offset for combination enumeration
    #[arg(long = "resume-from", alias = "resume-offset", default_value_t = 0)]
    pub resume_from: u64,

    /// Optional cap on combinations to evaluate this run
    #[arg(long = "max-combos", alias = "limit")]
    pub max_combos: Option<usize>,

    /// Enable dry-run mode (emit plan only)
    #[arg(long = "dry-run", default_value_t = false)]
    pub dry_run: bool,

    /// Reduce log noise (suppresses catalog/resume summaries)
    #[arg(long = "quiet", default_value_t = false)]
    pub quiet: bool,

    /// Control final metrics report emission (presets also update --report-top)
    #[arg(
        long = "report",
        alias = "report-metrics",
        value_enum,
        default_value = "full"
    )]
    pub report_metrics: ReportMetricsValue,

    /// Number of combinations to include in the final summary table
    #[arg(long = "top-k", alias = "report-top", default_value_t = 5)]
    pub top_k: usize,

    /// Force recompute even if existing results already cover the requested max-depth or CSV fingerprint
    #[arg(long = "force", alias = "force-recompute", default_value_t = false)]
    pub force: bool,

    /// Acknowledge that a newly engineered dataset differs from an existing
    /// barsmith_prepared.csv in the output directory; overwrite and continue.
    #[arg(long = "ack-new-df", default_value_t = false)]
    pub ack_new_df: bool,

    /// Enable numeric feature-to-feature comparisons (pairwise conditions)
    #[arg(
        long = "feature-pairs",
        alias = "enable-feature-pairs",
        default_value_t = false
    )]
    pub feature_pairs: bool,

    /// Maximum number of feature-to-feature comparison predicates to generate
    #[arg(long = "feature-pairs-limit", alias = "feature-pairs-max")]
    pub feature_pairs_limit: Option<usize>,

    /// Maximum allowed drawdown (in R units) for a combination to be stored
    /// in results_parquet and considered in top-results queries.
    #[arg(long = "max-drawdown", default_value_t = 30.0)]
    pub max_drawdown: f64,

    /// Optional drawdown ceiling applied only to reporting queries. When
    /// omitted, reporting uses --max-drawdown as its filter.
    #[arg(long = "max-drawdown-report")]
    pub max_drawdown_report: Option<f64>,

    /// Optional minimum Calmar ratio applied only to reporting queries.
    /// When omitted, reporting does not enforce a Calmar floor.
    #[arg(long = "min-calmar-report")]
    pub min_calmar_report: Option<f64>,

    /// Disable writing barsmith.log into the output directory. When set,
    /// logs are only emitted to stdout/stderr.
    #[arg(long = "no-file-log", default_value_t = false)]
    pub no_file_log: bool,

    /// Enable subset-based pruning of higher-depth combinations using
    /// zero-sample depth-2 pairs as dead prefixes.
    #[arg(
        long = "subset-pruning",
        alias = "enable-subset-pruning",
        default_value_t = false
    )]
    pub subset_pruning: bool,

    /// Control how much per-combination statistics detail is computed.
    ///
    /// In `full` mode, Barsmith computes the complete set of risk/shape
    /// metrics (Sharpe, Sortino, streaks, percentiles, ulcer/pain/BDE, etc.).
    /// In `core` mode, only essential fields used for ranking and storage
    /// (win rate, total_return, max_drawdown, calmar_ratio) are populated;
    /// the remaining fields are left at zero to keep evaluation cheaper
    /// while preserving the results schema.
    #[arg(long = "stats-detail", value_enum, default_value = "core")]
    pub stats_detail: StatsDetailValue,

    /// Emit eval_ms timing breakdowns to help profile performance hotspots.
    ///
    /// - off: no additional overhead (default)
    /// - coarse: low-overhead phase timings (build/scan/finalize)
    /// - fine: higher-overhead timing that also measures per-hit (on_hit) work
    #[arg(long = "profile-eval", value_enum, default_value = "off")]
    pub profile_eval: EvalProfileValue,

    /// Optional sampling rate for eval profiling. When > 1, only ~1/N
    /// combinations are instrumented (deterministically) to reduce overhead.
    #[arg(long = "profile-eval-sample-rate", default_value_t = 1)]
    pub profile_eval_sample_rate: usize,

    /// Starting capital in USD for equity simulation and dollar metrics.
    /// When omitted, a default (e.g., 100_000) is applied.
    #[arg(long = "capital")]
    pub capital: Option<f64>,

    /// Risk percentage per trade, applied to current equity.
    /// When omitted, a default (e.g., 1.0) is applied.
    #[arg(long = "risk-pct-per-trade")]
    pub risk_pct_per_trade: Option<f64>,

    /// Asset code for cost modeling (e.g., ES, MES).
    #[arg(long = "asset")]
    pub asset: Option<String>,

    /// Position sizing mode for equity simulation.
    ///
    /// - fractional: risk `--risk-pct-per-trade` of current equity each trade (legacy)
    /// - contracts: compute integer contracts from risk budget and stop distance
    #[arg(long = "position-sizing", value_enum, default_value = "fractional")]
    pub position_sizing: PositionSizingValue,

    /// Column name containing per-trade stop distance for contract sizing (in points or ticks).
    ///
    /// If omitted in `contracts` mode, Barsmith will try to infer a default for ATR-stop targets.
    #[arg(long = "stop-distance-column")]
    pub stop_distance_column: Option<String>,

    /// Unit for --stop-distance-column.
    #[arg(long = "stop-distance-unit", value_enum, default_value = "points")]
    pub stop_distance_unit: StopDistanceUnitValue,

    /// Minimum contracts to trade in contract sizing mode (default: 1).
    #[arg(long = "min-contracts", default_value_t = 1)]
    pub min_contracts: usize,

    /// Optional maximum contracts cap in contract sizing mode.
    #[arg(long = "max-contracts")]
    pub max_contracts: Option<usize>,

    /// Initial/overnight margin per contract in USD for contracts sizing (optional).
    ///
    /// When set, Barsmith caps contracts as floor(current_equity / margin_per_contract_dollar).
    #[arg(long = "margin-per-contract-dollar")]
    pub margin_per_contract_dollar: Option<f64>,

    /// Round-trip commission per trade in dollars (overrides asset default).
    #[arg(long = "commission-per-trade-dollar")]
    pub commission_per_trade_dollar: Option<f64>,

    /// Round-trip slippage per trade in dollars (overrides asset default).
    #[arg(long = "slippage-per-trade-dollar")]
    pub slippage_per_trade_dollar: Option<f64>,

    /// Round-trip total cost per trade in dollars (overrides commission+slippage).
    #[arg(long = "cost-per-trade-dollar")]
    pub cost_per_trade_dollar: Option<f64>,

    /// Disable cost model entirely and keep raw R semantics.
    #[arg(long = "no-costs", default_value_t = false)]
    pub no_costs: bool,

    /// Optional gating: only evaluate combinations that include at least one
    /// of the provided feature names (comma-delimited).
    ///
    /// Example: --require-any-features is_tribar_hl_green,is_tribar_hl_red
    #[arg(long = "require-any-features", value_delimiter = ',', num_args = 0..)]
    pub require_any_features: Vec<String>,

    /// Trade stacking behavior for `comb` evaluation.
    ///
    /// - stacking: treat every mask-hit bar as an independent trade sample (legacy).
    /// - no-stacking: enforce one open trade at a time using target exit indices.
    #[arg(long = "stacking-mode", value_enum, default_value = "no-stacking")]
    pub stacking_mode: StackingModeValue,
    // Zero-sample pruning, cross-run seeding, coverage checks, and
    // storage-backed membership reuse have been removed in favor of a
    // simpler, evaluation-only engine. The corresponding flags are no
    // longer exposed at the CLI level.
}

impl Cli {
    pub fn parse() -> Self {
        <Cli as Parser>::parse()
    }
}

impl CombArgs {
    pub fn into_config(self) -> Result<Config> {
        let include_date_start = parse_optional_date(self.date_start.as_deref())?;
        let include_date_end = parse_optional_date(self.date_end.as_deref())?;
        let logic_mode = self.logic.to_logic_mode();
        let direction = self.direction.to_direction();
        // Resolve how many combinations to include in the final summary table:
        // - When `--report off`, skip reporting entirely (top = 0).
        // - When `--report full`, honor `--top-k` (default 5) and clamp to >= 1.
        // - When `--report top10`, use 10 if the user left `--top-k` at its
        //   default (5), otherwise honor the explicit `--top-k` override.
        let report_top = match self.report_metrics {
            ReportMetricsValue::Off => 0,
            ReportMetricsValue::Full => self.top_k.max(1),
            ReportMetricsValue::Formula => self.top_k.max(1),
            ReportMetricsValue::Top10 => {
                let effective = if self.top_k == 5 { 10 } else { self.top_k };
                effective.max(1)
            }
            ReportMetricsValue::Top100 => {
                let effective = if self.top_k == 5 { 100 } else { self.top_k };
                effective.max(1)
            }
        };
        // Detect whether the user explicitly provided --resume-from/--resume-offset,
        // even if the value is zero, so we can treat that as an override of any
        // stored resume metadata.
        let explicit_resume_offset = detect_explicit_resume_flag(std::env::args());
        let min_sample_size = self.min_samples;
        let min_sample_size_report = self.min_samples_report.unwrap_or(min_sample_size);

        // Derive capital and risk percentage, applying sensible defaults when omitted.
        let capital = self.capital.unwrap_or(DEFAULT_CAPITAL_DOLLAR);
        let risk_pct = self
            .risk_pct_per_trade
            .unwrap_or(DEFAULT_RISK_PCT_PER_TRADE);
        let risk_per_trade_dollar = if risk_pct > 0.0 {
            Some(capital * risk_pct / 100.0)
        } else {
            None
        };

        // Derive asset-aware cost model parameters if requested. Costs are applied
        // per round-trip trade in USD and converted into R units using the
        // derived risk_per_trade_dollar when available.
        let mut asset_code: Option<String> = None;
        let mut effective_cost_dollar: Option<f64> = None;
        let mut tick_size: Option<f64> = None;
        let mut point_value: Option<f64> = None;
        let mut tick_value: Option<f64> = None;
        let mut margin_per_contract_dollar: Option<f64> = None;

        if !self.no_costs {
            if let Some(code) = &self.asset {
                let asset = find_asset(code)
                    .ok_or_else(|| anyhow::anyhow!("Unknown asset code '{}'", code))?;

                let base_commission = 2.0 * asset.ibkr_commission_per_side;
                let base_slippage = asset.default_slippage_ticks * asset.tick_value;
                tick_size = Some(asset.tick_size);
                point_value = Some(asset.point_value);
                tick_value = Some(asset.tick_value);
                margin_per_contract_dollar = Some(asset.margin_per_contract_dollar);

                let commission = self.commission_per_trade_dollar.unwrap_or(base_commission);
                let slippage = self.slippage_per_trade_dollar.unwrap_or(base_slippage);
                let cost = self.cost_per_trade_dollar.unwrap_or(commission + slippage);

                asset_code = Some(asset.code.to_string());
                effective_cost_dollar = Some(cost);
            } else if let Some(cost) = self.cost_per_trade_dollar {
                effective_cost_dollar = Some(cost);
            } else if self.commission_per_trade_dollar.is_some()
                || self.slippage_per_trade_dollar.is_some()
            {
                let commission = self.commission_per_trade_dollar.unwrap_or(0.0);
                let slippage = self.slippage_per_trade_dollar.unwrap_or(0.0);
                effective_cost_dollar = Some(commission + slippage);
            }
        } else if let Some(code) = &self.asset {
            // Even when cost modeling is disabled, retain the asset code and
            // tick size so engineered targets can still snap stops to the
            // correct price grid.
            let asset =
                find_asset(code).ok_or_else(|| anyhow::anyhow!("Unknown asset code '{}'", code))?;
            asset_code = Some(asset.code.to_string());
            tick_size = Some(asset.tick_size);
            point_value = Some(asset.point_value);
            tick_value = Some(asset.tick_value);
            margin_per_contract_dollar = Some(asset.margin_per_contract_dollar);
        }

        let position_sizing = self.position_sizing.to_mode();
        if matches!(position_sizing, PositionSizingMode::Contracts) && self.asset.is_none() {
            return Err(anyhow::anyhow!(
                "--position-sizing contracts requires --asset so point/tick values are known"
            ));
        }

        let stop_distance_column = if matches!(position_sizing, PositionSizingMode::Contracts) {
            self.stop_distance_column
                .or_else(|| infer_stop_distance_column(&self.target))
        } else {
            None
        };
        if matches!(position_sizing, PositionSizingMode::Contracts)
            && stop_distance_column.is_none()
        {
            return Err(anyhow::anyhow!(
                "--position-sizing contracts requires --stop-distance-column (or a target that infers it)"
            ));
        }

        let (cost_per_trade_dollar, cost_per_trade_r, dollars_per_r) = match position_sizing {
            PositionSizingMode::Fractional => {
                match (effective_cost_dollar, risk_per_trade_dollar) {
                    (Some(cost), Some(risk_dollar)) if risk_dollar > 0.0 => {
                        let cost_r = cost / risk_dollar;
                        (Some(cost), Some(cost_r), Some(risk_dollar))
                    }
                    (Some(cost), _) => (Some(cost), None, risk_per_trade_dollar),
                    (None, _) => (None, None, risk_per_trade_dollar),
                }
            }
            PositionSizingMode::Contracts => (effective_cost_dollar, None, risk_per_trade_dollar),
        };

        if !matches!(self.profile_eval, EvalProfileValue::Off) && self.profile_eval_sample_rate == 0
        {
            return Err(anyhow::anyhow!("--profile-eval-sample-rate must be >= 1"));
        }

        if self.s3_upload_each_batch && self.s3_output.is_none() {
            return Err(anyhow::anyhow!(
                "--s3-upload-each-batch requires --s3-output"
            ));
        }

        let target = if self.target == "atr_stop" {
            "2x_atr_tp_atr_stop".to_string()
        } else {
            self.target
        };

        Ok(Config {
            input_csv: self.csv_path.clone(),
            source_csv: Some(self.csv_path),
            direction,
            target,
            output_dir: self.output_dir,
            max_depth: self.max_depth,
            min_sample_size,
            min_sample_size_report,
            logic_mode,
            include_date_start,
            include_date_end,
            batch_size: self.batch_size.max(1),
            n_workers: normalize_workers(self.workers),
            auto_batch: self.auto_batch,
            early_exit_when_reused: self.early_exit_when_reused,
            resume_offset: self.resume_from,
            explicit_resume_offset,
            max_combos: self.max_combos,
            dry_run: self.dry_run,
            quiet: self.quiet,
            report_metrics: self.report_metrics.to_mode(),
            report_top,
            force_recompute: self.force,
            max_drawdown: self.max_drawdown,
            max_drawdown_report: self.max_drawdown_report,
            min_calmar_report: self.min_calmar_report,
            // Strict min pruning is now always enabled for scalar predicates;
            // the config field is retained only for metadata/debugging.
            strict_min_pruning: true,
            enable_feature_pairs: self.feature_pairs,
            feature_pairs_limit: self.feature_pairs_limit,
            enable_subset_pruning: self.subset_pruning,
            catalog_hash: None,
            stats_detail: self.stats_detail.to_mode(),
            eval_profile: self.profile_eval.to_mode(),
            eval_profile_sample_rate: self.profile_eval_sample_rate.max(1),
            s3_output: self.s3_output,
            s3_upload_each_batch: self.s3_upload_each_batch,
            capital_dollar: Some(capital),
            risk_pct_per_trade: Some(risk_pct),
            equity_time_years: None,
            asset: asset_code,
            risk_per_trade_dollar,
            cost_per_trade_dollar,
            cost_per_trade_r,
            dollars_per_r,
            tick_size,
            stacking_mode: self.stacking_mode.to_mode(),
            position_sizing,
            stop_distance_column,
            stop_distance_unit: self.stop_distance_unit.to_mode(),
            min_contracts: self.min_contracts.max(1),
            max_contracts: self.max_contracts,
            point_value,
            tick_value,
            margin_per_contract_dollar: self
                .margin_per_contract_dollar
                .or(margin_per_contract_dollar),
            require_any_features: {
                let mut names: Vec<String> = self
                    .require_any_features
                    .into_iter()
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect();
                names.sort();
                names.dedup();
                names
            },
        })
    }
}

fn infer_stop_distance_column(target: &str) -> Option<String> {
    match target {
        "2x_atr_tp_atr_stop" | "3x_atr_tp_atr_stop" | "atr_tp_atr_stop" | "atr_stop" => {
            Some("atr".to_string())
        }
        _ => None,
    }
}

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum EvalProfileValue {
    Off,
    Coarse,
    Fine,
}

impl EvalProfileValue {
    fn to_mode(self) -> EvalProfileMode {
        match self {
            EvalProfileValue::Off => EvalProfileMode::Off,
            EvalProfileValue::Coarse => EvalProfileMode::Coarse,
            EvalProfileValue::Fine => EvalProfileMode::Fine,
        }
    }
}

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum StackingModeValue {
    Stacking,
    #[value(name = "no-stacking")]
    NoStacking,
}

impl StackingModeValue {
    fn to_mode(self) -> StackingMode {
        match self {
            StackingModeValue::Stacking => StackingMode::Stacking,
            StackingModeValue::NoStacking => StackingMode::NoStacking,
        }
    }
}

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum PositionSizingValue {
    Fractional,
    Contracts,
}

impl PositionSizingValue {
    fn to_mode(self) -> PositionSizingMode {
        match self {
            PositionSizingValue::Fractional => PositionSizingMode::Fractional,
            PositionSizingValue::Contracts => PositionSizingMode::Contracts,
        }
    }
}

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum StopDistanceUnitValue {
    Points,
    Ticks,
}

impl StopDistanceUnitValue {
    fn to_mode(self) -> StopDistanceUnit {
        match self {
            StopDistanceUnitValue::Points => StopDistanceUnit::Points,
            StopDistanceUnitValue::Ticks => StopDistanceUnit::Ticks,
        }
    }
}

#[derive(ValueEnum, Clone, Copy, Debug)]
pub enum DirectionValue {
    Long,
    Short,
    Both,
}

impl DirectionValue {
    pub fn to_direction(self) -> Direction {
        match self {
            DirectionValue::Long => Direction::Long,
            DirectionValue::Short => Direction::Short,
            DirectionValue::Both => Direction::Both,
        }
    }
}

#[derive(ValueEnum, Clone, Copy, Debug)]
pub enum LogicModeValue {
    And,
    Or,
    Both,
}

impl LogicModeValue {
    fn to_logic_mode(self) -> LogicMode {
        match self {
            LogicModeValue::And => LogicMode::And,
            LogicModeValue::Or => LogicMode::Or,
            LogicModeValue::Both => LogicMode::Both,
        }
    }
}

#[derive(ValueEnum, Clone, Copy, Debug)]
pub enum ReportMetricsValue {
    Full,
    /// Only print ranked combination formulas (no metrics), honoring --top-k.
    Formula,
    Top10,
    Top100,
    Off,
}

impl ReportMetricsValue {
    fn to_mode(self) -> ReportMetricsMode {
        match self {
            ReportMetricsValue::Full => ReportMetricsMode::Full,
            ReportMetricsValue::Formula => ReportMetricsMode::FormulasOnly,
            ReportMetricsValue::Top10 => ReportMetricsMode::Full,
            ReportMetricsValue::Top100 => ReportMetricsMode::Full,
            ReportMetricsValue::Off => ReportMetricsMode::Off,
        }
    }
}

fn normalize_workers(value: Option<usize>) -> usize {
    value.unwrap_or_else(|| {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    })
}

fn detect_explicit_resume_flag<I, S>(args: I) -> bool
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    for arg in args {
        let raw = arg.as_ref();
        if raw == "--resume-from"
            || raw.starts_with("--resume-from=")
            || raw == "--resume-offset"
            || raw.starts_with("--resume-offset=")
        {
            return true;
        }
    }
    false
}

fn parse_optional_date(value: Option<&str>) -> Result<Option<NaiveDate>> {
    match value {
        Some(raw) => {
            let parsed = NaiveDate::parse_from_str(raw, "%Y-%m-%d")
                .with_context(|| format!("Invalid date format for {raw}. Expected YYYY-MM-DD"))?;
            Ok(Some(parsed))
        }
        None => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::NaiveDate;

    fn base_args() -> CombArgs {
        CombArgs {
            csv_path: PathBuf::from("dummy.csv"),
            direction: DirectionValue::Long,
            target: "next_bar_color_and_wicks".to_string(),
            output_dir: PathBuf::from("out"),
            s3_output: None,
            s3_upload_each_batch: false,
            max_depth: 3,
            min_samples: 100,
            min_samples_report: None,
            logic: LogicModeValue::And,
            date_start: None,
            date_end: None,
            batch_size: 20_000,
            auto_batch: false,
            early_exit_when_reused: false,
            workers: Some(4),
            resume_from: 0,
            max_combos: None,
            dry_run: false,
            quiet: false,
            report_metrics: ReportMetricsValue::Full,
            top_k: 5,
            force: false,
            ack_new_df: false,
            feature_pairs: false,
            feature_pairs_limit: None,
            max_drawdown: 50.0,
            max_drawdown_report: None,
            min_calmar_report: None,
            no_file_log: false,
            subset_pruning: false,
            stats_detail: StatsDetailValue::Full,
            profile_eval: EvalProfileValue::Off,
            profile_eval_sample_rate: 1,
            capital: None,
            risk_pct_per_trade: None,
            asset: None,
            position_sizing: PositionSizingValue::Fractional,
            stop_distance_column: None,
            stop_distance_unit: StopDistanceUnitValue::Points,
            min_contracts: 1,
            max_contracts: None,
            margin_per_contract_dollar: None,
            commission_per_trade_dollar: None,
            slippage_per_trade_dollar: None,
            cost_per_trade_dollar: None,
            no_costs: false,
            require_any_features: Vec::new(),
            stacking_mode: StackingModeValue::NoStacking,
        }
    }

    #[test]
    fn detect_explicit_resume_flag_matches_expected_patterns() {
        assert!(!detect_explicit_resume_flag([
            "barsmith_cli",
            "--csv",
            "foo.csv"
        ]));
        assert!(detect_explicit_resume_flag([
            "barsmith_cli",
            "--resume-from",
            "1000"
        ]));
        assert!(detect_explicit_resume_flag([
            "barsmith_cli",
            "--resume-from=0"
        ]));
        assert!(detect_explicit_resume_flag([
            "barsmith_cli",
            "--resume-offset=42"
        ]));
    }

    #[test]
    fn report_full_uses_top_k_and_clamps_to_one() {
        let mut args = base_args();
        args.report_metrics = ReportMetricsValue::Full;
        args.top_k = 0;
        let config = args.into_config().expect("config");
        assert!(
            matches!(config.report_metrics, ReportMetricsMode::Full),
            "report_metrics should be Full for full report mode"
        );
        assert_eq!(
            config.report_top, 1,
            "Full report should clamp top_k to at least 1"
        );
    }

    #[test]
    fn report_formula_uses_top_k_and_clamps_to_one() {
        let mut args = base_args();
        args.report_metrics = ReportMetricsValue::Formula;
        args.top_k = 0;
        let config = args.into_config().expect("config");
        assert!(
            matches!(config.report_metrics, ReportMetricsMode::FormulasOnly),
            "report_metrics should be FormulasOnly for formula report mode"
        );
        assert_eq!(
            config.report_top, 1,
            "Formula report should clamp top_k to at least 1"
        );
    }

    #[test]
    fn report_top10_uses_preset_when_top_k_is_default() {
        let mut args = base_args();
        args.report_metrics = ReportMetricsValue::Top10;
        args.top_k = 5; // default
        let config = args.into_config().expect("config");
        assert!(
            matches!(config.report_metrics, ReportMetricsMode::Full),
            "report_metrics should be Full for top10 preset"
        );
        assert_eq!(
            config.report_top, 10,
            "Top10 preset should default to 10 when top_k is left at its default"
        );
    }

    #[test]
    fn report_top10_respects_explicit_top_k_override() {
        let mut args = base_args();
        args.report_metrics = ReportMetricsValue::Top10;
        args.top_k = 3;
        let config = args.into_config().expect("config");
        assert!(
            matches!(config.report_metrics, ReportMetricsMode::Full),
            "report_metrics should be Full for top10 preset"
        );
        assert_eq!(
            config.report_top, 3,
            "Top10 preset should respect an explicit top_k override"
        );
    }

    #[test]
    fn report_top100_uses_preset_when_top_k_is_default() {
        let mut args = base_args();
        args.report_metrics = ReportMetricsValue::Top100;
        args.top_k = 5; // default
        let config = args.into_config().expect("config");
        assert!(
            matches!(config.report_metrics, ReportMetricsMode::Full),
            "report_metrics should be Full for top100 preset"
        );
        assert_eq!(
            config.report_top, 100,
            "Top100 preset should default to 100 when top_k is left at its default"
        );
    }

    #[test]
    fn report_top100_respects_explicit_top_k_override() {
        let mut args = base_args();
        args.report_metrics = ReportMetricsValue::Top100;
        args.top_k = 12;
        let config = args.into_config().expect("config");
        assert!(
            matches!(config.report_metrics, ReportMetricsMode::Full),
            "report_metrics should be Full for top100 preset"
        );
        assert_eq!(
            config.report_top, 12,
            "Top100 preset should respect an explicit top_k override"
        );
    }

    #[test]
    fn report_off_disables_reporting() {
        let mut args = base_args();
        args.report_metrics = ReportMetricsValue::Off;
        let config = args.into_config().expect("config");
        assert!(
            matches!(config.report_metrics, ReportMetricsMode::Off),
            "report_metrics should be Off when reporting is disabled"
        );
        assert_eq!(
            config.report_top, 0,
            "report_top should be zero when reporting is disabled"
        );
    }

    #[test]
    fn parse_optional_date_accepts_valid_yyyy_mm_dd() {
        let parsed = parse_optional_date(Some("2024-11-30"))
            .expect("parse should succeed")
            .expect("date should be present");
        let expected = NaiveDate::from_ymd_opt(2024, 11, 30).expect("valid date");
        assert_eq!(parsed, expected);
    }

    #[test]
    fn normalize_workers_prefers_explicit_value() {
        assert_eq!(normalize_workers(Some(2)), 2);
        assert!(
            normalize_workers(None) >= 1,
            "normalize_workers without explicit value should return at least 1"
        );
    }
}
