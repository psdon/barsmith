use std::path::PathBuf;

use chrono::NaiveDate;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum StackingMode {
    /// Treat every mask-hit bar as an independent trade sample (legacy comb semantics).
    Stacking,
    /// Enforce non-overlapping trades (at most one open trade at a time).
    NoStacking,
}

fn default_stacking_mode() -> StackingMode {
    StackingMode::NoStacking
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PositionSizingMode {
    /// Legacy mode: treat 1R as `risk_pct_per_trade` of current equity and apply RR directly to that.
    Fractional,
    /// Futures-style mode: compute integer contracts from risk budget and per-trade stop distance.
    Contracts,
}

fn default_position_sizing() -> PositionSizingMode {
    PositionSizingMode::Contracts
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum StopDistanceUnit {
    Points,
    Ticks,
}

fn default_stop_distance_unit() -> StopDistanceUnit {
    StopDistanceUnit::Points
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub input_csv: PathBuf,
    /// Optional canonical source CSV path used for fingerprinting.
    /// When set (CLI use), this stays bound to the original --csv argument
    /// even if input_csv is later rewritten to a prepared/engineered file.
    #[serde(default)]
    pub source_csv: Option<PathBuf>,
    pub direction: Direction,
    pub target: String,
    pub output_dir: PathBuf,
    pub max_depth: usize,
    /// Minimum samples required for a combination to be stored in the
    /// cumulative results. Combinations below this threshold are evaluated
    /// but not persisted.
    pub min_sample_size: usize,
    /// Minimum samples required for a combination to be considered
    /// "eligible" in reporting (top tables, overview). When not explicitly
    /// provided, this defaults to `min_sample_size`.
    #[serde(default)]
    pub min_sample_size_report: usize,
    pub logic_mode: LogicMode,
    pub include_date_start: Option<NaiveDate>,
    /// Inclusive end-date filter applied when loading the engineered dataset
    /// for evaluation. When set, only rows with date(timestamp) <= this
    /// value are retained in the in-memory dataset.
    pub include_date_end: Option<NaiveDate>,
    pub batch_size: usize,
    pub n_workers: usize,
    /// When true, Barsmith may adjust the effective batch size between
    /// iterations based on recent filter/eval timings. This never changes
    /// which combinations are evaluated, only how many are pulled per chunk.
    #[serde(default)]
    pub auto_batch: bool,
    /// When true, Barsmith may stop early once a sustained window of batches
    /// is almost entirely reused (very high reuse, negligible pruning). This
    /// can shorten long runs at the cost of potentially skipping a small tail
    /// of still-unevaluated combinations.
    #[serde(default)]
    pub early_exit_when_reused: bool,
    pub resume_offset: u64,
    /// True when the resume offset was explicitly provided by the caller (e.g. via CLI),
    /// even if that value is zero. This is used to distinguish "no override"
    /// from "explicitly requested resume-from=0" when deciding whether to
    /// apply stored resume metadata from DuckDB.
    #[serde(default)]
    pub explicit_resume_offset: bool,
    pub max_combos: Option<usize>,
    pub dry_run: bool,
    #[serde(default)]
    pub quiet: bool,
    #[serde(default)]
    pub report_metrics: ReportMetricsMode,
    #[serde(default = "default_report_top")]
    pub report_top: usize,
    #[serde(default)]
    pub force_recompute: bool,
    /// Maximum allowed drawdown (in R units) for a combination to be
    /// persisted into results_parquet. Any combination with max_drawdown
    /// above this threshold is never stored.
    #[serde(default = "default_max_drawdown")]
    pub max_drawdown: f64,
    /// Optional drawdown ceiling to apply when querying top results. When
    /// None, reporting uses `max_drawdown` as its filter so storage and
    /// reporting share the same threshold.
    #[serde(default)]
    pub max_drawdown_report: Option<f64>,
    /// Optional minimum Calmar ratio to apply when querying top results.
    /// When None, reporting does not enforce a Calmar floor.
    #[serde(default)]
    pub min_calmar_report: Option<f64>,
    /// When enabled, treat combinations (and scalar comparison predicates)
    /// with 0 < sample < min_sample_size as dead subsets and prune all
    /// supersets. This trades completeness of the under-sampled surface for
    /// more aggressive search-space trimming.
    #[serde(default)]
    pub strict_min_pruning: bool,
    /// Enable dynamic subset-based pruning for combination search.
    ///
    /// When true, depth-2 combinations that produce zero samples are cached
    /// as "dead pairs", and any higher-depth combination containing one of
    /// those pairs is skipped without full bitset evaluation. This is an
    /// optional, more aggressive pruning layer on top of strict_min_pruning.
    #[serde(default)]
    pub enable_subset_pruning: bool,
    /// Enable feature-to-feature comparisons (pairwise numeric predicates)
    #[serde(default)]
    pub enable_feature_pairs: bool,
    /// Optional cap on the number of feature-to-feature comparison
    /// predicates generated for this configuration.
    #[serde(default)]
    pub feature_pairs_limit: Option<usize>,
    /// Optional hash of the effective feature catalog (boolean, continuous,
    /// thresholds, and feature-to-feature predicates). When set, this is
    /// folded into the stable config hash so resume offsets are only reused
    /// when the catalog is identical.
    #[serde(default)]
    pub catalog_hash: Option<String>,
    /// Controls how much per-combination statistics detail is computed.
    ///
    /// In `Full` mode, Barsmith computes the complete set of risk/shape
    /// metrics (Sharpe, Sortino, streaks, percentiles, ulcer/pain/BDE, etc.).
    /// In `Core` mode, only essential fields used for ranking and storage
    /// (win rate, total_return, max_drawdown, equity Calmar) are populated;
    /// the remaining shape fields are left at zero to keep evaluation cheaper
    /// while preserving the results schema.
    #[serde(default)]
    pub stats_detail: StatsDetail,

    /// Optional eval-time profiling mode. When enabled, Barsmith will emit
    /// additional timing breakdowns for the evaluation phase (eval_ms).
    #[serde(default)]
    pub eval_profile: EvalProfileMode,

    /// Optional sampling rate for eval-time profiling. When > 1, only
    /// ~1/N combinations are instrumented (based on a deterministic hash
    /// of the combination indices) to reduce overhead.
    #[serde(default = "default_eval_profile_sample_rate")]
    pub eval_profile_sample_rate: usize,

    /// Optional S3 base URI to upload run artefacts to.
    ///
    /// This must be in the form `s3://bucket/prefix` (prefix optional).
    #[serde(default)]
    pub s3_output: Option<String>,

    /// When true, upload the newly produced Parquet batch part (and selected
    /// artefacts) to `s3_output` after every batch is ingested.
    #[serde(default)]
    pub s3_upload_each_batch: bool,

    /// Optional starting capital in USD used for equity-curve simulation and
    /// dollar-denominated statistics. When omitted, the CLI applies a default
    /// capital for combination runs and populates this field.
    #[serde(default)]
    pub capital_dollar: Option<f64>,

    /// Optional risk percentage per trade, applied to current equity when
    /// simulating the equity curve. When omitted, the CLI applies a default
    /// risk percentage for combination runs and populates this field.
    #[serde(default)]
    pub risk_pct_per_trade: Option<f64>,

    /// Optional time horizon in years for the engineered dataset, inferred
    /// from the timestamp range. This is used to annualize equity-based
    /// statistics such as CAGR, Sharpe, and Sortino.
    #[serde(default)]
    pub equity_time_years: Option<f64>,

    /// Optional asset code used for cost modeling (e.g., "ES", "MES").
    #[serde(default)]
    pub asset: Option<String>,

    /// Risk per trade in dollars (defines 1R in USD when set).
    #[serde(default)]
    pub risk_per_trade_dollar: Option<f64>,

    /// Round-trip cost per trade in dollars (commission + slippage).
    #[serde(default)]
    pub cost_per_trade_dollar: Option<f64>,

    /// Cost per trade in R units (typically cost_per_trade_dollar / risk_per_trade_dollar).
    #[serde(default)]
    pub cost_per_trade_r: Option<f64>,

    /// Dollars per one R (alias for risk_per_trade_dollar when set).
    #[serde(default)]
    pub dollars_per_r: Option<f64>,

    /// Optional tick size for the underlying asset (e.g., 0.25 for ES/MES).
    /// When set, target generators that depend on synthetic stop levels
    /// (such as `next_bar_color_and_wicks`) may snap those stops to the
    /// nearest valid tick to better reflect executable prices.
    #[serde(default)]
    pub tick_size: Option<f64>,

    /// Trade stacking behavior for combination evaluation.
    ///
    /// When `no_stacking`, Barsmith uses target-provided exit indices to
    /// ensure trades do not overlap (more realistic for live execution).
    #[serde(default = "default_stacking_mode")]
    pub stacking_mode: StackingMode,

    /// Position sizing model for equity simulation and dollar-denominated metrics.
    ///
    /// - `fractional` (default): per-trade risk is `risk_pct_per_trade` of current equity.
    /// - `contracts`: compute integer contracts from risk budget and stop distance.
    #[serde(default = "default_position_sizing")]
    pub position_sizing: PositionSizingMode,

    /// Column name for per-trade stop distance used in `contracts` sizing mode.
    ///
    /// This should represent the 1R stop distance per contract (either in points or ticks).
    #[serde(default)]
    pub stop_distance_column: Option<String>,

    /// Unit for `stop_distance_column` in `contracts` sizing mode.
    #[serde(default = "default_stop_distance_unit")]
    pub stop_distance_unit: StopDistanceUnit,

    /// Minimum contracts to trade in `contracts` sizing mode (default: 1).
    ///
    /// If the computed contract count is below this value, it is clamped up to `min_contracts`.
    #[serde(default = "default_min_contracts")]
    pub min_contracts: usize,

    /// Optional maximum contracts cap in `contracts` sizing mode.
    #[serde(default)]
    pub max_contracts: Option<usize>,

    /// Optional asset point value (e.g. ES=50, MES=5). Required for `contracts` sizing when stop distance is in points.
    #[serde(default)]
    pub point_value: Option<f64>,

    /// Optional asset tick value (e.g. ES=12.50, MES=1.25). Required for `contracts` sizing when stop distance is in ticks.
    #[serde(default)]
    pub tick_value: Option<f64>,

    /// Optional initial/overnight margin per contract in USD for `contracts` sizing mode.
    ///
    /// When set and > 0, Barsmith caps contracts as: floor(current_equity / margin_per_contract_dollar).
    #[serde(default)]
    pub margin_per_contract_dollar: Option<f64>,

    /// Optional gating: when non-empty, Barsmith will only evaluate combinations
    /// that include at least one of these feature names (after catalog pruning).
    /// This is an evaluation-time skip and does not change how combinations are
    /// enumerated or how resume offsets are computed.
    #[serde(default)]
    pub require_any_features: Vec<String>,
}

fn default_min_contracts() -> usize {
    1
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum Direction {
    Long,
    Short,
    Both,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum LogicMode {
    And,
    Or,
    Both,
}

#[allow(dead_code)]
impl LogicMode {
    pub fn requires_or(self) -> bool {
        matches!(self, LogicMode::Or | LogicMode::Both)
    }

    pub fn requires_and(self) -> bool {
        matches!(self, LogicMode::And | LogicMode::Both)
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
pub enum ReportMetricsMode {
    /// Emit full metrics blocks for each top combination.
    #[default]
    Full,
    /// Emit only the ranked combination formulas (no metrics), honoring --top-k.
    FormulasOnly,
    /// Disable reporting entirely.
    Off,
}

impl ReportMetricsMode {
    pub fn should_report(self) -> bool {
        !matches!(self, ReportMetricsMode::Off)
    }

    pub fn is_full(self) -> bool {
        matches!(self, ReportMetricsMode::Full)
    }

    pub fn is_formulas_only(self) -> bool {
        matches!(self, ReportMetricsMode::FormulasOnly)
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
pub enum StatsDetail {
    #[default]
    Core,
    Full,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
pub enum EvalProfileMode {
    #[default]
    Off,
    Coarse,
    Fine,
}

const fn default_report_top() -> usize {
    5
}

const fn default_max_drawdown() -> f64 {
    30.0
}

const fn default_eval_profile_sample_rate() -> usize {
    1
}
