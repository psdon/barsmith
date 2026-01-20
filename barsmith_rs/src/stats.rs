use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result, anyhow};
use serde::Serialize;

use crate::combinator::{Combination, IndexCombination};
use crate::config::{
    Config, Direction, EvalProfileMode, PositionSizingMode, StackingMode, StatsDetail,
    StopDistanceUnit,
};
use crate::data::ColumnarData;
use crate::feature::{ComparisonOperator, ComparisonSpec, FeatureDescriptor};
use crate::mask::{MaskBuffer, MaskCache};

#[cfg(all(target_arch = "aarch64", feature = "simd-eval"))]
use std::arch::aarch64::*;

thread_local! {
    static RETURNS_BUFFER: RefCell<Vec<f64>> = const { RefCell::new(Vec::new()) };
    static RISK_PER_CONTRACT_BUFFER: RefCell<Vec<f64>> = const { RefCell::new(Vec::new()) };
    static SORTED_RETURNS_BUFFER: RefCell<Vec<f64>> = const { RefCell::new(Vec::new()) };
}

#[derive(Debug, Clone)]
struct BitsetMask {
    words: Vec<u64>,
    len: usize,
    support: usize,
}

impl BitsetMask {
    fn from_bools(values: &[bool]) -> Self {
        let len = values.len();
        if len == 0 {
            return Self {
                words: Vec::new(),
                len: 0,
                support: 0,
            };
        }
        let words_len = len.div_ceil(64);
        let mut words = vec![0u64; words_len];
        for (idx, value) in values.iter().enumerate() {
            if *value {
                let word = idx / 64;
                let bit = idx % 64;
                words[word] |= 1u64 << bit;
            }
        }
        let support = words.iter().map(|w| w.count_ones() as usize).sum();
        Self {
            words,
            len,
            support,
        }
    }

    fn from_finite_f64(values: &[f64]) -> Self {
        let len = values.len();
        if len == 0 {
            return Self {
                words: Vec::new(),
                len: 0,
                support: 0,
            };
        }
        let words_len = len.div_ceil(64);
        let mut words = vec![0u64; words_len];
        for (idx, value) in values.iter().enumerate() {
            if value.is_finite() {
                let word = idx / 64;
                let bit = idx % 64;
                words[word] |= 1u64 << bit;
            }
        }
        let support = words.iter().map(|w| w.count_ones() as usize).sum();
        Self {
            words,
            len,
            support,
        }
    }
}

#[derive(Clone)]
pub struct EvaluationContext {
    data: Arc<ColumnarData>,
    mask_cache: Arc<MaskCache>,
    comparisons: Arc<HashMap<String, ComparisonSpec>>,
    target: Arc<Vec<bool>>,
    rewards: Option<Arc<Vec<f64>>>,
    risk_per_contract_dollar: Option<Arc<Vec<f64>>>,
    eligible: Option<Arc<Vec<bool>>>,
    eligible_bitset: Option<Arc<BitsetMask>>,
    reward_finite_bitset: Option<Arc<BitsetMask>>,
    stacking_mode: StackingMode,
    exit_indices: Option<Arc<Vec<usize>>>,
    row_count: usize,
    stats_detail: StatsDetail,
    position_sizing: PositionSizingMode,
    min_contracts: usize,
    max_contracts: Option<usize>,
    cost_per_trade_dollar: Option<f64>,
    margin_per_contract_dollar: Option<f64>,
    cost_per_trade_r: Option<f64>,
    dollars_per_r: Option<f64>,
    capital_dollar: Option<f64>,
    risk_pct_per_trade: Option<f64>,
    equity_time_years: Option<f64>,
}

impl EvaluationContext {
    pub fn new(
        data: Arc<ColumnarData>,
        mask_cache: Arc<MaskCache>,
        config: &Config,
        comparisons: Arc<HashMap<String, ComparisonSpec>>,
    ) -> Result<Self> {
        let position_sizing = config.position_sizing;
        let min_contracts = config.min_contracts.max(1);
        let max_contracts = config.max_contracts;

        let target = Arc::new(load_boolean_vector(&data, &config.target)?);
        let eligible = {
            let column = format!("{}_eligible", config.target);
            if data.has_column(&column) {
                Some(Arc::new(load_boolean_vector(&data, &column)?))
            } else {
                None
            }
        };
        let eligible_bitset = eligible
            .as_deref()
            .map(|values| BitsetMask::from_bools(values.as_slice()))
            .map(Arc::new);

        let stop_distance_unit = config.stop_distance_unit;
        let risk_per_contract_dollar = if matches!(position_sizing, PositionSizingMode::Contracts) {
            let stop_col = config.stop_distance_column.as_deref().ok_or_else(|| {
                anyhow!(
                    "Missing stop_distance_column in config for position_sizing=contracts. Provide --stop-distance-column (or use a target that infers it)."
                )
            })?;
            let stop_distance = load_float_vector(&data, stop_col)?;
            let multiplier = match stop_distance_unit {
                StopDistanceUnit::Points => config.point_value.ok_or_else(|| {
                    anyhow!("Missing point_value in config for contracts sizing. Provide --asset or set config.point_value.")
                })?,
                StopDistanceUnit::Ticks => config.tick_value.ok_or_else(|| {
                    anyhow!("Missing tick_value in config for contracts sizing. Provide --asset or set config.tick_value.")
                })?,
            };
            if !multiplier.is_finite() || multiplier <= 0.0 {
                return Err(anyhow!("Invalid stop-distance multiplier: {multiplier}"));
            }
            let mut rpc: Vec<f64> = Vec::with_capacity(stop_distance.len());
            for v in stop_distance {
                if v.is_finite() && v > 0.0 {
                    let dollars = v * multiplier;
                    if dollars.is_finite() && dollars > 0.0 {
                        rpc.push(dollars);
                    } else {
                        rpc.push(f64::NAN);
                    }
                } else {
                    rpc.push(f64::NAN);
                }
            }
            Some(Arc::new(rpc))
        } else {
            None
        };

        let reward_column = detect_reward_column(&data, config)?;
        let (rewards, reward_finite_bitset) = match reward_column {
            Some(column) => {
                let mut values = load_float_vector(&data, &column)?;
                let mut any_non_finite = false;
                match position_sizing {
                    PositionSizingMode::Fractional => {
                        if let Some(cost_r) = config.cost_per_trade_r {
                            if cost_r != 0.0 {
                                for v in &mut values {
                                    if v.is_finite() {
                                        *v -= cost_r;
                                    } else {
                                        any_non_finite = true;
                                    }
                                }
                            }
                        }
                    }
                    PositionSizingMode::Contracts => {
                        let rpc = risk_per_contract_dollar.as_deref().ok_or_else(|| {
                            anyhow!("Missing risk_per_contract_dollar series for contracts sizing")
                        })?;
                        let cost_dollar = config.cost_per_trade_dollar.unwrap_or(0.0);
                        for (idx, v) in values.iter_mut().enumerate() {
                            let rpc_i = rpc.get(idx).copied().unwrap_or(f64::NAN);
                            if !v.is_finite() {
                                any_non_finite = true;
                                continue;
                            }
                            if !rpc_i.is_finite() || rpc_i <= 0.0 {
                                *v = f64::NAN;
                                any_non_finite = true;
                                continue;
                            }
                            if cost_dollar != 0.0 {
                                *v -= cost_dollar / rpc_i;
                            }
                        }
                    }
                }
                if !any_non_finite {
                    any_non_finite = values.iter().any(|v| !v.is_finite());
                }
                let finite = if any_non_finite {
                    Some(Arc::new(BitsetMask::from_finite_f64(&values)))
                } else {
                    None
                };
                (Some(Arc::new(values)), finite)
            }
            None => (None, None),
        };

        let stacking_mode = config.stacking_mode;
        let exit_indices = if stacking_mode == StackingMode::NoStacking {
            let column = format!("{}_exit_i", config.target);
            if !data.has_column(&column) {
                return Err(anyhow!(
                    "Missing required '{}' column for --stacking-mode no-stacking. Re-generate the prepared dataset (barsmith_prepared.csv) with a feature-engineering step that emits exit indices.",
                    column
                ));
            }
            let ca = data.i64_column(&column)?;
            let mut values: Vec<usize> = Vec::with_capacity(ca.len());
            for opt in ca.into_iter() {
                let idx = match opt {
                    Some(v) if v >= 0 => v as usize,
                    _ => usize::MAX,
                };
                values.push(idx);
            }
            Some(Arc::new(values))
        } else {
            None
        };
        Ok(Self {
            data: Arc::clone(&data),
            mask_cache,
            comparisons,
            target,
            rewards,
            risk_per_contract_dollar,
            eligible,
            eligible_bitset,
            reward_finite_bitset,
            stacking_mode,
            exit_indices,
            row_count: data.approx_rows(),
            stats_detail: config.stats_detail,
            position_sizing,
            min_contracts,
            max_contracts,
            cost_per_trade_dollar: config.cost_per_trade_dollar,
            margin_per_contract_dollar: config.margin_per_contract_dollar,
            cost_per_trade_r: config.cost_per_trade_r,
            dollars_per_r: config.dollars_per_r,
            capital_dollar: config.capital_dollar,
            risk_pct_per_trade: config.risk_pct_per_trade,
            equity_time_years: config.equity_time_years,
        })
    }

    pub fn row_count(&self) -> usize {
        self.row_count
    }

    pub fn target(&self) -> &[bool] {
        self.target.as_ref()
    }

    pub fn rewards(&self) -> Option<&[f64]> {
        self.rewards.as_deref().map(|values| values.as_slice())
    }

    pub fn eligible(&self) -> Option<&[bool]> {
        self.eligible.as_deref().map(|values| values.as_slice())
    }

    pub fn stacking_mode(&self) -> StackingMode {
        self.stacking_mode
    }

    pub fn exit_indices(&self) -> Option<&[usize]> {
        self.exit_indices.as_deref().map(|values| values.as_slice())
    }

    fn eligible_bitset(&self) -> Option<&BitsetMask> {
        self.eligible_bitset.as_deref()
    }

    fn reward_finite_bitset(&self) -> Option<&BitsetMask> {
        self.reward_finite_bitset.as_deref()
    }

    pub fn position_sizing(&self) -> PositionSizingMode {
        self.position_sizing
    }

    pub fn min_contracts(&self) -> usize {
        self.min_contracts
    }

    pub fn max_contracts(&self) -> Option<usize> {
        self.max_contracts
    }

    pub fn margin_per_contract_dollar(&self) -> Option<f64> {
        self.margin_per_contract_dollar
    }

    pub fn risk_per_contract_dollar(&self) -> Option<&[f64]> {
        self.risk_per_contract_dollar
            .as_deref()
            .map(|values| values.as_slice())
    }

    #[allow(dead_code)]
    pub fn cost_per_trade_dollar(&self) -> Option<f64> {
        self.cost_per_trade_dollar
    }

    pub fn cost_per_trade_r(&self) -> Option<f64> {
        self.cost_per_trade_r
    }

    pub fn dollars_per_r(&self) -> Option<f64> {
        self.dollars_per_r
    }

    pub fn capital_dollar(&self) -> Option<f64> {
        self.capital_dollar
    }

    pub fn risk_pct_per_trade(&self) -> Option<f64> {
        self.risk_pct_per_trade
    }

    pub fn equity_time_years(&self) -> Option<f64> {
        self.equity_time_years
    }

    /// Return true if the named feature corresponds to a feature-to-feature
    /// comparison (i.e., its ComparisonSpec has a right-hand-side feature).
    /// Feature-to-constant thresholds return false.
    pub fn is_feature_pair(&self, feature: &str) -> bool {
        match self.comparisons.get(feature) {
            Some(spec) => spec.rhs_feature.is_some(),
            None => false,
        }
    }

    pub fn feature_mask(&self, feature: &str) -> Result<MaskBuffer> {
        if let Some(mask) = self.mask_cache.get(feature) {
            return Ok(mask);
        }
        if let Some(spec) = self.comparisons.get(feature) {
            return self.build_comparison_mask(feature, spec);
        }
        let column = self
            .data
            .boolean_column(feature)
            .with_context(|| format!("Feature column '{feature}' missing from dataset"))?;
        let mask = column
            .into_iter()
            .map(|value| value.unwrap_or(false))
            .collect();
        Ok(self.mask_cache.get_or_insert(feature, mask))
    }

    fn build_comparison_mask(&self, feature: &str, spec: &ComparisonSpec) -> Result<MaskBuffer> {
        // Feature-to-feature comparison
        if let Some(rhs) = &spec.rhs_feature {
            let left = self
                .data
                .float_column(&spec.base_feature)
                .with_context(|| {
                    format!(
                        "Numeric column '{}' missing for comparison",
                        spec.base_feature
                    )
                })?;
            let right = self
                .data
                .float_column(rhs)
                .with_context(|| format!("Numeric column '{}' missing for comparison", rhs))?;
            let mask = left
                .into_iter()
                .zip(&right)
                .map(|(l, r)| match (l, r) {
                    (Some(a), Some(b)) if a.is_finite() && b.is_finite() => {
                        apply_pair_operator(a, b, spec.operator)
                    }
                    _ => false,
                })
                .collect();
            return Ok(self.mask_cache.get_or_insert(feature, mask));
        }

        // Feature-to-threshold comparison
        let threshold = spec.threshold.unwrap_or(0.0);
        let column = self
            .data
            .float_column(&spec.base_feature)
            .with_context(|| {
                format!(
                    "Numeric column '{}' missing for comparison",
                    spec.base_feature
                )
            })?;
        let mask = column
            .into_iter()
            .map(|value| match value {
                Some(raw) if raw.is_finite() => apply_operator(raw, threshold, spec.operator),
                _ => false,
            })
            .collect();
        Ok(self.mask_cache.get_or_insert(feature, mask))
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct StatSummary {
    pub depth: usize,
    /// Number of bars where the combo mask is true (used for recall/density reporting).
    pub mask_hits: usize,
    pub total_bars: usize,
    pub profitable_bars: usize,
    pub unprofitable_bars: usize,
    /// Net-R win rate (% of trades with RR > 0).
    pub win_rate: f64,
    /// Target/label hit-rate (% of trades where the engineered target is true).
    pub label_hit_rate: f64,
    /// Number of trades where the engineered target is true.
    pub label_hits: usize,
    /// Number of trades where the engineered target is false.
    pub label_misses: usize,
    pub expectancy: f64,
    pub profit_factor: f64,
    /// Average reward on winning trades only (R per winning bar).
    pub avg_winning_rr: f64,
    pub calmar_ratio: f64,
    pub max_drawdown: f64,
    pub win_loss_ratio: f64,
    pub ulcer_index: f64,
    pub pain_ratio: f64,
    pub max_consecutive_wins: usize,
    pub max_consecutive_losses: usize,
    /// Average length of winning streaks (trades with RR > 0).
    pub avg_win_streak: f64,
    /// Average length of losing streaks (trades with RR < 0).
    pub avg_loss_streak: f64,
    /// Median R per trade for this combination.
    pub median_rr: f64,
    /// Average losing R per trade (negative value).
    pub avg_losing_rr: f64,
    /// 5th percentile R (downside tail).
    pub p05_rr: f64,
    /// 95th percentile R (upside tail).
    pub p95_rr: f64,
    pub largest_win: f64,
    pub largest_loss: f64,
    pub sample_quality: &'static str,
    pub total_return: f64,
    pub cost_per_trade_r: f64,
    pub dollars_per_r: f64,
    pub total_return_dollar: f64,
    pub max_drawdown_dollar: f64,
    pub expectancy_dollar: f64,
    pub final_capital: f64,
    pub total_return_pct: f64,
    pub cagr_pct: f64,
    pub max_drawdown_pct_equity: f64,
    pub calmar_equity: f64,
    pub sharpe_equity: f64,
    pub sortino_equity: f64,
}

impl StatSummary {
    #[allow(dead_code)]
    fn empty(depth: usize, sample_size: usize) -> Self {
        Self {
            depth,
            mask_hits: sample_size,
            total_bars: sample_size,
            profitable_bars: 0,
            unprofitable_bars: sample_size,
            win_rate: 0.0,
            label_hit_rate: 0.0,
            label_hits: 0,
            label_misses: sample_size,
            expectancy: 0.0,
            profit_factor: 0.0,
            avg_winning_rr: 0.0,
            calmar_ratio: 0.0,
            max_drawdown: 0.0,
            win_loss_ratio: 0.0,
            ulcer_index: 0.0,
            pain_ratio: 0.0,
            max_consecutive_wins: 0,
            max_consecutive_losses: 0,
            avg_win_streak: 0.0,
            avg_loss_streak: 0.0,
            median_rr: 0.0,
            avg_losing_rr: 0.0,
            p05_rr: 0.0,
            p95_rr: 0.0,
            largest_win: 0.0,
            largest_loss: 0.0,
            sample_quality: classify_sample(sample_size),
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

    fn under_min(depth: usize, total_bars: usize) -> Self {
        Self {
            depth,
            mask_hits: total_bars,
            total_bars,
            profitable_bars: 0,
            unprofitable_bars: total_bars,
            win_rate: 0.0,
            label_hit_rate: 0.0,
            label_hits: 0,
            label_misses: total_bars,
            expectancy: 0.0,
            profit_factor: 0.0,
            avg_winning_rr: 0.0,
            calmar_ratio: 0.0,
            max_drawdown: 0.0,
            win_loss_ratio: 0.0,
            ulcer_index: 0.0,
            pain_ratio: 0.0,
            max_consecutive_wins: 0,
            max_consecutive_losses: 0,
            avg_win_streak: 0.0,
            avg_loss_streak: 0.0,
            median_rr: 0.0,
            avg_losing_rr: 0.0,
            p05_rr: 0.0,
            p95_rr: 0.0,
            largest_win: 0.0,
            largest_loss: 0.0,
            sample_quality: classify_sample(total_bars),
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
}

pub fn evaluate_combination(
    combination: &Combination,
    ctx: &EvaluationContext,
    bitsets: &BitsetCatalog,
    min_sample_size: usize,
) -> Result<StatSummary> {
    let depth = combination.len();

    // Map feature descriptors to bitset references via the catalog.
    let mut combo_bitsets: Vec<&BitsetMask> = Vec::with_capacity(combination.len());
    for descriptor in combination {
        let name = descriptor.name.as_str();
        let mask = bitsets
            .get(name)
            .ok_or_else(|| anyhow!("Missing bitset for feature '{name}'"))?;
        combo_bitsets.push(mask);
    }

    // Reorder bitsets by ascending support (sparsest first) so that
    // intersections clear bits sooner and later ANDs become cheaper.
    combo_bitsets.sort_by_key(|m| m.support);

    Ok(evaluate_for_bitsets(
        depth,
        ctx,
        &combo_bitsets,
        min_sample_size,
    ))
}

pub fn evaluate_combination_indices(
    indices: &IndexCombination,
    ctx: &EvaluationContext,
    bitsets: &BitsetCatalog,
    min_sample_size: usize,
) -> Result<StatSummary> {
    let depth = indices.len();
    let mut combo_bitsets: Vec<&BitsetMask> = Vec::with_capacity(indices.len());
    for &idx in indices {
        let mask = bitsets
            .get_by_index(idx)
            .ok_or_else(|| anyhow!("Missing bitset for feature index {idx}"))?;
        combo_bitsets.push(mask);
    }

    // Reorder bitsets by ascending support to reduce intersection cost.
    combo_bitsets.sort_by_key(|m| m.support);

    Ok(evaluate_for_bitsets(
        depth,
        ctx,
        &combo_bitsets,
        min_sample_size,
    ))
}

#[derive(Debug, Default, Clone, Copy)]
pub struct EvalProfileTotals {
    pub combos_profiled: u64,
    pub build_ns: u64,
    pub scan_ns: u64,
    pub on_hit_ns: u64,
    pub finalize_ns: u64,
    pub mask_hits: u64,
    pub trades: u64,
}

impl EvalProfileTotals {
    pub fn add_assign(&mut self, other: Self) {
        self.combos_profiled += other.combos_profiled;
        self.build_ns += other.build_ns;
        self.scan_ns += other.scan_ns;
        self.on_hit_ns += other.on_hit_ns;
        self.finalize_ns += other.finalize_ns;
        self.mask_hits += other.mask_hits;
        self.trades += other.trades;
    }

    pub fn ms(self) -> (u64, u64, u64, u64) {
        (
            self.build_ns / 1_000_000,
            self.scan_ns / 1_000_000,
            self.on_hit_ns / 1_000_000,
            self.finalize_ns / 1_000_000,
        )
    }
}

fn should_profile_indices(indices: &IndexCombination, sample_rate: usize) -> bool {
    if sample_rate <= 1 {
        return true;
    }
    // Deterministic sampling to avoid RNG/atomics in hot loops.
    // This is not cryptographic; it just spreads combinations reasonably.
    let first = indices.first().copied().unwrap_or(0) as u64;
    let depth = indices.len() as u64;
    let hash = first
        .wrapping_mul(0x9E37_79B1_85EB_CA87)
        .wrapping_add(depth);
    (hash % sample_rate as u64) == 0
}

pub fn evaluate_combination_indices_profiled(
    indices: &IndexCombination,
    ctx: &EvaluationContext,
    bitsets: &BitsetCatalog,
    min_sample_size: usize,
    mode: EvalProfileMode,
    sample_rate: usize,
) -> Result<(StatSummary, EvalProfileTotals)> {
    let do_profile =
        mode != EvalProfileMode::Off && should_profile_indices(indices, sample_rate.max(1));
    if !do_profile {
        return Ok((
            evaluate_combination_indices(indices, ctx, bitsets, min_sample_size)?,
            EvalProfileTotals::default(),
        ));
    }

    let depth = indices.len();
    let build_start = Instant::now();
    let mut combo_bitsets: Vec<&BitsetMask> = Vec::with_capacity(indices.len());
    for &idx in indices {
        let mask = bitsets
            .get_by_index(idx)
            .ok_or_else(|| anyhow!("Missing bitset for feature index {idx}"))?;
        combo_bitsets.push(mask);
    }
    combo_bitsets.sort_by_key(|m| m.support);
    let build_ns = build_start.elapsed().as_nanos() as u64;

    let (stat, mut profile) =
        evaluate_for_bitsets_profiled(depth, ctx, &combo_bitsets, min_sample_size, mode);
    profile.combos_profiled = 1;
    profile.build_ns = build_ns;
    Ok((stat, profile))
}

#[allow(clippy::collapsible_else_if)]
fn evaluate_for_bitsets_profiled(
    depth: usize,
    ctx: &EvaluationContext,
    combo_bitsets: &[&BitsetMask],
    min_sample_size: usize,
    mode: EvalProfileMode,
) -> (StatSummary, EvalProfileTotals) {
    let mut profile = EvalProfileTotals::default();
    let target = ctx.target();
    let rewards = ctx.rewards();
    let eligible = ctx.eligible();
    let no_stacking = ctx.stacking_mode() == StackingMode::NoStacking;
    let exit_indices = if no_stacking {
        ctx.exit_indices()
            .expect("exit indices must be present when stacking_mode is NoStacking")
    } else {
        &[]
    };
    let gate_eligible = ctx.eligible_bitset();
    let gate_finite = ctx.reward_finite_bitset();
    let skip_eligible_check = gate_eligible.is_some();
    let skip_finite_check = gate_finite.is_some();
    let position_sizing = ctx.position_sizing();
    let risk_per_contract = ctx.risk_per_contract_dollar();
    let min_contracts = ctx.min_contracts();
    let max_contracts = ctx.max_contracts();
    let max_len = combo_bitsets
        .first()
        .map(|bitset| bitset.len.min(target.len()))
        .unwrap_or(0);

    if let Some(smallest) = combo_bitsets.first() {
        if smallest.support < min_sample_size {
            return (StatSummary::under_min(depth, smallest.support), profile);
        }
    }

    if combo_bitsets.is_empty() || max_len == 0 {
        let finalize_start = Instant::now();
        let stat = compute_statistics(
            depth,
            0,
            0,
            None,
            None,
            ctx.row_count(),
            ctx.stats_detail,
            position_sizing,
            ctx.dollars_per_r(),
            ctx.cost_per_trade_r(),
            ctx.capital_dollar(),
            ctx.risk_pct_per_trade(),
            ctx.equity_time_years(),
            min_contracts,
            max_contracts,
            ctx.margin_per_contract_dollar(),
        );
        profile.finalize_ns += finalize_start.elapsed().as_nanos() as u64;
        return (stat, profile);
    }

    #[cfg(feature = "simd-eval")]
    let use_simd = combo_bitsets.len() >= 2;
    #[cfg(not(feature = "simd-eval"))]
    let use_simd = false;

    if let Some(reward_series) = rewards {
        if matches!(ctx.stats_detail, StatsDetail::Core) {
            if no_stacking {
                let mut total = 0usize;
                let mut label_hits = 0usize;
                let mut acc = CoreStatsAccumulator::new(
                    position_sizing,
                    ctx.capital_dollar(),
                    ctx.risk_pct_per_trade(),
                    min_contracts,
                    max_contracts,
                    ctx.margin_per_contract_dollar(),
                );
                let mut next_free_idx = 0usize;

                let mut on_hit_inner = |idx: usize| {
                    if idx < next_free_idx {
                        return;
                    }
                    if !skip_eligible_check {
                        if let Some(mask) = eligible {
                            if idx < mask.len() && !mask[idx] {
                                return;
                            }
                        }
                    }
                    if idx >= reward_series.len() {
                        return;
                    }
                    let rr_net = reward_series[idx];
                    if !skip_finite_check && !rr_net.is_finite() {
                        return;
                    }

                    total += 1;
                    acc.total_bars += 1;
                    let rpc = if matches!(position_sizing, PositionSizingMode::Contracts) {
                        risk_per_contract.and_then(|values| values.get(idx).copied())
                    } else {
                        None
                    };
                    acc.push(rr_net, rpc);
                    if target[idx] {
                        label_hits += 1;
                    }

                    let exit_i = exit_indices[idx];
                    let candidate = if exit_i == usize::MAX || exit_i < idx {
                        idx.saturating_add(1)
                    } else {
                        exit_i
                    };
                    if candidate > next_free_idx {
                        next_free_idx = candidate;
                    }
                };

                let scan_start = Instant::now();
                let scan_total = if use_simd {
                    if matches!(mode, EvalProfileMode::Fine) {
                        let mut on_hit = |idx: usize| {
                            let start = Instant::now();
                            on_hit_inner(idx);
                            profile.on_hit_ns += start.elapsed().as_nanos() as u64;
                        };
                        scan_bitsets_simd_dyn_gated(
                            combo_bitsets,
                            max_len,
                            gate_eligible,
                            gate_finite,
                            &mut on_hit,
                        )
                    } else {
                        scan_bitsets_simd_dyn_gated(
                            combo_bitsets,
                            max_len,
                            gate_eligible,
                            gate_finite,
                            &mut on_hit_inner,
                        )
                    }
                } else if matches!(mode, EvalProfileMode::Fine) {
                    let mut on_hit = |idx: usize| {
                        let start = Instant::now();
                        on_hit_inner(idx);
                        profile.on_hit_ns += start.elapsed().as_nanos() as u64;
                    };
                    scan_bitsets_scalar_dyn_gated(
                        combo_bitsets,
                        max_len,
                        gate_eligible,
                        gate_finite,
                        &mut on_hit,
                    )
                } else {
                    scan_bitsets_scalar_dyn_gated(
                        combo_bitsets,
                        max_len,
                        gate_eligible,
                        gate_finite,
                        &mut on_hit_inner,
                    )
                };
                profile.scan_ns += scan_start.elapsed().as_nanos() as u64;
                profile.mask_hits += scan_total as u64;
                profile.trades += total as u64;

                let finalize_start = Instant::now();
                let stat = if total < min_sample_size {
                    StatSummary::under_min(depth, total)
                } else {
                    acc.finalize(depth, label_hits, ctx.equity_time_years())
                };
                profile.finalize_ns += finalize_start.elapsed().as_nanos() as u64;
                (stat, profile)
            } else {
                let mut total = 0usize;
                let mut label_hits = 0usize;
                let mut acc = CoreStatsAccumulator::new(
                    position_sizing,
                    ctx.capital_dollar(),
                    ctx.risk_pct_per_trade(),
                    min_contracts,
                    max_contracts,
                    ctx.margin_per_contract_dollar(),
                );

                let mut on_hit_inner = |idx: usize| {
                    if !skip_eligible_check {
                        if let Some(mask) = eligible {
                            if idx < mask.len() && !mask[idx] {
                                return;
                            }
                        }
                    }
                    if idx >= reward_series.len() {
                        return;
                    }
                    let rr_net = reward_series[idx];
                    if !skip_finite_check && !rr_net.is_finite() {
                        return;
                    }

                    total += 1;
                    acc.total_bars += 1;
                    let rpc = if matches!(position_sizing, PositionSizingMode::Contracts) {
                        risk_per_contract.and_then(|values| values.get(idx).copied())
                    } else {
                        None
                    };
                    acc.push(rr_net, rpc);
                    if target[idx] {
                        label_hits += 1;
                    }
                };

                let scan_start = Instant::now();
                let scan_total = if use_simd {
                    if matches!(mode, EvalProfileMode::Fine) {
                        let mut on_hit = |idx: usize| {
                            let start = Instant::now();
                            on_hit_inner(idx);
                            profile.on_hit_ns += start.elapsed().as_nanos() as u64;
                        };
                        scan_bitsets_simd_dyn_gated(
                            combo_bitsets,
                            max_len,
                            gate_eligible,
                            gate_finite,
                            &mut on_hit,
                        )
                    } else {
                        scan_bitsets_simd_dyn_gated(
                            combo_bitsets,
                            max_len,
                            gate_eligible,
                            gate_finite,
                            &mut on_hit_inner,
                        )
                    }
                } else if matches!(mode, EvalProfileMode::Fine) {
                    let mut on_hit = |idx: usize| {
                        let start = Instant::now();
                        on_hit_inner(idx);
                        profile.on_hit_ns += start.elapsed().as_nanos() as u64;
                    };
                    scan_bitsets_scalar_dyn_gated(
                        combo_bitsets,
                        max_len,
                        gate_eligible,
                        gate_finite,
                        &mut on_hit,
                    )
                } else {
                    scan_bitsets_scalar_dyn_gated(
                        combo_bitsets,
                        max_len,
                        gate_eligible,
                        gate_finite,
                        &mut on_hit_inner,
                    )
                };
                profile.scan_ns += scan_start.elapsed().as_nanos() as u64;
                profile.mask_hits += scan_total as u64;
                profile.trades += total as u64;

                let finalize_start = Instant::now();
                let stat = if total < min_sample_size {
                    StatSummary::under_min(depth, total)
                } else {
                    acc.finalize(depth, label_hits, ctx.equity_time_years())
                };
                profile.finalize_ns += finalize_start.elapsed().as_nanos() as u64;
                (stat, profile)
            }
        } else {
            RETURNS_BUFFER.with(|cell| {
                let mut returns = cell.borrow_mut();
                returns.clear();

                RISK_PER_CONTRACT_BUFFER.with(|risk_cell| {
                    let mut risks = risk_cell.borrow_mut();
                    risks.clear();
                    let want_risk = matches!(position_sizing, PositionSizingMode::Contracts);

                    if no_stacking {
                        let mut total = 0usize;
                        let mut label_hits = 0usize;
                        let mut next_free_idx = 0usize;

                        let mut on_hit_inner = |idx: usize| {
                            if idx < next_free_idx {
                                return;
                            }
                            if !skip_eligible_check {
                                if let Some(mask) = eligible {
                                    if idx < mask.len() && !mask[idx] {
                                        return;
                                    }
                                }
                            }
                            if idx >= reward_series.len() {
                                return;
                            }
                            let rr_net = reward_series[idx];
                            if !skip_finite_check && !rr_net.is_finite() {
                                return;
                            }
                            total += 1;
                            if target[idx] {
                                label_hits += 1;
                            }
                            returns.push(rr_net);
                            if want_risk {
                                let rpc = risk_per_contract
                                    .and_then(|values| values.get(idx).copied())
                                    .unwrap_or(f64::NAN);
                                risks.push(rpc);
                            }

                            let exit_i = exit_indices[idx];
                            let candidate = if exit_i == usize::MAX || exit_i < idx {
                                idx.saturating_add(1)
                            } else {
                                exit_i
                            };
                            if candidate > next_free_idx {
                                next_free_idx = candidate;
                            }
                        };

                        let scan_start = Instant::now();
                        let scan_total = if use_simd {
                            if matches!(mode, EvalProfileMode::Fine) {
                                let mut on_hit = |idx: usize| {
                                    let start = Instant::now();
                                    on_hit_inner(idx);
                                    profile.on_hit_ns += start.elapsed().as_nanos() as u64;
                                };
                                scan_bitsets_simd_dyn_gated(
                                    combo_bitsets,
                                    max_len,
                                    gate_eligible,
                                    gate_finite,
                                    &mut on_hit,
                                )
                            } else {
                                scan_bitsets_simd_dyn_gated(
                                    combo_bitsets,
                                    max_len,
                                    gate_eligible,
                                    gate_finite,
                                    &mut on_hit_inner,
                                )
                            }
                        } else if matches!(mode, EvalProfileMode::Fine) {
                            let mut on_hit = |idx: usize| {
                                let start = Instant::now();
                                on_hit_inner(idx);
                                profile.on_hit_ns += start.elapsed().as_nanos() as u64;
                            };
                            scan_bitsets_scalar_dyn_gated(
                                combo_bitsets,
                                max_len,
                                gate_eligible,
                                gate_finite,
                                &mut on_hit,
                            )
                        } else {
                            scan_bitsets_scalar_dyn_gated(
                                combo_bitsets,
                                max_len,
                                gate_eligible,
                                gate_finite,
                                &mut on_hit_inner,
                            )
                        };
                        profile.scan_ns += scan_start.elapsed().as_nanos() as u64;
                        profile.mask_hits += scan_total as u64;
                        profile.trades += total as u64;

                        let finalize_start = Instant::now();
                        let stat = if total < min_sample_size {
                            StatSummary::under_min(depth, total)
                        } else {
                            compute_statistics(
                                depth,
                                total,
                                label_hits,
                                Some(&returns[..]),
                                if want_risk { Some(&risks[..]) } else { None },
                                ctx.row_count(),
                                ctx.stats_detail,
                                position_sizing,
                                ctx.dollars_per_r(),
                                ctx.cost_per_trade_r(),
                                ctx.capital_dollar(),
                                ctx.risk_pct_per_trade(),
                                ctx.equity_time_years(),
                                min_contracts,
                                max_contracts,
                                ctx.margin_per_contract_dollar(),
                            )
                        };
                        profile.finalize_ns += finalize_start.elapsed().as_nanos() as u64;
                        (stat, profile)
                    } else {
                        let mut total = 0usize;
                        let mut label_hits = 0usize;

                        let mut on_hit_inner = |idx: usize| {
                            if !skip_eligible_check {
                                if let Some(mask) = eligible {
                                    if idx < mask.len() && !mask[idx] {
                                        return;
                                    }
                                }
                            }
                            if idx >= reward_series.len() {
                                return;
                            }
                            let rr_net = reward_series[idx];
                            if !skip_finite_check && !rr_net.is_finite() {
                                return;
                            }
                            total += 1;
                            if target[idx] {
                                label_hits += 1;
                            }
                            returns.push(rr_net);
                            if want_risk {
                                let rpc = risk_per_contract
                                    .and_then(|values| values.get(idx).copied())
                                    .unwrap_or(f64::NAN);
                                risks.push(rpc);
                            }
                        };

                        let scan_start = Instant::now();
                        let scan_total = if use_simd {
                            if matches!(mode, EvalProfileMode::Fine) {
                                let mut on_hit = |idx: usize| {
                                    let start = Instant::now();
                                    on_hit_inner(idx);
                                    profile.on_hit_ns += start.elapsed().as_nanos() as u64;
                                };
                                scan_bitsets_simd_dyn_gated(
                                    combo_bitsets,
                                    max_len,
                                    gate_eligible,
                                    gate_finite,
                                    &mut on_hit,
                                )
                            } else {
                                scan_bitsets_simd_dyn_gated(
                                    combo_bitsets,
                                    max_len,
                                    gate_eligible,
                                    gate_finite,
                                    &mut on_hit_inner,
                                )
                            }
                        } else if matches!(mode, EvalProfileMode::Fine) {
                            let mut on_hit = |idx: usize| {
                                let start = Instant::now();
                                on_hit_inner(idx);
                                profile.on_hit_ns += start.elapsed().as_nanos() as u64;
                            };
                            scan_bitsets_scalar_dyn_gated(
                                combo_bitsets,
                                max_len,
                                gate_eligible,
                                gate_finite,
                                &mut on_hit,
                            )
                        } else {
                            scan_bitsets_scalar_dyn_gated(
                                combo_bitsets,
                                max_len,
                                gate_eligible,
                                gate_finite,
                                &mut on_hit_inner,
                            )
                        };
                        profile.scan_ns += scan_start.elapsed().as_nanos() as u64;
                        profile.mask_hits += scan_total as u64;
                        profile.trades += total as u64;

                        let finalize_start = Instant::now();
                        let stat = if total < min_sample_size {
                            StatSummary::under_min(depth, total)
                        } else {
                            compute_statistics(
                                depth,
                                total,
                                label_hits,
                                Some(&returns[..]),
                                if want_risk { Some(&risks[..]) } else { None },
                                ctx.row_count(),
                                ctx.stats_detail,
                                position_sizing,
                                ctx.dollars_per_r(),
                                ctx.cost_per_trade_r(),
                                ctx.capital_dollar(),
                                ctx.risk_pct_per_trade(),
                                ctx.equity_time_years(),
                                min_contracts,
                                max_contracts,
                                ctx.margin_per_contract_dollar(),
                            )
                        };
                        profile.finalize_ns += finalize_start.elapsed().as_nanos() as u64;
                        (stat, profile)
                    }
                })
            })
        }
    } else {
        if no_stacking {
            let mut total = 0usize;
            let mut wins = 0usize;
            let mut next_free_idx = 0usize;

            let mut on_hit_inner = |idx: usize| {
                if idx < next_free_idx {
                    return;
                }
                if !skip_eligible_check {
                    if let Some(mask) = eligible {
                        if idx < mask.len() && !mask[idx] {
                            return;
                        }
                    }
                }
                total += 1;
                if target[idx] {
                    wins += 1;
                }

                let exit_i = exit_indices[idx];
                let candidate = if exit_i == usize::MAX || exit_i < idx {
                    idx.saturating_add(1)
                } else {
                    exit_i
                };
                if candidate > next_free_idx {
                    next_free_idx = candidate;
                }
            };

            let scan_start = Instant::now();
            let scan_total = if use_simd {
                if matches!(mode, EvalProfileMode::Fine) {
                    let mut on_hit = |idx: usize| {
                        let start = Instant::now();
                        on_hit_inner(idx);
                        profile.on_hit_ns += start.elapsed().as_nanos() as u64;
                    };
                    scan_bitsets_simd_dyn_gated(
                        combo_bitsets,
                        max_len,
                        gate_eligible,
                        None,
                        &mut on_hit,
                    )
                } else {
                    scan_bitsets_simd_dyn_gated(
                        combo_bitsets,
                        max_len,
                        gate_eligible,
                        None,
                        &mut on_hit_inner,
                    )
                }
            } else if matches!(mode, EvalProfileMode::Fine) {
                let mut on_hit = |idx: usize| {
                    let start = Instant::now();
                    on_hit_inner(idx);
                    profile.on_hit_ns += start.elapsed().as_nanos() as u64;
                };
                scan_bitsets_scalar_dyn_gated(
                    combo_bitsets,
                    max_len,
                    gate_eligible,
                    None,
                    &mut on_hit,
                )
            } else {
                scan_bitsets_scalar_dyn_gated(
                    combo_bitsets,
                    max_len,
                    gate_eligible,
                    None,
                    &mut on_hit_inner,
                )
            };
            profile.scan_ns += scan_start.elapsed().as_nanos() as u64;
            profile.mask_hits += scan_total as u64;
            profile.trades += total as u64;

            let finalize_start = Instant::now();
            let stat = if total < min_sample_size {
                StatSummary::under_min(depth, total)
            } else {
                compute_statistics(
                    depth,
                    total,
                    wins,
                    None,
                    None,
                    ctx.row_count(),
                    ctx.stats_detail,
                    position_sizing,
                    ctx.dollars_per_r(),
                    ctx.cost_per_trade_r(),
                    ctx.capital_dollar(),
                    ctx.risk_pct_per_trade(),
                    ctx.equity_time_years(),
                    min_contracts,
                    max_contracts,
                    ctx.margin_per_contract_dollar(),
                )
            };
            profile.finalize_ns += finalize_start.elapsed().as_nanos() as u64;
            (stat, profile)
        } else {
            let mut total = 0usize;
            let mut wins = 0usize;

            let mut on_hit_inner = |idx: usize| {
                if !skip_eligible_check {
                    if let Some(mask) = eligible {
                        if idx < mask.len() && !mask[idx] {
                            return;
                        }
                    }
                }
                total += 1;
                if target[idx] {
                    wins += 1;
                }
            };

            let scan_start = Instant::now();
            let scan_total = if use_simd {
                if matches!(mode, EvalProfileMode::Fine) {
                    let mut on_hit = |idx: usize| {
                        let start = Instant::now();
                        on_hit_inner(idx);
                        profile.on_hit_ns += start.elapsed().as_nanos() as u64;
                    };
                    scan_bitsets_simd_dyn_gated(
                        combo_bitsets,
                        max_len,
                        gate_eligible,
                        None,
                        &mut on_hit,
                    )
                } else {
                    scan_bitsets_simd_dyn_gated(
                        combo_bitsets,
                        max_len,
                        gate_eligible,
                        None,
                        &mut on_hit_inner,
                    )
                }
            } else if matches!(mode, EvalProfileMode::Fine) {
                let mut on_hit = |idx: usize| {
                    let start = Instant::now();
                    on_hit_inner(idx);
                    profile.on_hit_ns += start.elapsed().as_nanos() as u64;
                };
                scan_bitsets_scalar_dyn_gated(
                    combo_bitsets,
                    max_len,
                    gate_eligible,
                    None,
                    &mut on_hit,
                )
            } else {
                scan_bitsets_scalar_dyn_gated(
                    combo_bitsets,
                    max_len,
                    gate_eligible,
                    None,
                    &mut on_hit_inner,
                )
            };
            profile.scan_ns += scan_start.elapsed().as_nanos() as u64;
            profile.mask_hits += scan_total as u64;
            profile.trades += total as u64;

            let finalize_start = Instant::now();
            let stat = if total < min_sample_size {
                StatSummary::under_min(depth, total)
            } else {
                compute_statistics(
                    depth,
                    total,
                    wins,
                    None,
                    None,
                    ctx.row_count(),
                    ctx.stats_detail,
                    position_sizing,
                    ctx.dollars_per_r(),
                    ctx.cost_per_trade_r(),
                    ctx.capital_dollar(),
                    ctx.risk_pct_per_trade(),
                    ctx.equity_time_years(),
                    min_contracts,
                    max_contracts,
                    ctx.margin_per_contract_dollar(),
                )
            };
            profile.finalize_ns += finalize_start.elapsed().as_nanos() as u64;
            (stat, profile)
        }
    }
}

#[allow(dead_code)]
fn scan_bitsets_scalar_dyn(
    combo_bitsets: &[&BitsetMask],
    max_len: usize,
    on_hit: &mut dyn FnMut(usize),
) -> usize {
    if combo_bitsets.is_empty() || max_len == 0 {
        return 0;
    }

    let words_len = combo_bitsets[0].words.len();
    let mut total = 0usize;

    for word_index in 0..words_len {
        let mut combined = u64::MAX;
        for bitset in combo_bitsets {
            combined &= bitset.words[word_index];
        }
        let mut w = combined;
        while w != 0 {
            let tz = w.trailing_zeros() as usize;
            let idx = word_index * 64 + tz;
            if idx >= max_len {
                break;
            }
            w &= w - 1;

            total += 1;
            on_hit(idx);
        }
    }

    total
}

fn gate_word_allow_out_of_bounds_true(gate: &BitsetMask, word_index: usize) -> u64 {
    if word_index >= gate.words.len() {
        return u64::MAX;
    }
    let mut word = gate.words[word_index];
    if word_index + 1 == gate.words.len() {
        let rem = gate.len % 64;
        if rem != 0 {
            word |= !((1u64 << rem) - 1);
        }
    }
    word
}

fn scan_bitsets_scalar_dyn_gated(
    combo_bitsets: &[&BitsetMask],
    max_len: usize,
    gate_eligible: Option<&BitsetMask>,
    gate_finite: Option<&BitsetMask>,
    on_hit: &mut dyn FnMut(usize),
) -> usize {
    if combo_bitsets.is_empty() || max_len == 0 {
        return 0;
    }

    let words_len = max_len.div_ceil(64).min(combo_bitsets[0].words.len());
    let mut scan_total = 0usize;
    let rem = max_len % 64;
    let last_mask = if rem == 0 {
        u64::MAX
    } else {
        (1u64 << rem) - 1
    };

    for word_index in 0..words_len {
        let mut combined = u64::MAX;
        for bitset in combo_bitsets {
            combined &= bitset.words[word_index];
        }
        if word_index + 1 == words_len {
            combined &= last_mask;
        }
        scan_total += combined.count_ones() as usize;
        if combined == 0 {
            continue;
        }

        let mut gated = combined;
        if let Some(gate) = gate_eligible {
            gated &= gate_word_allow_out_of_bounds_true(gate, word_index);
        }
        if let Some(gate) = gate_finite {
            if word_index < gate.words.len() {
                gated &= gate.words[word_index];
            } else {
                gated = 0;
            }
        }

        let mut w = gated;
        while w != 0 {
            let tz = w.trailing_zeros() as usize;
            let idx = word_index * 64 + tz;
            w &= w - 1;
            on_hit(idx);
        }
    }

    scan_total
}

#[cfg(all(target_arch = "aarch64", feature = "simd-eval"))]
#[allow(dead_code)]
unsafe fn scan_bitsets_neon_dyn(
    combo_bitsets: &[&BitsetMask],
    max_len: usize,
    on_hit: &mut dyn FnMut(usize),
) -> usize {
    if combo_bitsets.is_empty() || max_len == 0 {
        return 0;
    }

    let words_len = combo_bitsets[0].words.len();
    let mut total = 0usize;

    let mut word_index = 0usize;
    while word_index + 1 < words_len {
        // Initialize combined = [!0, !0]
        let mut combined = unsafe { vdupq_n_u64(u64::MAX) };

        for bitset in combo_bitsets {
            let ptr = unsafe { bitset.words.as_ptr().add(word_index) };
            let vec = unsafe { vld1q_u64(ptr) };
            combined = unsafe { vandq_u64(combined, vec) };
        }

        let lane0 = unsafe { vgetq_lane_u64(combined, 0) };
        let lane1 = unsafe { vgetq_lane_u64(combined, 1) };

        let mut w0 = lane0;
        while w0 != 0 {
            let tz = w0.trailing_zeros() as usize;
            let idx = word_index * 64 + tz;
            if idx >= max_len {
                break;
            }
            w0 &= w0 - 1;
            total += 1;
            on_hit(idx);
        }

        let mut w1 = lane1;
        while w1 != 0 {
            let tz = w1.trailing_zeros() as usize;
            let idx = (word_index + 1) * 64 + tz;
            if idx >= max_len {
                break;
            }
            w1 &= w1 - 1;
            total += 1;
            on_hit(idx);
        }

        word_index += 2;
    }

    // Tail: fallback to scalar for remaining last word, if any.
    while word_index < words_len {
        let mut combined = u64::MAX;
        for bitset in combo_bitsets {
            combined &= bitset.words[word_index];
        }
        let mut w = combined;
        while w != 0 {
            let tz = w.trailing_zeros() as usize;
            let idx = word_index * 64 + tz;
            if idx >= max_len {
                break;
            }
            w &= w - 1;

            total += 1;
            on_hit(idx);
        }
        word_index += 1;
    }

    total
}

#[cfg(all(target_arch = "aarch64", feature = "simd-eval"))]
unsafe fn scan_bitsets_neon_dyn_gated(
    combo_bitsets: &[&BitsetMask],
    max_len: usize,
    gate_eligible: Option<&BitsetMask>,
    gate_finite: Option<&BitsetMask>,
    on_hit: &mut dyn FnMut(usize),
) -> usize {
    if combo_bitsets.is_empty() || max_len == 0 {
        return 0;
    }

    let words_len = max_len.div_ceil(64).min(combo_bitsets[0].words.len());
    let mut scan_total = 0usize;
    let rem = max_len % 64;
    let last_mask = if rem == 0 {
        u64::MAX
    } else {
        (1u64 << rem) - 1
    };

    let mut word_index = 0usize;
    while word_index + 1 < words_len {
        // Initialize combined = [!0, !0]
        let mut combined = unsafe { vdupq_n_u64(u64::MAX) };

        for bitset in combo_bitsets {
            let ptr = unsafe { bitset.words.as_ptr().add(word_index) };
            let vec = unsafe { vld1q_u64(ptr) };
            combined = unsafe { vandq_u64(combined, vec) };
        }

        let lane0 = unsafe { vgetq_lane_u64(combined, 0) };
        let mut lane1 = unsafe { vgetq_lane_u64(combined, 1) };
        if word_index + 1 + 1 == words_len {
            lane1 &= last_mask;
        }

        scan_total += lane0.count_ones() as usize;
        scan_total += lane1.count_ones() as usize;
        if (lane0 | lane1) == 0 {
            word_index += 2;
            continue;
        }

        let mut gated0 = lane0;
        let mut gated1 = lane1;
        if let Some(gate) = gate_eligible {
            gated0 &= gate_word_allow_out_of_bounds_true(gate, word_index);
            gated1 &= gate_word_allow_out_of_bounds_true(gate, word_index + 1);
        }
        if let Some(gate) = gate_finite {
            if word_index < gate.words.len() {
                gated0 &= gate.words[word_index];
            } else {
                gated0 = 0;
            }
            if word_index + 1 < gate.words.len() {
                gated1 &= gate.words[word_index + 1];
            } else {
                gated1 = 0;
            }
        }

        let mut w0 = gated0;
        while w0 != 0 {
            let tz = w0.trailing_zeros() as usize;
            let idx = word_index * 64 + tz;
            w0 &= w0 - 1;
            on_hit(idx);
        }

        let mut w1 = gated1;
        while w1 != 0 {
            let tz = w1.trailing_zeros() as usize;
            let idx = (word_index + 1) * 64 + tz;
            w1 &= w1 - 1;
            on_hit(idx);
        }

        word_index += 2;
    }

    // Tail: fallback to scalar for remaining last word, if any.
    while word_index < words_len {
        let mut combined = u64::MAX;
        for bitset in combo_bitsets {
            combined &= bitset.words[word_index];
        }
        if word_index + 1 == words_len {
            combined &= last_mask;
        }
        scan_total += combined.count_ones() as usize;
        if combined == 0 {
            word_index += 1;
            continue;
        }

        let mut gated = combined;
        if let Some(gate) = gate_eligible {
            gated &= gate_word_allow_out_of_bounds_true(gate, word_index);
        }
        if let Some(gate) = gate_finite {
            if word_index < gate.words.len() {
                gated &= gate.words[word_index];
            } else {
                gated = 0;
            }
        }

        let mut w = gated;
        while w != 0 {
            let tz = w.trailing_zeros() as usize;
            let idx = word_index * 64 + tz;
            w &= w - 1;
            on_hit(idx);
        }
        word_index += 1;
    }

    scan_total
}

#[cfg(all(target_arch = "aarch64", feature = "simd-eval"))]
#[allow(dead_code)]
fn scan_bitsets_simd_dyn(
    combo_bitsets: &[&BitsetMask],
    max_len: usize,
    on_hit: &mut dyn FnMut(usize),
) -> usize {
    unsafe { scan_bitsets_neon_dyn(combo_bitsets, max_len, on_hit) }
}

#[cfg(all(not(target_arch = "aarch64"), feature = "simd-eval"))]
#[allow(dead_code)]
fn scan_bitsets_simd_dyn(
    combo_bitsets: &[&BitsetMask],
    max_len: usize,
    on_hit: &mut dyn FnMut(usize),
) -> usize {
    // Fallback: simple scalar scan, same as scan_bitsets_scalar_dyn.
    scan_bitsets_scalar_dyn(combo_bitsets, max_len, on_hit)
}

#[cfg(not(feature = "simd-eval"))]
#[allow(dead_code)]
fn scan_bitsets_simd_dyn(
    combo_bitsets: &[&BitsetMask],
    max_len: usize,
    on_hit: &mut dyn FnMut(usize),
) -> usize {
    scan_bitsets_scalar_dyn(combo_bitsets, max_len, on_hit)
}

#[cfg(all(target_arch = "aarch64", feature = "simd-eval"))]
fn scan_bitsets_simd_dyn_gated(
    combo_bitsets: &[&BitsetMask],
    max_len: usize,
    gate_eligible: Option<&BitsetMask>,
    gate_finite: Option<&BitsetMask>,
    on_hit: &mut dyn FnMut(usize),
) -> usize {
    unsafe {
        scan_bitsets_neon_dyn_gated(combo_bitsets, max_len, gate_eligible, gate_finite, on_hit)
    }
}

#[cfg(all(not(target_arch = "aarch64"), feature = "simd-eval"))]
fn scan_bitsets_simd_dyn_gated(
    combo_bitsets: &[&BitsetMask],
    max_len: usize,
    gate_eligible: Option<&BitsetMask>,
    gate_finite: Option<&BitsetMask>,
    on_hit: &mut dyn FnMut(usize),
) -> usize {
    scan_bitsets_scalar_dyn_gated(combo_bitsets, max_len, gate_eligible, gate_finite, on_hit)
}

#[cfg(not(feature = "simd-eval"))]
fn scan_bitsets_simd_dyn_gated(
    combo_bitsets: &[&BitsetMask],
    max_len: usize,
    gate_eligible: Option<&BitsetMask>,
    gate_finite: Option<&BitsetMask>,
    on_hit: &mut dyn FnMut(usize),
) -> usize {
    scan_bitsets_scalar_dyn_gated(combo_bitsets, max_len, gate_eligible, gate_finite, on_hit)
}

#[allow(clippy::collapsible_else_if)]
fn evaluate_for_bitsets(
    depth: usize,
    ctx: &EvaluationContext,
    combo_bitsets: &[&BitsetMask],
    min_sample_size: usize,
) -> StatSummary {
    let target = ctx.target();
    let rewards = ctx.rewards();
    let eligible = ctx.eligible();
    let no_stacking = ctx.stacking_mode() == StackingMode::NoStacking;
    let exit_indices = if no_stacking {
        ctx.exit_indices()
            .expect("exit indices must be present when stacking_mode is NoStacking")
    } else {
        &[]
    };
    let gate_eligible = ctx.eligible_bitset();
    let gate_finite = ctx.reward_finite_bitset();
    let skip_eligible_check = gate_eligible.is_some();
    let skip_finite_check = gate_finite.is_some();
    let position_sizing = ctx.position_sizing();
    let risk_per_contract = ctx.risk_per_contract_dollar();
    let min_contracts = ctx.min_contracts();
    let max_contracts = ctx.max_contracts();
    // All bitsets are built from masks aligned to the target length, but
    // we still clamp to the smaller of the two for safety.
    let max_len = combo_bitsets
        .first()
        .map(|bitset| bitset.len.min(target.len()))
        .unwrap_or(0);

    // If the sparsest feature in this combination has support below the
    // minimum sample size, the intersection can never reach the threshold.
    // Reject these combinations up front without scanning any bits.
    if let Some(smallest) = combo_bitsets.first() {
        if smallest.support < min_sample_size {
            return StatSummary::under_min(depth, smallest.support);
        }
    }

    if combo_bitsets.is_empty() || max_len == 0 {
        return compute_statistics(
            depth,
            0,
            0,
            None,
            None,
            ctx.row_count(),
            ctx.stats_detail,
            position_sizing,
            ctx.dollars_per_r(),
            ctx.cost_per_trade_r(),
            ctx.capital_dollar(),
            ctx.risk_pct_per_trade(),
            ctx.equity_time_years(),
            min_contracts,
            max_contracts,
            ctx.margin_per_contract_dollar(),
        );
    }
    #[cfg(feature = "simd-eval")]
    let use_simd = combo_bitsets.len() >= 2;
    #[cfg(not(feature = "simd-eval"))]
    let use_simd = false;

    if let Some(reward_series) = rewards {
        // For core stats we stream metrics directly during bitset scanning to
        // avoid building an intermediate RR vector and a second pass. For full
        // detail we retain the existing path so that percentile, streak, and
        // other rich metrics can be computed from the full RR sequence.
        if matches!(ctx.stats_detail, StatsDetail::Core) {
            if no_stacking {
                let mut total = 0usize;
                let mut label_hits = 0usize;
                let mut acc = CoreStatsAccumulator::new(
                    position_sizing,
                    ctx.capital_dollar(),
                    ctx.risk_pct_per_trade(),
                    min_contracts,
                    max_contracts,
                    ctx.margin_per_contract_dollar(),
                );
                let mut next_free_idx = 0usize;

                let mut on_hit = |idx: usize| {
                    if idx < next_free_idx {
                        return;
                    }
                    if !skip_eligible_check {
                        if let Some(mask) = eligible {
                            if idx < mask.len() && !mask[idx] {
                                return;
                            }
                        }
                    }
                    if idx >= reward_series.len() {
                        return;
                    }
                    let rr_net = reward_series[idx];
                    if !skip_finite_check && !rr_net.is_finite() {
                        return;
                    }

                    total += 1;
                    acc.total_bars += 1;
                    let rpc = if matches!(position_sizing, PositionSizingMode::Contracts) {
                        risk_per_contract.and_then(|values| values.get(idx).copied())
                    } else {
                        None
                    };
                    acc.push(rr_net, rpc);
                    if target[idx] {
                        label_hits += 1;
                    }

                    let exit_i = exit_indices[idx];
                    let candidate = if exit_i == usize::MAX || exit_i < idx {
                        idx.saturating_add(1)
                    } else {
                        exit_i
                    };
                    if candidate > next_free_idx {
                        next_free_idx = candidate;
                    }
                };

                let scan_total = if use_simd {
                    #[cfg(feature = "simd-eval")]
                    {
                        scan_bitsets_simd_dyn_gated(
                            combo_bitsets,
                            max_len,
                            gate_eligible,
                            gate_finite,
                            &mut on_hit,
                        )
                    }
                    #[cfg(not(feature = "simd-eval"))]
                    {
                        scan_bitsets_scalar_dyn_gated(
                            combo_bitsets,
                            max_len,
                            gate_eligible,
                            gate_finite,
                            &mut on_hit,
                        )
                    }
                } else {
                    scan_bitsets_scalar_dyn_gated(
                        combo_bitsets,
                        max_len,
                        gate_eligible,
                        gate_finite,
                        &mut on_hit,
                    )
                };

                if total < min_sample_size {
                    let mut stat = StatSummary::under_min(depth, total);
                    stat.mask_hits = scan_total;
                    stat
                } else {
                    let mut stat = acc.finalize(depth, label_hits, ctx.equity_time_years());
                    stat.mask_hits = scan_total;
                    stat
                }
            } else {
                let mut total = 0usize;
                let mut label_hits = 0usize;
                let mut acc = CoreStatsAccumulator::new(
                    position_sizing,
                    ctx.capital_dollar(),
                    ctx.risk_pct_per_trade(),
                    min_contracts,
                    max_contracts,
                    ctx.margin_per_contract_dollar(),
                );

                let mut on_hit = |idx: usize| {
                    if !skip_eligible_check {
                        if let Some(mask) = eligible {
                            if idx < mask.len() && !mask[idx] {
                                return;
                            }
                        }
                    }
                    if idx >= reward_series.len() {
                        return;
                    }
                    let rr_net = reward_series[idx];
                    if !skip_finite_check && !rr_net.is_finite() {
                        return;
                    }

                    total += 1;
                    acc.total_bars += 1;
                    let rpc = if matches!(position_sizing, PositionSizingMode::Contracts) {
                        risk_per_contract.and_then(|values| values.get(idx).copied())
                    } else {
                        None
                    };
                    acc.push(rr_net, rpc);
                    if target[idx] {
                        label_hits += 1;
                    }
                };

                let scan_total = if use_simd {
                    #[cfg(feature = "simd-eval")]
                    {
                        scan_bitsets_simd_dyn_gated(
                            combo_bitsets,
                            max_len,
                            gate_eligible,
                            gate_finite,
                            &mut on_hit,
                        )
                    }
                    #[cfg(not(feature = "simd-eval"))]
                    {
                        scan_bitsets_scalar_dyn_gated(
                            combo_bitsets,
                            max_len,
                            gate_eligible,
                            gate_finite,
                            &mut on_hit,
                        )
                    }
                } else {
                    scan_bitsets_scalar_dyn_gated(
                        combo_bitsets,
                        max_len,
                        gate_eligible,
                        gate_finite,
                        &mut on_hit,
                    )
                };

                if total < min_sample_size {
                    let mut stat = StatSummary::under_min(depth, total);
                    stat.mask_hits = scan_total;
                    stat
                } else {
                    let mut stat = acc.finalize(depth, label_hits, ctx.equity_time_years());
                    stat.mask_hits = scan_total;
                    stat
                }
            }
        } else {
            RETURNS_BUFFER.with(|cell| {
                let mut returns = cell.borrow_mut();
                returns.clear();

                RISK_PER_CONTRACT_BUFFER.with(|risk_cell| {
                    let mut risks = risk_cell.borrow_mut();
                    risks.clear();
                    let want_risk = matches!(position_sizing, PositionSizingMode::Contracts);

                    if no_stacking {
                        let mut total = 0usize;
                        let mut label_hits = 0usize;
                        let mut next_free_idx = 0usize;

                        let mut on_hit = |idx: usize| {
                            if idx < next_free_idx {
                                return;
                            }
                            if !skip_eligible_check {
                                if let Some(mask) = eligible {
                                    if idx < mask.len() && !mask[idx] {
                                        return;
                                    }
                                }
                            }
                            if idx >= reward_series.len() {
                                return;
                            }
                            let rr_net = reward_series[idx];
                            if !skip_finite_check && !rr_net.is_finite() {
                                return;
                            }
                            total += 1;
                            if target[idx] {
                                label_hits += 1;
                            }
                            returns.push(rr_net);
                            if want_risk {
                                let rpc = risk_per_contract
                                    .and_then(|values| values.get(idx).copied())
                                    .unwrap_or(f64::NAN);
                                risks.push(rpc);
                            }

                            let exit_i = exit_indices[idx];
                            let candidate = if exit_i == usize::MAX || exit_i < idx {
                                idx.saturating_add(1)
                            } else {
                                exit_i
                            };
                            if candidate > next_free_idx {
                                next_free_idx = candidate;
                            }
                        };

                        let scan_total = if use_simd {
                            #[cfg(feature = "simd-eval")]
                            {
                                scan_bitsets_simd_dyn_gated(
                                    combo_bitsets,
                                    max_len,
                                    gate_eligible,
                                    gate_finite,
                                    &mut on_hit,
                                )
                            }
                            #[cfg(not(feature = "simd-eval"))]
                            {
                                scan_bitsets_scalar_dyn_gated(
                                    combo_bitsets,
                                    max_len,
                                    gate_eligible,
                                    gate_finite,
                                    &mut on_hit,
                                )
                            }
                        } else {
                            scan_bitsets_scalar_dyn_gated(
                                combo_bitsets,
                                max_len,
                                gate_eligible,
                                gate_finite,
                                &mut on_hit,
                            )
                        };

                        if total < min_sample_size {
                            let mut stat = StatSummary::under_min(depth, total);
                            stat.mask_hits = scan_total;
                            stat
                        } else {
                            let mut stat = compute_statistics(
                                depth,
                                total,
                                label_hits,
                                Some(&returns[..]),
                                if want_risk { Some(&risks[..]) } else { None },
                                ctx.row_count(),
                                ctx.stats_detail,
                                position_sizing,
                                ctx.dollars_per_r(),
                                ctx.cost_per_trade_r(),
                                ctx.capital_dollar(),
                                ctx.risk_pct_per_trade(),
                                ctx.equity_time_years(),
                                min_contracts,
                                max_contracts,
                                ctx.margin_per_contract_dollar(),
                            );
                            stat.mask_hits = scan_total;
                            stat
                        }
                    } else {
                        let mut total = 0usize;
                        let mut label_hits = 0usize;

                        let mut on_hit = |idx: usize| {
                            if !skip_eligible_check {
                                if let Some(mask) = eligible {
                                    if idx < mask.len() && !mask[idx] {
                                        return;
                                    }
                                }
                            }
                            if idx >= reward_series.len() {
                                return;
                            }
                            let rr_net = reward_series[idx];
                            if !skip_finite_check && !rr_net.is_finite() {
                                return;
                            }
                            total += 1;
                            if target[idx] {
                                label_hits += 1;
                            }
                            returns.push(rr_net);
                            if want_risk {
                                let rpc = risk_per_contract
                                    .and_then(|values| values.get(idx).copied())
                                    .unwrap_or(f64::NAN);
                                risks.push(rpc);
                            }
                        };

                        let scan_total = if use_simd {
                            #[cfg(feature = "simd-eval")]
                            {
                                scan_bitsets_simd_dyn_gated(
                                    combo_bitsets,
                                    max_len,
                                    gate_eligible,
                                    gate_finite,
                                    &mut on_hit,
                                )
                            }
                            #[cfg(not(feature = "simd-eval"))]
                            {
                                scan_bitsets_scalar_dyn_gated(
                                    combo_bitsets,
                                    max_len,
                                    gate_eligible,
                                    gate_finite,
                                    &mut on_hit,
                                )
                            }
                        } else {
                            scan_bitsets_scalar_dyn_gated(
                                combo_bitsets,
                                max_len,
                                gate_eligible,
                                gate_finite,
                                &mut on_hit,
                            )
                        };

                        if total < min_sample_size {
                            let mut stat = StatSummary::under_min(depth, total);
                            stat.mask_hits = scan_total;
                            stat
                        } else {
                            let mut stat = compute_statistics(
                                depth,
                                total,
                                label_hits,
                                Some(&returns[..]),
                                if want_risk { Some(&risks[..]) } else { None },
                                ctx.row_count(),
                                ctx.stats_detail,
                                position_sizing,
                                ctx.dollars_per_r(),
                                ctx.cost_per_trade_r(),
                                ctx.capital_dollar(),
                                ctx.risk_pct_per_trade(),
                                ctx.equity_time_years(),
                                min_contracts,
                                max_contracts,
                                ctx.margin_per_contract_dollar(),
                            );
                            stat.mask_hits = scan_total;
                            stat
                        }
                    }
                })
            })
        }
    } else {
        if no_stacking {
            let mut total = 0usize;
            let mut wins = 0usize;
            let mut next_free_idx = 0usize;

            let mut on_hit = |idx: usize| {
                if idx < next_free_idx {
                    return;
                }
                if !skip_eligible_check {
                    if let Some(mask) = eligible {
                        if idx < mask.len() && !mask[idx] {
                            return;
                        }
                    }
                }
                total += 1;
                if target[idx] {
                    wins += 1;
                }

                let exit_i = exit_indices[idx];
                let candidate = if exit_i == usize::MAX || exit_i < idx {
                    idx.saturating_add(1)
                } else {
                    exit_i
                };
                if candidate > next_free_idx {
                    next_free_idx = candidate;
                }
            };

            let scan_total = if use_simd {
                #[cfg(feature = "simd-eval")]
                {
                    scan_bitsets_simd_dyn_gated(
                        combo_bitsets,
                        max_len,
                        gate_eligible,
                        None,
                        &mut on_hit,
                    )
                }
                #[cfg(not(feature = "simd-eval"))]
                {
                    scan_bitsets_scalar_dyn_gated(
                        combo_bitsets,
                        max_len,
                        gate_eligible,
                        None,
                        &mut on_hit,
                    )
                }
            } else {
                scan_bitsets_scalar_dyn_gated(
                    combo_bitsets,
                    max_len,
                    gate_eligible,
                    None,
                    &mut on_hit,
                )
            };

            if total < min_sample_size {
                let mut stat = StatSummary::under_min(depth, total);
                stat.mask_hits = scan_total;
                stat
            } else {
                let mut stat = compute_statistics(
                    depth,
                    total,
                    wins,
                    None,
                    None,
                    ctx.row_count(),
                    ctx.stats_detail,
                    position_sizing,
                    ctx.dollars_per_r(),
                    ctx.cost_per_trade_r(),
                    ctx.capital_dollar(),
                    ctx.risk_pct_per_trade(),
                    ctx.equity_time_years(),
                    min_contracts,
                    max_contracts,
                    ctx.margin_per_contract_dollar(),
                );
                stat.mask_hits = scan_total;
                stat
            }
        } else {
            let mut total = 0usize;
            let mut wins = 0usize;

            let mut on_hit = |idx: usize| {
                if !skip_eligible_check {
                    if let Some(mask) = eligible {
                        if idx < mask.len() && !mask[idx] {
                            return;
                        }
                    }
                }
                total += 1;
                if target[idx] {
                    wins += 1;
                }
            };

            let scan_total = if use_simd {
                #[cfg(feature = "simd-eval")]
                {
                    scan_bitsets_simd_dyn_gated(
                        combo_bitsets,
                        max_len,
                        gate_eligible,
                        None,
                        &mut on_hit,
                    )
                }
                #[cfg(not(feature = "simd-eval"))]
                {
                    scan_bitsets_scalar_dyn_gated(
                        combo_bitsets,
                        max_len,
                        gate_eligible,
                        None,
                        &mut on_hit,
                    )
                }
            } else {
                scan_bitsets_scalar_dyn_gated(
                    combo_bitsets,
                    max_len,
                    gate_eligible,
                    None,
                    &mut on_hit,
                )
            };

            if total < min_sample_size {
                let mut stat = StatSummary::under_min(depth, total);
                stat.mask_hits = scan_total;
                stat
            } else {
                let mut stat = compute_statistics(
                    depth,
                    total,
                    wins,
                    None,
                    None,
                    ctx.row_count(),
                    ctx.stats_detail,
                    position_sizing,
                    ctx.dollars_per_r(),
                    ctx.cost_per_trade_r(),
                    ctx.capital_dollar(),
                    ctx.risk_pct_per_trade(),
                    ctx.equity_time_years(),
                    min_contracts,
                    max_contracts,
                    ctx.margin_per_contract_dollar(),
                );
                stat.mask_hits = scan_total;
                stat
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn compute_statistics(
    depth: usize,
    total_bars: usize,
    label_hits: usize,
    returns: Option<&[f64]>,
    risk_per_contract_dollar: Option<&[f64]>,
    _dataset_rows: usize,
    detail: StatsDetail,
    position_sizing: PositionSizingMode,
    dollars_per_r: Option<f64>,
    cost_per_trade_r: Option<f64>,
    capital_dollar: Option<f64>,
    risk_pct_per_trade: Option<f64>,
    equity_time_years: Option<f64>,
    min_contracts: usize,
    max_contracts: Option<usize>,
    margin_per_contract_dollar: Option<f64>,
) -> StatSummary {
    if let Some(filtered_rr) = returns {
        match detail {
            StatsDetail::Core => compute_core_statistics(
                depth,
                total_bars,
                filtered_rr,
                risk_per_contract_dollar,
                label_hits,
                position_sizing,
                capital_dollar,
                risk_pct_per_trade,
                equity_time_years,
                min_contracts,
                max_contracts,
                margin_per_contract_dollar,
            ),
            StatsDetail::Full => compute_full_statistics(
                depth,
                total_bars,
                filtered_rr,
                risk_per_contract_dollar,
                label_hits,
                dollars_per_r,
                cost_per_trade_r,
                capital_dollar,
                risk_pct_per_trade,
                equity_time_years,
                position_sizing,
                min_contracts,
                max_contracts,
                margin_per_contract_dollar,
            ),
        }
    } else {
        let wins = label_hits;
        let losses = total_bars.saturating_sub(wins);
        let expectancy_raw = if total_bars > 0 {
            let win_ratio = wins as f64 / total_bars as f64;
            (2.0 * win_ratio) - 1.0
        } else {
            0.0
        };
        let win_rate_raw = if total_bars > 0 {
            (wins as f64 / total_bars as f64) * 100.0
        } else {
            0.0
        };
        let profit_factor_raw = if win_rate_raw >= 100.0 {
            f64::INFINITY
        } else if win_rate_raw > 0.0 {
            win_rate_raw / (100.0 - win_rate_raw)
        } else {
            0.0
        };
        let expectancy = expectancy_raw;
        let win_rate = win_rate_raw;
        let label_hit_rate = win_rate_raw;
        let label_misses = total_bars.saturating_sub(label_hits);
        let profit_factor = profit_factor_raw;

        StatSummary {
            depth,
            mask_hits: total_bars,
            total_bars,
            profitable_bars: wins,
            unprofitable_bars: losses,
            win_rate,
            label_hit_rate,
            label_hits,
            label_misses,
            expectancy,
            profit_factor,
            avg_winning_rr: 0.0,
            calmar_ratio: 0.0,
            max_drawdown: 0.0,
            win_loss_ratio: 0.0,
            ulcer_index: 0.0,
            pain_ratio: 0.0,
            max_consecutive_wins: 0,
            max_consecutive_losses: 0,
            avg_win_streak: 0.0,
            avg_loss_streak: 0.0,
            median_rr: 0.0,
            avg_losing_rr: 0.0,
            p05_rr: 0.0,
            p95_rr: 0.0,
            largest_win: 0.0,
            largest_loss: 0.0,
            sample_quality: classify_sample(total_bars),
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
}

/// Lightweight accumulator for core statistics. This mirrors the logic in
/// `compute_core_statistics` but allows streaming updates over RR values
/// so that core metrics can be computed during bitset scanning without
/// materializing an intermediate RR vector.
struct CoreStatsAccumulator {
    total_bars: usize,
    total_return: f64,
    equity: f64,
    equity_peak: f64,
    max_drawdown: f64,
    profit_count: usize,
    loss_count: usize,
    simulate_equity: bool,
    position_sizing: PositionSizingMode,
    min_contracts: usize,
    max_contracts: Option<usize>,
    margin_per_contract_dollar: Option<f64>,
    capital_0: f64,
    risk_factor: f64,
    capital: f64,
    peak_capital: f64,
    max_drawdown_pct_equity: f64,
}

impl CoreStatsAccumulator {
    fn new(
        position_sizing: PositionSizingMode,
        capital_dollar: Option<f64>,
        risk_pct_per_trade: Option<f64>,
        min_contracts: usize,
        max_contracts: Option<usize>,
        margin_per_contract_dollar: Option<f64>,
    ) -> Self {
        let capital_0 = capital_dollar.unwrap_or(0.0);
        let risk_pct = risk_pct_per_trade.unwrap_or(0.0);
        let simulate_equity = capital_0 > 0.0 && risk_pct > 0.0;
        let risk_factor = risk_pct * 0.01;
        Self {
            total_bars: 0,
            total_return: 0.0,
            equity: 0.0,
            equity_peak: 0.0,
            max_drawdown: 0.0,
            profit_count: 0,
            loss_count: 0,
            simulate_equity,
            position_sizing,
            min_contracts: min_contracts.max(1),
            max_contracts,
            margin_per_contract_dollar,
            capital_0,
            risk_factor,
            capital: capital_0,
            peak_capital: capital_0,
            max_drawdown_pct_equity: 0.0,
        }
    }

    fn push(&mut self, rr: f64, risk_per_contract_dollar: Option<f64>) {
        self.total_return += rr;
        if rr > 0.0 {
            self.profit_count += 1;
        } else if rr < 0.0 {
            self.loss_count += 1;
        }

        // R-space equity and drawdown.
        self.equity += rr;
        if self.equity > self.equity_peak {
            self.equity_peak = self.equity;
        }
        let dd = self.equity - self.equity_peak;
        if dd < self.max_drawdown {
            self.max_drawdown = dd;
        }

        // Optional dollar equity simulation for true Calmar.
        if self.simulate_equity {
            let pnl = match self.position_sizing {
                PositionSizingMode::Fractional => {
                    let risk_i = self.capital * self.risk_factor;
                    rr * risk_i
                }
                PositionSizingMode::Contracts => {
                    let rpc = match risk_per_contract_dollar {
                        Some(v) if v.is_finite() && v > 0.0 => v,
                        _ => return,
                    };
                    let risk_budget = self.capital * self.risk_factor;
                    let raw = (risk_budget / rpc).floor();
                    let mut contracts = if raw.is_finite() && raw >= 0.0 {
                        raw as usize
                    } else {
                        0
                    };
                    if contracts < self.min_contracts {
                        contracts = self.min_contracts;
                    }
                    if let Some(max_contracts) = self.max_contracts {
                        contracts = contracts.min(max_contracts);
                    }
                    if let Some(margin) = self.margin_per_contract_dollar {
                        if margin.is_finite()
                            && margin > 0.0
                            && self.capital.is_finite()
                            && self.capital > 0.0
                        {
                            let cap = (self.capital / margin).floor();
                            if cap.is_finite() && cap >= 0.0 {
                                contracts = contracts.min(cap as usize);
                            }
                        }
                    }
                    if contracts == 0 {
                        return;
                    }
                    rr * rpc * (contracts as f64)
                }
            };
            let next_capital = self.capital + pnl;
            self.capital = next_capital;
            if self.capital > self.peak_capital {
                self.peak_capital = self.capital;
            }
            if self.peak_capital > 0.0 {
                let dd_pct = ((self.capital - self.peak_capital) / self.peak_capital) * 100.0;
                let dd_mag = -dd_pct;
                if dd_mag > self.max_drawdown_pct_equity {
                    self.max_drawdown_pct_equity = dd_mag;
                }
            }
        }
    }

    fn finalize(
        self,
        depth: usize,
        label_hits: usize,
        equity_time_years: Option<f64>,
    ) -> StatSummary {
        let total_bars = self.total_bars;

        // Equity-curve metrics driven by capital and risk%.
        let final_capital;
        let total_return_pct;
        let cagr_pct;
        let mut max_drawdown_pct_equity = self.max_drawdown_pct_equity;
        let calmar_equity;

        if self.simulate_equity && self.capital_0 > 0.0 && total_bars > 0 {
            let fc = self.capital;
            let tr_pct = ((fc / self.capital_0) - 1.0) * 100.0;

            let years = equity_time_years.unwrap_or(1.0).max(1e-9);
            let growth = if self.capital_0 > 0.0 {
                fc / self.capital_0
            } else {
                1.0
            };
            let cagr = if years > 0.0 && growth.is_finite() && growth > 0.0 {
                (growth.powf(1.0 / years) - 1.0) * 100.0
            } else {
                tr_pct
            };

            let calmar;
            if max_drawdown_pct_equity > 0.0 {
                calmar = cagr / max_drawdown_pct_equity;
            } else if cagr > 0.0 {
                calmar = f64::INFINITY;
            } else {
                calmar = 0.0;
            }

            final_capital = fc;
            total_return_pct = tr_pct;
            cagr_pct = cagr;
            calmar_equity = calmar;
        } else {
            // No capital/risk% context; keep equity metrics at zero.
            max_drawdown_pct_equity = 0.0;
            final_capital = 0.0;
            total_return_pct = 0.0;
            cagr_pct = 0.0;
            calmar_equity = 0.0;
        }

        let max_drawdown_abs = self.max_drawdown.abs();
        let win_rate_raw = if total_bars > 0 {
            (self.profit_count as f64 / total_bars as f64) * 100.0
        } else {
            0.0
        };
        let win_rate = win_rate_raw;

        let label_hits_count = label_hits;
        let label_misses = total_bars.saturating_sub(label_hits_count);
        let label_hit_rate_raw = if total_bars > 0 {
            (label_hits_count as f64 / total_bars as f64) * 100.0
        } else {
            0.0
        };
        let label_hit_rate = label_hit_rate_raw;

        StatSummary {
            depth,
            mask_hits: total_bars,
            total_bars,
            profitable_bars: self.profit_count,
            unprofitable_bars: self.loss_count,
            win_rate,
            label_hit_rate,
            label_hits: label_hits_count,
            label_misses,
            // Richer metrics are recomputed only for the reported top-K
            // combinations via the full-detail path.
            expectancy: 0.0,
            profit_factor: 0.0,
            avg_winning_rr: 0.0,
            calmar_ratio: calmar_equity,
            max_drawdown: max_drawdown_abs,
            win_loss_ratio: 0.0,
            ulcer_index: 0.0,
            pain_ratio: 0.0,
            max_consecutive_wins: 0,
            max_consecutive_losses: 0,
            avg_win_streak: 0.0,
            avg_loss_streak: 0.0,
            median_rr: 0.0,
            avg_losing_rr: 0.0,
            p05_rr: 0.0,
            p95_rr: 0.0,
            largest_win: 0.0,
            largest_loss: 0.0,
            sample_quality: classify_sample(total_bars),
            total_return: self.total_return,
            // No R→$ approximations in core; these remain zero until the full
            // statistics path is invoked for reporting.
            cost_per_trade_r: 0.0,
            dollars_per_r: 0.0,
            total_return_dollar: 0.0,
            max_drawdown_dollar: 0.0,
            expectancy_dollar: 0.0,
            final_capital,
            total_return_pct,
            cagr_pct,
            max_drawdown_pct_equity,
            calmar_equity,
            // Equity Sharpe/Sortino are only computed in the full-detail path.
            sharpe_equity: 0.0,
            sortino_equity: 0.0,
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn compute_core_statistics(
    depth: usize,
    _total_bars: usize,
    filtered_rr: &[f64],
    risk_per_contract_dollar: Option<&[f64]>,
    label_hits: usize,
    position_sizing: PositionSizingMode,
    capital_dollar: Option<f64>,
    risk_pct_per_trade: Option<f64>,
    equity_time_years: Option<f64>,
    min_contracts: usize,
    max_contracts: Option<usize>,
    margin_per_contract_dollar: Option<f64>,
) -> StatSummary {
    let mut acc = CoreStatsAccumulator::new(
        position_sizing,
        capital_dollar,
        risk_pct_per_trade,
        min_contracts,
        max_contracts,
        margin_per_contract_dollar,
    );
    for (idx, &rr) in filtered_rr.iter().enumerate() {
        acc.total_bars += 1;
        let rpc = match position_sizing {
            PositionSizingMode::Contracts => risk_per_contract_dollar
                .and_then(|values| values.get(idx).copied())
                .filter(|v| v.is_finite()),
            PositionSizingMode::Fractional => None,
        };
        acc.push(rr, rpc);
    }
    acc.finalize(depth, label_hits, equity_time_years)
}

#[allow(clippy::too_many_arguments)]
fn compute_full_statistics(
    depth: usize,
    total_bars: usize,
    filtered_rr: &[f64],
    risk_per_contract_dollar: Option<&[f64]>,
    label_hits: usize,
    dollars_per_r: Option<f64>,
    cost_per_trade_r: Option<f64>,
    capital_dollar: Option<f64>,
    risk_pct_per_trade: Option<f64>,
    equity_time_years: Option<f64>,
    position_sizing: PositionSizingMode,
    min_contracts: usize,
    max_contracts: Option<usize>,
    margin_per_contract_dollar: Option<f64>,
) -> StatSummary {
    let n = total_bars;
    let mut total_return = 0.0;
    let mut profit_sum = 0.0;
    let mut loss_sum = 0.0;
    let mut profit_count = 0usize;
    let mut loss_count = 0usize;

    // Drawdown-related accumulators.
    let mut equity = 0.0;
    let mut equity_peak = 0.0;
    let mut max_drawdown = 0.0; // most negative drawdown (in R)
    let mut dd_sum = 0.0;
    let mut dd_pct_sq_sum = 0.0;
    let mut dd_count = 0usize;
    let mut have_nonzero_peak = false;

    // Streak-related accumulators.
    let mut current_win_streak = 0usize;
    let mut current_loss_streak = 0usize;
    let mut max_consecutive_wins = 0usize;
    let mut max_consecutive_losses = 0usize;
    let mut total_win_streak_len = 0usize;
    let mut total_loss_streak_len = 0usize;
    let mut win_streak_count = 0usize;
    let mut loss_streak_count = 0usize;

    // Extremes.
    let mut largest_win = 0.0;
    let mut largest_loss = 0.0;

    for &rr in filtered_rr {
        total_return += rr;

        if rr > 0.0 {
            profit_sum += rr;
            profit_count += 1;
            if rr > largest_win {
                largest_win = rr;
            }
        } else if rr < 0.0 {
            let abs_rr = -rr;
            loss_sum += abs_rr;
            loss_count += 1;
            if abs_rr > largest_loss {
                largest_loss = abs_rr;
            }
        }

        // Equity curve and drawdowns.
        equity += rr;
        if equity > equity_peak {
            equity_peak = equity;
            if equity_peak.abs() >= f64::EPSILON {
                have_nonzero_peak = true;
            }
        }
        let dd = equity - equity_peak; // <= 0
        dd_sum += dd;
        dd_count += 1;
        if dd < max_drawdown {
            max_drawdown = dd;
        }
        if equity_peak.abs() >= f64::EPSILON {
            let pct = (dd / (equity_peak + f64::EPSILON)) * 100.0;
            dd_pct_sq_sum += pct * pct;
        }

        // Streak tracking: wins > 0, losses < 0, zero breaks both.
        if rr > 0.0 {
            if current_loss_streak > 0 {
                total_loss_streak_len += current_loss_streak;
                loss_streak_count += 1;
                if current_loss_streak > max_consecutive_losses {
                    max_consecutive_losses = current_loss_streak;
                }
                current_loss_streak = 0;
            }
            current_win_streak += 1;
        } else if rr < 0.0 {
            if current_win_streak > 0 {
                total_win_streak_len += current_win_streak;
                win_streak_count += 1;
                if current_win_streak > max_consecutive_wins {
                    max_consecutive_wins = current_win_streak;
                }
                current_win_streak = 0;
            }
            current_loss_streak += 1;
        } else {
            if current_win_streak > 0 {
                total_win_streak_len += current_win_streak;
                win_streak_count += 1;
                if current_win_streak > max_consecutive_wins {
                    max_consecutive_wins = current_win_streak;
                }
                current_win_streak = 0;
            }
            if current_loss_streak > 0 {
                total_loss_streak_len += current_loss_streak;
                loss_streak_count += 1;
                if current_loss_streak > max_consecutive_losses {
                    max_consecutive_losses = current_loss_streak;
                }
                current_loss_streak = 0;
            }
        }
    }

    // Flush any trailing streaks.
    if current_win_streak > 0 {
        total_win_streak_len += current_win_streak;
        win_streak_count += 1;
        if current_win_streak > max_consecutive_wins {
            max_consecutive_wins = current_win_streak;
        }
    }
    if current_loss_streak > 0 {
        total_loss_streak_len += current_loss_streak;
        loss_streak_count += 1;
        if current_loss_streak > max_consecutive_losses {
            max_consecutive_losses = current_loss_streak;
        }
    }

    let expectancy_raw = if n > 0 { total_return / n as f64 } else { 0.0 };
    let expectancy = expectancy_raw;

    // Profit factor and win/loss geometry.
    let profit_factor_raw = if loss_sum > 0.0 {
        profit_sum / loss_sum
    } else if profit_sum > 0.0 {
        f64::INFINITY
    } else {
        0.0
    };

    let avg_winning_rr = if profit_count > 0 {
        profit_sum / profit_count as f64
    } else {
        0.0
    };
    let avg_loss_abs = if loss_count > 0 {
        loss_sum / loss_count as f64
    } else {
        0.0
    };
    let avg_losing_rr = if loss_count > 0 { -avg_loss_abs } else { 0.0 };
    let win_loss_ratio_raw = if avg_loss_abs > 0.0 {
        avg_winning_rr / avg_loss_abs
    } else if avg_winning_rr > 0.0 {
        f64::INFINITY
    } else {
        0.0
    };

    // Drawdown-derived metrics.
    let max_drawdown_abs = max_drawdown.abs();
    let ulcer_index = if !have_nonzero_peak || dd_count == 0 {
        0.0
    } else {
        (dd_pct_sq_sum / dd_count as f64).sqrt()
    };
    let avg_drawdown = if dd_count > 0 {
        dd_sum / dd_count as f64
    } else {
        0.0
    };
    let pain_ratio = if avg_drawdown < 0.0 {
        total_return / avg_drawdown.abs()
    } else {
        0.0
    };
    let avg_win_streak = if win_streak_count > 0 {
        total_win_streak_len as f64 / win_streak_count as f64
    } else {
        0.0
    };
    let avg_loss_streak = if loss_streak_count > 0 {
        total_loss_streak_len as f64 / loss_streak_count as f64
    } else {
        0.0
    };

    // Distribution shape: median and simple 5th/95th percentiles.
    let (median_rr, p05_rr, p95_rr) = if n > 0 {
        percentile_triplet(filtered_rr)
    } else {
        (0.0, 0.0, 0.0)
    };

    let cost_r = cost_per_trade_r.unwrap_or(0.0);
    let dollars_per_r = dollars_per_r.unwrap_or(0.0);
    let mut total_return_dollar = if dollars_per_r > 0.0 {
        total_return * dollars_per_r
    } else {
        0.0
    };
    let mut max_drawdown_dollar = if dollars_per_r > 0.0 {
        max_drawdown_abs * dollars_per_r
    } else {
        0.0
    };
    let mut expectancy_dollar = if dollars_per_r > 0.0 {
        expectancy * dollars_per_r
    } else {
        0.0
    };

    // Net-R and label-based win rates.
    let win_rate_raw = if total_bars > 0 {
        (profit_count as f64 / total_bars as f64) * 100.0
    } else {
        0.0
    };
    let win_rate = win_rate_raw;

    let label_hits_count = label_hits;
    let label_misses = total_bars.saturating_sub(label_hits_count);
    let label_hit_rate_raw = if total_bars > 0 {
        (label_hits_count as f64 / total_bars as f64) * 100.0
    } else {
        0.0
    };
    let label_hit_rate = label_hit_rate_raw;

    // Equity-curve metrics driven by capital and risk%.
    let capital_0 = capital_dollar.unwrap_or(0.0);
    let risk_pct = risk_pct_per_trade.unwrap_or(0.0);
    let mut final_capital = 0.0;
    let mut total_return_pct = 0.0;
    let mut cagr_pct = 0.0;
    let mut max_drawdown_pct_equity = 0.0;
    let mut calmar_equity = 0.0;
    let mut sharpe_equity = 0.0;
    let mut sortino_equity = 0.0;

    if capital_0 > 0.0 && risk_pct > 0.0 && n > 0 {
        let mut capital = capital_0;
        let mut peak_capital = capital_0;
        let mut max_drawdown_dollar_sim = 0.0;
        let mut pnl_sum = 0.0;
        let mut eq_ret_sum = 0.0;
        let mut eq_ret_sq_sum = 0.0;
        let mut downside_sq_sum = 0.0;
        let mut downside_count = 0usize;

        for (idx, &rr) in filtered_rr.iter().enumerate() {
            let pnl = match position_sizing {
                PositionSizingMode::Fractional => {
                    let risk_i = capital * (risk_pct / 100.0);
                    rr * risk_i
                }
                PositionSizingMode::Contracts => {
                    let rpc = match risk_per_contract_dollar
                        .and_then(|values| values.get(idx).copied())
                    {
                        Some(v) if v.is_finite() && v > 0.0 => v,
                        _ => 0.0,
                    };
                    if rpc <= 0.0 {
                        0.0
                    } else {
                        let risk_budget = capital * (risk_pct / 100.0);
                        let raw = (risk_budget / rpc).floor();
                        let mut contracts = if raw.is_finite() && raw >= 0.0 {
                            raw as usize
                        } else {
                            0
                        };
                        let min_contracts = min_contracts.max(1);
                        if contracts < min_contracts {
                            contracts = min_contracts;
                        }
                        if let Some(max_contracts) = max_contracts {
                            contracts = contracts.min(max_contracts);
                        }
                        if let Some(margin) = margin_per_contract_dollar {
                            if margin.is_finite()
                                && margin > 0.0
                                && capital.is_finite()
                                && capital > 0.0
                            {
                                let cap = (capital / margin).floor();
                                if cap.is_finite() && cap >= 0.0 {
                                    contracts = contracts.min(cap as usize);
                                }
                            }
                        }
                        if contracts == 0 {
                            0.0
                        } else {
                            rr * rpc * (contracts as f64)
                        }
                    }
                }
            };
            let next_capital = capital + pnl;
            let ret = if capital > 0.0 {
                (next_capital / capital) - 1.0
            } else {
                0.0
            };

            pnl_sum += pnl;
            eq_ret_sum += ret;
            eq_ret_sq_sum += ret * ret;
            if ret < 0.0 {
                downside_sq_sum += ret * ret;
                downside_count += 1;
            }

            capital = next_capital;
            if capital > peak_capital {
                peak_capital = capital;
            }
            if peak_capital > 0.0 {
                let dd_pct = ((capital - peak_capital) / peak_capital) * 100.0;
                let dd_mag = -dd_pct;
                if dd_mag > max_drawdown_pct_equity {
                    max_drawdown_pct_equity = dd_mag;
                }
                let dd_dollar = capital - peak_capital;
                let dd_dollar_mag = -dd_dollar;
                if dd_dollar_mag > max_drawdown_dollar_sim {
                    max_drawdown_dollar_sim = dd_dollar_mag;
                }
            }
        }

        final_capital = capital;
        if capital_0 > 0.0 {
            total_return_pct = ((final_capital / capital_0) - 1.0) * 100.0;
        }

        total_return_dollar = pnl_sum;
        expectancy_dollar = pnl_sum / (n as f64);
        max_drawdown_dollar = max_drawdown_dollar_sim;

        let years = equity_time_years.unwrap_or(1.0).max(1e-9);
        let growth = if capital_0 > 0.0 {
            final_capital / capital_0
        } else {
            1.0
        };
        if years > 0.0 && growth.is_finite() && growth > 0.0 {
            cagr_pct = (growth.powf(1.0 / years) - 1.0) * 100.0;
        } else {
            cagr_pct = total_return_pct;
        }

        if max_drawdown_pct_equity > 0.0 {
            calmar_equity = cagr_pct / max_drawdown_pct_equity;
        } else if cagr_pct > 0.0 {
            calmar_equity = f64::INFINITY;
        } else {
            calmar_equity = 0.0;
        }

        let n_returns = n as f64;
        let mean = eq_ret_sum / n_returns;
        let var = (eq_ret_sq_sum / n_returns) - mean * mean;
        let std = var.max(0.0).sqrt();

        let downside_std = if downside_count > 0 {
            (downside_sq_sum / (downside_count as f64)).sqrt()
        } else {
            0.0
        };

        let trades_per_year = (n_returns / years).max(1e-9);
        let annual_scale = trades_per_year.sqrt();

        if std > 0.0 {
            sharpe_equity = (mean / std) * annual_scale;
        } else {
            sharpe_equity = 0.0;
        }

        sortino_equity = if downside_std > 0.0 {
            (mean / downside_std) * annual_scale
        } else if downside_std == 0.0 && mean > 0.0 {
            f64::INFINITY
        } else {
            0.0
        };
    }

    StatSummary {
        depth,
        mask_hits: total_bars,
        total_bars,
        profitable_bars: profit_count,
        unprofitable_bars: loss_count,
        win_rate,
        label_hit_rate,
        label_hits: label_hits_count,
        label_misses,
        expectancy,
        profit_factor: profit_factor_raw,
        avg_winning_rr,
        calmar_ratio: calmar_equity,
        max_drawdown: max_drawdown_abs,
        win_loss_ratio: win_loss_ratio_raw,
        ulcer_index,
        pain_ratio,
        max_consecutive_wins,
        max_consecutive_losses,
        avg_win_streak,
        avg_loss_streak,
        median_rr,
        avg_losing_rr,
        p05_rr,
        p95_rr,
        largest_win,
        largest_loss,
        sample_quality: classify_sample(total_bars),
        total_return,
        cost_per_trade_r: cost_r,
        dollars_per_r,
        total_return_dollar,
        max_drawdown_dollar,
        expectancy_dollar,
        final_capital,
        total_return_pct,
        cagr_pct,
        max_drawdown_pct_equity,
        calmar_equity,
        sharpe_equity,
        sortino_equity,
    }
}

fn percentile_triplet(filtered_rr: &[f64]) -> (f64, f64, f64) {
    SORTED_RETURNS_BUFFER.with(|cell| {
        let mut buf = cell.borrow_mut();
        buf.clear();
        buf.extend_from_slice(filtered_rr);
        let len = buf.len();
        if len == 0 {
            return (0.0, 0.0, 0.0);
        }

        let len_minus_one = len - 1;
        let idx_p05 = ((len_minus_one as f64) * 0.05).round() as usize;
        let idx_p95 = ((len_minus_one as f64) * 0.95).round() as usize;
        let idx_p05 = idx_p05.min(len - 1);
        let idx_p95 = idx_p95.min(len - 1);

        // Median via selection:
        // - First, select the middle element.
        // - For even lengths, recover the lower neighbor as the maximum
        //   element in the left partition.
        let mid = len / 2;
        buf.select_nth_unstable_by(mid, |a, b| {
            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
        });
        let median = if len % 2 == 0 {
            let mid_val = buf[mid];
            let mut max_left = buf[0];
            for &v in &buf[1..mid] {
                if v > max_left {
                    max_left = v;
                }
            }
            (max_left + mid_val) / 2.0
        } else {
            buf[mid]
        };

        // 5th and 95th percentiles via selection.
        buf.select_nth_unstable_by(idx_p05, |a, b| {
            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
        });
        let p05 = buf[idx_p05];

        buf.select_nth_unstable_by(idx_p95, |a, b| {
            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
        });
        let p95 = buf[idx_p95];

        (median, p05, p95)
    })
}

#[cfg(test)]
fn round_to(value: f64, decimals: i32) -> f64 {
    if !value.is_finite() {
        return value;
    }
    let factor = 10f64.powi(decimals);
    let scaled = value * factor;
    let floor = scaled.floor();
    let diff = scaled - floor;
    let epsilon = 1e-9;
    let rounded = if (diff - 0.5).abs() < epsilon {
        let floor_even = ((floor as i64) & 1) == 0;
        if floor_even { floor } else { floor + 1.0 }
    } else {
        scaled.round()
    };
    rounded / factor
}

fn classify_sample(total_bars: usize) -> &'static str {
    match total_bars {
        n if n >= 100 => "excellent",
        n if n >= 50 => "good",
        n if n >= 30 => "fair",
        _ => "poor",
    }
}

fn load_boolean_vector(data: &ColumnarData, column: &str) -> Result<Vec<bool>> {
    Ok(data
        .boolean_column(column)?
        .into_iter()
        .map(|value| value.unwrap_or(false))
        .collect())
}

fn load_float_vector(data: &ColumnarData, column: &str) -> Result<Vec<f64>> {
    Ok(data
        .float_column(column)?
        .into_iter()
        .map(|value| value.unwrap_or(0.0))
        .collect())
}

/// In-memory catalog of bitset masks for all features in the current run.
/// This is built once from the boolean masks and then shared read-only
/// across all worker threads, avoiding per-combination locking and
/// reference counting.
#[derive(Clone)]
pub struct BitsetCatalog {
    bitsets: Vec<BitsetMask>,
    name_to_index: HashMap<String, usize>,
}

impl BitsetCatalog {
    fn get(&self, feature: &str) -> Option<&BitsetMask> {
        self.name_to_index
            .get(feature)
            .and_then(|&idx| self.bitsets.get(idx))
    }

    fn get_by_index(&self, index: usize) -> Option<&BitsetMask> {
        self.bitsets.get(index)
    }
}

pub fn build_bitset_catalog(
    ctx: &EvaluationContext,
    features: &[FeatureDescriptor],
) -> Result<BitsetCatalog> {
    let mut bitsets = Vec::with_capacity(features.len());
    let mut name_to_index = HashMap::with_capacity(features.len());

    for (idx, descriptor) in features.iter().enumerate() {
        let name = descriptor.name.as_str();
        let mask = ctx.feature_mask(name)?;
        let bitset = BitsetMask::from_bools(mask.as_ref());
        bitsets.push(bitset);
        // Map feature name to its first index; this mirrors how combinations
        // reference features by descriptor name while allowing fast index
        // lookups for the evaluation hot path.
        name_to_index.entry(name.to_string()).or_insert(idx);
    }

    Ok(BitsetCatalog {
        bitsets,
        name_to_index,
    })
}

fn apply_operator(value: f64, threshold: f64, operator: ComparisonOperator) -> bool {
    match operator {
        ComparisonOperator::GreaterThan => value > threshold,
        ComparisonOperator::LessThan => value < threshold,
        ComparisonOperator::GreaterEqual => value >= threshold,
        ComparisonOperator::LessEqual => value <= threshold,
    }
}

fn apply_pair_operator(left: f64, right: f64, operator: ComparisonOperator) -> bool {
    match operator {
        ComparisonOperator::GreaterThan => left > right,
        ComparisonOperator::LessThan => left < right,
        ComparisonOperator::GreaterEqual => left >= right,
        ComparisonOperator::LessEqual => left <= right,
    }
}

pub(crate) fn detect_reward_column(data: &ColumnarData, config: &Config) -> Result<Option<String>> {
    const FALLBACK: [&str; 4] = ["rr", "reward", "r_multiple", "returns"];
    let mut candidates: Vec<String> = Vec::new();
    candidates.push(format!("rr_{}", config.target));
    match config.direction {
        Direction::Long => candidates.push("rr_long".into()),
        Direction::Short => candidates.push("rr_short".into()),
        Direction::Both => {
            candidates.push("rr_long".into());
            candidates.push("rr_short".into());
        }
    }
    candidates.extend(FALLBACK.iter().map(|candidate| (*candidate).to_string()));
    for candidate in candidates {
        let normalized = candidate.trim();
        if normalized.is_empty() {
            continue;
        }
        if data.has_column(normalized) {
            return Ok(Some(normalized.to_string()));
        }
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{EvalProfileMode, LogicMode, ReportMetricsMode, StackingMode};
    use std::path::PathBuf;
    use tempfile::tempdir;

    fn dummy_config(target: &str, direction: Direction) -> Config {
        Config {
            input_csv: PathBuf::from("dummy.csv"),
            source_csv: None,
            direction,
            target: target.to_string(),
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
            stats_detail: crate::config::StatsDetail::Full,
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
            position_sizing: crate::config::PositionSizingMode::Fractional,
            stop_distance_column: None,
            stop_distance_unit: crate::config::StopDistanceUnit::Points,
            min_contracts: 1,
            max_contracts: None,
            point_value: None,
            tick_value: None,
            margin_per_contract_dollar: None,
            require_any_features: Vec::new(),
        }
    }

    #[test]
    fn bitset_mask_from_bools_sets_bits_correctly() {
        let mask = BitsetMask::from_bools(&[true, false, true, false, false, true]);
        assert_eq!(mask.len, 6);
        assert_eq!(mask.words.len(), 1);
        let word = mask.words[0];
        assert_eq!(word & 1, 1, "bit 0 should be set");
        assert_eq!((word >> 2) & 1, 1, "bit 2 should be set");
        assert_eq!((word >> 5) & 1, 1, "bit 5 should be set");
    }

    #[test]
    fn classify_sample_buckets_match_thresholds() {
        assert_eq!(classify_sample(0), "poor");
        assert_eq!(classify_sample(29), "poor");
        assert_eq!(classify_sample(30), "fair");
        assert_eq!(classify_sample(49), "fair");
        assert_eq!(classify_sample(50), "good");
        assert_eq!(classify_sample(99), "good");
        assert_eq!(classify_sample(100), "excellent");
        assert_eq!(classify_sample(1_000), "excellent");
    }

    #[test]
    fn round_to_uses_bankers_rounding_for_ties() {
        let a = round_to(0.125, 2);
        let b = round_to(0.135, 2);
        assert!((a - 0.12).abs() < 1e-9, "0.125 should round to 0.12");
        assert!((b - 0.14).abs() < 1e-9, "0.135 should round to 0.14");
    }

    #[test]
    fn apply_operator_and_pair_operator_match_comparisons() {
        use crate::feature::ComparisonOperator;

        assert!(apply_operator(2.0, 1.0, ComparisonOperator::GreaterThan));
        assert!(!apply_operator(0.5, 1.0, ComparisonOperator::GreaterThan));
        assert!(apply_operator(1.0, 1.0, ComparisonOperator::GreaterEqual));
        assert!(apply_pair_operator(1.0, 2.0, ComparisonOperator::LessThan));
        assert!(!apply_pair_operator(3.0, 2.0, ComparisonOperator::LessThan));
    }

    #[test]
    fn detect_reward_column_prefers_target_specific_rr() -> Result<()> {
        let dir = tempdir()?;
        let csv_path = dir.path().join("rr.csv");
        // Include both a target-specific rr column and a generic rr_long.
        std::fs::write(&csv_path, "rr_next_bar_color_and_wicks,rr_long\n0.1,0.2\n")?;

        let data = ColumnarData::load(&csv_path)?;
        let config = dummy_config("next_bar_color_and_wicks", Direction::Long);
        let detected = detect_reward_column(&data, &config)?;
        assert_eq!(
            detected.as_deref(),
            Some("rr_next_bar_color_and_wicks"),
            "target-specific rr_<target> should be preferred when present"
        );
        Ok(())
    }

    #[test]
    fn detect_reward_column_falls_back_to_directional_then_generic() -> Result<()> {
        let dir = tempdir()?;
        let csv_path = dir.path().join("rr_fallback.csv");
        // No rr_<target> column; should prefer rr_long for Long direction.
        std::fs::write(&csv_path, "rr_long,rr\n0.1,0.2\n")?;

        let data = ColumnarData::load(&csv_path)?;
        let config = dummy_config("some_other_target", Direction::Long);
        let detected = detect_reward_column(&data, &config)?;
        assert_eq!(
            detected.as_deref(),
            Some("rr_long"),
            "rr_long should be chosen when rr_<target> is absent"
        );
        Ok(())
    }

    #[test]
    fn percentile_triplet_matches_sort_reference() {
        let values = vec![-2.0, -1.0, 0.5, 1.0, 3.0, 5.0, 8.0, 9.0, 10.0, 12.0];

        let (median_sel, p05_sel, p95_sel) = percentile_triplet(&values);

        let mut sorted = values.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let len = sorted.len();
        let mid = len / 2;
        let median_ref = if len % 2 == 0 {
            (sorted[mid - 1] + sorted[mid]) / 2.0
        } else {
            sorted[mid]
        };
        let idx_p05 = ((len - 1) as f64 * 0.05).round() as usize;
        let idx_p95 = ((len - 1) as f64 * 0.95).round() as usize;
        let p05_ref = sorted[idx_p05.min(len - 1)];
        let p95_ref = sorted[idx_p95.min(len - 1)];

        assert!((median_sel - median_ref).abs() < 1e-9);
        assert!((p05_sel - p05_ref).abs() < 1e-9);
        assert!((p95_sel - p95_ref).abs() < 1e-9);
    }

    #[test]
    fn net_r_win_rate_and_label_hit_rate_diverge_when_labels_and_profit_disagree() {
        // Three trades in R-space: one loss, one win, one flat.
        let rr = vec![-0.5, 1.0, 0.0];
        let total_bars = rr.len();
        // Engineered target is true on the last two trades (win + flat).
        let label_hits = 2usize;

        let stats = compute_full_statistics(
            1,          // depth
            total_bars, // total_bars
            &rr,        // filtered_rr (already net R)
            None,       // risk_per_contract_dollar
            label_hits, // label hits
            None,       // dollars_per_r
            None,       // cost_per_trade_r
            None,       // capital_dollar
            None,       // risk_pct_per_trade
            None,       // equity_time_years
            crate::config::PositionSizingMode::Fractional,
            1,
            None,
            None,
        );

        assert_eq!(stats.total_bars, 3);
        // Net-profitable trades: only the +1.0R.
        assert_eq!(stats.profitable_bars, 1);
        assert_eq!(stats.unprofitable_bars, 1);
        // Win rate is based on net R, not labels: 1/3.
        let expected_win_rate = 100.0 / 3.0;
        assert!((stats.win_rate - expected_win_rate).abs() < 1e-9);

        // Label metrics reflect target behaviour (2/3 hits).
        assert_eq!(stats.label_hits, 2);
        assert_eq!(stats.label_misses, 1);
        let expected_label_hit_rate = 200.0 / 3.0;
        assert!((stats.label_hit_rate - expected_label_hit_rate).abs() < 1e-9);
    }

    #[test]
    fn contracts_equity_simulation_respects_margin_cap() {
        // One winning trade with 1R=rpc dollars and contracts computed from risk budget,
        // then capped by margin_per_contract_dollar.
        let rr = vec![1.0];
        let rpc = vec![100.0];
        let stats = compute_full_statistics(
            1,
            1,
            &rr,
            Some(&rpc),
            0,
            None,
            None,
            Some(10_000.0),
            Some(10.0),
            Some(1.0),
            crate::config::PositionSizingMode::Contracts,
            1,
            None,
            Some(2_500.0),
        );

        // risk budget = 10k*10% = 1k => floor(1k/100)=10 by risk
        // margin cap = floor(10k/2500)=4 => pnl = 1R * $100 * 4 = $400
        assert!((stats.final_capital - 10_400.0).abs() < 1e-9);
    }

    #[test]
    fn evaluation_uses_eligible_and_finite_rr_as_trade_denominator() -> Result<()> {
        let dir = tempdir()?;
        let csv_path = dir.path().join("eligibility.csv");
        let csv = "\
feature_a,highlow_or_atr,highlow_or_atr_eligible,rr_highlow_or_atr\n\
true,true,true,2.0\n\
true,false,true,-1.0\n\
true,false,false,NaN\n\
true,false,false,NaN\n\
true,true,true,2.0\n";
        std::fs::write(&csv_path, csv)?;

        let data = Arc::new(ColumnarData::load(&csv_path)?);
        let mut config = dummy_config("highlow_or_atr", Direction::Long);
        config.stats_detail = StatsDetail::Full;
        let mask_cache = Arc::new(MaskCache::with_max_entries(128));
        let ctx = EvaluationContext::new(
            Arc::clone(&data),
            mask_cache,
            &config,
            Arc::new(HashMap::new()),
        )?;

        let features = vec![FeatureDescriptor::boolean("feature_a", "test")];
        let bitsets = build_bitset_catalog(&ctx, &features)?;
        let combo = vec![features[0].clone()];
        let stats = evaluate_combination(&combo, &ctx, &bitsets, 3)?;

        assert_eq!(stats.mask_hits, 5, "combo mask should match all 5 rows");
        assert_eq!(
            stats.total_bars, 3,
            "only eligible rows with finite RR count as trades"
        );
        assert_eq!(stats.profitable_bars, 2);
        assert_eq!(stats.unprofitable_bars, 1);
        assert_eq!(stats.label_hits, 2);
        assert_eq!(stats.label_misses, 1);
        Ok(())
    }

    #[test]
    fn evaluation_no_stacking_skips_overlapping_trades_using_exit_indices() -> Result<()> {
        let dir = tempdir()?;
        let csv_path = dir.path().join("no_stacking.csv");
        let csv = "\
feature_a,highlow_or_atr,highlow_or_atr_eligible,rr_highlow_or_atr,highlow_or_atr_exit_i\n\
true,false,true,1.0,4\n\
true,false,true,1.0,4\n\
true,false,true,1.0,4\n\
true,false,true,1.0,4\n\
true,false,true,1.0,4\n";
        std::fs::write(&csv_path, csv)?;

        let data = Arc::new(ColumnarData::load(&csv_path)?);
        let mask_cache = Arc::new(MaskCache::with_max_entries(128));
        let features = vec![FeatureDescriptor::boolean("feature_a", "test")];

        let mut config_stacking = dummy_config("highlow_or_atr", Direction::Long);
        config_stacking.stats_detail = StatsDetail::Full;
        config_stacking.stacking_mode = StackingMode::Stacking;
        let ctx_stacking = EvaluationContext::new(
            Arc::clone(&data),
            Arc::clone(&mask_cache),
            &config_stacking,
            Arc::new(HashMap::new()),
        )?;
        let bitsets_stacking = build_bitset_catalog(&ctx_stacking, &features)?;
        let combo = vec![features[0].clone()];
        let stats_stacking = evaluate_combination(&combo, &ctx_stacking, &bitsets_stacking, 1)?;

        let mut config_no_stacking = dummy_config("highlow_or_atr", Direction::Long);
        config_no_stacking.stats_detail = StatsDetail::Full;
        config_no_stacking.stacking_mode = StackingMode::NoStacking;
        let ctx_no_stacking = EvaluationContext::new(
            Arc::clone(&data),
            mask_cache,
            &config_no_stacking,
            Arc::new(HashMap::new()),
        )?;
        let bitsets_no_stacking = build_bitset_catalog(&ctx_no_stacking, &features)?;
        let stats_no_stacking =
            evaluate_combination(&combo, &ctx_no_stacking, &bitsets_no_stacking, 1)?;

        assert_eq!(stats_stacking.mask_hits, 5);
        assert_eq!(stats_stacking.total_bars, 5);
        assert_eq!(stats_no_stacking.mask_hits, 5);
        assert_eq!(
            stats_no_stacking.total_bars, 2,
            "expected only idx=0 and idx=4 to be eligible under no-stacking"
        );
        Ok(())
    }
}
