use std::collections::HashMap;
use std::path::Path;

use crate::thresholds;
use anyhow::Result;
use barsmith_rs::config::Config;
use barsmith_rs::feature::{
    ComparisonOperator, ComparisonSpec, FeatureCategory, FeatureDescriptor,
    generate_unordered_feature_comparisons,
};
use polars::prelude::*;
use std::collections::HashSet;
use tracing::info;

pub struct FeatureCatalog;

pub struct CatalogBuild {
    pub descriptors: Vec<FeatureDescriptor>,
    pub comparison_specs: HashMap<String, ComparisonSpec>,
}

impl FeatureCatalog {
    pub fn build_with_dataset(dataset_path: &Path, config: &Config) -> Result<CatalogBuild> {
        // Load engineered dataset to audit boolean/continuous coverage against the catalog.
        let df = CsvReader::from_path(dataset_path)?
            .with_ignore_errors(true)
            .has_header(true)
            .finish()?;
        audit_boolean_definitions(&df);
        audit_boolean_coverage(&df);
        audit_continuous_definitions(&df);
        audit_continuous_coverage(&df);

        // Start from the boolean catalog, then drop any flags that are
        // dataset-trivial (always true/false) or exact duplicates of another
        // boolean mask. This keeps the permutation surface focused on
        // informative features without changing the available continuous or
        // comparison descriptors.
        let mut descriptors = Self::boolean_descriptors();
        descriptors = prune_boolean_constants_and_duplicates(descriptors, &df);

        let threshold_catalog = thresholds::generate_threshold_catalog(dataset_path)?;
        let mut comparison_specs = threshold_catalog.specs;
        descriptors.extend(threshold_catalog.descriptors);

        // Optionally augment the catalog with feature-to-feature comparisons
        // (e.g., 9ema>200sma), controlled via the main Config.
        if config.enable_feature_pairs {
            let mut available_pair_features: Vec<&str> = Vec::new();
            let mut missing_pair_features: Vec<&str> = Vec::new();
            let mut non_numeric_pair_features: Vec<&str> = Vec::new();
            let declared_pair_names = pairwise_declared_feature_names();
            let declared_pair_set: HashSet<&str> = declared_pair_names.iter().copied().collect();

            // Treat all explicitly curated pairwise numeric features as bases
            // for feature-to-feature comparisons, but only if they exist and
            // are truly numeric in the engineered dataset.
            for &name in &declared_pair_names {
                match df.column(name) {
                    Ok(series) => match series.dtype() {
                        DataType::Float32 | DataType::Float64 => {
                            // Reuse the same sanity check as for continuous
                            // features: skip columns that are effectively
                            // boolean 0/1, even if they are stored as floats.
                            if is_binary_01_series(&series) {
                                non_numeric_pair_features.push(name);
                            } else {
                                available_pair_features.push(name);
                            }
                        }
                        _ => {
                            non_numeric_pair_features.push(name);
                        }
                    },
                    Err(_) => {
                        missing_pair_features.push(name);
                    }
                }
            }

            if !missing_pair_features.is_empty() {
                info!(
                    "⚠️  Pairwise numeric configuration entries missing from engineered dataset (not present as columns):"
                );
                for name in &missing_pair_features {
                    info!("   - {}", name);
                }
                info!(
                    "💡 Consider adding these to engineer.rs or removing them from the pairwise numeric configuration"
                );
            }

            if !non_numeric_pair_features.is_empty() {
                info!(
                    "⚠️  Pairwise numeric configuration entries are not numeric in the engineered dataset (skipping for feature-pairs):"
                );
                for name in &non_numeric_pair_features {
                    if let Ok(series) = df.column(name) {
                        info!("   - {} (dtype={:?})", name, series.dtype());
                    } else {
                        info!("   - {}", name);
                    }
                }
                info!(
                    "💡 Ensure these columns are exported as float32/float64 if you want them in feature-to-feature comparisons"
                );
            }

            // Also surface any numeric columns that *could* be used for
            // feature-to-feature comparisons but are not listed in
            // the pairwise numeric configuration, mirroring the continuous coverage
            // audit. This helps keep the pairwise catalog in sync with the
            // engineered dataframe.
            let mut unexpected_pair_candidates = Vec::new();
            for series in df.get_columns() {
                let name = series.name();
                if declared_pair_set.contains(name) {
                    continue;
                }
                match series.dtype() {
                    DataType::Float32 | DataType::Float64 => {
                        if !is_binary_01_series(series) {
                            unexpected_pair_candidates.push(name.to_string());
                        }
                    }
                    _ => {}
                }
            }
            if !unexpected_pair_candidates.is_empty() {
                unexpected_pair_candidates.sort();
                info!(
                    "⚠️  Found {} numeric features in dataframe but not in the pairwise numeric configuration (skipped for feature-to-feature predicates):",
                    unexpected_pair_candidates.len()
                );
                for name in &unexpected_pair_candidates {
                    info!("   - {}", name);
                }
                info!(
                    "💡 Consider adding these to the pairwise numeric configuration in custom_rs::features if you want feature-to-feature comparisons for them"
                );
            }

            if !available_pair_features.is_empty() {
                info!("🎯 Feature-pair candidates:");
                for name in &available_pair_features {
                    info!("   - {}", name);
                }

                // Only strict inequality for feature-to-feature comparisons to
                // reduce redundancy; >= and <= add little information on top
                // of > and < for most numeric pairs.
                let ops = [
                    ComparisonOperator::GreaterThan,
                    ComparisonOperator::LessThan,
                ];
                let (pair_descriptors, pair_specs) = generate_unordered_feature_comparisons(
                    &available_pair_features,
                    &ops,
                    None,
                    "Feature-to-feature comparison",
                );

                // Filter generated comparisons through the per-feature pairing
                // rules, and respect the optional feature_pairs_limit from the
                // main Config.
                let mut descriptor_by_name: HashMap<String, FeatureDescriptor> = pair_descriptors
                    .into_iter()
                    .map(|d| (d.name.clone(), d))
                    .collect();

                let mut filtered_descriptors: Vec<FeatureDescriptor> = Vec::new();
                let mut filtered_specs: HashMap<String, ComparisonSpec> = HashMap::new();
                let limit = config.feature_pairs_limit.unwrap_or(usize::MAX);
                let mut emitted = 0usize;

                for (name, spec) in pair_specs {
                    let left = spec.base_feature.as_str();
                    let right = match spec.rhs_feature.as_ref() {
                        Some(rhs) => rhs.as_str(),
                        None => continue,
                    };

                    if !pair_allowed(left, right) {
                        continue;
                    }

                    if emitted >= limit {
                        break;
                    }

                    if let Some(descriptor) = descriptor_by_name.remove(&name) {
                        filtered_descriptors.push(descriptor);
                    }
                    filtered_specs.insert(name, spec);
                    emitted += 1;
                }

                let pair_count = filtered_specs.len();

                let preview_names: Vec<String> = filtered_descriptors
                    .iter()
                    .take(5)
                    .map(|d| d.name.clone())
                    .collect();

                descriptors.extend(filtered_descriptors);
                for (key, spec) in filtered_specs {
                    comparison_specs.entry(key).or_insert(spec);
                }

                let op_symbols: Vec<&str> = ops
                    .iter()
                    .map(|op| match op {
                        ComparisonOperator::GreaterThan => ">",
                        ComparisonOperator::LessThan => "<",
                        _ => "?", // unreachable given current ops
                    })
                    .collect();
                let total_pairs = available_pair_features.len()
                    * (available_pair_features.len().saturating_sub(1))
                    / 2;
                let theoretical_predicates = total_pairs * op_symbols.len();
                info!(
                    dataset = %dataset_path.display(),
                    pair_feature_count = available_pair_features.len(),
                    operators = ?op_symbols,
                    theoretical_predicates,
                    pair_condition_count = pair_count,
                    pair_limit = ?config.feature_pairs_limit,
                    "Added feature-to-feature comparisons to catalog"
                );

                // Emit a small preview of the generated predicates, mirroring the
                // threshold summary style.
                if pair_count > 0 {
                    info!("🎯 Sample feature-to-feature predicates:");
                    for name in &preview_names {
                        info!("   - {}", name);
                    }
                    if pair_count > 5 {
                        info!("   ... and {} more", pair_count.saturating_sub(5));
                    }
                }
            } else {
                info!(
                    dataset = %dataset_path.display(),
                    "Feature pairs enabled but no PAIRWISE_NUMERIC_FEATURES present in dataset; skipping feature-to-feature comparisons"
                );
            }
        }

        info!(
            dataset = %dataset_path.display(),
            boolean_features = descriptors
                .iter()
                .filter(|d| matches!(d.category, FeatureCategory::Boolean))
                .count(),
            feature_vs_constant = descriptors
                .iter()
                .filter(|d| matches!(d.category, FeatureCategory::FeatureVsConstant))
                .count(),
            feature_vs_feature = descriptors
                .iter()
                .filter(|d| matches!(d.category, FeatureCategory::FeatureVsFeature))
                .count(),
            catalog_total = descriptors.len(),
            "Feature catalog generated"
        );

        Ok(CatalogBuild {
            descriptors,
            comparison_specs,
        })
    }

    pub fn boolean_descriptors() -> Vec<FeatureDescriptor> {
        BOOLEAN_FEATURES
            .iter()
            .map(|name| FeatureDescriptor::boolean(name, BOOLEAN_NOTE))
            .collect()
    }

    pub fn boolean_names() -> &'static [&'static str] {
        BOOLEAN_FEATURES
    }

    pub fn descriptors_for(names: &[&str]) -> Vec<FeatureDescriptor> {
        names
            .iter()
            .map(|name| {
                if BOOLEAN_FEATURES.contains(name) {
                    FeatureDescriptor::boolean(name, BOOLEAN_NOTE)
                } else if CONTINUOUS_FEATURES.contains(name) {
                    FeatureDescriptor::new(*name, FeatureCategory::Continuous, CONTINUOUS_NOTE)
                } else {
                    // Fallback: treat unknown names as boolean flags. This is primarily
                    // used by parity tests that request a curated set of legacy features
                    // which may not all be part of the core BOOLEAN_FEATURES catalog.
                    FeatureDescriptor::boolean(name, BOOLEAN_NOTE)
                }
            })
            .collect()
    }
}

fn boolean_mask_from_series(series: &Series) -> Option<Vec<bool>> {
    match series.dtype() {
        DataType::Boolean => {
            let ca = series.bool().ok()?;
            Some(ca.into_iter().map(|v| v.unwrap_or(false)).collect())
        }
        DataType::Int8
        | DataType::Int16
        | DataType::Int32
        | DataType::Int64
        | DataType::UInt8
        | DataType::UInt16
        | DataType::UInt32
        | DataType::UInt64 => {
            if !is_binary_01_series(series) {
                return None;
            }
            let ca = series.i64().ok()?;
            Some(
                ca.into_iter()
                    .map(|v| matches!(v, Some(value) if value == 1))
                    .collect(),
            )
        }
        DataType::Float32 | DataType::Float64 => {
            if !is_binary_01_series(series) {
                return None;
            }
            let ca = series.f64().ok()?;
            Some(
                ca.into_iter()
                    .map(|v| matches!(v, Some(value) if value == 1.0))
                    .collect(),
            )
        }
        _ => None,
    }
}

fn prune_boolean_constants_and_duplicates(
    descriptors: Vec<FeatureDescriptor>,
    df: &DataFrame,
) -> Vec<FeatureDescriptor> {
    let mut kept: Vec<FeatureDescriptor> = Vec::new();
    let mut mask_index: HashMap<Vec<bool>, String> = HashMap::new();
    let mut constant_dropped: Vec<String> = Vec::new();
    let mut duplicate_dropped: Vec<(String, String)> = Vec::new();

    for descriptor in descriptors.into_iter() {
        if descriptor.category != FeatureCategory::Boolean {
            kept.push(descriptor);
            continue;
        }

        let name = descriptor.name.clone();
        let series = match df.column(&name) {
            Ok(col) => col,
            Err(_) => {
                kept.push(descriptor);
                continue;
            }
        };

        let mask = match boolean_mask_from_series(series) {
            Some(mask) => mask,
            None => {
                kept.push(descriptor);
                continue;
            }
        };

        if mask.is_empty() {
            kept.push(descriptor);
            continue;
        }

        let all_true = mask.iter().all(|v| *v);
        let all_false = mask.iter().all(|v| !*v);
        if all_true || all_false {
            constant_dropped.push(name);
            continue;
        }

        if let Some(existing) = mask_index.get(&mask) {
            duplicate_dropped.push((name, existing.clone()));
            continue;
        }

        mask_index.insert(mask, descriptor.name.clone());
        kept.push(descriptor);
    }

    let kept_boolean = kept
        .iter()
        .filter(|d| d.category == FeatureCategory::Boolean)
        .count();
    let any_kept_boolean = kept_boolean > 0;

    if !any_kept_boolean {
        return kept;
    }

    if !constant_dropped.is_empty() {
        constant_dropped.sort();
        info!(
            "⚠️  Dropped {} constant boolean features for this dataset (mask always true/false):",
            constant_dropped.len()
        );
        for name in &constant_dropped {
            info!("   - {}", name);
        }
    }

    if !duplicate_dropped.is_empty() {
        info!(
            "⚠️  Dropped {} duplicate boolean features with identical masks:",
            duplicate_dropped.len()
        );
        for (dup, canonical) in &duplicate_dropped {
            info!("   - {} (duplicate of {})", dup, canonical);
        }
    }

    kept
}

fn audit_boolean_coverage(df: &DataFrame) {
    let known: HashSet<&str> = BOOLEAN_FEATURES.iter().copied().collect();
    let mut unexpected = Vec::new();

    for series in df.get_columns() {
        let name = series.name();
        if known.contains(name) {
            continue;
        }
        match series.dtype() {
            DataType::Boolean => unexpected.push(name.to_string()),
            DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64 => {
                // Heuristic: treat integer columns with only 0/1 as boolean flags.
                if let Ok(values) = series.i64() {
                    let mut has_zero = false;
                    let mut has_one = false;
                    let mut other = false;
                    for v in values.into_iter().flatten() {
                        match v {
                            0 => has_zero = true,
                            1 => has_one = true,
                            _ => {
                                other = true;
                                break;
                            }
                        }
                    }
                    if !other && (has_zero || has_one) {
                        unexpected.push(name.to_string());
                    }
                }
            }
            _ => {}
        }
    }

    if !unexpected.is_empty() {
        unexpected.sort();
        info!(
            "⚠️  Found {} boolean features (0/1 values) in dataframe but not in BOOLEAN_FEATURES:",
            unexpected.len()
        );
        for name in &unexpected {
            info!("   - {}", name);
        }
        info!(
            "💡 Consider adding these to BOOLEAN_FEATURES in custom_rs::features if you want them in the catalog"
        );
    }
}

fn audit_continuous_coverage(df: &DataFrame) {
    let known_ranges: HashSet<&str> = CONTINUOUS_FEATURES.iter().copied().collect();
    let mut unexpected = Vec::new();

    for series in df.get_columns() {
        let name = series.name();
        if known_ranges.contains(name) {
            continue;
        }
        match series.dtype() {
            DataType::Float32 | DataType::Float64 => {
                let values = series.f64().ok();
                if values.is_none() {
                    continue;
                }
                let values = values.unwrap();
                let mut finite: Vec<f64> = values
                    .into_iter()
                    .flatten()
                    .filter(|v| v.is_finite())
                    .collect();
                if finite.len() < 2 {
                    continue;
                }
                finite.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let n = finite.len();
                let p1 = finite[(0.01 * (n - 1) as f64) as usize];
                let p99 = finite[(0.99 * (n - 1) as f64) as usize];
                let data_min = finite[0];
                let data_max = finite[n - 1];
                unexpected.push((name.to_string(), p1, p99, data_min, data_max));
            }
            _ => {}
        }
    }

    if !unexpected.is_empty() {
        info!(
            "⚠️  Found {} continuous features in dataframe but not in CONTINUOUS_FEATURES/feature_ranges.json:",
            unexpected.len()
        );
        for (name, p1, p99, data_min, data_max) in &unexpected {
            info!(
                "   - {:<34} P1-P99: [{:10.4}, {:10.4}]  Data: [{:10.4}, {:10.4}]",
                name, p1, p99, data_min, data_max
            );
        }
        info!(
            "💡 Consider adding these to feature_ranges.json if you want to include them in threshold testing"
        );
    }
}

fn audit_continuous_definitions(df: &DataFrame) {
    let mut missing = Vec::new();
    let mut non_numeric_or_binary = Vec::new();

    for &name in CONTINUOUS_FEATURES {
        match df.column(name) {
            Ok(series) => match series.dtype() {
                DataType::Float32 | DataType::Float64 => {
                    // Sanity check: continuous features should not be effectively
                    // boolean (only 0/1 or true/false). If they are, treat them
                    // as mis-specified for the continuous catalog.
                    if is_binary_01_series(series) {
                        non_numeric_or_binary.push((name, series.dtype().clone()));
                    }
                }
                _ => non_numeric_or_binary.push((name, series.dtype().clone())),
            },
            Err(_) => missing.push(name),
        }
    }

    if !missing.is_empty() {
        info!(
            "⚠️  CONTINUOUS_FEATURES entries missing from engineered dataset (not present as columns):"
        );
        for name in &missing {
            info!("   - {}", name);
        }
        info!("💡 Consider adding these to engineer.rs or removing them from CONTINUOUS_FEATURES");
    }

    if !non_numeric_or_binary.is_empty() {
        info!(
            "⚠️  CONTINUOUS_FEATURES entries are not suitable continuous numerics in the engineered dataset (non-float or effectively boolean 0/1):"
        );
        for (name, dtype) in &non_numeric_or_binary {
            info!("   - {} (dtype={:?})", name, dtype);
        }
        info!(
            "💡 Ensure these columns are exported as float32/float64 with rich value ranges (not just 0/1) if you want them in the continuous catalog"
        );
    }
}

fn audit_boolean_definitions(df: &DataFrame) {
    let mut missing = Vec::new();
    let mut non_boolean = Vec::new();

    for &name in BOOLEAN_FEATURES {
        match df.column(name) {
            Ok(series) => match series.dtype() {
                DataType::Boolean => {}
                DataType::Int8
                | DataType::Int16
                | DataType::Int32
                | DataType::Int64
                | DataType::UInt8
                | DataType::UInt16
                | DataType::UInt32
                | DataType::UInt64 => {
                    // Treat integer columns as valid boolean definitions only if
                    // they contain only 0/1.
                    let mut ok = false;
                    if let Ok(values) = series.i64() {
                        let mut has_zero_or_one = false;
                        let mut other = false;
                        for v in values.into_iter().flatten() {
                            match v {
                                0 | 1 => {
                                    has_zero_or_one = true;
                                }
                                _ => {
                                    other = true;
                                    break;
                                }
                            }
                        }
                        ok = has_zero_or_one && !other;
                    }
                    if !ok {
                        non_boolean.push((name, series.dtype().clone()));
                    }
                }
                _ => {
                    non_boolean.push((name, series.dtype().clone()));
                }
            },
            Err(_) => missing.push(name),
        }
    }

    if !missing.is_empty() {
        info!(
            "⚠️  BOOLEAN_FEATURES entries missing from engineered dataset (not present as columns):"
        );
        for name in &missing {
            info!("   - {}", name);
        }
        info!(
            "💡 Consider adding these flags to engineer.rs or removing them from BOOLEAN_FEATURES"
        );
    }

    if !non_boolean.is_empty() {
        info!("⚠️  BOOLEAN_FEATURES entries are not boolean/0-1 in the engineered dataset:");
        for (name, dtype) in &non_boolean {
            info!("   - {} (dtype={:?})", name, dtype);
        }
        info!(
            "💡 Ensure these columns are exported as booleans or 0/1 integers if you want them in the boolean catalog"
        );
    }
}

/// Heuristic to detect series that are effectively boolean, even if they are
/// stored as numeric types. Returns true when all observed finite values are
/// in {0, 1}.
fn is_binary_01_series(series: &Series) -> bool {
    match series.dtype() {
        DataType::Boolean => true,
        DataType::Int8
        | DataType::Int16
        | DataType::Int32
        | DataType::Int64
        | DataType::UInt8
        | DataType::UInt16
        | DataType::UInt32
        | DataType::UInt64 => {
            if let Ok(values) = series.i64() {
                let mut seen = false;
                for v in values.into_iter().flatten() {
                    if v != 0 && v != 1 {
                        return false;
                    }
                    seen = true;
                }
                seen
            } else {
                false
            }
        }
        DataType::Float32 | DataType::Float64 => {
            if let Ok(values) = series.f64() {
                let mut seen = false;
                for v in values.into_iter().flatten() {
                    if v != 0.0 && v != 1.0 {
                        return false;
                    }
                    seen = true;
                }
                seen
            } else {
                false
            }
        }
        _ => false,
    }
}

const BOOLEAN_NOTE: &str = "See docs §3.4.1-§3.4.5";
const CONTINUOUS_NOTE: &str = "See docs §3.4.6-§3.4.8";

pub const BOOLEAN_FEATURES: &[&str] = &[
    // NOTE: This list controls *combination enumeration* for boolean predicates.
    // - Only names listed here become boolean feature descriptors in the catalog.
    // - Removing a name here removes it from permutation search (and pruning/audits),
    //   but does not remove the underlying column from `barsmith_prepared.csv`.
    // - To stop exporting a column entirely, change `custom_rs::engineer` instead.
    // Candle / sequence / pattern structure
    "is_tribar",
    "is_tribar_green",
    "is_tribar_red",
    "is_tribar_hl",
    "is_tribar_hl_green",
    "is_tribar_hl_red",
    "consecutive_green_2",
    "consecutive_green_3",
    "consecutive_red_2",
    "consecutive_red_3",
    "prev_tribar",
    "prev_green",
    "higher_high",
    "higher_low",
    "lower_high",
    "lower_low",
    "is_hammer",
    "is_shooting_star",
    "bullish_engulfing",
    "bearish_engulfing",
    // ADX / KF-ADX dynamics
    "adx_rising",
    "adx_accelerating",
    "kf_adx_increasing",
    "kf_adx_decreasing",
    "kf_adx_accelerating",
    "kf_adx_decelerating",
    "kf_adx_trend_emerging",
    "kf_adx_trend_fading",
    // Volatility / ATR dynamics
    "is_kf_atr_squeeze",
    "is_kf_atr_c2c_squeeze",
    "is_kf_atr_volatility_spike",
    "is_kf_atr_volatility_drop",
    "is_kf_atr_c2c_spike",
    "is_kf_atr_c2c_drop",
    "kf_atr_c2c_contracting",
    "kf_atr_c2c_expanding",
    "kf_atr_contracting",
    "kf_atr_expanding",
    "kf_volatility_divergence",
    // KF trend / momentum / regimes
    "kf_momentum_increasing",
    "kf_momentum_decreasing",
    "kf_slope_increasing",
    "kf_slope_decreasing",
    "kf_trending_volatile",
    "kf_trending_quiet",
    "kf_ranging_volatile",
    "kf_ranging_quiet",
    "is_kf_strong_trend_low_vol",
    "is_kf_breakout_potential",
    "is_kf_consolidation",
    "kf_trending_c2c_volatile",
    "kf_trending_c2c_quiet",
    "is_kf_gap_trading_opportunity",
    "is_kf_smooth_trend",
    "is_kf_positive_surprise",
    "is_kf_negative_surprise",
    // Oscillator / MACD / Bollinger events
    "is_rsi_oversold_recovery",
    "is_rsi_overbought_recovery",
    "stoch_bullish_cross",
    "stoch_bearish_cross",
    "macd_cross_up",
    "macd_cross_down",
    "is_bb_squeeze",
];

pub const CONTINUOUS_FEATURES: &[&str] = &[
    // NOTE: This list does *not* control scalar-threshold enumeration by itself.
    // Scalar thresholds are driven by `feature_ranges.json` (see `custom_rs::thresholds`).
    // This list is used for:
    // - audits/coverage reporting, and
    // - defining the "core numeric set" used by `drop_rows_with_nan_in_core`.
    "rsi_14",
    "rsi_7",
    "rsi_21",
    "momentum_14",
    "momentum_score",
    "roc_5",
    "roc_10",
    "adx",
    "kf_adx",
    "trend_strength",
    "adx_sma",
    "kf_trend_momentum",
    "kf_trend_volatility_ratio",
    "macd",
    "macd_signal",
    "macd_hist",
    "stoch_k",
    "stoch_d",
    "kf_price_deviation",
    "kf_vs_9ema",
    "kf_vs_200sma",
    "kf_innovation_abs",
    "kf_innovation",
    "kf_adx_deviation",
    "kf_adx_innovation_abs",
    "kf_adx_momentum_5",
    "kf_atr_pct",
    "kf_atr_c2c_pct",
    "kf_atr_vs_c2c",
    "kf_atr_deviation",
    "kf_atr_momentum_5",
    "kf_atr_c2c_momentum_5",
    "kf_atr_innovation",
    "kf_atr_c2c_innovation",
    "atr_pct",
    "atr_c2c_pct",
    "bar_range_pct",
    "volatility_20_cv",
    "body_size_pct",
    "body_vs_max_wick_ratio",
    "body_to_total_wick",
    "body_atr_ratio",
    "consecutive_green",
    "bb_position",
    "wicks_diff",
    "wicks_diff_sma14",
    "kf_wicks_smooth",
    "price_vs_200sma_dev",
    "price_vs_9ema_dev",
    "9ema_to_200sma",
    "upper_shadow_ratio",
    "lower_shadow_ratio",
    "atr_pct_mean50",
    "atr_c2c_mean50",
    "kf_atr",
    "kf_atr_c2c",
    "bb_std",
    "ext",
    "ext_sma14",
    "macd_hist_delta_1",
];

/// Core price/level features that act as default anchors for pairwise
/// numeric comparisons. Most other numeric features are, by default,
/// compared against this set.
pub const PAIRWISE_BASE_NUMERIC_FEATURES: &[&str] = &[
    // NOTE: Pairwise numeric predicates are only generated when `--feature-pairs`
    // is enabled. These lists control *combination enumeration* for feature-vs-feature
    // predicates.
    // - Removing a name here prevents it from participating in pairwise comparisons,
    //   but does not remove the underlying column from `barsmith_prepared.csv`.
    "close",
    "open",
    "high",
    "low",
    "kf_smooth", // Kalman Filter smoothed price
];

/// Additional numeric features that are eligible for feature-to-feature
/// comparisons. By default, these are compared against the base anchors
/// in `PAIRWISE_BASE_NUMERIC_FEATURES` unless a dedicated rule overrides
/// their partner set.
pub const PAIRWISE_EXTRA_NUMERIC_FEATURES: &[&str] = &[
    // See note on `PAIRWISE_BASE_NUMERIC_FEATURES`: this list directly affects
    // which feature-vs-feature comparisons are emitted into the permutation catalog.
    // Core price/MA/volatility structure and closely related price-level
    // numerics that make sense to compare directly against price anchors.
    "9ema",
    "20ema",
    "50ema",
    "200sma",
    "atr_c2c",
    "atr_c2c_mean50",
    "bb_std",
    "kf_atr",
    "kf_atr_c2c",
    "wicks_diff",
    "wicks_diff_sma14",
    "lower_shadow_ratio",
    "upper_shadow_ratio",
    "price_vs_9ema_dev",
    "price_vs_200sma_dev",
    "9ema_to_200sma",
    "kf_price_deviation",
    "kf_vs_9ema",
    "kf_vs_200sma",
    "kf_innovation_abs",
];

/// Per-feature rules for pairwise numeric comparisons. Each rule controls how
/// a given feature participates in feature-to-feature predicates:
///
/// - `use_default`: whether to include `PAIRWISE_BASE_NUMERIC_FEATURES` as
///   candidate partners.
/// - `include`: additional partner features to allow beyond the default set.
/// - `exclude`: partners to remove from the allowed set; the special token
///   `"self"` maps to the rule's `feature` name.
#[derive(Debug)]
pub struct PairwiseRule {
    pub feature: &'static str,
    pub use_default: bool,
    pub include: &'static [&'static str],
    pub exclude: &'static [&'static str],
}

/// Curated overrides for specific numeric features. Features not listed here
/// implicitly behave as if they had:
///
///   { use_default: true, include: &[], exclude: &["self"] }
///
/// which means they compare against the default price anchors but not against
/// themselves.
pub const PAIRWISE_NUMERIC_RULES: &[PairwiseRule] = &[
    PairwiseRule {
        feature: "close",
        use_default: true,
        include: &[],
        exclude: &["self"],
    },
    PairwiseRule {
        feature: "open",
        use_default: true,
        include: &[],
        exclude: &["self"],
    },
    PairwiseRule {
        feature: "high",
        use_default: true,
        include: &[],
        exclude: &["self"],
    },
    PairwiseRule {
        feature: "low",
        use_default: true,
        include: &[],
        exclude: &["self"],
    },
];

/// Resolve the rule for a given feature name, if any.
fn find_pairwise_rule(name: &str) -> Option<&'static PairwiseRule> {
    PAIRWISE_NUMERIC_RULES
        .iter()
        .find(|rule| rule.feature == name)
}

/// Return the full set of feature names that are declared as eligible for
/// pairwise numeric comparisons (used for coverage/audit).
fn pairwise_declared_feature_names() -> Vec<&'static str> {
    let mut names: Vec<&'static str> = Vec::new();

    // Always include the base numeric anchors.
    names.extend_from_slice(PAIRWISE_BASE_NUMERIC_FEATURES);

    // Include all extra numeric features that should participate in
    // feature-to-feature comparisons (typically against the base anchors).
    names.extend_from_slice(PAIRWISE_EXTRA_NUMERIC_FEATURES);

    // Include all rule features and their explicit includes.
    for rule in PAIRWISE_NUMERIC_RULES {
        names.push(rule.feature);
        names.extend_from_slice(rule.include);
    }

    // Ensure uniqueness and stable ordering.
    names.sort_unstable();
    names.dedup();
    names
}

/// Compute the allowed partner set for a given feature, before intersecting
/// with the actually available numeric columns in the dataset.
fn allowed_partners_for(feature: &str) -> Vec<&'static str> {
    let rule = find_pairwise_rule(feature);

    let use_default = rule.map(|r| r.use_default).unwrap_or(true);
    let mut partners: Vec<&'static str> = Vec::new();
    if use_default {
        partners.extend_from_slice(PAIRWISE_BASE_NUMERIC_FEATURES);
    }
    if let Some(r) = rule {
        partners.extend_from_slice(r.include);
    }

    let mut exclude: HashSet<&str> = HashSet::new();
    if let Some(r) = rule {
        for &raw in r.exclude {
            if raw == "self" {
                exclude.insert(feature);
            } else {
                exclude.insert(raw);
            }
        }
    } else {
        // Unlisted features exclude themselves by default.
        exclude.insert(feature);
    }

    partners
        .into_iter()
        .filter(|p| !exclude.contains(p))
        .collect()
}

/// Determine whether a pair (left, right) should be eligible for
/// feature-to-feature predicates under the current configuration. This uses
/// the symmetric rule: a pair is allowed if either side lists the other as an
/// allowed partner.
fn pair_allowed(left: &str, right: &str) -> bool {
    if left == right {
        return false;
    }

    let left_partners = allowed_partners_for(left);
    if left_partners.contains(&right) {
        return true;
    }

    let right_partners = allowed_partners_for(right);
    right_partners.contains(&left)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn df_from_series(columns: Vec<(&str, Series)>) -> DataFrame {
        let series: Vec<Series> = columns.into_iter().map(|(_, s)| s).collect();
        DataFrame::new(series).expect("failed to build test DataFrame")
    }

    fn names(descriptors: &[FeatureDescriptor]) -> Vec<String> {
        descriptors.iter().map(|d| d.name.clone()).collect()
    }

    #[test]
    fn constant_true_boolean_is_pruned() {
        let s_true = Series::new("const_true", &[true, true, true]);
        let s_mixed = Series::new("mixed", &[true, false, true]);
        let df = df_from_series(vec![("const_true", s_true), ("mixed", s_mixed)]);

        let descriptors = vec![
            FeatureDescriptor::boolean("const_true", "test"),
            FeatureDescriptor::boolean("mixed", "test"),
        ];

        let pruned = prune_boolean_constants_and_duplicates(descriptors, &df);
        let names = names(&pruned);
        assert!(
            !names.contains(&"const_true".to_string()),
            "constant-true flag should be pruned"
        );
        assert!(
            names.contains(&"mixed".to_string()),
            "non-constant boolean flag should be kept"
        );
    }

    #[test]
    fn constant_false_boolean_is_pruned() {
        let s_false = Series::new("const_false", &[false, false, false]);
        let s_mixed = Series::new("mixed2", &[false, true, false]);
        let df = df_from_series(vec![("const_false", s_false), ("mixed2", s_mixed)]);

        let descriptors = vec![
            FeatureDescriptor::boolean("const_false", "test"),
            FeatureDescriptor::boolean("mixed2", "test"),
        ];

        let pruned = prune_boolean_constants_and_duplicates(descriptors, &df);
        let names = names(&pruned);
        assert!(
            !names.contains(&"const_false".to_string()),
            "constant-false flag should be pruned"
        );
        assert!(
            names.contains(&"mixed2".to_string()),
            "non-constant boolean flag should be kept"
        );
    }

    #[test]
    fn duplicate_boolean_masks_prune_to_single_canonical() {
        let s_a = Series::new("a", &[true, false, true, false]);
        let s_b = Series::new("b", &[true, false, true, false]);
        let s_c = Series::new("c", &[false, false, true, true]);
        let df = df_from_series(vec![("a", s_a), ("b", s_b), ("c", s_c)]);

        let descriptors = vec![
            FeatureDescriptor::boolean("a", "test"),
            FeatureDescriptor::boolean("b", "test"),
            FeatureDescriptor::boolean("c", "test"),
        ];

        let pruned = prune_boolean_constants_and_duplicates(descriptors, &df);
        let names = names(&pruned);
        assert!(
            names.contains(&"a".to_string()) || names.contains(&"b".to_string()),
            "one of the duplicate flags should be kept"
        );
        assert!(
            !(names.contains(&"a".to_string()) && names.contains(&"b".to_string())),
            "both duplicate flags should not be kept simultaneously"
        );
        assert!(
            names.contains(&"c".to_string()),
            "independent boolean flag should be kept"
        );
    }

    #[test]
    fn integer_binary_series_treated_as_boolean_and_pruned_when_constant() {
        let s_const = Series::new("ib_const", &[0i64, 0, 0, 0]);
        let s_mixed = Series::new("ib_mixed", &[0i64, 1, 0, 1]);
        let df = df_from_series(vec![("ib_const", s_const), ("ib_mixed", s_mixed)]);

        let descriptors = vec![
            FeatureDescriptor::boolean("ib_const", "test"),
            FeatureDescriptor::boolean("ib_mixed", "test"),
        ];

        let pruned = prune_boolean_constants_and_duplicates(descriptors, &df);
        let names = names(&pruned);
        assert!(
            !names.contains(&"ib_const".to_string()),
            "constant 0/1 integer flag should be pruned"
        );
        assert!(
            names.contains(&"ib_mixed".to_string()),
            "non-constant 0/1 integer flag should be kept"
        );
    }

    #[test]
    fn float_binary_series_treated_as_boolean_and_pruned_when_constant() {
        let s_const = Series::new("fb_const", &[1.0f64, 1.0, 1.0]);
        let s_mixed = Series::new("fb_mixed", &[0.0f64, 1.0, 0.0]);
        let df = df_from_series(vec![("fb_const", s_const), ("fb_mixed", s_mixed)]);

        let descriptors = vec![
            FeatureDescriptor::boolean("fb_const", "test"),
            FeatureDescriptor::boolean("fb_mixed", "test"),
        ];

        let pruned = prune_boolean_constants_and_duplicates(descriptors, &df);
        let names = names(&pruned);
        assert!(
            !names.contains(&"fb_const".to_string()),
            "constant 0/1 float flag should be pruned"
        );
        assert!(
            names.contains(&"fb_mixed".to_string()),
            "non-constant 0/1 float flag should be kept"
        );
    }

    #[test]
    fn non_binary_integer_series_not_treated_as_boolean() {
        let s_int = Series::new("int_other", &[0i64, 1, 2, 3]);
        let df = df_from_series(vec![("int_other", s_int)]);

        let descriptors = vec![FeatureDescriptor::boolean("int_other", "test")];
        let pruned = prune_boolean_constants_and_duplicates(descriptors, &df);
        let names = names(&pruned);
        assert!(
            names.contains(&"int_other".to_string()),
            "non-binary integer column should not be treated as boolean for pruning"
        );
    }

    #[test]
    fn non_binary_float_series_not_treated_as_boolean() {
        let s_float = Series::new("float_other", &[0.1f64, 0.0, 1.0, 0.3]);
        let df = df_from_series(vec![("float_other", s_float)]);

        let descriptors = vec![FeatureDescriptor::boolean("float_other", "test")];
        let pruned = prune_boolean_constants_and_duplicates(descriptors, &df);
        let names = names(&pruned);
        assert!(
            names.contains(&"float_other".to_string()),
            "non-binary float column should not be treated as boolean for pruning"
        );
    }

    #[test]
    fn missing_boolean_column_is_left_untouched() {
        let s_other = Series::new("other", &[true, false, true]);
        let df = df_from_series(vec![("other", s_other)]);

        let descriptors = vec![FeatureDescriptor::boolean("missing_flag", "test")];
        let pruned = prune_boolean_constants_and_duplicates(descriptors, &df);
        let names = names(&pruned);
        assert!(
            names.contains(&"missing_flag".to_string()),
            "flags without a corresponding column in the dataframe should be preserved"
        );
    }

    #[test]
    fn empty_dataframe_keeps_boolean_descriptors() {
        let s_empty_bool: Series = Series::new_empty("empty_flag", &DataType::Boolean);
        let df = df_from_series(vec![("empty_flag", s_empty_bool)]);

        let descriptors = vec![FeatureDescriptor::boolean("empty_flag", "test")];
        let pruned = prune_boolean_constants_and_duplicates(descriptors, &df);
        let names = names(&pruned);
        assert!(
            names.contains(&"empty_flag".to_string()),
            "boolean flags over empty datasets should be preserved"
        );
    }

    #[test]
    fn non_boolean_descriptors_are_not_affected_by_pruning() {
        let s_bool = Series::new("flag", &[true, false, true]);
        let df = df_from_series(vec![("flag", s_bool)]);

        let descriptors = vec![
            FeatureDescriptor::boolean("flag", "test"),
            FeatureDescriptor::new("continuous_feature", FeatureCategory::Continuous, "note"),
        ];

        let pruned = prune_boolean_constants_and_duplicates(descriptors, &df);
        let names = names(&pruned);
        assert!(
            names.contains(&"continuous_feature".to_string()),
            "non-boolean descriptors should pass through pruning unchanged"
        );
    }

    #[test]
    fn allowed_partners_apply_default_and_self_exclusion() {
        // Unlisted features should default to comparing against the base
        // numeric anchors while excluding themselves.
        let partners = allowed_partners_for("momentum_score");
        assert!(
            partners.contains(&"close"),
            "default partners should include core price anchors"
        );
        assert!(
            !partners.contains(&"momentum_score"),
            "features should not list themselves as partners by default"
        );
    }

    #[test]
    fn bb_position_and_consecutive_green_do_not_pair_with_each_other() {
        let bb_partners = allowed_partners_for("bb_position");
        assert!(
            bb_partners.contains(&"close"),
            "bb_position should still compare against core price anchors"
        );
        assert!(
            !bb_partners.contains(&"bb_position"),
            "bb_position should not list itself as a partner"
        );
        assert!(
            !bb_partners.contains(&"consecutive_green"),
            "bb_position should not pair directly with consecutive_green"
        );

        let cg_partners = allowed_partners_for("consecutive_green");
        assert!(
            cg_partners.contains(&"close"),
            "consecutive_green should still compare against core price anchors"
        );
        assert!(
            !cg_partners.contains(&"consecutive_green"),
            "consecutive_green should not list itself as a partner"
        );
        assert!(
            !cg_partners.contains(&"bb_position"),
            "consecutive_green should not pair directly with bb_position"
        );
    }

    #[test]
    fn pairwise_declared_feature_names_cover_bases_and_rules() {
        let names = pairwise_declared_feature_names();
        assert!(
            names.contains(&"close"),
            "declared names should include base numeric anchors"
        );
        assert!(
            names.contains(&"9ema"),
            "declared names should include at least one extra numeric feature"
        );
    }
}
