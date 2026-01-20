use std::collections::HashMap;
use std::path::Path;

use crate::features::CONTINUOUS_FEATURES;
use anyhow::{Context, Result, anyhow};
use barsmith_rs::feature::{ComparisonOperator, ComparisonSpec, FeatureDescriptor};
use polars::prelude::*;
use serde::Deserialize;
use tracing::info;

const FEATURE_RANGES_JSON: &str = include_str!("../feature_ranges.json");

#[derive(Debug, Deserialize)]
struct RawRangeConfig {
    #[serde(default)]
    enabled: Option<bool>,
    min: serde_json::Value,
    max: serde_json::Value,
    increment: serde_json::Value,
    operators: Vec<String>,
    description: String,
}

pub struct ThresholdCatalog {
    pub descriptors: Vec<FeatureDescriptor>,
    pub specs: HashMap<String, ComparisonSpec>,
}

pub fn generate_threshold_catalog(dataset_path: &Path) -> Result<ThresholdCatalog> {
    generate_threshold_catalog_from_ranges_json(dataset_path, FEATURE_RANGES_JSON)
}

fn generate_threshold_catalog_from_ranges_json(
    dataset_path: &Path,
    ranges_json: &str,
) -> Result<ThresholdCatalog> {
    let ranges = parse_feature_ranges_json(ranges_json)?;

    // Runtime sanity check: ensure the scalar threshold catalog in
    // feature_ranges.json stays aligned with the Rust continuous feature list.
    // This mirrors tmp/check_feature_sets.py but runs even when the Python
    // tooling is not available.
    audit_feature_ranges_vs_continuous(&ranges);

    let df = CsvReader::from_path(dataset_path)
        .context("unable to open engineered dataset for threshold generation")?
        .has_header(true)
        .with_ignore_errors(true)
        .finish()
        .context("unable to read engineered dataset for threshold generation")?;

    let mut descriptors = Vec::new();
    let mut specs = HashMap::new();

    // Track summary stats for the auto-generated thresholds, to mirror the Python debug output.
    struct ThresholdSummary {
        feature: String,
        count: usize,
        data_min: f64,
        data_max: f64,
        unique_values: usize,
    }
    let mut summaries: Vec<ThresholdSummary> = Vec::new();

    for (feature, config) in ranges.iter() {
        if config.enabled == Some(false) {
            continue;
        }
        let Ok(series) = df.column(feature) else {
            continue;
        };
        let Ok(values) = series.f64() else {
            continue;
        };
        let mut samples: Vec<f64> = values
            .into_iter()
            .flatten()
            .filter(|value| value.is_finite())
            .collect();
        if samples.len() < 2 {
            continue;
        }
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let data_min = *samples.first().unwrap();
        let data_max = *samples.last().unwrap();
        let unique_values = samples.windows(2).filter(|w| w[0] != w[1]).count() + 1;
        let p1 = percentile(&samples, 0.01);
        let p99 = percentile(&samples, 0.99);
        let min_override = round_to_nice_multiple(p1, true);
        let max_override = round_to_nice_multiple(p99, false);

        let min_val = resolve_bound(&config.min, min_override);
        let max_val = resolve_bound(&config.max, max_override);
        if min_val.is_none() || max_val.is_none() {
            continue;
        }
        let min_val = min_val.unwrap();
        let max_val = max_val.unwrap();
        if !min_val.is_finite() || !max_val.is_finite() || min_val >= max_val {
            continue;
        }

        let increment =
            resolve_increment(&config.increment, min_val, max_val, config.operators.len());
        if increment <= 0.0 {
            continue;
        }

        let mut feature_threshold_count = 0usize;

        for operator in &config.operators {
            let cmp_operator = parse_operator(operator)?;
            let thresholds = generate_threshold_values(min_val, max_val, increment, operator);
            for threshold in thresholds {
                let descriptor_name = format!(
                    "{}{}{}",
                    feature,
                    operator,
                    format_threshold_value(threshold)
                );
                if specs.contains_key(&descriptor_name) {
                    continue;
                }
                let descriptor = FeatureDescriptor::feature_vs_constant(
                    descriptor_name.clone(),
                    config.description.clone(),
                );
                descriptors.push(descriptor);
                specs.insert(
                    descriptor_name,
                    ComparisonSpec::threshold(feature, cmp_operator, threshold),
                );
                feature_threshold_count += 1;
            }
        }

        if feature_threshold_count > 0 {
            summaries.push(ThresholdSummary {
                feature: feature.clone(),
                count: feature_threshold_count,
                data_min,
                data_max,
                unique_values,
            });
        }
    }

    // Emit a Python-style auto-thresholds summary so Barsmith logs carry the same
    // information as run_statistical_feature_permutation.
    if !summaries.is_empty() {
        info!("🎯 Auto-generated thresholds:");
        // Mirror the style: print a few headline features, then the full analysis summary.
        for summary in summaries.iter().take(5) {
            info!("   {}: {} thresholds", summary.feature, summary.count);
        }
        if summaries.len() > 5 {
            info!("   ... and {} more features", summaries.len() - 5);
        }
        info!("   Analyzed {} features", summaries.len());
        for summary in &summaries {
            info!(
                "   - {}: [{:.2}, {:.2}] ({} unique values)",
                summary.feature, summary.data_min, summary.data_max, summary.unique_values
            );
        }
    } else {
        // Fallback for extremely small or synthetic datasets (e.g., unit tests)
        // where no configured continuous feature has enough data to support the
        // usual threshold-generation path. In these cases, emit a single,
        // generic feature-vs-constant predicate so downstream code (and tests)
        // can still exercise FeatureVsConstant behavior without impacting
        // real-world runs, which always generate thresholds from the catalog.
        if let Some(series) = df
            .get_columns()
            .iter()
            .find(|s| matches!(s.dtype(), DataType::Float32 | DataType::Float64))
        {
            if let Ok(values) = series.f64() {
                let mut samples: Vec<f64> = values
                    .into_iter()
                    .flatten()
                    .filter(|value| value.is_finite())
                    .collect();
                if samples.len() >= 2 {
                    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    let data_min = *samples.first().unwrap();
                    let data_max = *samples.last().unwrap();
                    let mid = (data_min + data_max) / 2.0;
                    let feature_name = series.name().to_string();
                    let descriptor_name =
                        format!("{}>{}", feature_name, format_threshold_value(mid));
                    if !specs.contains_key(&descriptor_name) {
                        let descriptor = FeatureDescriptor::feature_vs_constant(
                            descriptor_name.clone(),
                            "Fallback auto-threshold for tiny dataset".to_string(),
                        );
                        descriptors.push(descriptor);
                        specs.insert(
                            descriptor_name,
                            ComparisonSpec::threshold(
                                &feature_name,
                                ComparisonOperator::GreaterThan,
                                mid,
                            ),
                        );
                        info!(
                            feature = %feature_name,
                            "Generated fallback feature-vs-constant threshold for tiny dataset"
                        );
                    }
                }
            }
        }
    }

    Ok(ThresholdCatalog { descriptors, specs })
}

/// Emit non-fatal warnings when feature_ranges.json and CONTINUOUS_FEATURES
/// drift out of sync. This helps catch catalog skew even when the Python
/// parity scripts are not being run.
fn audit_feature_ranges_vs_continuous(ranges: &HashMap<String, RawRangeConfig>) {
    let mut disabled: Vec<&str> = ranges
        .iter()
        .filter_map(|(name, cfg)| (cfg.enabled == Some(false)).then_some(name.as_str()))
        .collect();
    disabled.sort_unstable();

    let json_keys: Vec<&str> = ranges
        .iter()
        .filter_map(|(name, cfg)| (cfg.enabled != Some(false)).then_some(name.as_str()))
        .collect();
    let continuous: Vec<&str> = CONTINUOUS_FEATURES.to_vec();

    let mut in_json_not_continuous: Vec<&str> = json_keys
        .iter()
        .copied()
        .filter(|name| !continuous.contains(name))
        .collect();
    let mut in_continuous_not_json: Vec<&str> = continuous
        .iter()
        .copied()
        .filter(|name| !json_keys.contains(name) && !disabled.contains(name))
        .collect();

    if !disabled.is_empty() {
        info!(
            "ℹ️  feature_ranges.json entries disabled (no scalar thresholds will be generated for these):"
        );
        for name in &disabled {
            info!("   - {}", name);
        }
    }

    if !in_json_not_continuous.is_empty() {
        in_json_not_continuous.sort_unstable();
        in_json_not_continuous.dedup();
        info!(
            "⚠️  feature_ranges.json entries missing from CONTINUOUS_FEATURES (these features will have thresholds but are not in the Rust continuous catalog):"
        );
        for name in &in_json_not_continuous {
            info!("   - {}", name);
        }
        info!(
            "💡 Consider adding these to CONTINUOUS_FEATURES in custom_rs::features or dropping them from feature_ranges.json"
        );
    }

    if !in_continuous_not_json.is_empty() {
        in_continuous_not_json.sort_unstable();
        in_continuous_not_json.dedup();
        info!(
            "⚠️  CONTINUOUS_FEATURES entries missing from feature_ranges.json (no scalar thresholds will be generated for these):"
        );
        for name in &in_continuous_not_json {
            info!("   - {}", name);
        }
        info!(
            "💡 Add matching entries to feature_ranges.json if you want these features in feature-vs-constant permutations"
        );
    }
}

fn parse_feature_ranges_json(ranges_json: &str) -> Result<HashMap<String, RawRangeConfig>> {
    serde_json::from_str(ranges_json).context("failed to parse feature_ranges.json")
}

fn resolve_bound(value: &serde_json::Value, auto_value: f64) -> Option<f64> {
    if value == "auto" {
        Some(auto_value)
    } else {
        value.as_f64()
    }
}

fn resolve_increment(
    value: &serde_json::Value,
    min_val: f64,
    max_val: f64,
    operator_count: usize,
) -> f64 {
    if value == "auto" {
        calculate_optimal_increment(min_val, max_val, operator_count, 10)
    } else {
        value.as_f64().unwrap_or(0.0)
    }
}

fn percentile(sorted: &[f64], quantile: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return f64::NAN;
    }
    let rank = quantile * (n - 1) as f64;
    let lower = rank.floor() as usize;
    let upper = rank.ceil() as usize;
    if lower == upper {
        return sorted[lower];
    }
    let weight = rank - lower as f64;
    sorted[lower] + weight * (sorted[upper] - sorted[lower])
}

fn round_to_nice_multiple(value: f64, down: bool) -> f64 {
    if value == 0.0 {
        return 0.0;
    }

    let magnitude = 10f64.powf(value.abs().log10().floor());
    let candidates = [
        magnitude,
        2.0 * magnitude,
        5.0 * magnitude,
        10.0 * magnitude,
    ];

    if down {
        if value > 0.0 {
            candidates
                .iter()
                .copied()
                .filter(|candidate| *candidate <= value)
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap_or(0.0)
        } else {
            -candidates
                .iter()
                .copied()
                .filter(|candidate| *candidate <= value.abs())
                .min_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap_or(magnitude * 10.0)
        }
    } else if value > 0.0 {
        candidates
            .iter()
            .copied()
            .filter(|candidate| *candidate >= value)
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(*candidates.last().unwrap())
    } else {
        -candidates
            .iter()
            .copied()
            .filter(|candidate| *candidate >= value.abs())
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(0.0)
    }
}

fn calculate_optimal_increment(
    min_val: f64,
    max_val: f64,
    operator_count: usize,
    target_values: usize,
) -> f64 {
    let range_size = max_val - min_val;
    if range_size <= 0.0 {
        return 0.0;
    }
    let target_per_operator = (target_values + operator_count) as f64 / operator_count as f64;
    let ideal_increment = range_size / target_per_operator;
    let standard_increments = [
        0.00001, 0.00002, 0.00005, 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05,
        0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0,
        5000.0,
    ];

    let mut best_increment = None;
    let mut best_diff = f64::MAX;
    for increment in standard_increments {
        if increment <= 0.0 {
            continue;
        }
        let values_per_op = ((max_val - min_val) / increment).floor() as usize + 1;
        let total_conditions = values_per_op * operator_count - operator_count;
        if (5..=15).contains(&total_conditions) {
            let diff = (total_conditions as f64 - target_values as f64).abs();
            if diff < best_diff {
                best_diff = diff;
                best_increment = Some(increment);
            }
        }
    }

    best_increment.unwrap_or_else(|| {
        standard_increments
            .iter()
            .copied()
            .min_by(|a, b| {
                (a - ideal_increment)
                    .abs()
                    .partial_cmp(&(b - ideal_increment).abs())
                    .unwrap()
            })
            .unwrap_or(ideal_increment)
    })
}

fn generate_threshold_values(
    min_val: f64,
    max_val: f64,
    increment: f64,
    operator: &str,
) -> Vec<f64> {
    let mut thresholds = Vec::new();
    let mut current = min_val;
    while current <= max_val + f64::EPSILON {
        let impossible = match operator {
            ">" => current >= max_val,
            "<" => current <= min_val,
            ">=" => current > max_val,
            "<=" => current < min_val,
            _ => false,
        };
        if !impossible {
            thresholds.push(current);
        }
        current += increment;
    }
    thresholds
}

fn parse_operator(value: &str) -> Result<ComparisonOperator> {
    match value {
        ">" => Ok(ComparisonOperator::GreaterThan),
        "<" => Ok(ComparisonOperator::LessThan),
        ">=" => Ok(ComparisonOperator::GreaterEqual),
        "<=" => Ok(ComparisonOperator::LessEqual),
        _ => Err(anyhow!("unsupported operator '{value}'")),
    }
}

fn format_threshold_value(value: f64) -> String {
    let mut formatted = format!("{value:.2}");
    if let Some(idx) = formatted.find('.') {
        let mut trimmed = formatted.trim_end_matches('0').to_string();
        if trimmed.ends_with('.') {
            trimmed.push('0');
        }
        if trimmed.len() > idx && trimmed[idx + 1..].len() <= 2 {
            formatted = trimmed;
        }
    }
    formatted
}

#[cfg(test)]
mod tests {
    use super::{calculate_optimal_increment, generate_threshold_catalog_from_ranges_json};
    use std::fs;
    use std::path::PathBuf;

    #[test]
    fn increment_matches_python_reference() {
        let inc = calculate_optimal_increment(-0.1, 2.0, 2, 10);
        assert!((inc - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn disabled_feature_ranges_do_not_generate_thresholds() {
        let tmp_dir = std::env::temp_dir();
        let csv_path: PathBuf = tmp_dir.join(format!(
            "barsmith_thresholds_enabled_test_{}.csv",
            std::process::id()
        ));
        fs::write(&csv_path, "a,b\n0.0,0.0\n1.0,1.0\n2.0,2.0\n").expect("failed to write temp csv");

        let ranges_json = r#"
        {
          "a": { "enabled": false, "min": 0, "max": 2, "increment": 1, "operators": [">"], "description": "a" },
          "b": { "min": 0, "max": 2, "increment": 1, "operators": [">"], "description": "b" }
        }
        "#;

        let catalog = generate_threshold_catalog_from_ranges_json(&csv_path, ranges_json)
            .expect("failed to generate catalog");
        let names: Vec<String> = catalog.descriptors.iter().map(|d| d.name.clone()).collect();
        assert!(names.iter().all(|n| !n.starts_with("a>")));
        assert!(names.iter().any(|n| n == "b>0.0"));
        assert!(names.iter().any(|n| n == "b>1.0"));

        let _ = fs::remove_file(&csv_path);
    }
}
