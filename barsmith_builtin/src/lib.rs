use std::collections::HashMap;
use std::fs::{self, File};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};
use barsmith_rs::feature::{ComparisonOperator, ComparisonSpec, FeatureDescriptor};
use barsmith_rs::{Config, PermutationPipeline};
use polars::prelude::*;
use tracing::info;

#[derive(Clone, Copy, Debug, Default)]
pub struct BuiltinPipelineOptions {
    pub ack_new_df: bool,
}

const TARGET_NEXT_BAR_UP: &str = "next_bar_up";
const TARGET_NEXT_BAR_DOWN: &str = "next_bar_down";
// Compatibility alias used by older configs / examples.
const TARGET_NEXT_BAR_COLOR_AND_WICKS: &str = "next_bar_color_and_wicks";

const SUPPORTED_TARGETS: [&str; 3] = [
    TARGET_NEXT_BAR_UP,
    TARGET_NEXT_BAR_DOWN,
    TARGET_NEXT_BAR_COLOR_AND_WICKS,
];

fn normalize_target(target: &str) -> &str {
    match target {
        TARGET_NEXT_BAR_COLOR_AND_WICKS => TARGET_NEXT_BAR_UP,
        other => other,
    }
}

fn ensure_target_supported(target: &str) -> Result<()> {
    let normalized = normalize_target(target);
    if SUPPORTED_TARGETS.contains(&target) || SUPPORTED_TARGETS.contains(&normalized) {
        return Ok(());
    }
    Err(anyhow!(
        "Unsupported target '{target}'. Supported targets: {}",
        SUPPORTED_TARGETS.join(", ")
    ))
}

fn read_base_ohlcv(path: &Path) -> Result<DataFrame> {
    let lazy = LazyCsvReader::new(path)
        .has_header(true)
        .with_try_parse_dates(true)
        .with_ignore_errors(true)
        .finish()
        .with_context(|| format!("Failed to initialize CSV reader for {}", path.display()))?;
    let df = lazy
        .collect()
        .with_context(|| format!("Failed to collect CSV data from {}", path.display()))?;
    Ok(df)
}

fn series_to_f64(series: &Series) -> Result<Vec<f64>> {
    match series.dtype() {
        DataType::Float64 => Ok(series
            .f64()
            .context("Failed to interpret as f64")?
            .into_iter()
            .map(|v| v.unwrap_or(f64::NAN))
            .collect()),
        DataType::Float32 => Ok(series
            .f32()
            .context("Failed to interpret as f32")?
            .into_iter()
            .map(|v| v.map(|x| x as f64).unwrap_or(f64::NAN))
            .collect()),
        DataType::Int64 => Ok(series
            .i64()
            .context("Failed to interpret as i64")?
            .into_iter()
            .map(|v| v.map(|x| x as f64).unwrap_or(f64::NAN))
            .collect()),
        DataType::Int32 => Ok(series
            .i32()
            .context("Failed to interpret as i32")?
            .into_iter()
            .map(|v| v.map(|x| x as f64).unwrap_or(f64::NAN))
            .collect()),
        DataType::UInt64 => Ok(series
            .u64()
            .context("Failed to interpret as u64")?
            .into_iter()
            .map(|v| v.map(|x| x as f64).unwrap_or(f64::NAN))
            .collect()),
        DataType::UInt32 => Ok(series
            .u32()
            .context("Failed to interpret as u32")?
            .into_iter()
            .map(|v| v.map(|x| x as f64).unwrap_or(f64::NAN))
            .collect()),
        other => Err(anyhow!(
            "Unsupported numeric dtype for {}: {other:?}",
            series.name()
        )),
    }
}

fn compute_sma(values: &[f64], window: usize) -> Vec<f64> {
    let mut out = vec![f64::NAN; values.len()];
    if window == 0 || values.is_empty() {
        return out;
    }
    let mut sum = 0.0f64;
    let mut count = 0usize;
    for i in 0..values.len() {
        let v = values[i];
        if v.is_finite() {
            sum += v;
            count += 1;
        }
        if i >= window {
            let drop = values[i - window];
            if drop.is_finite() {
                sum -= drop;
                count = count.saturating_sub(1);
            }
        }
        if i + 1 >= window && count == window {
            out[i] = sum / window as f64;
        }
    }
    out
}

fn quantile(values: &[f64], q: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut clean: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
    if clean.is_empty() {
        return None;
    }
    clean.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let q = q.clamp(0.0, 1.0);
    let idx = ((clean.len() - 1) as f64 * q).round() as usize;
    clean.get(idx).copied()
}

pub fn prepare_dataset_with_options(
    config: &Config,
    options: BuiltinPipelineOptions,
) -> Result<PathBuf> {
    ensure_target_supported(&config.target)?;

    fs::create_dir_all(&config.output_dir)
        .with_context(|| format!("Failed to create {}", config.output_dir.display()))?;
    let out_path = config.output_dir.join("barsmith_prepared.csv");

    if out_path.exists() && !options.ack_new_df {
        return Err(anyhow!(
            "Prepared dataset already exists at {}. Re-run with --ack-new-df to overwrite.",
            out_path.display()
        ));
    }

    let source_path = config.source_csv.as_ref().unwrap_or(&config.input_csv);
    let mut df = read_base_ohlcv(source_path)?;

    for col in ["open", "high", "low", "close", "volume"] {
        if df.column(col).is_err() {
            return Err(anyhow!(
                "Missing required OHLCV column '{col}' in {}",
                source_path.display()
            ));
        }
    }

    let open = series_to_f64(df.column("open")?)?;
    let high = series_to_f64(df.column("high")?)?;
    let low = series_to_f64(df.column("low")?)?;
    let close = series_to_f64(df.column("close")?)?;
    let _volume = series_to_f64(df.column("volume")?)?;

    let n = close.len();
    if n == 0 {
        return Err(anyhow!("Empty dataset: {}", source_path.display()));
    }

    let mut is_green = Vec::with_capacity(n);
    let mut is_red = Vec::with_capacity(n);
    let mut range = Vec::with_capacity(n);
    let mut body = Vec::with_capacity(n);
    let mut abs_body = Vec::with_capacity(n);
    let mut is_inside_bar = Vec::with_capacity(n);
    let mut is_range_expansion = Vec::with_capacity(n);
    for i in 0..n {
        let o = open[i];
        let c = close[i];
        let h = high[i];
        let l = low[i];
        let r = if h.is_finite() && l.is_finite() {
            h - l
        } else {
            f64::NAN
        };
        let b = if c.is_finite() && o.is_finite() {
            c - o
        } else {
            f64::NAN
        };

        is_green.push(c.is_finite() && o.is_finite() && c > o);
        is_red.push(c.is_finite() && o.is_finite() && c < o);
        range.push(r);
        body.push(b);
        abs_body.push(b.abs());

        if i == 0 {
            is_inside_bar.push(false);
            is_range_expansion.push(false);
        } else {
            let prev_h = high[i - 1];
            let prev_l = low[i - 1];
            is_inside_bar.push(
                h.is_finite()
                    && l.is_finite()
                    && prev_h.is_finite()
                    && prev_l.is_finite()
                    && h < prev_h
                    && l > prev_l,
            );

            let prev_r = range[i - 1];
            is_range_expansion.push(r.is_finite() && prev_r.is_finite() && r > prev_r);
        }
    }

    let sma_9 = compute_sma(&close, 9);
    let sma_20 = compute_sma(&close, 20);
    let sma_50 = compute_sma(&close, 50);

    let mut eligible = Vec::with_capacity(n);
    let mut exit_i = Vec::with_capacity(n);
    for i in 0..n {
        if i + 1 < n {
            eligible.push(true);
            exit_i.push((i + 1) as i64);
        } else {
            eligible.push(false);
            exit_i.push(-1);
        }
    }

    let mut next_bar_up = Vec::with_capacity(n);
    let mut next_bar_down = Vec::with_capacity(n);
    let mut rr_next_bar_up = Vec::with_capacity(n);
    let mut rr_next_bar_down = Vec::with_capacity(n);
    for i in 0..n {
        if i + 1 >= n {
            next_bar_up.push(false);
            next_bar_down.push(false);
            rr_next_bar_up.push(f64::NAN);
            rr_next_bar_down.push(f64::NAN);
            continue;
        }

        let o_next = open[i + 1];
        let c_next = close[i + 1];
        let up = o_next.is_finite() && c_next.is_finite() && c_next > o_next;
        let down = o_next.is_finite() && c_next.is_finite() && c_next < o_next;
        next_bar_up.push(up);
        next_bar_down.push(down);
        rr_next_bar_up.push(if up { 1.0 } else { -1.0 });
        rr_next_bar_down.push(if down { 1.0 } else { -1.0 });
    }

    df.with_column(Series::new("is_green", is_green))?;
    df.with_column(Series::new("is_red", is_red))?;
    df.with_column(Series::new("is_inside_bar", is_inside_bar))?;
    df.with_column(Series::new("is_range_expansion", is_range_expansion))?;
    df.with_column(Series::new("range", range))?;
    df.with_column(Series::new("body", body))?;
    df.with_column(Series::new("abs_body", abs_body))?;
    df.with_column(Series::new("sma_9", sma_9))?;
    df.with_column(Series::new("sma_20", sma_20))?;
    df.with_column(Series::new("sma_50", sma_50))?;

    df.with_column(Series::new(TARGET_NEXT_BAR_UP, next_bar_up))?;
    df.with_column(Series::new(TARGET_NEXT_BAR_DOWN, next_bar_down))?;
    // Compatibility alias (same semantics as next_bar_up in the builtin engine).
    df.with_column(Series::new(
        TARGET_NEXT_BAR_COLOR_AND_WICKS,
        df.column(TARGET_NEXT_BAR_UP)?.clone(),
    ))?;

    let eligible_up = format!("{}_eligible", TARGET_NEXT_BAR_UP);
    let eligible_down = format!("{}_eligible", TARGET_NEXT_BAR_DOWN);
    let eligible_alias = format!("{}_eligible", TARGET_NEXT_BAR_COLOR_AND_WICKS);
    df.with_column(Series::new(&eligible_up, eligible.clone()))?;
    df.with_column(Series::new(&eligible_down, eligible.clone()))?;
    df.with_column(Series::new(&eligible_alias, eligible.clone()))?;

    let exit_up = format!("{}_exit_i", TARGET_NEXT_BAR_UP);
    let exit_down = format!("{}_exit_i", TARGET_NEXT_BAR_DOWN);
    let exit_alias = format!("{}_exit_i", TARGET_NEXT_BAR_COLOR_AND_WICKS);
    df.with_column(Series::new(&exit_up, exit_i.clone()))?;
    df.with_column(Series::new(&exit_down, exit_i.clone()))?;
    df.with_column(Series::new(&exit_alias, exit_i.clone()))?;

    let rr_up = format!("rr_{}", TARGET_NEXT_BAR_UP);
    let rr_down = format!("rr_{}", TARGET_NEXT_BAR_DOWN);
    let rr_alias = format!("rr_{}", TARGET_NEXT_BAR_COLOR_AND_WICKS);
    df.with_column(Series::new(&rr_up, rr_next_bar_up))?;
    df.with_column(Series::new(&rr_down, rr_next_bar_down))?;
    df.with_column(Series::new(&rr_alias, df.column(&rr_up)?.clone()))?;

    let file = File::create(&out_path)
        .with_context(|| format!("Failed to create {}", out_path.display()))?;
    CsvWriter::new(file)
        .include_header(true)
        .finish(&mut df)
        .with_context(|| format!("Failed to write {}", out_path.display()))?;

    info!(
        rows = df.height(),
        path = %out_path.display(),
        "Prepared engineered dataset written for combination run"
    );

    Ok(out_path)
}

pub struct FeatureCatalog {
    pub descriptors: Vec<FeatureDescriptor>,
    pub comparison_specs: HashMap<String, ComparisonSpec>,
}

impl FeatureCatalog {
    pub fn build_with_dataset(prepared_csv: &Path, config: &Config) -> Result<Self> {
        ensure_target_supported(&config.target)?;

        // Read only once (cheap for the small builtin catalog).
        let df = barsmith_rs::data::ColumnarData::load(prepared_csv)
            .with_context(|| format!("Failed to load prepared CSV {}", prepared_csv.display()))?
            .data_frame();
        let frame = df.as_ref();

        let mut descriptors: Vec<FeatureDescriptor> = Vec::new();
        let mut specs: HashMap<String, ComparisonSpec> = HashMap::new();

        // Built-in booleans.
        let bools = [
            ("is_green", "close > open"),
            ("is_red", "close < open"),
            ("is_inside_bar", "high < prev_high && low > prev_low"),
            ("is_range_expansion", "range > prev_range"),
        ];
        for (name, note) in bools {
            if frame.column(name).is_ok() {
                descriptors.push(FeatureDescriptor::boolean(name, note));
            }
        }

        // Built-in scalar thresholds on basic numeric columns.
        for base in ["volume", "range", "abs_body"] {
            let series = frame.column(base).with_context(|| {
                format!("Missing base numeric column '{base}' in prepared dataset")
            })?;
            let values = series_to_f64(series)?;
            let p50 = quantile(&values, 0.50);
            let p75 = quantile(&values, 0.75);

            for (q, label) in [(p50, "p50"), (p75, "p75")] {
                let Some(threshold) = q else {
                    continue;
                };
                if !threshold.is_finite() {
                    continue;
                }
                let name = format!("{base}>{label}");
                let note = format!("{base} greater than {label}");
                descriptors.push(FeatureDescriptor::feature_vs_constant(name.clone(), note));
                specs.insert(
                    name,
                    ComparisonSpec::threshold(base, ComparisonOperator::GreaterThan, threshold),
                );
            }
        }

        // Optional feature pairs: compare a few common columns.
        if config.enable_feature_pairs {
            let candidates: Vec<&str> = ["close", "sma_9", "sma_20", "sma_50"]
                .into_iter()
                .filter(|name| frame.column(name).is_ok())
                .collect();

            let (pair_desc, pair_specs) =
                barsmith_rs::feature::generate_unordered_feature_comparisons(
                    &candidates,
                    &[ComparisonOperator::GreaterThan],
                    config.feature_pairs_limit,
                    "builtin feature pair",
                );
            descriptors.extend(pair_desc);
            specs.extend(pair_specs);
        }

        Ok(Self {
            descriptors,
            comparison_specs: specs,
        })
    }
}

pub fn run_builtin_pipeline_with_options(
    config: Config,
    options: BuiltinPipelineOptions,
) -> Result<()> {
    let mut config = config;
    let prepared_csv = prepare_dataset_with_options(&config, options)?;
    config.input_csv = prepared_csv.clone();

    let catalog = FeatureCatalog::build_with_dataset(&config.input_csv, &config)?;

    let pipeline_config = config;
    let descriptors = catalog.descriptors;
    let comparison_specs = catalog.comparison_specs;

    let builder = std::thread::Builder::new()
        .name("barsmith-pipeline".to_string())
        .stack_size(32 * 1024 * 1024);

    let handle = builder
        .spawn(move || -> Result<()> {
            let mut pipeline =
                PermutationPipeline::new(pipeline_config, descriptors, comparison_specs);
            pipeline.run()
        })
        .map_err(|err| anyhow!("failed to spawn barsmith pipeline thread: {err}"))?;

    handle
        .join()
        .map_err(|_| anyhow!("barsmith pipeline thread panicked"))?
}
