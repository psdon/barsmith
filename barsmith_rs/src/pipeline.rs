use std::collections::{HashMap, HashSet};
use std::fmt::Write;
use std::path::PathBuf;
use std::sync::{Arc, mpsc};
use std::thread;
use std::time::Instant;

use ahash::AHashSet;
use anyhow::{Context, Result, anyhow};
use rayon::ThreadPoolBuilder;
use rayon::prelude::*;
use tracing::{info, warn};

use std::fs;

use crate::combinator::{FeaturePools, IndexCombinationBatcher, total_combinations};
use crate::config::{Config, EvalProfileMode, PositionSizingMode, StatsDetail};
use crate::data::ColumnarData;
use crate::feature::{ComparisonOperator, ComparisonSpec, FeatureCategory, FeatureDescriptor};
use crate::mask::MaskCache;
use crate::progress::ProgressTracker;
use crate::s3::S3Destination;
use crate::stats::{self, EvaluationContext, StatSummary};
use crate::storage::{CumulativeStore, ResultRow};
use sha2::{Digest, Sha256};

fn required_columns_for_catalog(
    config: &Config,
    data: &ColumnarData,
    feature_pools: &FeaturePools,
    comparison_specs: &HashMap<String, ComparisonSpec>,
    reward_col: Option<&str>,
) -> Vec<String> {
    let mut keep_columns: Vec<String> = Vec::new();
    // Always keep the target column.
    keep_columns.push(config.target.clone());
    // Keep the detected reward column, if any.
    if let Some(col) = reward_col {
        keep_columns.push(col.to_string());
    }

    // Preserve eligibility masks when present so trade denominators can't drift
    // if target-specific engineering semantics change.
    let eligible = format!("{}_eligible", config.target);
    if data.has_column(&eligible) {
        keep_columns.push(eligible);
    }
    let eligible_long = format!("{}_eligible_long", config.target);
    if data.has_column(&eligible_long) {
        keep_columns.push(eligible_long);
    }
    let eligible_short = format!("{}_eligible_short", config.target);
    if data.has_column(&eligible_short) {
        keep_columns.push(eligible_short);
    }

    // Preserve target-provided exit indices so `--stacking-mode no-stacking`
    // can enforce non-overlapping trades.
    let exit_i = format!("{}_exit_i", config.target);
    if data.has_column(&exit_i) {
        keep_columns.push(exit_i);
    }
    let exit_i_long = format!("{}_exit_i_long", config.target);
    if data.has_column(&exit_i_long) {
        keep_columns.push(exit_i_long);
    }
    let exit_i_short = format!("{}_exit_i_short", config.target);
    if data.has_column(&exit_i_short) {
        keep_columns.push(exit_i_short);
    }

    if matches!(config.position_sizing, PositionSizingMode::Contracts) {
        if let Some(col) = config.stop_distance_column.as_deref() {
            if data.has_column(col) {
                keep_columns.push(col.to_string());
            }
        }
    }

    // Keep boolean feature columns and any numeric bases/RHS used by
    // comparison predicates in the current catalog.
    for desc in feature_pools.descriptors() {
        match desc.category {
            FeatureCategory::Boolean => keep_columns.push(desc.name.clone()),
            FeatureCategory::FeatureVsConstant | FeatureCategory::FeatureVsFeature => {
                if let Some(spec) = comparison_specs.get(&desc.name) {
                    keep_columns.push(spec.base_feature.clone());
                    if let Some(rhs) = spec.rhs_feature.as_ref() {
                        keep_columns.push(rhs.clone());
                    }
                }
            }
            FeatureCategory::Continuous => {}
        }
    }

    // Deduplicate while preserving first-seen order.
    let mut seen_cols = HashSet::new();
    keep_columns.retain(|name| seen_cols.insert(name.clone()));
    keep_columns
}

/// Per-batch timing snapshot used by the BatchTuner.
/// This mirrors the progress log timing fields but stays pure so it can be
/// unit-tested easily without depending on logging.
#[derive(Clone, Debug)]
struct BatchTimingSnapshot {
    enumeration_ms: u64,
    filter_ms: u64,
    eval_ms: u64,
    ingest_ms: u64,
    prune_subset_ms: u64,
    prune_struct_ms: u64,
}

/// Simple heuristic batch-size tuner. It never changes which combinations
/// are evaluated, only how many are requested per producer batch.
#[derive(Clone, Debug)]
struct BatchTuner {
    min_batch: usize,
    max_batch: usize,
    shrink_factor: f32,
    grow_factor: f32,
    target_total_ms: f32,
    history_len: usize,
}

impl BatchTuner {
    fn new(initial_batch: usize) -> Self {
        // Keep a sensible global upper bound but treat the configured
        // batch size as the minimum floor. Auto-batch tuning may grow the
        // batch size above this value when it is safe to do so, but it
        // will never shrink below the user-specified batch_size.
        // Allow growth above the initial batch size so that --auto-batch
        // can aggressively increase batch_size when total_ms is far below
        // the target.
        let hard_max = initial_batch.saturating_mul(4).max(200_000usize);
        let min_batch = initial_batch.max(1);
        let max_batch = hard_max.max(initial_batch.max(1));
        Self {
            min_batch,
            max_batch,
            shrink_factor: 0.5,
            grow_factor: 2.0,
            target_total_ms: 15_000.0,
            history_len: 5,
        }
    }

    fn recommend(&self, current_batch: usize, snapshots: &[BatchTimingSnapshot]) -> usize {
        if snapshots.is_empty() {
            return current_batch.max(1);
        }

        let len = snapshots.len();
        let start = len.saturating_sub(self.history_len);
        let window = &snapshots[start..];

        let mut sum_enum = 0u64;
        let mut sum_filter = 0u64;
        let mut sum_eval = 0u64;
        let mut sum_ingest = 0u64;
        let mut _sum_subset = 0u64;
        let mut _sum_struct = 0u64;
        for snap in window {
            sum_enum += snap.enumeration_ms;
            sum_filter += snap.filter_ms;
            sum_eval += snap.eval_ms;
            sum_ingest += snap.ingest_ms;
            _sum_subset += snap.prune_subset_ms;
            _sum_struct += snap.prune_struct_ms;
        }
        let count = window.len() as u64;
        if count == 0 {
            return current_batch.max(1);
        }

        let mean_total_ms: f32 =
            (sum_enum + sum_filter + sum_eval + sum_ingest) as f32 / count as f32;

        let mut proposed = current_batch.max(1);

        // Use mean total_ms to steer batch size toward target_total_ms.
        let hi = self.target_total_ms * 2.0;
        let lo = self.target_total_ms * 0.5;

        if mean_total_ms > hi {
            // Too slow: shrink the batch.
            let shrunk = (proposed as f32 * self.shrink_factor).round() as usize;
            if shrunk < proposed {
                proposed = shrunk.max(1);
            }
        } else if mean_total_ms < lo {
            // Too fast: grow the batch.
            let grown = (proposed as f32 * self.grow_factor).round() as usize;
            if grown > proposed {
                proposed = grown;
            }
        }

        // Clamp to configured bounds.
        if proposed < self.min_batch {
            proposed = self.min_batch;
        }
        if proposed > self.max_batch {
            proposed = self.max_batch;
        }
        proposed
    }
}

fn infer_years_from_dataset(data: &ColumnarData) -> Option<f64> {
    let df = data.data_frame();
    let frame = df.as_ref();

    // Prefer a column literally named "timestamp" when present.
    let mut candidate = if data.has_column("timestamp") {
        frame.column("timestamp").ok()
    } else {
        None
    };

    // Fallback: first datetime-typed column in the frame.
    if candidate.is_none() {
        for series in frame.get_columns() {
            if matches!(series.dtype(), polars::prelude::DataType::Datetime(_, _)) {
                candidate = Some(series);
                break;
            }
        }
    }

    let series = candidate?;
    let (unit, _) = match series.dtype() {
        polars::prelude::DataType::Datetime(unit, tz) => (*unit, tz.clone()),
        _ => return None,
    };

    let ca = series.datetime().ok()?;
    if ca.is_empty() {
        return None;
    }

    let first = ca.get(0)?;
    let last = ca.get(ca.len().saturating_sub(1))?;
    let delta_raw = (last - first).abs() as f64;
    if !delta_raw.is_finite() || delta_raw <= 0.0 {
        return None;
    }

    use polars::prelude::TimeUnit;
    let seconds = match unit {
        TimeUnit::Nanoseconds => delta_raw / 1e9,
        TimeUnit::Microseconds => delta_raw / 1e6,
        TimeUnit::Milliseconds => delta_raw / 1e3,
    };
    let years = seconds / (365.25 * 24.0 * 3600.0);
    if years > 0.0 && years.is_finite() {
        Some(years)
    } else {
        None
    }
}

pub struct PermutationPipeline {
    config: Config,
    feature_pools: FeaturePools,
    comparison_specs: Arc<HashMap<String, ComparisonSpec>>,
    /// Per-feature scalar bound metadata used to enforce structural
    /// constraints such as "at most one lower and one upper bound per
    /// numeric base feature".
    ///
    /// For each feature index, this records an optional (base_id, bound_kind)
    /// pair when the feature represents a scalar threshold on a numeric
    /// base. Feature-vs-feature predicates and booleans are left as None.
    scalar_bound_meta: Vec<Option<(u16, BoundKind)>>,
    /// Total number of distinct base features that participate in scalar
    /// bound constraints. This bounds the per-combination scratch storage
    /// used by `combo_respects_scalar_bounds`.
    scalar_base_count: u16,
}

/// Direction of a scalar threshold bound for structural constraints.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum BoundKind {
    Lower,
    Upper,
}

fn bound_kind_for_operator(op: ComparisonOperator) -> Option<BoundKind> {
    match op {
        ComparisonOperator::GreaterThan | ComparisonOperator::GreaterEqual => {
            Some(BoundKind::Lower)
        }
        ComparisonOperator::LessThan | ComparisonOperator::LessEqual => Some(BoundKind::Upper),
    }
}

/// Simple LRU-style cache for under-min depth-2 pairs used for subset-based
/// pruning. Keys are encoded as u64: i | (j << 32) with i < j.
struct SubsetPruningCache {
    keys: std::collections::VecDeque<u64>,
    set: AHashSet<u64>,
    capacity: usize,
}

const SUBSET_CACHE_CAPACITY: usize = 5_000_000;

impl SubsetPruningCache {
    fn new(capacity: usize) -> Self {
        Self {
            keys: std::collections::VecDeque::new(),
            set: AHashSet::new(),
            capacity,
        }
    }

    fn encode_pair(i: usize, j: usize) -> u64 {
        debug_assert!(i < j);
        (i as u64) | ((j as u64) << 32)
    }

    fn insert_pair(&mut self, i: usize, j: usize) {
        if i >= j {
            return;
        }
        let key = Self::encode_pair(i, j);
        if self.set.contains(&key) {
            return;
        }
        self.set.insert(key);
        self.keys.push_back(key);
        if self.keys.len() > self.capacity {
            if let Some(old) = self.keys.pop_front() {
                self.set.remove(&old);
            }
        }
    }

    fn view(&self) -> &AHashSet<u64> {
        &self.set
    }

    fn len(&self) -> usize {
        self.keys.len()
    }

    fn keys_snapshot(&self) -> Vec<u64> {
        self.keys.iter().copied().collect()
    }

    fn save_to_file(&self, path: &std::path::Path) -> Result<()> {
        // Binary layout:
        // [u32 version][u32 reserved][u64 count][u64 key0]...[u64 keyN]
        let version: u32 = 1;
        let reserved: u32 = 0;
        let count: u64 = self.keys.len() as u64;
        let mut buf = Vec::with_capacity(16 + self.keys.len() * 8);
        buf.extend_from_slice(&version.to_le_bytes());
        buf.extend_from_slice(&reserved.to_le_bytes());
        buf.extend_from_slice(&count.to_le_bytes());
        for &key in self.keys.iter() {
            buf.extend_from_slice(&key.to_le_bytes());
        }
        fs::write(path, &buf).with_context(|| {
            format!("Failed to write subset pruning cache to {}", path.display())
        })?;
        Ok(())
    }

    fn load_from_file(path: &std::path::Path, capacity: usize) -> Result<Self> {
        if !path.exists() {
            return Ok(Self::new(capacity));
        }
        let data = fs::read(path).with_context(|| {
            format!(
                "Failed to read subset pruning cache from {}",
                path.display()
            )
        })?;
        if data.len() < 16 {
            return Ok(Self::new(capacity));
        }
        let version = u32::from_le_bytes(data[0..4].try_into().unwrap());
        let _reserved = u32::from_le_bytes(data[4..8].try_into().unwrap());
        let count = u64::from_le_bytes(data[8..16].try_into().unwrap());
        if version != 1 {
            return Ok(Self::new(capacity));
        }
        let available = (data.len() - 16) / 8;
        let take = std::cmp::min(available, std::cmp::min(count as usize, capacity));
        let mut cache = Self::new(capacity);
        for idx in 0..take {
            let start = 16 + idx * 8;
            let end = start + 8;
            let key = u64::from_le_bytes(data[start..end].try_into().unwrap());
            if cache.set.insert(key) {
                cache.keys.push_back(key);
            }
        }
        Ok(cache)
    }
}

/// Background saver that persists subset pruning cache snapshots without
/// blocking the main evaluation loop. Snapshots are written to a temporary
/// file and then atomically renamed into place.
struct SubsetCacheSaver {
    tx: mpsc::SyncSender<Vec<u64>>,
}

impl SubsetCacheSaver {
    fn new(path: PathBuf) -> (Self, thread::JoinHandle<()>) {
        let (tx, rx) = mpsc::sync_channel::<Vec<u64>>(1);
        let builder = thread::Builder::new()
            .name("subset-cache-saver".to_string())
            // Give the saver thread a larger stack in case downstream
            // logging or filesystem layers use recursion.
            .stack_size(8 * 1024 * 1024);
        let handle = builder
            .spawn(move || {
                while let Ok(snapshot) = rx.recv() {
                    let start = Instant::now();
                    let version: u32 = 1;
                    let reserved: u32 = 0;
                    let count: u64 = snapshot.len() as u64;
                    let mut buf = Vec::with_capacity(16 + snapshot.len() * 8);
                    buf.extend_from_slice(&version.to_le_bytes());
                    buf.extend_from_slice(&reserved.to_le_bytes());
                    buf.extend_from_slice(&count.to_le_bytes());
                    for key in snapshot {
                        buf.extend_from_slice(&key.to_le_bytes());
                    }
                    let tmp_path = path.with_extension("bin.tmp");
                    let write_result =
                        fs::write(&tmp_path, &buf).and_then(|_| fs::rename(&tmp_path, &path));
                    let elapsed_ms = (start.elapsed().as_secs_f32() * 1000.0).round() as u64;
                    match write_result {
                        Ok(_) => {
                            info!(
                                entries = %format_int(count as u128),
                                save_ms = %format_int(elapsed_ms),
                                path = %path.display(),
                                "Subset pruning cache async save completed"
                            );
                        }
                        Err(error) => {
                            warn!(
                                ?error,
                                save_ms = %format_int(elapsed_ms),
                                path = %path.display(),
                                "Subset pruning cache async save failed"
                            );
                        }
                    }
                }
            })
            .expect("failed to spawn subset-cache-saver thread");
        (SubsetCacheSaver { tx }, handle)
    }

    fn enqueue_blocking(&self, snapshot: Vec<u64>) {
        if snapshot.is_empty() {
            return;
        }
        let _ = self.tx.send(snapshot);
    }
}

/// Messages sent from the evaluation loop to the writer thread to decouple
/// Parquet/DuckDB I/O from combination evaluation. This allows the evaluator
/// to start working on the next batch while the previous batch is being
/// ingested.
enum StoreMsg {
    Ingest {
        combinations: Vec<String>,
        stats: Vec<StatSummary>,
        enumerated_count: usize,
        batch_start_offset: u64,
    },
    Flush,
}

impl PermutationPipeline {
    pub fn new(
        config: Config,
        features: Vec<FeatureDescriptor>,
        comparison_specs: HashMap<String, ComparisonSpec>,
    ) -> Self {
        let mut config = config;
        let catalog_hash = Self::compute_catalog_hash(&features, &comparison_specs);
        config.catalog_hash = Some(catalog_hash);

        let feature_pools = FeaturePools::new(features);
        let (scalar_bound_meta, scalar_base_count) =
            Self::build_scalar_bound_metadata(&feature_pools, &comparison_specs);
        Self {
            config,
            feature_pools,
            comparison_specs: Arc::new(comparison_specs),
            scalar_bound_meta,
            scalar_base_count,
        }
    }

    /// Precompute scalar-bound metadata used by `combo_respects_scalar_bounds`.
    /// For each catalog feature index, record an optional (base_id, bound_kind)
    /// pair when the feature represents a scalar threshold on a numeric base
    /// feature. This lets the structural constraint run in O(depth) with
    /// simple array lookups instead of repeated HashMap/string work.
    fn build_scalar_bound_metadata(
        feature_pools: &FeaturePools,
        comparison_specs: &HashMap<String, ComparisonSpec>,
    ) -> (Vec<Option<(u16, BoundKind)>>, u16) {
        use std::collections::HashMap;

        let descriptors = feature_pools.descriptors();
        let mut meta: Vec<Option<(u16, BoundKind)>> = vec![None; descriptors.len()];

        let mut base_ids: HashMap<&str, u16> = HashMap::new();
        let mut next_base_id: u16 = 0;

        for (idx, desc) in descriptors.iter().enumerate() {
            if desc.category != FeatureCategory::FeatureVsConstant {
                continue;
            }
            let spec = match comparison_specs.get(&desc.name) {
                Some(s) => s,
                None => continue,
            };
            // Skip feature-to-feature predicates; we only care about
            // scalar thresholds on a single numeric base.
            if spec.rhs_feature.is_some() {
                continue;
            }
            let kind = match bound_kind_for_operator(spec.operator) {
                Some(k) => k,
                None => continue,
            };
            let base = spec.base_feature.as_str();
            let base_id = *base_ids.entry(base).or_insert_with(|| {
                let id = next_base_id;
                next_base_id = next_base_id.saturating_add(1);
                id
            });
            meta[idx] = Some((base_id, kind));
        }

        (meta, next_base_id)
    }

    fn subset_cache_path(&self) -> Option<PathBuf> {
        let mut path = self.config.output_dir.clone();
        let hash = self.config.catalog_hash.as_deref()?;
        path.push(format!("subset_pruning_cache_{hash}.bin"));
        Some(path)
    }

    /// Enforce simple scalar-threshold structure on combinations:
    /// for each numeric base feature, allow at most one lower bound (>, >=)
    /// and at most one upper bound (<, <=). This permits brackets like
    /// `adx>20 && adx<50` but rejects redundant same-direction ladders such
    /// as `adx>20 && adx>40`.
    fn combo_respects_scalar_bounds(&self, indices: &[usize]) -> bool {
        if indices.len() < 2 || self.scalar_base_count == 0 {
            return true;
        }

        // Allocation-free implementation: for the (typically small) max_depth used
        // in Barsmith runs, do an O(depth²) pairwise scan and reject duplicate
        // bounds (same base_id + same bound direction) within a combination.
        for (pos_a, &feature_a) in indices.iter().enumerate() {
            let meta_a = match self.scalar_bound_meta.get(feature_a).and_then(|v| *v) {
                Some(tuple) => tuple,
                None => continue,
            };
            for &feature_b in indices.iter().skip(pos_a + 1) {
                let meta_b = match self.scalar_bound_meta.get(feature_b).and_then(|v| *v) {
                    Some(tuple) => tuple,
                    None => continue,
                };
                if meta_a.0 == meta_b.0 && meta_a.1 == meta_b.1 {
                    return false;
                }
            }
        }

        true
    }

    /// Build a human-readable label for a combination described by feature
    /// indices into the current feature catalog.
    fn combination_label_from_indices(&self, indices: &[usize]) -> String {
        let descriptors = self.feature_pools.descriptors();
        let mut parts = Vec::with_capacity(indices.len());
        for &idx in indices {
            if let Some(desc) = descriptors.get(idx) {
                parts.push(desc.name.as_str());
            }
        }
        parts.join(" && ")
    }

    /// Compute a stable hash of the effective feature catalog used for this run.
    /// This folds in:
    ///   - feature descriptors (name + category)
    ///   - comparison specs (threshold and feature-to-feature predicates)
    ///
    /// Any change in the catalog yields a different hash regardless of ordering.
    /// The hash is embedded into `Config` and used by the `CumulativeStore` to
    /// ensure resume offsets are only reused when the catalog is identical.
    fn compute_catalog_hash(
        descriptors: &[FeatureDescriptor],
        comparison_specs: &HashMap<String, ComparisonSpec>,
    ) -> String {
        let mut parts: Vec<String> = Vec::new();

        for d in descriptors {
            let cat = match d.category {
                FeatureCategory::Boolean => "B",
                FeatureCategory::Continuous => "N",
                FeatureCategory::FeatureVsConstant => "S",
                FeatureCategory::FeatureVsFeature => "P",
            };
            parts.push(format!("F|{cat}|{}", d.name));
        }

        for (name, spec) in comparison_specs {
            let op = match spec.operator {
                crate::feature::ComparisonOperator::GreaterThan => ">",
                crate::feature::ComparisonOperator::LessThan => "<",
                crate::feature::ComparisonOperator::GreaterEqual => ">=",
                crate::feature::ComparisonOperator::LessEqual => "<=",
            };
            let rhs = spec.rhs_feature.as_deref().unwrap_or("");
            let thr = spec
                .threshold
                .map(|t| format!("{:.10}", t))
                .unwrap_or_else(|| "".to_string());
            parts.push(format!("S|{name}|{}|{op}|{rhs}|{thr}", spec.base_feature));
        }

        parts.sort();
        let joined = parts.join("\n");
        let digest = Sha256::digest(joined.as_bytes());
        hex::encode(digest)
    }

    pub fn run(&mut self) -> Result<()> {
        let verbose = !self.config.quiet;

        if self.config.s3_upload_each_batch && self.config.s3_output.is_none() {
            return Err(anyhow!(
                "s3_upload_each_batch=true requires s3_output to be set"
            ));
        }

        if verbose {
            info!(
                target = %self.config.target,
                direction = ?self.config.direction,
                logic_mode = ?self.config.logic_mode,
                max_depth = %format_int(self.config.max_depth as u64),
                min_sample = %format_int(self.config.min_sample_size as u64),
                n_workers = %format_int(self.config.n_workers as u64),
                batch_size = %format_int(self.config.batch_size as u64),
                resume_offset = %format_int(self.config.resume_offset),
                limit = ?self.config.max_combos,
                output = %self.config.output_dir.display(),
                "Initialized permutation pipeline configuration"
            );
        } else {
            info!(
                target = %self.config.target,
                direction = ?self.config.direction,
                output = %self.config.output_dir.display(),
                "Starting permutation pipeline"
            );
        }

        let pruned_data = {
            let raw_data = ColumnarData::load(&self.config.input_csv)
                .with_context(|| "Failed to load dataset in columnar form")?;

            // Apply optional date filters to the engineered dataset at load time
            // so that the in-memory dataset, statistics, and reporting all see the
            // same time window. The prepared CSV on disk always contains the full
            // engineered history.
            let filtered_data = raw_data
                .filter_by_date_range(self.config.include_date_start, self.config.include_date_end)
                .with_context(|| "Failed to apply date filters to dataset")?;
            // raw_data is no longer needed after filtering; drop it eagerly so
            // that only the filtered frame and later the pruned frame remain
            // resident.
            drop(raw_data);

            // Infer the effective time horizon in years from the timestamp
            // range of the filtered engineered dataset so equity-based metrics
            // (CAGR, equity Sharpe/Sortino, Calmar) can be annualized
            // consistently.
            if self.config.equity_time_years.is_none() {
                if let Some(years) = infer_years_from_dataset(&filtered_data) {
                    self.config.equity_time_years = Some(years);
                }
            }

            // Detect the reward column up front so we can ensure it is
            // preserved when pruning the dataset down to just the columns
            // needed for this catalog (target, rewards, boolean features, and
            // comparison bases/RHS).
            let reward_col = stats::detect_reward_column(&filtered_data, &self.config)?;
            let keep_columns = required_columns_for_catalog(
                &self.config,
                &filtered_data,
                &self.feature_pools,
                self.comparison_specs.as_ref(),
                reward_col.as_deref(),
            );

            let pruned_data = filtered_data
                .prune_to_columns(&keep_columns)
                .with_context(|| "Failed to prune dataset to required columns")?;
            // filtered_data is not needed after we build the pruned frame; drop
            // it so that only the pruned dataset remains in memory.
            drop(filtered_data);
            pruned_data
        };
        let data = Arc::new(pruned_data);

        info!(
            columns = %format_int(data.column_names().len() as u64),
            approx_rows = %format_int(data.approx_rows() as u64),
            "Dataset loaded"
        );

        if self.config.dry_run {
            let feature_stats = summarize_features(self.feature_pools.descriptors());
            let theoretical = total_combinations(feature_stats.total, self.config.max_depth);
            if verbose {
                info!(
                    boolean_features = %format_int(feature_stats.boolean as u64),
                    feature_vs_constant = %format_int(feature_stats.scalar_comparisons as u64),
                    feature_vs_feature = %format_int(feature_stats.pair_comparisons as u64),
                    catalog_total = %format_int(feature_stats.total as u64),
                    theoretical_combos = %format_int(theoretical),
                    "Feature catalog detected"
                );
                info!(
                    batch_size = %format_int(self.config.batch_size as u64),
                    "Batch streaming parameters"
                );
            }
            info!(
                max_depth = %format_int(self.config.max_depth as u64),
                batch_size = %format_int(self.config.batch_size as u64),
                "Dry run: configuration summarized"
            );
            return Ok(());
        }

        // Size the mask cache based on the current feature catalog so we avoid
        // hitting the generational clear in MaskCache for large catalogs.
        // Each feature typically corresponds to a single boolean mask entry, so
        // a small multiple of the feature count is sufficient headroom while
        // keeping a hard upper bound to avoid unbounded growth.
        let feature_capacity = self.feature_pools.descriptors().len();
        let mask_cache_capacity = (feature_capacity.saturating_mul(4)).max(8_192);
        let mask_cache = Arc::new(MaskCache::with_max_entries(mask_cache_capacity));
        let eval_ctx = EvaluationContext::new(
            Arc::clone(&data),
            mask_cache,
            &self.config,
            Arc::clone(&self.comparison_specs),
        )
        .with_context(|| "Failed to initialize evaluation context")?;

        // Prune comparison predicates (feature-vs-constant and feature-vs-feature)
        // that are constant over this dataset, strictly below min_sample when
        // configured to do so, or duplicates of an earlier predicate.
        let original_features = self.feature_pools.descriptors().to_vec();
        let (pruned_features, comparison_pruning) = prune_comparison_features(
            &original_features,
            &eval_ctx,
            &self.config,
            self.comparison_specs.as_ref(),
        );
        self.feature_pools = FeaturePools::new(pruned_features);

        // Build a shared bitset catalog for all remaining features once, so
        // evaluation can use plain references without per-combination locking
        // or reference counting.
        let bitset_catalog =
            stats::build_bitset_catalog(&eval_ctx, self.feature_pools.descriptors())
                .with_context(|| "Failed to build bitset catalog for evaluation")?;

        let feature_stats = summarize_features(self.feature_pools.descriptors());
        let theoretical = total_combinations(feature_stats.total, self.config.max_depth);
        if verbose {
            log_target_stats(&self.config.target, &eval_ctx);
            if comparison_pruning.total_dropped() > 0 {
                comparison_pruning.log_summary();
            }
            info!(
                boolean_features = %format_int(feature_stats.boolean as u64),
                feature_vs_constant = %format_int(feature_stats.scalar_comparisons as u64),
                feature_vs_feature = %format_int(feature_stats.pair_comparisons as u64),
                catalog_total = %format_int(feature_stats.total as u64),
                theoretical_combos = %format_int(theoretical),
                "Feature catalog detected"
            );
            info!(
                batch_size = %format_int(self.config.batch_size as u64),
                "Batch streaming parameters"
            );
            log_analysis_overview(self.feature_pools.descriptors(), &eval_ctx, &self.config);
            info!("Analysis overview logged; inspecting existing cumulative results...");
        }

        // Detect existing Parquet result batches (if any) before opening the
        // cumulative store so we can report whether this run is extending a
        // prior surface or starting from scratch.
        let mut prior_parquet_batches: usize = 0;
        let results_dir = self.config.output_dir.join("results_parquet");
        if results_dir.exists() {
            if let Ok(entries) = fs::read_dir(&results_dir) {
                for entry in entries.flatten() {
                    let name = entry.file_name().to_string_lossy().into_owned();
                    if name.starts_with("part-") && name.ends_with(".parquet") {
                        prior_parquet_batches += 1;
                    }
                }
            }
        }
        let had_existing_results = prior_parquet_batches > 0;
        if verbose {
            if had_existing_results {
                if self.config.force_recompute {
                    info!(
                        prior_batches = %format_int(prior_parquet_batches as u64),
                        "Found existing results_parquet batches; force_recompute=true will clear prior results for this run"
                    );
                } else {
                    info!(
                        prior_batches = %format_int(prior_parquet_batches as u64),
                        "Found existing results_parquet batches; this run will extend the cumulative surface"
                    );
                }
            } else {
                info!("No existing results_parquet batches; starting fresh cumulative surface");
            }
        }
        let prior_results_retained = had_existing_results && !self.config.force_recompute;

        let (store, resume_offset) = CumulativeStore::new(&self.config)?;
        // Ensure subsequent uses of this Config within this run do not
        // trigger another force_recompute wipe when opening additional
        // CumulativeStore handles (e.g., for final reporting).
        self.config.force_recompute = false;
        let skip_evaluation = false;

        // Optional gating: only evaluate combinations that include at least one
        // of the provided feature names. This does not change enumeration, only
        // whether a given enumerated combination is evaluated.
        let required_feature_mask: Option<Vec<bool>> =
            if self.config.require_any_features.is_empty() {
                None
            } else {
                let descriptors = self.feature_pools.descriptors();
                let mut indices = HashSet::new();
                let mut missing: Vec<String> = Vec::new();
                for name in &self.config.require_any_features {
                    match descriptors.iter().position(|d| d.name == *name) {
                        Some(idx) => {
                            indices.insert(idx);
                        }
                        None => missing.push(name.clone()),
                    }
                }
                if !missing.is_empty() {
                    return Err(anyhow!(
                        "Unknown required feature name(s) for --require-any-features: {:?}. \
                     These must exist in the effective catalog after pruning.",
                        missing
                    ));
                }
                let mut mask = vec![false; descriptors.len()];
                for idx in indices {
                    if idx < mask.len() {
                        mask[idx] = true;
                    }
                }
                if verbose {
                    info!(
                        required_count = %format_int(self.config.require_any_features.len() as u64),
                        "Gating evaluation to combinations containing at least one required feature"
                    );
                }
                Some(mask)
            };

        // If there is config-specific resume metadata and the caller has not
        // explicitly provided a resume offset (via CLI or other means),
        // apply the stored offset so that evaluation extends the existing
        // combination stream instead of restarting from zero.
        if !self.config.explicit_resume_offset
            && self.config.resume_offset == 0
            && resume_offset > 0
        {
            self.config.resume_offset = resume_offset;
            if verbose {
                info!(resume_offset, "Resuming from cumulative metadata");
            }
        }

        let total_to_process = {
            let total = theoretical;
            if self.config.resume_offset as u128 > total {
                warn!(
                    resume_offset = %format_int(self.config.resume_offset),
                    theoretical_total = %format_int(total),
                    "Resume offset exceeds theoretical combination count; no new combinations will be enumerated this run"
                );
            }
            let remaining = total.saturating_sub(self.config.resume_offset as u128);
            if verbose {
                info!(
                    resume_offset = %format_int(self.config.resume_offset),
                    remaining_combos = %format_int(remaining),
                    "Combination window prepared"
                );
            }
            if remaining == 0 && self.config.resume_offset > 0 {
                info!(
                    resume_offset = %format_int(self.config.resume_offset),
                    "All theoretical combinations have already been enumerated for this configuration"
                );
            }
            if remaining > u64::MAX as u128 {
                return Err(anyhow!(
                    "Remaining combination count ({}) exceeds u64::MAX. \
                     Consider running depths separately or reducing catalog size.",
                    remaining
                ));
            }
            Some(remaining as u64)
        };

        // If there are retained prior results on disk for this configuration,
        // always emit a cumulative summary before evaluating any new
        // combinations. This mirrors the final report but lets you inspect
        // the current state of the surface at the very start of the run.
        if self.config.report_metrics.should_report() && prior_results_retained {
            let report_top = self.config.report_top.max(1);
            let dd_report = self
                .config
                .max_drawdown_report
                .unwrap_or(self.config.max_drawdown);
            let min_calmar = self.config.min_calmar_report;
            match store.top_results(
                report_top,
                self.config.min_sample_size,
                dd_report,
                min_calmar,
            ) {
                Ok(rows) => {
                    info!("📊 Existing cumulative results before this run (partial catalog)");
                    if self.config.report_metrics.is_full() {
                        let rows = if self.config.stats_detail == StatsDetail::Core {
                            match self.recompute_full_stats_for_rows(&rows, &data, &bitset_catalog)
                            {
                                Ok(updated) => updated,
                                Err(error) => {
                                    warn!(
                                        ?error,
                                        "Failed to recompute full statistics for existing cumulative results; falling back to stored metrics"
                                    );
                                    rows
                                }
                            }
                        } else {
                            rows
                        };
                        log_top_results(
                            &rows,
                            eval_ctx.row_count(),
                            self.config.position_sizing,
                            self.config.dollars_per_r,
                            self.config.cost_per_trade_r,
                            self.config.cost_per_trade_dollar,
                        );
                    } else if self.config.report_metrics.is_formulas_only() {
                        log_top_formulas(&rows);
                    }
                }
                Err(error) => {
                    warn!(
                        ?error,
                        "Failed to summarize existing cumulative results before resume"
                    );
                }
            }
        }

        let start_time = Instant::now();
        let mut batches_processed = 0usize;
        let mut tracker = ProgressTracker::new(&self.config);
        let mut total_enumerated: u64 = 0;
        let mut total_evaluated: u64 = 0;
        let mut total_enum_secs: f32 = 0.0;
        let mut total_filter_secs: f32 = 0.0;
        let mut total_eval_secs: f32 = 0.0;
        let mut total_ingest_secs: f32 = 0.0;
        let mut total_subset_save_secs: f32 = 0.0;
        let mut timing_history: Vec<BatchTimingSnapshot> = Vec::new();
        let mut effective_batch_size = self.config.batch_size.max(1);
        let batch_tuner = if self.config.auto_batch {
            Some(BatchTuner::new(effective_batch_size))
        } else {
            None
        };

        let subset_cache_path = if self.config.enable_subset_pruning {
            self.subset_cache_path()
        } else {
            None
        };
        let mut subset_cache = if self.config.enable_subset_pruning {
            if let Some(path) = &subset_cache_path {
                let load_start = Instant::now();
                match SubsetPruningCache::load_from_file(path, SUBSET_CACHE_CAPACITY) {
                    Ok(cache) => {
                        let load_ms = (load_start.elapsed().as_secs_f32() * 1000.0).round() as u64;
                        info!(
                            entries = cache.len(),
                            load_ms = %format_int(load_ms),
                            path = %path.display(),
                            "Subset pruning cache loaded"
                        );
                        Some(cache)
                    }
                    Err(error) => {
                        let load_ms = (load_start.elapsed().as_secs_f32() * 1000.0).round() as u64;
                        warn!(
                            ?error,
                            load_ms = %format_int(load_ms),
                            path = %path.display(),
                            "Failed to load subset pruning cache; starting empty"
                        );
                        Some(SubsetPruningCache::new(SUBSET_CACHE_CAPACITY))
                    }
                }
            } else {
                Some(SubsetPruningCache::new(SUBSET_CACHE_CAPACITY))
            }
        } else {
            None
        };
        let mut subset_cache_saver: Option<SubsetCacheSaver> = None;
        let mut subset_cache_handle: Option<thread::JoinHandle<()>> = None;
        if self.config.enable_subset_pruning {
            if let Some(path) = subset_cache_path.clone() {
                let (saver, handle) = SubsetCacheSaver::new(path);
                subset_cache_saver = Some(saver);
                subset_cache_handle = Some(handle);
            }
        }

        if !skip_evaluation {
            // Spawn a dedicated writer thread that owns the CumulativeStore and
            // ingests evaluated batches. This lets the evaluator overlap Parquet
            // / DuckDB I/O for batch N with enumeration and evaluation of batch
            // N+1.
            let (tx, rx) = mpsc::sync_channel::<StoreMsg>(2);

            let writer_handle = {
                let mut store = store;
                let s3_destination = if self.config.s3_upload_each_batch {
                    self.config
                        .s3_output
                        .as_deref()
                        .map(S3Destination::parse)
                        .transpose()?
                } else {
                    None
                };
                let output_dir = self.config.output_dir.clone();
                let catalog_hash = self.config.catalog_hash.clone();
                let mut s3_uploaded_prepared = false;
                let builder = thread::Builder::new()
                    .name("cumulative-writer".to_string())
                    // Writer owns Polars/DuckDB work; give it a generous stack.
                    .stack_size(16 * 1024 * 1024);
                builder
                    .spawn(move || -> Result<()> {
                        for msg in rx {
                            match msg {
                                StoreMsg::Ingest {
                                    combinations,
                                    stats,
                                    enumerated_count,
                                    batch_start_offset,
                                } => {
                                    let parquet_path = store.ingest_with_enumerated(
                                        &combinations,
                                        &stats,
                                        enumerated_count,
                                        batch_start_offset,
                                    )?;
                                    if let Some(dest) = &s3_destination {
                                        let start = Instant::now();
                                        let mut uploaded = 0usize;
                                        let mut upload_ms: u64 = 0;

                                        let prepared_path =
                                            output_dir.join("barsmith_prepared.csv");
                                        if !s3_uploaded_prepared && prepared_path.exists() {
                                            upload_ms +=
                                                dest.cp(&prepared_path, "barsmith_prepared.csv")?;
                                            uploaded += 1;
                                            s3_uploaded_prepared = true;
                                        }

                                        if let Some(path) = parquet_path.as_ref() {
                                            let filename = path
                                                .file_name()
                                                .and_then(|v| v.to_str())
                                                .unwrap_or("part.parquet");
                                            upload_ms += dest
                                                .cp(path, &format!("results_parquet/{filename}"))?;
                                            uploaded += 1;
                                        }

                                        let duckdb_path = output_dir.join("cumulative.duckdb");
                                        if duckdb_path.exists() {
                                            // Persist a consistent snapshot before uploading.
                                            store.flush()?;
                                            let tmp_path =
                                                output_dir.join("cumulative.duckdb.s3tmp");
                                            fs::copy(&duckdb_path, &tmp_path).with_context(
                                                || {
                                                    format!(
                                                        "Failed to copy DuckDB store from {} to {}",
                                                        duckdb_path.display(),
                                                        tmp_path.display()
                                                    )
                                                },
                                            )?;
                                            upload_ms += dest.cp(&tmp_path, "cumulative.duckdb")?;
                                            let _ = fs::remove_file(&tmp_path);
                                            uploaded += 1;
                                        }

                                        if let Some(hash) = catalog_hash.as_deref() {
                                            let filename =
                                                format!("subset_pruning_cache_{hash}.bin");
                                            let cache_path = output_dir.join(&filename);
                                            if cache_path.exists() {
                                                upload_ms += dest.cp(&cache_path, &filename)?;
                                                uploaded += 1;
                                            }
                                        }

                                        let total_ms =
                                            (start.elapsed().as_secs_f32() * 1000.0).round() as u64;
                                        info!(
                                            uploaded = %format_int(uploaded as u64),
                                            s3_upload_ms = %format_int(upload_ms),
                                            total_ms = %format_int(total_ms),
                                            "S3 upload completed"
                                        );
                                    }
                                }
                                StoreMsg::Flush => {
                                    store.flush()?;
                                    if let Some(dest) = &s3_destination {
                                        let duckdb_path = output_dir.join("cumulative.duckdb");
                                        if duckdb_path.exists() {
                                            dest.cp(&duckdb_path, "cumulative.duckdb")?;
                                        }
                                    }
                                    break;
                                }
                            }
                        }
                        Ok(())
                    })
                    .expect("failed to spawn cumulative-writer thread")
            };

            let mut batcher = IndexCombinationBatcher::new(
                &self.feature_pools,
                self.config.max_depth,
                self.config.resume_offset,
            );
            let pool = ThreadPoolBuilder::new()
                .num_threads(self.config.n_workers.max(1))
                .build()
                .context("Failed to build worker pool")?;

            loop {
                // Global enumeration cursor at the start of this batch,
                // including any non-zero resume offset.
                let batch_start_offset = tracker.processed() as u64;
                let enum_start = Instant::now();
                let batch = match batcher.next_batch(effective_batch_size) {
                    Some(batch) => batch,
                    None => break,
                };
                let enum_secs = enum_start.elapsed().as_secs_f32();

                // The combination iterator already produces each logical
                // combination exactly once in lexicographic order. Every
                // enumerated combination in this batch is considered once
                // for evaluation; there is no membership or reuse set.
                let enumerated_count = batch.len();
                // Measure subset-based and structural pruning separately so we can
                // see how much of the filter phase is dominated by each.
                let filter_start = Instant::now();
                let subset_start = filter_start;
                // Structural and subset-based pruning: precompute prune flags
                // for this batch based on:
                //   - scalar threshold structure per base feature
                //   - zero-sample depth-2 pairs from earlier in the run
                // Use u8 instead of Vec<bool> to avoid bit-packing overhead in hot loops.
                // 0 = keep, 1 = prune.
                let mut prune_flags: Vec<u8> = vec![0; batch.len()];

                // Subset-based pruning only applies for depth >= 3.
                if let Some(cache) = &subset_cache {
                    let snapshot = cache.view();
                    // Parallel subset-based pruning over the current batch, writing into
                    // the shared prune_flags vector (no intermediate flag allocations).
                    pool.install(|| {
                        prune_flags.par_iter_mut().zip(batch.par_iter()).for_each(
                            |(flag, indices)| {
                                let depth = indices.len();
                                if depth < 3 {
                                    return;
                                }
                                // Generate all depth-2 subsets (i,j) and check membership.
                                for a in 0..depth {
                                    for b in (a + 1)..depth {
                                        let i = indices[a];
                                        let j = indices[b];
                                        let key = SubsetPruningCache::encode_pair(i, j);
                                        if snapshot.contains(&key) {
                                            *flag = 1;
                                            return;
                                        }
                                    }
                                }
                            },
                        )
                    });
                }

                let subset_secs = subset_start.elapsed().as_secs_f32();
                let struct_start = Instant::now();

                // Structural pruning: reject combinations that contain more
                // than one lower bound or more than one upper bound on the
                // same base feature (e.g., adx>20 && adx>40).
                if !batch.is_empty() {
                    let required_mask = required_feature_mask.as_deref();
                    pool.install(|| {
                        prune_flags.par_iter_mut().zip(batch.par_iter()).for_each(
                            |(flag, indices)| {
                                if *flag != 0 {
                                    return;
                                }
                                if let Some(mask) = required_mask {
                                    // Optional gating: skip combos that do not include any required feature.
                                    if !indices.iter().any(|&idx| idx < mask.len() && mask[idx]) {
                                        *flag = 1;
                                        return;
                                    }
                                }
                                if !self.combo_respects_scalar_bounds(indices) {
                                    *flag = 1;
                                }
                            },
                        )
                    });
                }
                let struct_secs = struct_start.elapsed().as_secs_f32();
                let filter_secs = subset_secs + struct_secs;
                let batch_pruned = prune_flags.iter().filter(|&&v| v != 0).count();

                total_enumerated += enumerated_count as u64;
                // total_evaluated is updated after we know how many combinations
                // we actually evaluated.

                let eval_profile_mode = self.config.eval_profile;
                let eval_profile_sample_rate = self.config.eval_profile_sample_rate.max(1);
                let eval_start = Instant::now();
                let (summaries, eval_profile_totals): (
                    Vec<Option<StatSummary>>,
                    Option<stats::EvalProfileTotals>,
                ) = if batch.is_empty() {
                    (Vec::new(), None)
                } else if matches!(eval_profile_mode, EvalProfileMode::Off) {
                    // Share the precomputed bitset catalog across all worker
                    // threads; combinations reference bitsets by index via
                    // this catalog without additional locking.
                    let bitsets = &bitset_catalog;
                    let min_sample_size = self.config.min_sample_size;
                    let summaries = pool.install(|| {
                        batch
                            .par_iter()
                            .zip(prune_flags.par_iter())
                            .map(|(indices, &prune)| {
                                if prune != 0 {
                                    Ok(None)
                                } else {
                                    stats::evaluate_combination_indices(
                                        indices,
                                        &eval_ctx,
                                        bitsets,
                                        min_sample_size,
                                    )
                                    .map(Some)
                                }
                            })
                            .collect::<Result<Vec<_>>>()
                    })?;
                    (summaries, None)
                } else {
                    let bitsets = &bitset_catalog;
                    let min_sample_size = self.config.min_sample_size;
                    let profiled = pool.install(|| {
                        batch
                            .par_iter()
                            .zip(prune_flags.par_iter())
                            .map(|(indices, &prune)| {
                                if prune != 0 {
                                    Ok(None)
                                } else {
                                    stats::evaluate_combination_indices_profiled(
                                        indices,
                                        &eval_ctx,
                                        bitsets,
                                        min_sample_size,
                                        eval_profile_mode,
                                        eval_profile_sample_rate,
                                    )
                                    .map(Some)
                                }
                            })
                            .collect::<Result<Vec<_>>>()
                    })?;

                    let mut summaries: Vec<Option<StatSummary>> =
                        Vec::with_capacity(profiled.len());
                    let mut totals = stats::EvalProfileTotals::default();
                    for maybe in profiled {
                        match maybe {
                            None => summaries.push(None),
                            Some((stat, profile)) => {
                                totals.add_assign(profile);
                                summaries.push(Some(stat));
                            }
                        }
                    }

                    (summaries, Some(totals))
                };
                let eval_secs = eval_start.elapsed().as_secs_f32();

                // Split evaluated combinations into those that meet the
                // storage threshold and those that are under-min. Both sets
                // count as "evaluated" for timing/logging, but only the
                // storage-eligible subset is persisted to cumulative results.
                let mut store_combinations: Vec<String> = Vec::new();
                let mut store_stats: Vec<stats::StatSummary> = Vec::new();
                let mut batch_evaluated: u64 = 0;
                let mut subset_save_secs: f32 = 0.0;
                if let Some(cache) = &mut subset_cache {
                    // Insert new under-min depth-2 pairs into the cache so
                    // higher-depth supersets can be pruned cheaply.
                    let min_sample_size = self.config.min_sample_size;
                    for (indices, maybe_stat) in batch.iter().zip(summaries.iter()) {
                        if let Some(stat) = maybe_stat {
                            if indices.len() == 2 && stat.total_bars < min_sample_size {
                                let i = indices[0];
                                let j = indices[1];
                                cache.insert_pair(i, j);
                            }
                        }
                    }

                    // Persist the updated subset pruning cache snapshot after each batch
                    // via the background saver so that an interrupted run still leaves
                    // behind the latest known dead pairs. This is measured separately
                    // from filter/eval timing and reported as subset_save_ms in logs.
                    if cache.len() > 0 {
                        let save_start = Instant::now();
                        if let Some(saver) = subset_cache_saver.as_ref() {
                            let snapshot = cache.keys_snapshot();
                            saver.enqueue_blocking(snapshot);
                        } else if let Some(path) = &subset_cache_path {
                            // Fallback to a synchronous save when no saver is available.
                            if let Err(error) = cache.save_to_file(path) {
                                warn!(
                                    ?error,
                                    path = %path.display(),
                                    "Failed to persist subset pruning cache after batch"
                                );
                            }
                        }
                        subset_save_secs = save_start.elapsed().as_secs_f32();
                    }
                }

                let dd_store = self.config.max_drawdown;
                for (indices, maybe_stat) in batch.iter().zip(summaries.into_iter()) {
                    if let Some(stat) = maybe_stat {
                        batch_evaluated += 1;
                        if stat.total_bars >= self.config.min_sample_size
                            && stat.max_drawdown <= dd_store
                        {
                            let label = self.combination_label_from_indices(indices);
                            store_combinations.push(label);
                            store_stats.push(stat);
                        }
                    }
                }

                total_evaluated += batch_evaluated;

                let ingest_start = Instant::now();
                tx.send(StoreMsg::Ingest {
                    combinations: store_combinations,
                    stats: store_stats,
                    enumerated_count,
                    batch_start_offset,
                })
                .expect("writer thread dropped");
                let continue_running = tracker.record_batch(enumerated_count);
                let ingest_secs = ingest_start.elapsed().as_secs_f32();

                total_enum_secs += enum_secs;
                total_filter_secs += filter_secs;
                total_eval_secs += eval_secs;
                total_ingest_secs += ingest_secs;
                total_subset_save_secs += subset_save_secs;

                // Track timings for optional auto-batch tuning.
                if total_to_process.is_some() {
                    timing_history.push(BatchTimingSnapshot {
                        enumeration_ms: (enum_secs * 1000.0).round() as u64,
                        filter_ms: (filter_secs * 1000.0).round() as u64,
                        eval_ms: (eval_secs * 1000.0).round() as u64,
                        ingest_ms: (ingest_secs * 1000.0).round() as u64,
                        prune_subset_ms: (subset_secs * 1000.0).round() as u64,
                        prune_struct_ms: (struct_secs * 1000.0).round() as u64,
                    });
                    // Keep the timing window reasonably small so tuning remains responsive.
                    if timing_history.len() > 32 {
                        let excess = timing_history.len() - 32;
                        timing_history.drain(0..excess);
                    }

                    // Periodically adjust the effective batch size when auto-batch is enabled.
                    if let Some(tuner) = &batch_tuner {
                        // Avoid thrashing: only adjust every few batches and once we have history.
                        if batches_processed > 0 && batches_processed % 10 == 0 {
                            let recommended =
                                tuner.recommend(effective_batch_size, &timing_history);
                            if recommended != effective_batch_size {
                                if verbose {
                                    info!(
                                        old_batch_size = %format_int(effective_batch_size as u64),
                                        new_batch_size = %format_int(recommended as u64),
                                        "Auto-batch adjusted effective batch size based on recent timings"
                                    );
                                }
                                effective_batch_size = recommended.max(1);
                            }
                        }
                    }
                }

                batches_processed += 1;

                if verbose {
                    let elapsed_secs = start_time.elapsed().as_secs_f32();
                    let elapsed_human = format_duration(elapsed_secs);
                    let enumerated = tracker.processed_since_start() as u64;
                    // Global enumeration cursor including any non-zero resume offset.
                    let resume_offset = tracker.processed() as u64;
                    // Track the depth (number of features) for combinations in this batch.
                    let current_depth = batch.first().map(|c| c.len()).unwrap_or(0);
                    let total_secs =
                        enum_secs + filter_secs + eval_secs + ingest_secs + subset_save_secs;
                    let (
                        eval_profile_build_ms,
                        eval_profile_scan_ms,
                        eval_profile_on_hit_ms,
                        eval_profile_finalize_ms,
                    ) = eval_profile_totals.unwrap_or_default().ms();
                    let eval_profile_combos =
                        eval_profile_totals.map(|p| p.combos_profiled).unwrap_or(0);
                    let eval_profile_mask_hits =
                        eval_profile_totals.map(|p| p.mask_hits).unwrap_or(0);
                    let eval_profile_trades = eval_profile_totals.map(|p| p.trades).unwrap_or(0);
                    let eval_profile_enabled = eval_profile_totals.is_some();

                    let (remaining_opt, eta_secs) =
                        total_to_process.map_or((None, None), |total| {
                            if enumerated > 0 && total > enumerated {
                                let remaining = total.saturating_sub(enumerated);
                                let ratio = total as f32 / enumerated as f32;
                                let eta = elapsed_secs * (ratio - 1.0);
                                (Some(remaining), Some(eta))
                            } else {
                                (None, None)
                            }
                        });

                    match (remaining_opt, eta_secs) {
                        (Some(remaining), Some(eta)) => {
                            let eta_human = format_duration(eta);
                            if eval_profile_enabled {
                                info!(
                                    resume_offset = %format_int(resume_offset),
                                    enumerated = %format_int(enumerated),
                                    eta = %eta_human,
                                    elapsed = %elapsed_human,
                                    eval_ms = %format_int((eval_secs * 1000.0).round() as u64),
                                    eval_profile = ?eval_profile_mode,
                                    eval_profile_sample_rate = %format_int(eval_profile_sample_rate as u64),
                                    eval_profiled_combos = %format_int(eval_profile_combos),
                                    eval_profile_mask_hits = %format_int(eval_profile_mask_hits),
                                    eval_profile_trades = %format_int(eval_profile_trades),
                                    eval_build_ms = %format_int(eval_profile_build_ms),
                                    eval_scan_ms = %format_int(eval_profile_scan_ms),
                                    eval_on_hit_ms = %format_int(eval_profile_on_hit_ms),
                                    eval_finalize_ms = %format_int(eval_profile_finalize_ms),
                                    enumeration_ms = %format_int((enum_secs * 1000.0).round() as u64),
                                    prune = %format_int(batch_pruned as u64),
                                    prune_ms = %format_int((filter_secs * 1000.0).round() as u64),
                                    prune_subset_ms = %format_int((subset_secs * 1000.0).round() as u64),
                                    prune_struct_ms = %format_int((struct_secs * 1000.0).round() as u64),
                                    subset_save_ms = %format_int((subset_save_secs * 1000.0).round() as u64),
                                    ingest_ms = %format_int((ingest_secs * 1000.0).round() as u64),
                                    total_ms = %format_int((total_secs * 1000.0).round() as u64),
                                    remaining = %format_int(remaining as u128),
                                    last_batch = %format_int(batch.len() as u64),
                                    batches_processed = %format_int(batches_processed as u64),
                                    current_depth = %format_int(current_depth as u64),
                                    "Processed batch"
                                );
                            } else {
                                info!(
                                    resume_offset = %format_int(resume_offset),
                                    enumerated = %format_int(enumerated),
                                    eta = %eta_human,
                                    elapsed = %elapsed_human,
                                    eval_ms = %format_int((eval_secs * 1000.0).round() as u64),
                                    enumeration_ms = %format_int((enum_secs * 1000.0).round() as u64),
                                    prune = %format_int(batch_pruned as u64),
                                    prune_ms = %format_int((filter_secs * 1000.0).round() as u64),
                                    prune_subset_ms = %format_int((subset_secs * 1000.0).round() as u64),
                                    prune_struct_ms = %format_int((struct_secs * 1000.0).round() as u64),
                                    subset_save_ms = %format_int((subset_save_secs * 1000.0).round() as u64),
                                    ingest_ms = %format_int((ingest_secs * 1000.0).round() as u64),
                                    total_ms = %format_int((total_secs * 1000.0).round() as u64),
                                    remaining = %format_int(remaining as u128),
                                    last_batch = %format_int(batch.len() as u64),
                                    batches_processed = %format_int(batches_processed as u64),
                                    current_depth = %format_int(current_depth as u64),
                                    "Processed batch"
                                );
                            }
                        }
                        (Some(remaining), None) => {
                            if eval_profile_enabled {
                                info!(
                                    resume_offset = %format_int(resume_offset),
                                    enumerated = %format_int(enumerated),
                                    elapsed = %elapsed_human,
                                    eval_ms = %format_int((eval_secs * 1000.0).round() as u64),
                                    eval_profile = ?eval_profile_mode,
                                    eval_profile_sample_rate = %format_int(eval_profile_sample_rate as u64),
                                    eval_profiled_combos = %format_int(eval_profile_combos),
                                    eval_profile_mask_hits = %format_int(eval_profile_mask_hits),
                                    eval_profile_trades = %format_int(eval_profile_trades),
                                    eval_build_ms = %format_int(eval_profile_build_ms),
                                    eval_scan_ms = %format_int(eval_profile_scan_ms),
                                    eval_on_hit_ms = %format_int(eval_profile_on_hit_ms),
                                    eval_finalize_ms = %format_int(eval_profile_finalize_ms),
                                    enumeration_ms = %format_int((enum_secs * 1000.0).round() as u64),
                                    prune = %format_int(batch_pruned as u64),
                                    prune_ms = %format_int((filter_secs * 1000.0).round() as u64),
                                    prune_subset_ms = %format_int((subset_secs * 1000.0).round() as u64),
                                    prune_struct_ms = %format_int((struct_secs * 1000.0).round() as u64),
                                    subset_save_ms = %format_int((subset_save_secs * 1000.0).round() as u64),
                                    ingest_ms = %format_int((ingest_secs * 1000.0).round() as u64),
                                    total_ms = %format_int((total_secs * 1000.0).round() as u64),
                                    remaining = %format_int(remaining as u128),
                                    last_batch = %format_int(batch.len() as u64),
                                    batches_processed = %format_int(batches_processed as u64),
                                    current_depth = %format_int(current_depth as u64),
                                    "Processed batch"
                                );
                            } else {
                                info!(
                                    resume_offset = %format_int(resume_offset),
                                    enumerated = %format_int(enumerated),
                                    elapsed = %elapsed_human,
                                    eval_ms = %format_int((eval_secs * 1000.0).round() as u64),
                                    enumeration_ms = %format_int((enum_secs * 1000.0).round() as u64),
                                    prune = %format_int(batch_pruned as u64),
                                    prune_ms = %format_int((filter_secs * 1000.0).round() as u64),
                                    prune_subset_ms = %format_int((subset_secs * 1000.0).round() as u64),
                                    prune_struct_ms = %format_int((struct_secs * 1000.0).round() as u64),
                                    subset_save_ms = %format_int((subset_save_secs * 1000.0).round() as u64),
                                    ingest_ms = %format_int((ingest_secs * 1000.0).round() as u64),
                                    total_ms = %format_int((total_secs * 1000.0).round() as u64),
                                    remaining = %format_int(remaining as u128),
                                    last_batch = %format_int(batch.len() as u64),
                                    batches_processed = %format_int(batches_processed as u64),
                                    current_depth = %format_int(current_depth as u64),
                                    "Processed batch"
                                );
                            }
                        }
                        (None, _) => {
                            if eval_profile_enabled {
                                info!(
                                    resume_offset = %format_int(resume_offset),
                                    enumerated = %format_int(enumerated),
                                    elapsed = %elapsed_human,
                                    eval_ms = %format_int((eval_secs * 1000.0).round() as u64),
                                    eval_profile = ?eval_profile_mode,
                                    eval_profile_sample_rate = %format_int(eval_profile_sample_rate as u64),
                                    eval_profiled_combos = %format_int(eval_profile_combos),
                                    eval_profile_mask_hits = %format_int(eval_profile_mask_hits),
                                    eval_profile_trades = %format_int(eval_profile_trades),
                                    eval_build_ms = %format_int(eval_profile_build_ms),
                                    eval_scan_ms = %format_int(eval_profile_scan_ms),
                                    eval_on_hit_ms = %format_int(eval_profile_on_hit_ms),
                                    eval_finalize_ms = %format_int(eval_profile_finalize_ms),
                                    enumeration_ms = %format_int((enum_secs * 1000.0).round() as u64),
                                    prune = %format_int(batch_pruned as u64),
                                    prune_ms = %format_int((filter_secs * 1000.0).round() as u64),
                                    prune_subset_ms = %format_int((subset_secs * 1000.0).round() as u64),
                                    prune_struct_ms = %format_int((struct_secs * 1000.0).round() as u64),
                                    subset_save_ms = %format_int((subset_save_secs * 1000.0).round() as u64),
                                    ingest_ms = %format_int((ingest_secs * 1000.0).round() as u64),
                                    total_ms = %format_int((total_secs * 1000.0).round() as u64),
                                    last_batch = %format_int(batch.len() as u64),
                                    batches_processed = %format_int(batches_processed as u64),
                                    current_depth = %format_int(current_depth as u64),
                                    "Processed batch"
                                );
                            } else {
                                info!(
                                    resume_offset = %format_int(resume_offset),
                                    enumerated = %format_int(enumerated),
                                    elapsed = %elapsed_human,
                                    eval_ms = %format_int((eval_secs * 1000.0).round() as u64),
                                    enumeration_ms = %format_int((enum_secs * 1000.0).round() as u64),
                                    prune = %format_int(batch_pruned as u64),
                                    prune_ms = %format_int((filter_secs * 1000.0).round() as u64),
                                    prune_subset_ms = %format_int((subset_secs * 1000.0).round() as u64),
                                    prune_struct_ms = %format_int((struct_secs * 1000.0).round() as u64),
                                    subset_save_ms = %format_int((subset_save_secs * 1000.0).round() as u64),
                                    ingest_ms = %format_int((ingest_secs * 1000.0).round() as u64),
                                    total_ms = %format_int((total_secs * 1000.0).round() as u64),
                                    last_batch = %format_int(batch.len() as u64),
                                    batches_processed = %format_int(batches_processed as u64),
                                    current_depth = %format_int(current_depth as u64),
                                    "Processed batch"
                                );
                            }
                        }
                    }
                }
                if !continue_running {
                    warn!("Reached configured limit; stopping early");
                    break;
                }
            }

            // Signal the writer to flush and wait for it to finish so that
            // all results are durable before we compute the final report.
            if tx.send(StoreMsg::Flush).is_ok() {
                drop(tx);
            }
            let writer_result = writer_handle
                .join()
                .map_err(|_| anyhow::anyhow!("writer thread panicked"))?;
            writer_result?;
        }

        if !skip_evaluation {
            if verbose {
                let mut buffer = String::new();
                let _ = writeln!(buffer, "📊  Evaluation summary (this run):");
                let _ = writeln!(
                    buffer,
                    "   Enumerated: {}",
                    format_int(total_enumerated as u128)
                );
                let _ = writeln!(
                    buffer,
                    "   Evaluated:  {}",
                    format_int(total_evaluated as u128)
                );
                let _ = writeln!(
                    buffer,
                    "   Phase timing (approx): enum={}s, filter={}s, eval={}s, ingest={}s, subset_save={}s",
                    format_duration(total_enum_secs),
                    format_duration(total_filter_secs),
                    format_duration(total_eval_secs),
                    format_duration(total_ingest_secs),
                    format_duration(total_subset_save_secs),
                );
                info!("{}", buffer);
            } else {
                info!(
                    enumerated = %format_int(total_enumerated as u128),
                    total_evaluated = %format_int(total_evaluated as u128),
                    enum_secs = %format_duration(total_enum_secs),
                    filter_secs = %format_duration(total_filter_secs),
                    eval_secs = %format_duration(total_eval_secs),
                    ingest_secs = %format_duration(total_ingest_secs),
                    "Evaluation summary"
                );
            }
        }

        let total_elapsed = start_time.elapsed().as_secs_f32();
        let total_elapsed_human = format_duration(total_elapsed);

        if self.config.enable_subset_pruning {
            // Force a final snapshot and wait for the background saver to
            // flush it so that the latest state is durable before exit.
            if let Some(cache) = subset_cache.as_ref() {
                if let Some(saver) = subset_cache_saver.take() {
                    let snapshot = cache.keys_snapshot();
                    saver.enqueue_blocking(snapshot);
                    // Dropping `saver` here closes the channel, allowing the
                    // saver thread to observe EOF and exit cleanly.
                }
            } else {
                // No cache snapshot to persist, but still drop the saver so
                // that the background thread can exit.
                let _ = subset_cache_saver.take();
            }
            if let Some(handle) = subset_cache_handle.take() {
                let _ = handle.join();
            }
        }

        info!(
            elapsed = %total_elapsed_human,
            total_batches = %format_int(batches_processed as u64),
            total_combos = %format_int(tracker.processed_since_start() as u128),
            resume_offset = %format_int(tracker.start_offset() as u64),
            "Permutation run complete"
        );

        if self.config.report_metrics.should_report() {
            let report_top = self.config.report_top.max(1);
            let dd_report = self
                .config
                .max_drawdown_report
                .unwrap_or(self.config.max_drawdown);
            let min_calmar = self.config.min_calmar_report;
            match CumulativeStore::new(&self.config).and_then(|(store, _)| {
                store.top_results(
                    report_top,
                    self.config.min_sample_size,
                    dd_report,
                    min_calmar,
                )
            }) {
                Ok(rows) => {
                    if prior_results_retained {
                        info!(
                            "Final top results include combinations from previous runs (cumulative surface)"
                        );
                    } else if had_existing_results {
                        info!(
                            "Previous Parquet batches were cleared for this run; top results reflect only this run"
                        );
                    } else {
                        info!("Top results reflect only this run");
                    }
                    if self.config.report_metrics.is_full() {
                        let rows = match self.recompute_full_stats_for_rows(
                            &rows,
                            &data,
                            &bitset_catalog,
                        ) {
                            Ok(updated) => updated,
                            Err(error) => {
                                warn!(
                                    ?error,
                                    "Failed to recompute full statistics for final top results; falling back to stored metrics"
                                );
                                rows
                            }
                        };
                        log_top_results(
                            &rows,
                            eval_ctx.row_count(),
                            self.config.position_sizing,
                            self.config.dollars_per_r,
                            self.config.cost_per_trade_r,
                            self.config.cost_per_trade_dollar,
                        );
                    } else if self.config.report_metrics.is_formulas_only() {
                        log_top_formulas(&rows);
                    }
                }
                Err(error) => warn!(?error, "Failed to summarize cumulative results"),
            }
        }

        Ok(())
    }
}

impl PermutationPipeline {
    /// When running in core-stats mode, recompute full statistics for a small
    /// set of top-ranked combinations prior to reporting so that the printed
    /// table includes the complete metric set. This never mutates stored
    /// results; it is a read-only, in-process refinement for display.
    fn recompute_full_stats_for_rows(
        &self,
        rows: &[ResultRow],
        data: &Arc<ColumnarData>,
        bitsets: &stats::BitsetCatalog,
    ) -> Result<Vec<ResultRow>> {
        if rows.is_empty() {
            return Ok(Vec::new());
        }

        // Build a fresh evaluation context configured for full-detail stats.
        let feature_capacity = self.feature_pools.descriptors().len();
        let mask_cache_capacity = (feature_capacity.saturating_mul(4)).max(8_192);
        let mask_cache = Arc::new(MaskCache::with_max_entries(mask_cache_capacity));
        let mut full_config = self.config.clone();
        full_config.stats_detail = StatsDetail::Full;
        let eval_ctx = EvaluationContext::new(
            Arc::clone(data),
            mask_cache,
            &full_config,
            Arc::clone(&self.comparison_specs),
        )?;

        // Map feature names to indices for quick lookup.
        let descriptors = self.feature_pools.descriptors();
        let mut name_to_index: HashMap<&str, usize> = HashMap::new();
        for (idx, desc) in descriptors.iter().enumerate() {
            name_to_index.insert(desc.name.as_str(), idx);
        }

        let mut updated_rows = Vec::with_capacity(rows.len());
        for row in rows {
            // Convert "a && b && c" back into catalog indices.
            let mut indices: Vec<usize> = Vec::new();
            for name in row.combination.split(" && ") {
                if let Some(&idx) = name_to_index.get(name) {
                    indices.push(idx);
                } else {
                    indices.clear();
                    break;
                }
            }

            if indices.is_empty() {
                // If we cannot map the combination back to indices, fall back
                // to the stored row without modification.
                updated_rows.push(row.clone());
                continue;
            }

            match stats::evaluate_combination_indices(
                &indices,
                &eval_ctx,
                bitsets,
                self.config.min_sample_size,
            ) {
                Ok(stat) => {
                    let mut r = row.clone();
                    r.depth = stat.depth as u32;
                    r.mask_hits = stat.mask_hits as u64;
                    r.total_bars = stat.total_bars as u64;
                    r.profitable_bars = stat.profitable_bars as u64;
                    r.label_hit_rate = stat.label_hit_rate;
                    r.label_hits = stat.label_hits as u64;
                    r.label_misses = stat.label_misses as u64;
                    r.win_rate = stat.win_rate;
                    r.expectancy = stat.expectancy;
                    r.total_return = stat.total_return;
                    r.max_drawdown = stat.max_drawdown;
                    r.profit_factor = stat.profit_factor;
                    r.calmar_ratio = stat.calmar_ratio;
                    r.win_loss_ratio = stat.win_loss_ratio;
                    r.ulcer_index = stat.ulcer_index;
                    r.pain_ratio = stat.pain_ratio;
                    r.max_consecutive_wins = stat.max_consecutive_wins as u64;
                    r.max_consecutive_losses = stat.max_consecutive_losses as u64;
                    r.avg_winning_rr = stat.avg_winning_rr;
                    r.avg_win_streak = stat.avg_win_streak;
                    r.avg_loss_streak = stat.avg_loss_streak;
                    r.median_rr = stat.median_rr;
                    r.avg_losing_rr = stat.avg_losing_rr;
                    r.p05_rr = stat.p05_rr;
                    r.p95_rr = stat.p95_rr;
                    r.largest_win = stat.largest_win;
                    r.largest_loss = stat.largest_loss;
                    r.final_capital = stat.final_capital;
                    r.total_return_pct = stat.total_return_pct;
                    r.cagr_pct = stat.cagr_pct;
                    r.max_drawdown_pct_equity = stat.max_drawdown_pct_equity;
                    r.calmar_equity = stat.calmar_equity;
                    r.sharpe_equity = stat.sharpe_equity;
                    r.sortino_equity = stat.sortino_equity;
                    updated_rows.push(r);
                }
                Err(_) => {
                    // On any evaluation error, fall back to stored metrics for this row.
                    updated_rows.push(row.clone());
                }
            }
        }

        Ok(updated_rows)
    }
}

struct FeatureStatsSummary {
    boolean: usize,
    scalar_comparisons: usize,
    pair_comparisons: usize,
    total: usize,
}

fn summarize_features(features: &[FeatureDescriptor]) -> FeatureStatsSummary {
    let mut stats = FeatureStatsSummary {
        boolean: 0,
        scalar_comparisons: 0,
        pair_comparisons: 0,
        total: features.len(),
    };

    for descriptor in features {
        match descriptor.category {
            FeatureCategory::Boolean => {
                stats.boolean += 1;
            }
            FeatureCategory::FeatureVsConstant => {
                stats.scalar_comparisons += 1;
            }
            FeatureCategory::FeatureVsFeature => {
                stats.pair_comparisons += 1;
            }
            // Continuous features are not part of the permutation catalog surface,
            // so we do not include them in the high-level counts to avoid confusion.
            FeatureCategory::Continuous => {}
        }
    }
    stats
}

fn format_int<T: Into<u128>>(value: T) -> String {
    let s = value.into().to_string();
    let len = s.len();
    if len <= 3 {
        return s;
    }
    let mut out = String::with_capacity(len + len / 3);
    let mut count = 0usize;
    for ch in s.chars().rev() {
        if count == 3 {
            out.push(',');
            count = 0;
        }
        out.push(ch);
        count += 1;
    }
    out.chars().rev().collect()
}

fn format_duration(seconds: f32) -> String {
    if !seconds.is_finite() || seconds < 0.0 {
        return "unknown".to_string();
    }
    let total = seconds.round() as u64;
    let days = total / 86_400;
    let hours = (total % 86_400) / 3_600;
    let minutes = (total % 3_600) / 60;
    let secs = total % 60;

    if days > 0 {
        // Include days, hours, and minutes for very long durations.
        format!("{d}d {h:02}h {m:02}m", d = days, h = hours, m = minutes)
    } else if hours > 0 {
        // For multi-hour runs, show hours and minutes.
        format!("{h}h {m:02}m", h = hours, m = minutes)
    } else if minutes > 0 {
        // For shorter runs, show minutes and seconds.
        format!("{m}m {s:02}s", m = minutes, s = secs)
    } else {
        // Sub-minute durations stay in seconds.
        format!("{s}s", s = secs)
    }
}

fn log_target_stats(target_name: &str, ctx: &EvaluationContext) {
    let target = ctx.target();
    let total = target.len();
    let wins = target.iter().filter(|value| **value).count();
    let losses = total.saturating_sub(wins);
    let win_rate = if total > 0 {
        wins as f64 / total as f64 * 100.0
    } else {
        0.0
    };
    let reward_available = ctx.rewards().is_some();
    let win_rate_fmt = format!("{win_rate:.2}%");
    info!(
        target = %target_name,
        total_bars = total,
        profitable = wins,
        unprofitable = losses,
        win_rate = win_rate_fmt.as_str(),
        reward_column = reward_available,
        "Target statistics ready"
    );

    let mut buffer = String::new();
    let _ = writeln!(buffer, "📊 Target statistics:");
    let _ = writeln!(buffer, "   Total bars: {}", total);
    let _ = writeln!(buffer, "   Profitable: {} ({:.1}%)", wins, win_rate);
    let _ = writeln!(
        buffer,
        "   Unprofitable: {} ({:.1}%)",
        losses,
        100.0 - win_rate
    );
    info!("{}", buffer);
}

fn sample_size_for_mask(ctx: &EvaluationContext, mask: &[bool]) -> usize {
    let target = ctx.target();
    let rewards = ctx.rewards();
    let eligible = ctx.eligible();
    let mut total = 0usize;

    if let Some(reward_series) = rewards {
        for (idx, include) in mask.iter().enumerate() {
            if !*include || idx >= target.len() {
                continue;
            }
            if let Some(eligible_mask) = eligible {
                if idx < eligible_mask.len() && !eligible_mask[idx] {
                    continue;
                }
            }
            let rr_value = match reward_series.get(idx).copied() {
                Some(value) if value.is_finite() => value,
                _ => continue,
            };
            let _ = rr_value; // value is only used to validate finiteness
            total += 1;
        }
    } else {
        for (idx, include) in mask.iter().enumerate() {
            if !*include || idx >= target.len() {
                continue;
            }
            if let Some(eligible_mask) = eligible {
                if idx < eligible_mask.len() && !eligible_mask[idx] {
                    continue;
                }
            }
            if idx < target.len() {
                total += 1;
            }
        }
    }
    total
}

struct SamplingSummary {
    boolean_total: usize,
    comparison_total: usize,
    boolean_dropped: usize,
    comparison_dropped: usize,
    threshold_total: usize,
    threshold_dropped: usize,
    pair_total: usize,
    pair_dropped: usize,
}

impl SamplingSummary {
    fn boolean_eligible(&self) -> usize {
        self.boolean_total.saturating_sub(self.boolean_dropped)
    }
    fn threshold_eligible(&self) -> usize {
        self.threshold_total.saturating_sub(self.threshold_dropped)
    }
    fn pair_eligible(&self) -> usize {
        self.pair_total.saturating_sub(self.pair_dropped)
    }
}

fn compute_sampling_summary(
    features: &[FeatureDescriptor],
    ctx: &EvaluationContext,
    min_sample_size: usize,
) -> SamplingSummary {
    let mut summary = SamplingSummary {
        boolean_total: 0,
        comparison_total: 0,
        boolean_dropped: 0,
        comparison_dropped: 0,
        threshold_total: 0,
        threshold_dropped: 0,
        pair_total: 0,
        pair_dropped: 0,
    };

    for descriptor in features {
        match descriptor.category {
            FeatureCategory::Boolean => {
                summary.boolean_total += 1;
                if let Ok(mask) = ctx.feature_mask(descriptor.name.as_str()) {
                    let sample = sample_size_for_mask(ctx, &mask);
                    if sample < min_sample_size {
                        summary.boolean_dropped += 1;
                    }
                }
            }
            FeatureCategory::FeatureVsConstant | FeatureCategory::FeatureVsFeature => {
                summary.comparison_total += 1;
                let is_pair = ctx.is_feature_pair(descriptor.name.as_str());
                if is_pair {
                    summary.pair_total += 1;
                } else {
                    summary.threshold_total += 1;
                }
                if let Ok(mask) = ctx.feature_mask(descriptor.name.as_str()) {
                    let sample = sample_size_for_mask(ctx, &mask);
                    if sample < min_sample_size {
                        summary.comparison_dropped += 1;
                        if is_pair {
                            summary.pair_dropped += 1;
                        } else {
                            summary.threshold_dropped += 1;
                        }
                    }
                }
            }
            FeatureCategory::Continuous => {}
        }
    }
    summary
}

struct ComparisonPruningSummary {
    constants_dropped: usize,
    under_min_dropped: usize,
    duplicates_dropped: usize,
    dead_families: usize,
    constant_examples: Vec<String>,
    under_min_examples: Vec<String>,
    duplicate_examples: Vec<String>,
    dead_family_examples: Vec<String>,
}

impl ComparisonPruningSummary {
    fn total_dropped(&self) -> usize {
        self.constants_dropped + self.under_min_dropped + self.duplicates_dropped
    }

    fn log_summary(&self) {
        let mut buffer = String::new();
        let _ = writeln!(buffer, "♻️  Comparison catalog pruning:");
        if self.constants_dropped > 0 {
            let _ = writeln!(
                buffer,
                "   - Dropped {} constant predicates (always true/false for this dataset)",
                format_int(self.constants_dropped as u64)
            );
            if !self.constant_examples.is_empty() {
                let _ = writeln!(
                    buffer,
                    "     Constant examples (up to {}):",
                    self.constant_examples.len()
                );
                for name in &self.constant_examples {
                    let _ = writeln!(buffer, "       - {name}");
                }
            }
        }
        if self.under_min_dropped > 0 {
            let _ = writeln!(
                buffer,
                "   - Dropped {} predicates below min_samples in strict mode",
                format_int(self.under_min_dropped as u64)
            );
            if !self.under_min_examples.is_empty() {
                let _ = writeln!(
                    buffer,
                    "     Under-min examples (up to {}):",
                    self.under_min_examples.len()
                );
                for name in &self.under_min_examples {
                    let _ = writeln!(buffer, "       - {name}");
                }
            }
        }
        if self.duplicates_dropped > 0 {
            let _ = writeln!(
                buffer,
                "   - Dropped {} duplicate predicates with identical masks",
                format_int(self.duplicates_dropped as u64)
            );
            if !self.duplicate_examples.is_empty() {
                let _ = writeln!(
                    buffer,
                    "     Duplicate examples (up to {}):",
                    self.duplicate_examples.len()
                );
                for name in &self.duplicate_examples {
                    let _ = writeln!(buffer, "       - {name}");
                }
            }
        }
        if self.dead_families > 0 {
            let _ = writeln!(
                buffer,
                "   - {} base features have no remaining scalar thresholds (all constant/under-min)",
                format_int(self.dead_families as u64)
            );
            if !self.dead_family_examples.is_empty() {
                let _ = writeln!(
                    buffer,
                    "     Base features with no remaining thresholds (up to {}):",
                    self.dead_family_examples.len()
                );
                for name in &self.dead_family_examples {
                    let _ = writeln!(buffer, "       - {name}");
                }
            }
        }
        info!("{}", buffer);
    }
}

fn prune_comparison_features(
    features: &[FeatureDescriptor],
    ctx: &EvaluationContext,
    config: &Config,
    comparison_specs: &HashMap<String, ComparisonSpec>,
) -> (Vec<FeatureDescriptor>, ComparisonPruningSummary) {
    #[derive(Default)]
    struct FamilyStats {
        total: usize,
        keepable: usize,
    }

    let mut kept: Vec<FeatureDescriptor> = Vec::with_capacity(features.len());
    let mut mask_index: HashMap<Vec<bool>, String> = HashMap::new();
    let mut families: HashMap<String, FamilyStats> = HashMap::new();

    let mut constant_examples: Vec<String> = Vec::new();
    let mut under_min_examples: Vec<String> = Vec::new();
    let mut duplicate_examples: Vec<String> = Vec::new();

    let mut constants_dropped = 0usize;
    let mut under_min_dropped = 0usize;
    let mut duplicates_dropped = 0usize;

    for descriptor in features {
        match descriptor.category {
            FeatureCategory::Boolean => {
                // Apply the same min-sample gating to boolean features as to
                // scalar predicates: drop flags that are non-empty but cannot
                // ever meet the storage floor on their own.
                let name = descriptor.name.as_str();
                let mask_arc = match ctx.feature_mask(name) {
                    Ok(mask) => mask,
                    Err(_) => {
                        kept.push(descriptor.clone());
                        continue;
                    }
                };
                let mask: Vec<bool> = mask_arc.as_ref().clone();
                if mask.is_empty() {
                    kept.push(descriptor.clone());
                    continue;
                }
                let sample = sample_size_for_mask(ctx, &mask);
                if sample < config.min_sample_size {
                    // Under-min boolean features are pruned from the catalog
                    // so they do not participate in any higher-depth
                    // combinations.
                    continue;
                }
                kept.push(descriptor.clone());
            }
            FeatureCategory::Continuous => kept.push(descriptor.clone()),
            FeatureCategory::FeatureVsConstant | FeatureCategory::FeatureVsFeature => {
                let name = descriptor.name.as_str();
                let spec = match comparison_specs.get(name) {
                    Some(spec) => spec,
                    None => {
                        kept.push(descriptor.clone());
                        continue;
                    }
                };

                let mask_arc = match ctx.feature_mask(name) {
                    Ok(mask) => mask,
                    Err(_) => {
                        kept.push(descriptor.clone());
                        continue;
                    }
                };
                let mask: Vec<bool> = mask_arc.as_ref().clone();
                if mask.is_empty() {
                    kept.push(descriptor.clone());
                    continue;
                }

                let all_true = mask.iter().all(|v| *v);
                let all_false = mask.iter().all(|v| !*v);
                let sample = sample_size_for_mask(ctx, &mask);

                let is_constant = all_true || all_false;
                let is_pair = spec.rhs_feature.is_some();

                if !is_pair {
                    let entry = families.entry(spec.base_feature.clone()).or_default();
                    entry.total += 1;
                    if !is_constant && sample >= config.min_sample_size {
                        entry.keepable += 1;
                    }
                }

                if is_constant {
                    constants_dropped += 1;
                    constant_examples.push(name.to_string());
                    continue;
                }

                if sample < config.min_sample_size {
                    under_min_dropped += 1;
                    under_min_examples.push(name.to_string());
                    continue;
                }

                if let Some(canonical) = mask_index.get(&mask) {
                    duplicates_dropped += 1;
                    duplicate_examples.push(format!("{name} (canonical: {canonical})"));
                    continue;
                } else {
                    mask_index.insert(mask, name.to_string());
                }

                kept.push(descriptor.clone());
            }
        }
    }

    let mut dead_families = 0usize;
    let mut dead_family_examples: Vec<String> = Vec::new();
    for (base, stats) in families.iter() {
        if stats.total > 0 && stats.keepable == 0 {
            dead_families += 1;
            dead_family_examples.push(base.clone());
        }
    }

    (
        kept,
        ComparisonPruningSummary {
            constants_dropped,
            under_min_dropped,
            duplicates_dropped,
            dead_families,
            constant_examples,
            under_min_examples,
            duplicate_examples,
            dead_family_examples,
        },
    )
}

fn log_analysis_overview(features: &[FeatureDescriptor], ctx: &EvaluationContext, config: &Config) {
    let sampling = compute_sampling_summary(features, ctx, config.min_sample_size);

    info!("Preparing feature lists...");
    info!("Boolean features (catalog): {}", sampling.boolean_total);
    info!(
        "Feature vs constant conditions (catalog): {}",
        sampling.threshold_total
    );
    info!(
        "Feature vs feature predicates (catalog): {}",
        sampling.pair_total
    );
    if sampling.boolean_dropped > 0 {
        info!(
            "   🔍 Dropped {} boolean features below min sample size",
            sampling.boolean_dropped
        );
    }
    if sampling.threshold_dropped > 0 {
        info!(
            "   🔍 Dropped {} feature-vs-constant conditions below min sample size",
            sampling.threshold_dropped
        );
    }
    if sampling.pair_dropped > 0 {
        info!(
            "   🔍 Dropped {} feature-vs-feature predicates below min sample size",
            sampling.pair_dropped
        );
    }

    let mut buffer = String::new();
    let _ = writeln!(buffer, "\n🔬 Mixed Features Analysis");
    let _ = writeln!(
        buffer,
        "   Boolean features (eligible): {}",
        sampling.boolean_eligible()
    );
    let _ = writeln!(
        buffer,
        "   Feature vs constant (eligible): {}",
        sampling.threshold_eligible()
    );
    let _ = writeln!(
        buffer,
        "   Feature vs feature (eligible): {}",
        sampling.pair_eligible()
    );
    let _ = writeln!(buffer, "   Max depth: {}", config.max_depth);
    let _ = writeln!(
        buffer,
        "   Min samples (eligibility): {}",
        config.min_sample_size
    );
    let _ = writeln!(buffer, "   Logic mode: {:?}", config.logic_mode);
    let _ = writeln!(buffer, "   Reporting mode: {:?}", config.report_metrics);
    info!("{}", buffer);
}

fn log_top_results(
    rows: &[ResultRow],
    dataset_rows: usize,
    position_sizing: PositionSizingMode,
    dollars_per_r: Option<f64>,
    cost_per_trade_r: Option<f64>,
    cost_per_trade_dollar: Option<f64>,
) {
    if rows.is_empty() {
        info!("No cumulative permutation results available yet");
        return;
    }

    let mut buffer = String::new();
    let direction = rows
        .first()
        .map(|row| row.direction.to_uppercase())
        .unwrap_or_else(|| "N/A".to_string());
    let _ = writeln!(
        buffer,
        "\n📈 TOP {} COMBINATIONS ({}) - Sorted by EQUITY CALMAR:",
        format_int(rows.len() as u128),
        direction
    );
    let _ = writeln!(
        buffer,
        "======================================================================\n"
    );

    let dollars_per_r = dollars_per_r.unwrap_or(0.0);
    let cost_per_trade_r = cost_per_trade_r.unwrap_or(0.0);
    let has_dollar_model = dollars_per_r > 0.0;
    let cost_per_trade_dollar = cost_per_trade_dollar.unwrap_or(0.0);

    for (idx, row) in rows.iter().enumerate() {
        let win_rate_pct = row.win_rate;
        let label_hit_rate_pct = row.label_hit_rate;
        let largest_loss_abs = row.largest_loss;
        let mask_hits = if row.mask_hits > 0 {
            row.mask_hits
        } else {
            row.total_bars
        };
        let (matched_bars, dataset_bars, coverage_pct) = if dataset_rows > 0 {
            let pct = (mask_hits as f64 / dataset_rows as f64) * 100.0;
            (mask_hits, dataset_rows, pct)
        } else {
            (mask_hits, 0, 0.0)
        };
        let _ = writeln!(buffer, "Rank {}: {}", idx + 1, row.combination);
        let _ = writeln!(
            buffer,
            "  Offset: {}",
            format_int(row.resume_offset as u128)
        );
        let _ = writeln!(
            buffer,
            "  Bars matching combo mask: {} ({:.2}% of dataset)",
            format_int(matched_bars as u128),
            coverage_pct
        );
        let _ = writeln!(
            buffer,
            "  Trades (eligible & finite RR): {}",
            format_int(row.total_bars as u128)
        );
        let _ = writeln!(
            buffer,
            "  Win Rate: {win_rate_pct:.2}% ({}/{} bars)",
            format_int(row.profitable_bars as u128),
            format_int(row.total_bars as u128)
        );
        let _ = writeln!(
            buffer,
            "  Target hit-rate: {label_hit_rate_pct:.2}% ({}/{} bars)",
            format_int(row.label_hits as u128),
            format_int(row.total_bars as u128)
        );
        let _ = writeln!(
            buffer,
            "  Expectancy: {:.3}R | Avg win: {:.3}R | Avg loss: {:.3}R",
            row.expectancy, row.avg_winning_rr, row.avg_losing_rr
        );
        let _ = writeln!(
            buffer,
            "  Total R: {:.1}R | Max DD: {:.1}R | Profit factor: {:.3}",
            row.total_return, row.max_drawdown, row.profit_factor
        );
        let _ = writeln!(
            buffer,
            "  R-dist: median {:.3}R | p05 {:.3}R | p95 {:.3}R | avg loss {:.3}R",
            row.median_rr, row.p05_rr, row.p95_rr, row.avg_losing_rr
        );
        match position_sizing {
            PositionSizingMode::Fractional => {
                if has_dollar_model && cost_per_trade_r > 0.0 {
                    let cost_dollar = cost_per_trade_r * dollars_per_r;
                    let _ = writeln!(
                        buffer,
                        "  Cost model: {:.3}R/trade (~${:.2})",
                        cost_per_trade_r, cost_dollar
                    );
                }
            }
            PositionSizingMode::Contracts => {
                if cost_per_trade_dollar > 0.0 {
                    let _ = writeln!(
                        buffer,
                        "  Cost model: ${:.2}/contract round-trip",
                        cost_per_trade_dollar
                    );
                }
            }
        }
        if row.final_capital > 0.0 && row.total_return_pct != 0.0 {
            let final_capital_str = format_int(row.final_capital.round() as u128);
            let _ = writeln!(
                buffer,
                "  Equity: Final ${} | Total {:.1}% | CAGR {:.2}%",
                final_capital_str, row.total_return_pct, row.cagr_pct
            );
            let _ = writeln!(
                buffer,
                "  Equity DD: Max {:.1}% | Calmar (equity): {:.2}",
                row.max_drawdown_pct_equity, row.calmar_equity
            );
            let _ = writeln!(
                buffer,
                "  Equity Sharpe/Sortino: {:.2} / {:.2}",
                row.sharpe_equity, row.sortino_equity
            );
        }
        let _ = writeln!(buffer, "  Win/Loss: {:.2}", row.win_loss_ratio);
        let _ = writeln!(
            buffer,
            "  Drawdown shape: Pain {:.2} | Ulcer {:.2}",
            row.pain_ratio, row.ulcer_index
        );
        let _ = writeln!(
            buffer,
            "  Recall: {} / {} bars ({:.2}% of dataset)",
            format_int(matched_bars as u128),
            format_int(dataset_bars as u128),
            coverage_pct
        );
        let trades_per_1000 = if dataset_bars > 0 {
            (row.total_bars as f64 * 1000.0) / dataset_bars as f64
        } else {
            0.0
        };
        let _ = writeln!(buffer, "  Density: {:.2} trades/1000 bars", trades_per_1000);
        let _ = writeln!(
            buffer,
            "  Streaks W/L: {}/{} (avg {:.2}/{:.2}) | Largest Win/Loss: {:.2}R / {:.2}R",
            format_int(row.max_consecutive_wins as u128),
            format_int(row.max_consecutive_losses as u128),
            row.avg_win_streak,
            row.avg_loss_streak,
            row.largest_win,
            largest_loss_abs
        );
        if idx + 1 < rows.len() {
            let _ = writeln!(buffer);
        }
    }

    let _ = writeln!(
        buffer,
        "\n======================================================================"
    );
    info!("{}", buffer);
}

fn log_top_formulas(rows: &[ResultRow]) {
    if rows.is_empty() {
        info!("No cumulative permutation results available yet");
        return;
    }

    let mut buffer = String::new();
    let direction = rows
        .first()
        .map(|row| row.direction.to_uppercase())
        .unwrap_or_else(|| "N/A".to_string());
    let _ = writeln!(
        buffer,
        "\n📈 TOP {} FORMULAS ({}) - Sorted by CALMAR RATIO:",
        rows.len(),
        direction
    );
    let _ = writeln!(
        buffer,
        "======================================================================\n"
    );

    for (idx, row) in rows.iter().enumerate() {
        let _ = writeln!(buffer, "Rank {}: {}", idx + 1, row.combination);
    }

    let _ = writeln!(
        buffer,
        "\n======================================================================"
    );
    info!("{}", buffer);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Direction, EvalProfileMode, LogicMode, ReportMetricsMode, StackingMode};
    use crate::feature::{FeatureCategory, FeatureDescriptor};
    use std::path::PathBuf;
    use tempfile::tempdir;

    // --- BatchTuner tests -------------------------------------------------

    fn snapshot(filter_ms: u64, eval_ms: u64) -> BatchTimingSnapshot {
        BatchTimingSnapshot {
            enumeration_ms: 10,
            filter_ms,
            eval_ms,
            ingest_ms: 0,
            prune_subset_ms: filter_ms,
            prune_struct_ms: 0,
        }
    }

    #[test]
    fn batch_tuner_keeps_batch_when_no_history() {
        let tuner = BatchTuner::new(50_000);
        let recommended = tuner.recommend(50_000, &[]);
        assert_eq!(
            recommended, 50_000,
            "without history, tuner should keep the current batch size"
        );
    }

    #[test]
    fn batch_tuner_grows_in_reuse_heavy_region() {
        let tuner = BatchTuner::new(50_000);
        let snaps = vec![snapshot(10, 0), snapshot(12, 0), snapshot(8, 0)];
        let recommended = tuner.recommend(50_000, &snaps);
        assert!(
            recommended > 50_000,
            "cheap reuse region should allow the batch size to grow"
        );
    }

    #[test]
    fn batch_tuner_shrinks_when_filter_dominates() {
        let tuner = BatchTuner::new(100_000);
        let snaps = vec![
            snapshot(25_000, 15_000),
            snapshot(28_000, 12_000),
            snapshot(26_000, 14_000),
        ];
        // Simulate a grown batch size; tuner should recommend shrinking
        // towards the floor (initial batch size) but never below it.
        let recommended = tuner.recommend(200_000, &snaps);
        assert!(
            recommended < 200_000,
            "filter-bound batches should trigger a shrink recommendation"
        );
        assert!(
            recommended >= 100_000,
            "shrink recommendation should not go below the configured floor (initial batch size)"
        );
    }

    #[test]
    fn batch_tuner_grows_when_both_filter_and_eval_are_cheap() {
        let tuner = BatchTuner::new(20_000);
        let snaps = vec![snapshot(100, 200), snapshot(80, 150), snapshot(90, 100)];
        let recommended = tuner.recommend(20_000, &snaps);
        assert!(
            recommended > 20_000,
            "balanced cheap region should allow the batch size to grow"
        );
    }

    #[test]
    fn batch_tuner_respects_min_and_max_bounds() {
        let tuner = BatchTuner::new(10_000);
        // Strong filter pressure tries to shrink aggressively, but the tuner must
        // not go below its internal min_batch.
        let snaps = vec![
            snapshot(5_000, 10),
            snapshot(4_800, 20),
            snapshot(4_900, 15),
        ];
        let recommended = tuner.recommend(10_000, &snaps);
        assert!(
            recommended >= tuner.min_batch,
            "recommended batch size should never fall below the configured min_batch"
        );

        let tuner_large = BatchTuner::new(300_000);
        let snaps_large = vec![snapshot(10, 0), snapshot(12, 0), snapshot(8, 0)];
        let recommended_large = tuner_large.recommend(300_000, &snaps_large);
        assert!(
            recommended_large <= tuner_large.max_batch,
            "recommended batch size should never exceed the configured max_batch"
        );
    }

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
    fn required_columns_preserve_target_eligibility_masks_when_present() -> Result<()> {
        let dir = tempdir()?;
        let csv_path = dir.path().join("eligible_prune.csv");
        std::fs::write(
            &csv_path,
            "timestamp,highlow_or_atr,rr_highlow_or_atr,highlow_or_atr_eligible,highlow_or_atr_eligible_long,highlow_or_atr_eligible_short,extra\n\
             2024-01-01T00:00:00Z,true,1.0,true,true,false,1\n\
             2024-01-02T00:00:00Z,false,0.0,false,false,true,2\n",
        )?;

        let data = ColumnarData::load(&csv_path)?;
        let config = dummy_config("highlow_or_atr", Direction::Long);
        let feature_pools = FeaturePools::new(Vec::<FeatureDescriptor>::new());
        let comparison_specs: HashMap<String, ComparisonSpec> = HashMap::new();

        let keep = required_columns_for_catalog(
            &config,
            &data,
            &feature_pools,
            &comparison_specs,
            Some("rr_highlow_or_atr"),
        );
        let pruned = data.prune_to_columns(&keep)?;

        assert!(pruned.has_column("highlow_or_atr"));
        assert!(pruned.has_column("rr_highlow_or_atr"));
        assert!(pruned.has_column("highlow_or_atr_eligible"));
        assert!(pruned.has_column("highlow_or_atr_eligible_long"));
        assert!(pruned.has_column("highlow_or_atr_eligible_short"));
        assert!(
            !pruned.has_column("extra"),
            "unrequired columns should be pruned away"
        );
        Ok(())
    }

    #[test]
    fn summarize_features_splits_scalar_and_pair_comparisons() {
        // One boolean feature.
        let bool_feature = FeatureDescriptor {
            name: "is_green".to_string(),
            category: FeatureCategory::Boolean,
            note: "test".to_string(),
        };
        // Scalar comparison: rsi_14 > 20.
        let scalar_name = "rsi_14>20".to_string();
        let scalar_desc = FeatureDescriptor {
            name: scalar_name.clone(),
            category: FeatureCategory::FeatureVsConstant,
            note: "scalar".to_string(),
        };
        // Pair comparison: 9ema > 200sma.
        let pair_name = "9ema>200sma".to_string();
        let pair_desc = FeatureDescriptor {
            name: pair_name.clone(),
            category: FeatureCategory::FeatureVsFeature,
            note: "pair".to_string(),
        };
        let features = vec![bool_feature, scalar_desc, pair_desc];
        let stats = summarize_features(&features);

        assert_eq!(stats.boolean, 1);
        assert_eq!(stats.scalar_comparisons, 1);
        assert_eq!(stats.pair_comparisons, 1);
        assert_eq!(stats.total, 3);
    }

    #[test]
    fn require_any_features_mask_gates_combinations() {
        let required_mask = [false, true, false, false];
        let includes_required = vec![0usize, 1usize];
        let excludes_required = vec![0usize, 2usize, 3usize];

        let has_any = |indices: &[usize]| {
            indices
                .iter()
                .any(|&idx| idx < required_mask.len() && required_mask[idx])
        };

        assert!(has_any(&includes_required));
        assert!(!has_any(&excludes_required));
    }

    #[test]
    fn log_top_results_handles_sample_percentage() {
        let rows = vec![ResultRow {
            direction: "long".to_string(),
            target: "is_green".to_string(),
            combination: "a && b".to_string(),
            resume_offset: 0,
            depth: 2,
            mask_hits: 100,
            total_bars: 100,
            profitable_bars: 60,
            win_rate: 60.0,
            label_hit_rate: 65.0,
            label_hits: 65,
            label_misses: 35,
            expectancy: 0.1,
            total_return: 10.0,
            max_drawdown: 5.0,
            profit_factor: 1.5,
            calmar_ratio: 2.0,
            win_loss_ratio: 1.8,
            ulcer_index: 10.0,
            pain_ratio: 5.0,
            max_consecutive_wins: 3,
            max_consecutive_losses: 2,
            avg_winning_rr: 1.0,
            avg_win_streak: 2.0,
            avg_loss_streak: 1.0,
            median_rr: 0.1,
            avg_losing_rr: -0.5,
            p05_rr: -1.0,
            p95_rr: 2.0,
            largest_win: 2.0,
            largest_loss: 1.0,
            final_capital: 0.0,
            total_return_pct: 0.0,
            cagr_pct: 0.0,
            max_drawdown_pct_equity: 0.0,
            calmar_equity: 0.0,
            sharpe_equity: 0.0,
            sortino_equity: 0.0,
        }];

        // Just ensure this does not panic and can be called with a non-zero dataset size.
        log_top_results(
            &rows,
            1_000,
            PositionSizingMode::Fractional,
            None,
            None,
            None,
        );
    }
}
