use std::collections::HashSet;
use std::fs::{self, File};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, anyhow};
use chrono::Utc;
use duckdb::{Connection, params};
use polars::io::parquet::{ParquetCompression, ParquetWriter};
use polars::prelude::*;
use sha2::{Digest, Sha256};
use tracing::{info, warn};

use crate::config::{Config, Direction};
use crate::stats::StatSummary;

/// Logical namespace used for resume metadata.
///
/// Older versions of Barsmith used a stable `config_hash` derived from the
/// full `Config` (including `catalog_hash`) plus a separate `csv_hash` to
/// scope resume offsets. Newer builds key resume behaviour purely on the
/// CSV fingerprint and collapse any legacy `(config_hash, csv_hash)` rows
/// into a single per‑CSV entry under this namespace.
const CSV_RESUME_CONFIG_NAMESPACE: &str = "csv_only";

#[derive(Debug, Clone)]
pub struct ResultRow {
    pub direction: String,
    pub target: String,
    pub combination: String,
    pub resume_offset: u64,
    pub depth: u32,
    /// Raw combo-mask support (used for recall reporting). This is populated
    /// during full-stat recomputation; when absent, callers may fall back to
    /// `total_bars`.
    pub mask_hits: u64,
    pub total_bars: u64,
    pub profitable_bars: u64,
    pub win_rate: f64,
    pub label_hit_rate: f64,
    pub label_hits: u64,
    pub label_misses: u64,
    pub expectancy: f64,
    pub total_return: f64,
    pub max_drawdown: f64,
    pub profit_factor: f64,
    pub calmar_ratio: f64,
    pub win_loss_ratio: f64,
    pub ulcer_index: f64,
    pub pain_ratio: f64,
    pub max_consecutive_wins: u64,
    pub max_consecutive_losses: u64,
    pub avg_winning_rr: f64,
    pub avg_win_streak: f64,
    pub avg_loss_streak: f64,
    pub median_rr: f64,
    pub avg_losing_rr: f64,
    pub p05_rr: f64,
    pub p95_rr: f64,
    pub largest_win: f64,
    pub largest_loss: f64,
    pub final_capital: f64,
    pub total_return_pct: f64,
    pub cagr_pct: f64,
    pub max_drawdown_pct_equity: f64,
    pub calmar_equity: f64,
    pub sharpe_equity: f64,
    pub sortino_equity: f64,
}

pub struct CumulativeStore {
    results_dir: PathBuf,
    duckdb_conn: Connection,
    config_hash: String,
    csv_hash: String,
    direction: String,
    target: String,
    batch_counter: usize,
    // Snapshot of config knobs that are useful for metadata/debugging.
    min_sample_size: usize,
    strict_min_pruning: bool,
}

impl CumulativeStore {
    pub fn new(config: &Config) -> Result<(Self, u64)> {
        fs::create_dir_all(&config.output_dir)?;

        let results_dir = config.output_dir.join("results_parquet");
        let duckdb_path = config.output_dir.join("cumulative.duckdb");

        // When force_recompute is enabled, wipe any existing cumulative state
        // for this output directory so the run behaves like a fresh start:
        // - remove the DuckDB catalog file
        // - clear existing Parquet batch files
        if config.force_recompute {
            if duckdb_path.exists() {
                let _ = fs::remove_file(&duckdb_path);
            }
            if results_dir.exists() {
                for entry in fs::read_dir(&results_dir)? {
                    let entry = entry?;
                    let path = entry.path();
                    if path.is_file() {
                        let _ = fs::remove_file(path);
                    }
                }
            }
        }

        fs::create_dir_all(&results_dir)?;

        let conn = Connection::open(&duckdb_path)
            .with_context(|| format!("Unable to open {}", duckdb_path.display()))?;

        info!(
            db_path = %duckdb_path.display(),
            "CumulativeStore::new opened DuckDB connection"
        );

        // Loosen DuckDB's default expression depth guard to handle the large
        // UNION view over many Parquet parts and complex predicates. This does
        // not change query semantics, only how deep an expression tree DuckDB
        // will accept before bailing out during parsing/optimization.
        conn.execute("SET max_expression_depth TO 100000", [])?;
        // Allow the DuckDB catalog backing the cumulative store to use up to
        // 30 GiB of memory for query execution. This helps avoid per-query
        // OOMs when scanning large Parquet-backed views, at the cost of a
        // higher process-level memory ceiling.
        conn.execute("SET memory_limit TO '30GB'", [])?;

        conn.execute(
            "CREATE TABLE IF NOT EXISTS metadata (
                config_hash TEXT NOT NULL,
                csv_hash TEXT NOT NULL,
                processed BIGINT NOT NULL,
                last_updated TIMESTAMP NOT NULL,
                PRIMARY KEY (config_hash, csv_hash)
            )",
            [],
        )?;

        // Ensure newer debug columns exist even for metadata tables created
        // by older builds. These are used for observability and do not
        // participate in the primary key.
        conn.execute(
            "ALTER TABLE metadata ADD COLUMN IF NOT EXISTS min_sample_size INTEGER",
            [],
        )?;
        conn.execute(
            "ALTER TABLE metadata ADD COLUMN IF NOT EXISTS strict_min_pruning BOOLEAN",
            [],
        )?;

        let csv_path = config.source_csv.as_ref().unwrap_or(&config.input_csv);
        let csv_hash = csv_fingerprint(csv_path)?;

        // Enforce that a single output directory is bound to a single CSV fingerprint
        // unless the caller explicitly opts in to recomputing on a different dataset.
        enforce_csv_consistency(&conn, &csv_hash, config.force_recompute)?;

        // Migrate any legacy metadata keyed by per‑config hashes into a single
        // per‑CSV namespace so that resume offsets depend only on the input
        // CSV fingerprint.
        if !config.force_recompute {
            migrate_metadata_to_csv_namespace(
                &conn,
                &csv_hash,
                config.min_sample_size,
                config.strict_min_pruning,
            )?;
        }

        // Newer builds no longer rely on a stable per‑config hash for resume;
        // resume offsets are scoped by CSV fingerprint only.
        let config_hash = CSV_RESUME_CONFIG_NAMESPACE.to_string();

        let resume_offset = if config.force_recompute {
            0
        } else {
            query_resume_offset(&conn, &config_hash, &csv_hash)?
        };

        let batch_counter = if config.force_recompute {
            0
        } else {
            existing_batch_count(&results_dir)?
        };

        let store = Self {
            results_dir: results_dir.clone(),
            duckdb_conn: conn,
            config_hash,
            csv_hash,
            direction: format_direction(config.direction),
            target: config.target.clone(),
            batch_counter,
            min_sample_size: config.min_sample_size,
            strict_min_pruning: config.strict_min_pruning,
        };
        info!("CumulativeStore::new refreshing results view...");
        store.refresh_view()?;
        info!("CumulativeStore::new refresh_view completed");

        // If metadata indicates that some combinations were already processed but
        // there are no Parquet result parts on disk, warn so callers understand
        // that index-based resume is relying on now-missing batches (e.g., files
        // were manually removed).
        if resume_offset > 0 && !has_parquet_files(&results_dir)? {
            warn!(
                resume_offset,
                "Resume metadata reports processed combinations but no Parquet result parts were found on disk; \
                 prior batches may have been removed"
            );
        }

        Ok((store, resume_offset))
    }

    fn current_min_sample_size(&self) -> usize {
        self.min_sample_size
    }

    fn current_strict_min_pruning(&self) -> bool {
        self.strict_min_pruning
    }

    /// Expose the CSV fingerprint for this store so callers can bind
    /// membership lookups to the correct dataset.
    pub fn csv_hash(&self) -> &str {
        &self.csv_hash
    }

    pub fn ingest(&mut self, combinations: &[String], stats: &[StatSummary]) -> Result<()> {
        // This method is retained for compatibility but now assumes `combos.len() == stats.len()`
        // and that the caller has already accounted for any skipped combinations in the
        // metadata update. New code should prefer `ingest_with_enumerated`.
        let _ = self.ingest_with_enumerated(combinations, stats, combinations.len(), 0)?;
        Ok(())
    }

    /// Ingest a batch of *evaluated* combinations, while also tracking how many
    /// combinations were *enumerated* in the global stream. This allows resume
    /// offsets to remain index-based even when some combinations are skipped
    /// due to prior evaluation (combination-key reuse).
    pub fn ingest_with_enumerated(
        &mut self,
        combinations: &[String],
        stats: &[StatSummary],
        enumerated_count: usize,
        batch_start_offset: u64,
    ) -> Result<Option<PathBuf>> {
        let mut build_ms: u64 = 0;
        let mut parquet_ms: u64 = 0;
        let mut meta_ms: u64 = 0;
        let mut parquet_path: Option<PathBuf> = None;

        if !combinations.is_empty() {
            let build_start = Instant::now();
            let mut df = self.build_batch_frame(combinations, stats, batch_start_offset)?;
            build_ms = (build_start.elapsed().as_secs_f32() * 1000.0).round() as u64;

            let filename = format!("part-{:016}.parquet", self.batch_counter);
            let file_path = self.results_dir.join(filename);
            self.batch_counter += 1;
            parquet_path = Some(file_path.clone());

            let mut file = File::create(&file_path)
                .with_context(|| format!("Unable to create {}", file_path.display()))?;
            let parquet_start = Instant::now();
            ParquetWriter::new(&mut file)
                .with_compression(ParquetCompression::Zstd(None))
                .finish(&mut df)
                .context("Failed to write Parquet batch")?;
            parquet_ms = (parquet_start.elapsed().as_secs_f32() * 1000.0).round() as u64;
        }

        if enumerated_count > 0 {
            let meta_start = Instant::now();
            self.update_metadata(enumerated_count as i64)?;
            meta_ms = (meta_start.elapsed().as_secs_f32() * 1000.0).round() as u64;
        }

        let total_ms = build_ms + parquet_ms + meta_ms;
        info!(
            ingest_build_ms = %build_ms,
            ingest_parquet_ms = %parquet_ms,
            ingest_meta_ms = %meta_ms,
            ingest_total_ms = %total_ms,
            stored = %combinations.len(),
            enumerated = %enumerated_count,
            "Ingest batch timing"
        );

        Ok(parquet_path)
    }

    pub fn flush(&mut self) -> Result<()> {
        self.duckdb_conn.execute("CHECKPOINT", [])?;
        Ok(())
    }

    pub fn top_results(
        &self,
        limit: usize,
        min_sample_size: usize,
        max_drawdown: f64,
        min_calmar: Option<f64>,
    ) -> Result<Vec<ResultRow>> {
        if limit == 0 || !has_parquet_files(&self.results_dir)? {
            return Ok(Vec::new());
        }
        let sql_base = "\
            SELECT
                direction,
                target,
                combination,
                depth,
                total_bars,
                profitable_bars,
                win_rate,
                max_drawdown,
                calmar_ratio,
                resume_offset
            FROM results
            WHERE total_bars >= ?
              AND max_drawdown <= ?";

        let (sql, with_calmar) = match min_calmar {
            Some(_) => (
                format!("{sql_base} AND calmar_ratio >= ? ORDER BY calmar_ratio DESC LIMIT ?"),
                true,
            ),
            None => (
                format!("{sql_base} ORDER BY calmar_ratio DESC LIMIT ?"),
                false,
            ),
        };
        let mut stmt = self.duckdb_conn.prepare(&sql)?;
        let mut rows = if with_calmar {
            let min_calmar = min_calmar.unwrap();
            stmt.query(params![
                min_sample_size as i64,
                max_drawdown,
                min_calmar,
                limit as i64
            ])?
        } else {
            stmt.query(params![min_sample_size as i64, max_drawdown, limit as i64])?
        };

        let mut out = Vec::new();
        while let Some(row) = rows.next()? {
            let total_bars = row.get::<_, i64>(4)? as u64;
            out.push(ResultRow {
                direction: row.get(0)?,
                target: row.get(1)?,
                combination: row.get(2)?,
                depth: row.get::<_, i32>(3)? as u32,
                mask_hits: total_bars,
                total_bars,
                profitable_bars: row.get::<_, i64>(5)? as u64,
                win_rate: row.get(6)?,
                max_drawdown: row.get(7)?,
                calmar_ratio: row.get(8)?,
                resume_offset: row.get::<_, i64>(9)? as u64,
                // The remaining fields are populated later via full-stat
                // recomputation; initialize them to neutral defaults here.
                label_hit_rate: 0.0,
                label_hits: 0,
                label_misses: 0,
                expectancy: 0.0,
                total_return: 0.0,
                profit_factor: 0.0,
                win_loss_ratio: 0.0,
                ulcer_index: 0.0,
                pain_ratio: 0.0,
                max_consecutive_wins: 0,
                max_consecutive_losses: 0,
                avg_winning_rr: 0.0,
                avg_win_streak: 0.0,
                avg_loss_streak: 0.0,
                median_rr: 0.0,
                avg_losing_rr: 0.0,
                p05_rr: 0.0,
                p95_rr: 0.0,
                largest_win: 0.0,
                largest_loss: 0.0,
                final_capital: 0.0,
                total_return_pct: 0.0,
                cagr_pct: 0.0,
                max_drawdown_pct_equity: 0.0,
                calmar_equity: 0.0,
                sharpe_equity: 0.0,
                sortino_equity: 0.0,
            });
        }
        Ok(out)
    }

    /// Load the set of already-evaluated combinations for this CSV/target/direction.
    /// This is used for testing and verification purposes; cross-run deduplication
    /// in the pipeline is handled via the in-memory reuse set.
    pub fn existing_combinations(&self) -> Result<HashSet<String>> {
        let mut existing = HashSet::new();
        if !has_parquet_files(&self.results_dir)? {
            return Ok(existing);
        }

        let sql = "\
            SELECT DISTINCT combination \
            FROM results \
            WHERE csv_hash = ? \
              AND direction = ? \
              AND target = ?";

        let mut stmt = self.duckdb_conn.prepare(sql)?;
        let mut rows = stmt.query(params![&self.csv_hash, &self.direction, &self.target])?;
        while let Some(row) = rows.next()? {
            let combination: String = row.get(0)?;
            existing.insert(combination);
        }

        Ok(existing)
    }

    pub fn refresh_view(&self) -> Result<()> {
        if !has_parquet_files(&self.results_dir)? {
            // No batches yet; leave any existing view in place.
            return Ok(());
        }

        // Collect and sort candidate Parquet parts.
        let mut candidates: Vec<PathBuf> = Vec::new();
        for entry in fs::read_dir(&self.results_dir)? {
            let entry = entry?;
            let name = entry.file_name().to_string_lossy().into_owned();
            if name.starts_with("part-") && name.ends_with(".parquet") {
                candidates.push(entry.path());
            }
        }
        candidates.sort();

        let mut good_paths: Vec<String> = Vec::new();
        for path in candidates {
            let display_str = path.display().to_string();
            let escaped = display_str.replace('\'', "''");
            let probe_sql = format!("SELECT COUNT(*) FROM read_parquet('{escaped}')");
            match self.duckdb_conn.prepare(&probe_sql)?.query([]) {
                Ok(_) => good_paths.push(escaped),
                Err(error) => {
                    warn!(
                        file = %display_str,
                        ?error,
                        "Skipping corrupt or unreadable Parquet batch"
                    );
                }
            }
        }

        if good_paths.is_empty() {
            // All parts are unreadable; drop the view so downstream queries fail fast.
            self.duckdb_conn
                .execute("DROP VIEW IF EXISTS results", [])?;
            return Ok(());
        }

        let mut union_sql = String::new();
        for (idx, path) in good_paths.iter().enumerate() {
            if idx > 0 {
                union_sql.push_str(" UNION ALL ");
            }
            union_sql.push_str(&format!("SELECT * FROM read_parquet('{path}')"));
        }
        let view_sql = format!("CREATE OR REPLACE VIEW results AS {union_sql}");
        self.duckdb_conn.execute(&view_sql, [])?;
        Ok(())
    }

    fn update_metadata(&mut self, delta: i64) -> Result<()> {
        self.duckdb_conn.execute(
            "INSERT INTO metadata (config_hash, csv_hash, processed, last_updated, min_sample_size, strict_min_pruning)
             VALUES (?, ?, ?, CURRENT_TIMESTAMP, ?, ?)
             ON CONFLICT(config_hash, csv_hash) DO UPDATE
             SET processed = metadata.processed + excluded.processed,
                 last_updated = excluded.last_updated,
                 min_sample_size = excluded.min_sample_size,
                 strict_min_pruning = excluded.strict_min_pruning",
            params![
                &self.config_hash,
                &self.csv_hash,
                delta,
                self.current_min_sample_size() as i64,
                self.current_strict_min_pruning()
            ],
        )?;
        Ok(())
    }

    fn build_batch_frame(
        &self,
        combinations: &[String],
        stats: &[StatSummary],
        batch_start_offset: u64,
    ) -> Result<DataFrame> {
        let count = combinations.len();
        let combination_text = combinations.to_vec();

        let csv_col = vec![self.csv_hash.clone(); count];
        let dir_col = vec![self.direction.clone(); count];
        let target_col = vec![self.target.clone(); count];
        let processed_at = vec![Utc::now().format("%Y-%m-%dT%H:%M:%S%.3fZ").to_string(); count];
        let resume_offsets: Vec<u64> = (0..count)
            .map(|idx| batch_start_offset + idx as u64)
            .collect();

        let mut columns = vec![
            Series::new("csv_hash", csv_col),
            Series::new("direction", dir_col),
            Series::new("target", target_col),
            Series::new("combination", combination_text),
            Series::new("processed_at", processed_at),
            Series::new("resume_offset", resume_offsets),
        ];

        macro_rules! push_series {
            ($name:expr, $iter:expr) => {
                columns.push(Series::new($name, $iter));
            };
        }

        push_series!(
            "depth",
            stats
                .iter()
                .map(|stat| stat.depth as u32)
                .collect::<Vec<_>>()
        );
        push_series!(
            "total_bars",
            stats
                .iter()
                .map(|stat| stat.total_bars as u64)
                .collect::<Vec<_>>()
        );
        push_series!(
            "profitable_bars",
            stats
                .iter()
                .map(|stat| stat.profitable_bars as u64)
                .collect::<Vec<_>>()
        );
        push_series!(
            "win_rate",
            stats.iter().map(|stat| stat.win_rate).collect::<Vec<_>>()
        );
        push_series!(
            "max_drawdown",
            stats
                .iter()
                .map(|stat| stat.max_drawdown)
                .collect::<Vec<_>>()
        );
        push_series!(
            "calmar_ratio",
            stats
                .iter()
                .map(|stat| stat.calmar_ratio)
                .collect::<Vec<_>>()
        );

        DataFrame::new(columns).context("Failed to build batch DataFrame")
    }
}

fn existing_batch_count(results_dir: &Path) -> Result<usize> {
    let mut count = 0usize;
    if results_dir.exists() {
        for entry in fs::read_dir(results_dir)? {
            let entry = entry?;
            if entry.file_name().to_string_lossy().starts_with("part-") {
                count += 1;
            }
        }
    }
    Ok(count)
}

fn has_parquet_files(results_dir: &Path) -> Result<bool> {
    if !results_dir.exists() {
        return Ok(false);
    }
    for entry in fs::read_dir(results_dir)? {
        let entry = entry?;
        if entry.file_name().to_string_lossy().starts_with("part-") {
            return Ok(true);
        }
    }
    Ok(false)
}

fn query_resume_offset(conn: &Connection, config_hash: &str, csv_hash: &str) -> Result<u64> {
    // Resume offsets are now keyed by CSV fingerprint only. The `config_hash`
    // parameter is retained for back-compat but ignored for lookups; any
    // legacy rows are migrated into the CSV-only namespace by
    // `migrate_metadata_to_csv_namespace`.
    let _ = config_hash;
    let sql = "\
        SELECT processed \
        FROM metadata \
        WHERE csv_hash = ?";
    let mut stmt = conn.prepare(sql)?;
    let mut rows = stmt.query(params![csv_hash])?;
    if let Some(row) = rows.next()? {
        let processed: i64 = row.get(0)?;
        Ok(processed.max(0) as u64)
    } else {
        Ok(0)
    }
}

fn enforce_csv_consistency(conn: &Connection, csv_hash: &str, force: bool) -> Result<()> {
    // If there is no metadata yet, there is nothing to enforce.
    let mut stmt = conn.prepare("SELECT DISTINCT csv_hash FROM metadata")?;
    let mut rows = stmt.query([])?;
    let mut seen: Vec<String> = Vec::new();
    while let Some(row) = rows.next()? {
        let existing: String = row.get(0)?;
        if !seen.contains(&existing) {
            seen.push(existing);
        }
    }

    // Ignore legacy entries that were written before we prefixed hashes with a
    // scheme tag. This prevents old rows (which used a different fingerprinting
    // strategy) from blocking reuse for the new raw-CSV-based fingerprints.
    let relevant: Vec<&String> = if csv_hash.contains(':') {
        seen.iter().filter(|value| value.contains(':')).collect()
    } else {
        seen.iter().collect()
    };

    if relevant.is_empty() {
        return Ok(());
    }

    // If all existing hashes match the current one, we are fine.
    if relevant
        .iter()
        .all(|existing| existing.as_str() == csv_hash)
    {
        return Ok(());
    }

    if force {
        // Caller opted in to reusing this output directory despite a different CSV.
        return Ok(());
    }

    Err(anyhow!(
        "Existing cumulative metadata in this output directory was created from a different CSV. \
         Run with --force-recompute or choose a fresh --output-dir for this dataset."
    ))
}

/// Collapse any legacy `(config_hash, csv_hash)` metadata rows into a single
/// per‑CSV entry keyed by `CSV_RESUME_CONFIG_NAMESPACE`.
fn migrate_metadata_to_csv_namespace(
    conn: &Connection,
    csv_hash: &str,
    min_sample_size: usize,
    strict_min_pruning: bool,
) -> Result<()> {
    // Collect all existing rows for this CSV fingerprint.
    let mut stmt = conn.prepare(
        "SELECT config_hash, processed \
         FROM metadata \
         WHERE csv_hash = ?",
    )?;
    let mut rows = stmt.query(params![csv_hash])?;
    let mut max_processed: i64 = 0;
    let mut found = false;
    let mut already_migrated_only = true;

    while let Some(row) = rows.next()? {
        let existing_config: String = row.get(0)?;
        let processed: i64 = row.get(1)?;
        if existing_config != CSV_RESUME_CONFIG_NAMESPACE {
            already_migrated_only = false;
        }
        if !found || processed > max_processed {
            max_processed = processed;
            found = true;
        }
    }

    // No rows to migrate.
    if !found {
        return Ok(());
    }

    // If the only existing row is already in the CSV-only namespace, keep it.
    if already_migrated_only {
        return Ok(());
    }

    // Collapse all rows for this CSV hash into a single entry keyed by the
    // CSV-only namespace, preserving the maximum processed index so we never
    // regress resume offsets.
    conn.execute("DELETE FROM metadata WHERE csv_hash = ?", params![csv_hash])?;

    conn.execute(
        "INSERT INTO metadata (config_hash, csv_hash, processed, last_updated, min_sample_size, strict_min_pruning)
         VALUES (?, ?, ?, CURRENT_TIMESTAMP, ?, ?)",
        params![
            CSV_RESUME_CONFIG_NAMESPACE,
            csv_hash,
            max_processed.max(0),
            min_sample_size as i64,
            strict_min_pruning
        ],
    )?;

    Ok(())
}

fn format_direction(direction: Direction) -> String {
    match direction {
        Direction::Long => "long",
        Direction::Short => "short",
        Direction::Both => "both",
    }
    .to_string()
}

pub fn stable_config_hash(config: &Config) -> Result<String> {
    let mut normalized = config.clone();
    // Strip out fields that do not affect which combinations are evaluated
    // so that changing runtime knobs (batch size, worker count, reporting
    // options) does not force a new resume namespace.
    normalized.resume_offset = 0;
    normalized.dry_run = false;
    normalized.batch_size = 0;
    normalized.n_workers = 0;
    normalized.auto_batch = false;
    normalized.early_exit_when_reused = false;
    normalized.max_combos = None;
    normalized.quiet = false;
    normalized.report_metrics = crate::config::ReportMetricsMode::Full;
    normalized.report_top = 0;
    // min_sample_size controls which combinations are persisted to results
    // and therefore must always be part of the stable config hash.
    normalized.explicit_resume_offset = false;
    normalized.output_dir = PathBuf::from("__OUTPUT__");
    normalized.input_csv = PathBuf::from("__INPUT__");
    normalized.source_csv = Some(PathBuf::from("__SOURCE__"));
    normalized.force_recompute = false;
    normalized.s3_output = None;
    normalized.s3_upload_each_batch = false;
    let serialized = serde_json::to_vec(&normalized)?;
    Ok(hex::encode(Sha256::digest(serialized)))
}

fn csv_fingerprint(path: &Path) -> Result<String> {
    let mut file = File::open(path)
        .with_context(|| format!("Unable to open {} for fingerprinting", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 8192];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    // Prefix the hash so we can distinguish new, raw-CSV-based fingerprints
    // from legacy bare hex digests written by older builds.
    let hex = hex::encode(hasher.finalize());
    Ok(format!("raw:{}", hex))
}
