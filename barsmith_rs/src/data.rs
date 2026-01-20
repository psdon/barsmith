use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result};
use chrono::{DateTime, NaiveDate, Utc};
use polars::prelude::*;

#[derive(Debug, Clone)]
pub struct DataSetMetadata {
    column_names: Arc<Vec<String>>,
    approx_rows: usize,
}

#[derive(Clone)]
pub struct ColumnarData {
    frame: Arc<DataFrame>,
    metadata: DataSetMetadata,
}

impl ColumnarData {
    pub fn load(path: &Path) -> Result<Self> {
        let lazy = LazyCsvReader::new(path)
            .has_header(true)
            .with_try_parse_dates(true)
            .with_ignore_errors(true)
            .finish()
            .with_context(|| format!("Failed to initialize CSV reader for {}", path.display()))?;

        let df = lazy
            .collect()
            .with_context(|| format!("Failed to collect columnar data from {}", path.display()))?;

        let column_names = df
            .get_columns()
            .iter()
            .map(|series| series.name().to_string())
            .collect::<Vec<_>>();

        let metadata = DataSetMetadata {
            column_names: Arc::new(column_names),
            approx_rows: df.height(),
        };

        Ok(Self {
            frame: Arc::new(df),
            metadata,
        })
    }

    pub fn metadata(&self) -> DataSetMetadata {
        self.metadata.clone()
    }

    pub fn column_names(&self) -> &[String] {
        self.metadata.column_names.as_ref()
    }

    pub fn approx_rows(&self) -> usize {
        self.metadata.approx_rows
    }

    pub fn data_frame(&self) -> Arc<DataFrame> {
        Arc::clone(&self.frame)
    }

    pub fn has_column(&self, name: &str) -> bool {
        self.metadata.column_names.iter().any(|col| col == name)
    }

    pub fn boolean_column(&self, name: &str) -> Result<BooleanChunked> {
        self.frame
            .column(name)
            .with_context(|| format!("Missing boolean column '{name}'"))?
            .bool()
            .cloned()
            .context("Failed to interpret column as boolean")
    }

    pub fn float_column(&self, name: &str) -> Result<Float64Chunked> {
        self.frame
            .column(name)
            .with_context(|| format!("Missing float column '{name}'"))?
            .f64()
            .cloned()
            .context("Failed to interpret column as float")
    }

    pub fn i64_column(&self, name: &str) -> Result<Int64Chunked> {
        self.frame
            .column(name)
            .with_context(|| format!("Missing int column '{name}'"))?
            .i64()
            .cloned()
            .context("Failed to interpret column as i64")
    }

    /// Return a new ColumnarData containing only rows whose calendar date
    /// (derived from a timestamp/datetime column) lies within the optional
    /// [start, end] range. When both bounds are None, the original dataset
    /// is returned unchanged.
    pub fn filter_by_date_range(
        &self,
        start: Option<NaiveDate>,
        end: Option<NaiveDate>,
    ) -> Result<Self> {
        if start.is_none() && end.is_none() {
            return Ok(self.clone());
        }

        // Prefer an explicit "timestamp" column when present; otherwise fall
        // back to the first datetime-typed column in the frame.
        let df = self.data_frame();
        let frame = df.as_ref();

        let mut series_opt = if self.has_column("timestamp") {
            frame.column("timestamp").ok()
        } else {
            None
        };

        if series_opt.is_none() {
            for candidate in frame.get_columns() {
                if matches!(candidate.dtype(), DataType::Datetime(_, _)) {
                    series_opt = Some(candidate);
                    break;
                }
            }
        }

        let series = series_opt
            .with_context(|| "Missing required timestamp/datetime column for date filtering")?;

        let mut keep: Vec<bool> = Vec::with_capacity(series.len());

        match series.dtype() {
            DataType::Datetime(unit, _) => {
                let ca = series
                    .datetime()
                    .with_context(|| "Failed to interpret timestamp column as datetime")?;
                for opt_v in ca.into_iter() {
                    let ts = match opt_v {
                        Some(v) => v,
                        None => {
                            keep.push(false);
                            continue;
                        }
                    };
                    // Convert the integer timestamp into a NaiveDate via the
                    // configured time unit.
                    let (secs, nsecs) = match unit {
                        TimeUnit::Nanoseconds => {
                            let secs = ts / 1_000_000_000;
                            let nsecs = (ts % 1_000_000_000) as u32;
                            (secs, nsecs)
                        }
                        TimeUnit::Microseconds => {
                            let secs = ts / 1_000_000;
                            let nsecs = (ts % 1_000_000) as u32 * 1_000;
                            (secs, nsecs)
                        }
                        TimeUnit::Milliseconds => {
                            let secs = ts / 1_000;
                            let nsecs = (ts % 1_000) as u32 * 1_000_000;
                            (secs, nsecs)
                        }
                    };
                    let dt = match DateTime::<Utc>::from_timestamp(secs, nsecs) {
                        Some(v) => v,
                        None => {
                            keep.push(false);
                            continue;
                        }
                    };
                    let d = dt.date_naive();

                    let mut ok = true;
                    if let Some(s) = start {
                        if d < s {
                            ok = false;
                        }
                    }
                    if let Some(e) = end {
                        if d > e {
                            ok = false;
                        }
                    }
                    keep.push(ok);
                }
            }
            _ => {
                for value in series.iter() {
                    use polars::prelude::AnyValue;

                    let raw = match value {
                        AnyValue::String(s) => s,
                        AnyValue::StringOwned(ref s) => s.as_str(),
                        AnyValue::Null => {
                            keep.push(false);
                            continue;
                        }
                        other => {
                            return Err(anyhow::anyhow!(
                                "Timestamp column must be UTF-8 strings for date filtering (got {:?})",
                                other.dtype()
                            ));
                        }
                    };
                    let parsed = chrono::DateTime::parse_from_rfc3339(raw)
                        .with_context(|| format!("Failed to parse timestamp '{raw}' as RFC3339"))?;
                    let d = parsed.date_naive();

                    let mut ok = true;
                    if let Some(s) = start {
                        if d < s {
                            ok = false;
                        }
                    }
                    if let Some(e) = end {
                        if d > e {
                            ok = false;
                        }
                    }
                    keep.push(ok);
                }
            }
        }

        let mask = BooleanChunked::from_slice("date_filter", &keep);
        let mut filtered = frame
            .filter(&mask)
            .with_context(|| "Failed to filter dataframe to requested date range")?;

        // If the dataset was sliced away from the front (include_date_start),
        // remap any target-provided exit-index columns so they remain valid
        // within the filtered (0-based) row coordinates. This keeps
        // `--stacking-mode no-stacking` correct after date filtering.
        if let Some(start_offset) = keep.iter().position(|flag| *flag) {
            if start_offset > 0 {
                let offset = start_offset as i64;
                let names: Vec<String> = filtered
                    .get_column_names()
                    .iter()
                    .map(|s| s.to_string())
                    .collect();
                for name in names {
                    if !(name.ends_with("_exit_i")
                        || name.ends_with("_exit_i_long")
                        || name.ends_with("_exit_i_short"))
                    {
                        continue;
                    }
                    let Ok(col) = filtered.column(&name) else {
                        continue;
                    };
                    if !matches!(col.dtype(), DataType::Int64) {
                        continue;
                    }
                    let ca = col
                        .i64()
                        .with_context(|| format!("Failed to interpret '{name}' as i64"))?
                        .clone();
                    let adjusted = Int64Chunked::from_iter_options(
                        name.as_str(),
                        ca.into_iter().map(|opt| match opt {
                            Some(v) if v >= offset => Some(v - offset),
                            _ => None,
                        }),
                    )
                    .into_series();
                    if filtered.column(&name).is_ok() {
                        filtered = filtered
                            .drop(&name)
                            .with_context(|| format!("Failed to drop '{name}' for remapping"))?;
                    }
                    filtered.with_column(adjusted).with_context(|| {
                        format!("Failed to update remapped exit index column '{name}'")
                    })?;
                }
            }
        }

        let column_names = filtered
            .get_columns()
            .iter()
            .map(|s| s.name().to_string())
            .collect::<Vec<_>>();

        let metadata = DataSetMetadata {
            column_names: Arc::new(column_names),
            approx_rows: filtered.height(),
        };

        Ok(Self {
            frame: Arc::new(filtered),
            metadata,
        })
    }

    /// Return a new ColumnarData containing only the selected columns. This is
    /// useful for dropping unused engineered fields on large datasets to
    /// reduce memory pressure. The original ColumnarData is left unchanged.
    pub fn prune_to_columns<S: AsRef<str>>(&self, keep: &[S]) -> Result<Self> {
        if keep.is_empty() {
            return Ok(self.clone());
        }
        let names: Vec<&str> = keep.iter().map(|s| s.as_ref()).collect();
        let df = self
            .frame
            .select(&names)
            .with_context(|| "Failed to prune dataframe to selected columns")?;

        let column_names = df
            .get_columns()
            .iter()
            .map(|series| series.name().to_string())
            .collect::<Vec<_>>();

        let metadata = DataSetMetadata {
            column_names: Arc::new(column_names),
            approx_rows: df.height(),
        };

        Ok(Self {
            frame: Arc::new(df),
            metadata,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::NaiveDate;
    use tempfile::tempdir;

    #[test]
    fn prune_to_columns_keeps_requested_columns_and_rows() -> Result<()> {
        let temp_dir = tempdir()?;
        let csv_path = temp_dir.path().join("sample.csv");
        std::fs::write(
            &csv_path,
            "timestamp,a,b,c\n2024-01-01T00:00:00Z,1,2,3\n2024-01-01T00:30:00Z,4,5,6\n",
        )?;

        let original = ColumnarData::load(&csv_path)?;
        assert_eq!(original.column_names().len(), 4);
        assert_eq!(original.approx_rows(), 2);

        let pruned = original.prune_to_columns(&["timestamp", "a"])?;
        let cols = pruned.column_names();
        assert_eq!(cols.len(), 2);
        assert!(cols.contains(&"timestamp".to_string()));
        assert!(cols.contains(&"a".to_string()));
        assert_eq!(pruned.approx_rows(), 2);

        Ok(())
    }

    #[test]
    fn filter_by_date_range_retains_only_dates_within_bounds() -> Result<()> {
        let temp_dir = tempdir()?;
        let csv_path = temp_dir.path().join("sample_dates.csv");
        std::fs::write(
            &csv_path,
            "timestamp,a\n\
             2023-12-31T23:30:00Z,1\n\
             2024-01-01T00:00:00Z,2\n\
             2024-06-01T00:00:00Z,3\n\
             2025-01-01T00:00:00Z,4\n",
        )?;

        let original = ColumnarData::load(&csv_path)?;
        assert_eq!(original.approx_rows(), 4);

        let start = NaiveDate::from_ymd_opt(2024, 1, 1).unwrap();
        let end = NaiveDate::from_ymd_opt(2024, 12, 31).unwrap();
        let filtered = original.filter_by_date_range(Some(start), Some(end))?;
        assert_eq!(filtered.approx_rows(), 2);

        Ok(())
    }
}
