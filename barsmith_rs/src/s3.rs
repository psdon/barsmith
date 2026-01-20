use std::path::Path;
use std::process::Command;
use std::time::Instant;

use anyhow::{Context, Result, anyhow};

#[derive(Debug, Clone)]
pub struct S3Destination {
    base: String,
}

impl S3Destination {
    pub fn parse(value: &str) -> Result<Self> {
        let trimmed = value.trim();
        if !trimmed.starts_with("s3://") {
            return Err(anyhow!("S3 output must start with s3:// (got '{trimmed}')"));
        }
        let without = &trimmed["s3://".len()..];
        let bucket = without.split('/').next().unwrap_or("");
        if bucket.is_empty() {
            return Err(anyhow!(
                "S3 output must include a bucket name (got '{trimmed}')"
            ));
        }
        let base = trimmed.trim_end_matches('/').to_string();
        Ok(Self { base })
    }

    pub fn object_uri(&self, relative: &str) -> String {
        let rel = relative.trim_start_matches('/');
        format!("{}/{}", self.base, rel)
    }

    pub fn cp(&self, local_path: &Path, relative: &str) -> Result<u64> {
        let uri = self.object_uri(relative);
        let start = Instant::now();
        let output = Command::new("aws")
            .args(["s3", "cp"])
            .arg(local_path)
            .arg(&uri)
            .args(["--only-show-errors"])
            .output()
            .with_context(|| "Failed to spawn aws CLI (is awscli installed and on PATH?)")?;
        if !output.status.success() {
            return Err(anyhow!(
                "aws s3 cp failed (status={})\nstdout:\n{}\nstderr:\n{}",
                output.status,
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            ));
        }
        Ok((start.elapsed().as_secs_f32() * 1000.0).round() as u64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn s3_destination_parses_and_joins() -> Result<()> {
        let dest = S3Destination::parse("s3://my-bucket/some/prefix/")?;
        assert_eq!(
            dest.object_uri("results_parquet/part-0001.parquet"),
            "s3://my-bucket/some/prefix/results_parquet/part-0001.parquet"
        );
        assert_eq!(
            dest.object_uri("/cumulative.duckdb"),
            "s3://my-bucket/some/prefix/cumulative.duckdb"
        );
        Ok(())
    }

    #[test]
    fn s3_destination_rejects_missing_bucket() {
        assert!(S3Destination::parse("s3://").is_err());
        assert!(S3Destination::parse("http://bucket").is_err());
    }
}
