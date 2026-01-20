use std::path::Path;
use std::process::Command;
use tempfile::tempdir;

fn repo_root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root")
}

#[test]
fn cli_dry_run_prepares_dataset() {
    let repo_root = repo_root();
    let dataset = repo_root
        .join("tests")
        .join("data")
        .join("es_30m_sample.csv");
    assert!(
        dataset.exists(),
        "Expected sample dataset at {}",
        dataset.display()
    );

    let temp_dir = tempdir().expect("tempdir");
    let output_dir = temp_dir.path().join("parity_es_check");
    let cli_status = Command::new("cargo")
        .args([
            "run",
            "-p",
            "barsmith_cli",
            "--",
            "comb",
            "--csv",
            dataset.to_str().expect("dataset path"),
            "--direction",
            "long",
            "--target",
            "next_bar_color_and_wicks",
            "--position-sizing",
            "fractional",
            "--output-dir",
            output_dir.to_str().expect("output dir"),
            "--max-depth",
            "3",
            "--min-samples",
            "500",
            "--logic",
            "and",
            "--date-start",
            "2024-01-01",
            "--date-end",
            "2024-12-31",
            "--batch-size",
            "50",
            "--workers",
            "1",
            "--max-combos",
            "1",
            "--dry-run",
        ])
        .current_dir(repo_root)
        .status()
        .expect("failed to run barsmith_cli");
    assert!(
        cli_status.success(),
        "barsmith_cli dry-run failed: {cli_status:?}"
    );

    let prepared_csv = output_dir.join("barsmith_prepared.csv");
    assert!(
        prepared_csv.exists(),
        "expected barsmith_prepared.csv at {}",
        prepared_csv.display()
    );
}
