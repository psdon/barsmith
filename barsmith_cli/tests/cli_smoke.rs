use std::path::{Path, PathBuf};
use std::process::Command;

use tempfile::tempdir;

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root")
        .to_path_buf()
}

#[test]
fn cli_runs_on_sample_dataset() {
    let sample_csv = workspace_root()
        .join("tests")
        .join("data")
        .join("ohlcv_tiny.csv");
    assert!(
        sample_csv.exists(),
        "sample CSV missing at {}",
        sample_csv.display()
    );

    let temp_dir = tempdir().expect("temp output dir");
    let output_dir = temp_dir.path().join("barsmith_output");

    let mut cmd = if let Some(bin) = option_env!("CARGO_BIN_EXE_barsmith_cli") {
        Command::new(bin)
    } else {
        let mut cmd = Command::new("cargo");
        cmd.args(["run", "-p", "barsmith_cli", "--"]);
        cmd
    };

    let status = cmd
        .args([
            "comb",
            "--csv",
            sample_csv.to_str().expect("sample"),
            "--direction",
            "long",
            "--target",
            "next_bar_color_and_wicks",
            "--position-sizing",
            "fractional",
            "--output-dir",
            output_dir.to_str().expect("output"),
            "--max-depth",
            "2",
            "--min-samples",
            "25",
            "--logic",
            "and",
            "--batch-size",
            "25",
            "--workers",
            "1",
            "--max-combos",
            "10",
            "--dry-run",
        ])
        .current_dir(workspace_root())
        .status()
        .expect("failed to spawn barsmith_cli");

    assert!(status.success(), "barsmith_cli exited with {status:?}");

    let prepared_csv = output_dir.join("barsmith_prepared.csv");
    assert!(
        prepared_csv.exists(),
        "expected engineered CSV at {}",
        prepared_csv.display()
    );
}
