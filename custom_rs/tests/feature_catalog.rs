use std::path::PathBuf;
use std::process::Command;

#[test]
fn python_feature_catalog_matches() {
    let crate_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let repo_root = crate_dir
        .parent()
        .expect("workspace root")
        .parent()
        .expect("repo root")
        .to_path_buf();
    let script = repo_root.join("tmp").join("check_feature_sets.py");
    if !script.exists() {
        eprintln!(
            "Skipping python_feature_catalog_matches; audit script missing at {}",
            script.display()
        );
        return;
    }

    let output = Command::new("uv")
        .arg("run")
        .arg("python")
        .arg(script)
        .current_dir(&repo_root)
        .output()
        .expect("failed to spawn feature catalog audit");

    if !output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        panic!(
            "feature catalog audit failed with {}.\nSTDOUT:\n{}\nSTDERR:\n{}",
            output.status, stdout, stderr
        );
    }
}
