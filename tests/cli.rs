//! CLI smoke tests — run the actual binary on offline paths.

use std::process::Command;

fn wetware(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_wetware"))
        .args(args)
        .output()
        .expect("failed to run the wetware binary")
}

#[test]
fn sample_info() {
    let out = wetware(&["--sample", "info"]);
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("\"source\": \"synthetic sample\""),
        "stdout: {stdout}"
    );
    assert!(stdout.contains("\"neurons\": 600"), "stdout: {stdout}");
}

#[test]
fn sample_run_timeseries() {
    let out = wetware(&["--sample", "run", "timeseries"]);
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("\"task\": \"timeseries\""),
        "stdout: {stdout}"
    );
    assert!(stdout.contains("\"nrmse\""), "stdout: {stdout}");
}

#[test]
fn rejects_unknown_task() {
    let out = wetware(&["run", "bogus"]);
    assert!(!out.status.success());
}

#[test]
fn sample_flag_is_global() {
    // `--sample` after the subcommand, the way argparse accepted it
    let out = wetware(&["info", "--sample"]);
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("\"synthetic sample\""), "stdout: {stdout}");
}
