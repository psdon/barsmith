mod cli;
mod stats_detail;

use std::fs::OpenOptions;
use std::path::PathBuf;
use std::process::Command;

use anyhow::{Result, anyhow};
use barsmith_builtin::{BuiltinPipelineOptions, run_builtin_pipeline_with_options};
use cli::{Cli, Commands};
use tracing_appender::non_blocking;
use tracing_subscriber::{EnvFilter, prelude::*};

fn init_tracing(log_file: Option<PathBuf>) -> Result<()> {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));

    let stdout_layer = tracing_subscriber::fmt::layer().with_writer(std::io::stdout);

    if let Some(path) = log_file {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|err| anyhow!("failed to create log directory {parent:?}: {err}"))?;
        }
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(|err| anyhow!("failed to open log file {path:?}: {err}"))?;
        let (non_blocking_writer, guard) = non_blocking(file);
        // Leak the guard so the non-blocking writer stays alive for the
        // duration of the process without additional plumbing.
        let _guard = Box::leak(Box::new(guard));
        let file_layer = tracing_subscriber::fmt::layer().with_writer(non_blocking_writer);
        tracing_subscriber::registry()
            .with(filter)
            .with(stdout_layer)
            .with(file_layer)
            .try_init()
            .map_err(|err| anyhow!("failed to initialize tracing: {err}"))
    } else {
        tracing_subscriber::registry()
            .with(filter)
            .with(stdout_layer)
            .try_init()
            .map_err(|err| anyhow!("failed to initialize tracing: {err}"))
    }
}

#[derive(Debug)]
struct ProcInfo {
    ppid: i32,
    command: String,
}

fn query_process(pid: i32) -> Option<ProcInfo> {
    let output = Command::new("ps")
        .args(["-ww", "-o", "ppid=", "-o", "command=", "-p"])
        .arg(pid.to_string())
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let raw = String::from_utf8_lossy(&output.stdout);
    let line = raw.lines().next()?.trim_start();
    if line.is_empty() {
        return None;
    }

    let ppid_token = line.split_whitespace().next()?;
    let ppid: i32 = ppid_token.parse().ok()?;
    let cmd_start = line.find(ppid_token)? + ppid_token.len();
    let command = line[cmd_start..].trim().to_string();

    Some(ProcInfo { ppid, command })
}

fn command_exe(command: &str) -> Option<&str> {
    command.split_whitespace().next()
}

fn command_exe_basename(command: &str) -> Option<&str> {
    let exe = command_exe(command)?;
    Some(exe.rsplit('/').next().unwrap_or(exe))
}

fn is_caffeinate_process(command: &str) -> bool {
    command_exe_basename(command) == Some("caffeinate")
}

fn is_cargo_process(command: &str) -> bool {
    command_exe_basename(command) == Some("cargo")
}

fn is_likely_shell(command: &str) -> bool {
    let Some(exe) = command.split_whitespace().next() else {
        return false;
    };
    exe.ends_with("/zsh")
        || exe.ends_with("/bash")
        || exe.ends_with("/sh")
        || exe.ends_with("/fish")
        || exe.ends_with("/dash")
        || exe == "zsh"
        || exe == "bash"
        || exe == "sh"
        || exe == "fish"
        || exe == "dash"
}

fn collect_ancestor_processes(max_depth: usize) -> Vec<ProcInfo> {
    let mut results = Vec::new();
    let mut pid = unsafe { libc::getppid() } as i32;
    for _ in 0..max_depth {
        if pid <= 1 {
            break;
        }
        let Some(info) = query_process(pid) else {
            break;
        };
        pid = info.ppid;
        let should_stop = is_likely_shell(&info.command);
        results.push(info);
        if should_stop {
            break;
        }
    }
    results
}

fn detect_launcher_command(ancestors: &[ProcInfo]) -> Option<&str> {
    ancestors
        .iter()
        .find(|info| {
            is_caffeinate_process(&info.command)
                && info.command.contains("cargo")
                && info.command.contains("barsmith_cli")
        })
        .or_else(|| {
            ancestors.iter().find(|info| {
                is_cargo_process(&info.command)
                    && info.command.contains("barsmith_cli")
                    && info.command.contains("run")
            })
        })
        .map(|info| info.command.as_str())
}

fn log_invocation(log_file: Option<&PathBuf>) {
    let cwd = std::env::current_dir().ok();
    let argv: Vec<String> = std::env::args_os()
        .map(|arg| arg.to_string_lossy().into_owned())
        .collect();
    let ancestors = collect_ancestor_processes(6);

    tracing::info!("==================== new barsmith_cli run ====================");
    tracing::info!(
        version = env!("CARGO_PKG_VERSION"),
        cwd = ?cwd,
        log_file = ?log_file,
        argv = ?argv,
        "barsmith_cli invoked"
    );
    tracing::info!("command_line={}", argv.join(" "));

    if let Ok(launcher) = std::env::var("BARSMITH_LAUNCHER_COMMAND") {
        tracing::info!("launcher_command={launcher}");
    } else if let Some(launcher) = detect_launcher_command(&ancestors) {
        tracing::info!("launcher_command={launcher}");
    }

    if argv.len() >= 2 {
        tracing::info!(
            "cargo_repro_command=cargo run --release -p barsmith_cli -- {}",
            argv[1..].join(" ")
        );
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    let log_file = match &cli.command {
        Commands::Comb(args) => {
            if args.no_file_log {
                None
            } else {
                Some(args.output_dir.join("barsmith.log"))
            }
        }
    };

    init_tracing(log_file.clone())?;
    log_invocation(log_file.as_ref());

    match cli.command {
        Commands::Comb(args) => {
            let ack_new_df = args.ack_new_df;
            let config = args.into_config()?;
            run_builtin_pipeline_with_options(config, BuiltinPipelineOptions { ack_new_df })
        }
    }
}
