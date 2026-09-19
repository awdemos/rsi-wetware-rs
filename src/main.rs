//! Command line interface: `wetware --help`.
//!
//! Subcommands: `download` (fetch and cache the real connectome), `info`
//! (connectome stats), `run <task>` (train a readout on a task and report the
//! score), `optimize <task>` (the RSI improvement loop, levels l1–l5),
//! `adapt` (L4 online-adaptation drift demo), `bank` (experience bank).
//! `--sample` runs everything on the offline synthetic stand-in.

use std::path::PathBuf;

use clap::{Parser, Subcommand, ValueEnum};

use wetware::data::{self, Connectome};
use wetware::rsi::bank::Bank;
use wetware::rsi::optimize::{Level, OptimizeOpts};
use wetware::tasks::{digits, timeseries};
use wetware::{Metric, Report, Result};

#[derive(Parser)]
#[command(
    name = "wetware",
    version,
    about = "Compute on a real fruit-fly brain."
)]
struct Cli {
    /// use the offline synthetic stand-in, not the real brain
    #[arg(long, global = true)]
    sample: bool,

    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// fetch and cache the real connectome
    Download {
        /// re-download even if cached
        #[arg(long)]
        force: bool,
    },
    /// connectome stats
    Info,
    /// train a readout on a task and report the score
    Run {
        /// digits | timeseries
        task: Task,
    },
    /// run the RSI improvement loop over the task's harness (levels l1–l5)
    Optimize {
        /// digits | timeseries
        task: Task,
        /// number of L1 random-search trials
        #[arg(long, default_value_t = 16)]
        budget: usize,
        /// highest autonomy level to exercise
        #[arg(long, value_enum, default_value = "l5")]
        level: Level,
        /// master seed (splits + sampling); fixed for reproducibility
        #[arg(long, default_value_t = 0)]
        seed: u64,
        /// seeds per acceptance check (median gates acceptance)
        #[arg(long, default_value_t = 3)]
        verify_seeds: usize,
        /// experience-bank path (default: the shared cache)
        #[arg(long)]
        bank: Option<PathBuf>,
        /// JSONL event log (default: human lines on stderr only)
        #[arg(long)]
        log: Option<PathBuf>,
        /// suppress human-readable progress lines
        #[arg(long)]
        quiet: bool,
    },
    /// L4 demo: online adaptation under distribution drift, naive vs gated
    Adapt {
        /// master seed
        #[arg(long, default_value_t = 0)]
        seed: u64,
        /// JSONL event log
        #[arg(long)]
        log: Option<PathBuf>,
        /// suppress human-readable progress lines
        #[arg(long)]
        quiet: bool,
    },
    /// print the experience bank and the policy derived from it
    Bank {
        /// experience-bank path (default: the shared cache)
        #[arg(long)]
        path: Option<PathBuf>,
    },
}

#[derive(Copy, Clone, ValueEnum)]
enum Task {
    Digits,
    Timeseries,
}

fn brain(sample: bool) -> Result<Connectome> {
    if sample {
        Ok(data::load_sample(600, 0.013, 0))
    } else {
        data::load()
    }
}

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Download { force } => {
            let path = data::download(force)?;
            println!("connectome cached at {}", path.display());
        }
        Cmd::Info => {
            let conn = brain(cli.sample)?;
            print_info(&conn, cli.sample);
        }
        Cmd::Run { task } => {
            let conn = brain(cli.sample)?;
            let report = match task {
                Task::Digits => digits::demo(&conn, &Default::default())?,
                Task::Timeseries => timeseries::demo(&conn, &Default::default())?,
            };
            print_json(&round_report(&report, 4));
        }
        Cmd::Optimize {
            task,
            budget,
            level,
            seed,
            verify_seeds,
            bank,
            log,
            quiet,
        } => {
            let conn = brain(cli.sample)?;
            let opts = OptimizeOpts {
                budget,
                level,
                seed,
                verify_seeds,
                bank_path: bank,
                log_path: log,
                human: !quiet,
            };
            let report = wetware::rsi::optimize::run(&conn, task_name(task), &opts)?;
            print_json(&round_report(&report, 6));
        }
        Cmd::Adapt {
            seed,
            log,
            quiet,
        } => {
            let conn = brain(cli.sample)?;
            let opts = wetware::rsi::adapt::AdaptOpts {
                seed,
                log_path: log,
                human: !quiet,
            };
            let report = wetware::rsi::adapt::run(&conn, &opts)?;
            print_json(&round_report(&report, 6));
        }
        Cmd::Bank { path } => {
            let path = match path {
                Some(p) => p,
                None => data::cache_dir()?.join("rsi-bank.jsonl"),
            };
            let bank = Bank::load(&path)?;
            print!("{}", bank.describe());
        }
    }
    Ok(())
}

fn task_name(task: Task) -> &'static str {
    match task {
        Task::Digits => "digits",
        Task::Timeseries => "timeseries",
    }
}

fn print_info(conn: &Connectome, sample: bool) {
    let n = conn.n;
    let nnz = conn.w.iter().filter(|&&v| v != 0.0).count();
    let density = nnz as f64 / (n * n) as f64;
    let sum: f64 = conn.w.iter().filter(|&&v| v != 0.0).sum();
    let mean = if nnz > 0 { sum / nnz as f64 } else { 0.0 };
    let report: Report = vec![
        (
            "source".into(),
            Metric::S(if sample {
                "synthetic sample".into()
            } else {
                "larval Drosophila connectome (Winding 2023)".into()
            }),
        ),
        ("neurons".into(), Metric::I(n as i64)),
        ("connections".into(), Metric::I(nnz as i64)),
        ("density".into(), Metric::F(round(density, 4))),
        (
            "mean_synapses_per_connection".into(),
            Metric::F(round(mean, 2)),
        ),
    ];
    print_json(&report);
}

fn round(x: f64, digits: i32) -> f64 {
    let m = 10f64.powi(digits);
    (x * m).round() / m
}

fn round_report(report: &Report, digits: i32) -> Report {
    report
        .iter()
        .map(|(k, v)| {
            let v = match v {
                Metric::F(x) => Metric::F(round(*x, digits)),
                other => other.clone(),
            };
            (k.clone(), v)
        })
        .collect()
}

/// Pretty-print a report the way Python's `json.dumps(..., indent=2)` does.
fn print_json(report: &Report) {
    println!("{{");
    for (i, (k, v)) in report.iter().enumerate() {
        let comma = if i + 1 == report.len() { "" } else { "," };
        match v {
            Metric::S(s) => println!("  \"{k}\": \"{s}\"{comma}"),
            Metric::I(x) => println!("  \"{k}\": {x}{comma}"),
            Metric::F(x) => println!("  \"{k}\": {x:?}{comma}"),
        }
    }
    println!("}}");
}
