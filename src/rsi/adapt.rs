//! Standalone L4 demo: watch a deployed readout meet drift, with and without
//! the retention gate.
//!
//! `wetware adapt` drives the champion-default digits config through a
//! deployment stream whose class mix shifts mid-flight (digits 0–4 first,
//! then 5–9). Two identical readouts adapt online by recursive least squares:
//! the naive one keeps every update; the gated one only persists a change
//! that beats the accepted readout on a fixed validation set. The report
//! shows both final scores plus how many adaptations were applied vs rolled
//! back — the measured value of safe inheritance.

use std::path::PathBuf;

use serde_json::json;

use crate::Result;
use crate::data::Connectome;
use crate::{Metric, Report};
use crate::rsi::evaluate::{Evaluator, TaskKind};
use crate::rsi::observe::Observer;
use crate::rsi::optimize::rehearsal_l4;

/// Knobs for `wetware adapt`.
pub struct AdaptOpts {
    /// master seed (splits, stream order)
    pub seed: u64,
    /// JSONL event-log path
    pub log_path: Option<PathBuf>,
    /// human-readable progress on stderr
    pub human: bool,
}

/// Run the drift demo on the digits task and return the report.
pub fn run(conn: &Connectome, opts: &AdaptOpts) -> Result<Report> {
    let mut obs = Observer::new(opts.log_path.as_deref(), opts.human)?;
    let ev = Evaluator::new(conn, TaskKind::Digits, opts.seed)?;
    let cfg = crate::rsi::config::task_default("digits");
    obs.event(
        "adapt_started",
        json!({"task": "digits", "drift": "classes 0-4 then 5-9", "seed": opts.seed}),
    );
    let r = rehearsal_l4(&ev, &cfg, &mut obs)?;
    let naive = r.get("naive_val").and_then(|v| v.as_f64()).unwrap_or(f64::NAN);
    let gated = r.get("gated_val").and_then(|v| v.as_f64()).unwrap_or(f64::NAN);
    let adaptations = r.get("adaptations").and_then(|v| v.as_u64()).unwrap_or(0);
    let rollbacks = r.get("rollbacks").and_then(|v| v.as_u64()).unwrap_or(0);
    let report: Report = vec![
        ("task".into(), Metric::S("digits".into())),
        ("mode".into(), Metric::S("l4_deployment_rehearsal".into())),
        ("l4_naive_val".into(), Metric::F(naive)),
        ("l4_gated_val".into(), Metric::F(gated)),
        ("gate_advantage".into(), Metric::F(gated - naive)),
        ("adaptations".into(), Metric::I(adaptations as i64)),
        ("rollbacks".into(), Metric::I(rollbacks as i64)),
    ];
    obs.finish(json!({
        "mode": "l4_deployment_rehearsal",
        "naive_val": naive,
        "gated_val": gated,
        "adaptations": adaptations,
        "rollbacks": rollbacks,
    }));
    Ok(report)
}
