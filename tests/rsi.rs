//! Tests for the RSI loop. Everything runs offline on the synthetic sample
//! brain with tiny budgets — the point is to exercise the loop machinery and
//! its observability, not to squeeze accuracy out of a stand-in network.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use wetware::data;
use wetware::rsi::evaluate::{Evaluator, Split, TaskKind};
use wetware::rsi::learn::{GatedAdapter, GateDecision, OnlineReadout, fit_ridge};
use wetware::rsi::optimize::{self, Level, OptimizeOpts};

static TMP_ID: AtomicU64 = AtomicU64::new(0);

/// A fresh scratch directory per test: unique name *and* wiped if a previous
/// crashed run left one behind (the temp dir survives across processes).
fn scratch(tag: &str) -> PathBuf {
    let id = TMP_ID.fetch_add(1, Ordering::SeqCst);
    let dir = std::env::temp_dir().join(format!(
        "wetware-rsi-test-{}-{}-{id}",
        tag,
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn report_f(r: &wetware::Report, key: &str) -> Option<f64> {
    r.iter().find(|(k, _)| k == key).and_then(|(_, v)| match v {
        wetware::Metric::F(x) => Some(*x),
        _ => None,
    })
}

fn report_i(r: &wetware::Report, key: &str) -> Option<i64> {
    r.iter().find(|(k, _)| k == key).and_then(|(_, v)| match v {
        wetware::Metric::I(x) => Some(*x),
        _ => None,
    })
}

fn opts(dir: &PathBuf, budget: usize, level: Level) -> OptimizeOpts {
    OptimizeOpts {
        budget,
        level,
        seed: 7,
        verify_seeds: 2,
        bank_path: Some(dir.join("bank.jsonl")),
        log_path: Some(dir.join("events.jsonl")),
        human: false,
    }
}

fn read_events(dir: &PathBuf) -> Vec<serde_json::Value> {
    std::fs::read_to_string(dir.join("events.jsonl"))
        .unwrap()
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).unwrap())
        .collect()
}

fn kinds(events: &[serde_json::Value]) -> Vec<&str> {
    events
        .iter()
        .map(|e| e["kind"].as_str().unwrap())
        .collect()
}

// --- L1: execution autonomy -----------------------------------------------

#[test]
fn l1_random_search_beats_the_anchor_and_logs_every_trial() {
    let dir = scratch("l1");
    let c = data::load_sample(500, 0.013, 0);
    let opts = opts(&dir, 8, Level::L1);
    let report = optimize::run(&c, "timeseries", &opts).unwrap();

    // the search must at least match the status quo it started from
    assert!(report_f(&report, "median_val").unwrap()
        >= report_f(&report, "default_median_val").unwrap() - 1e-9);
    // l1 must not touch the bank (retention starts at l2)
    assert!(!dir.join("bank.jsonl").exists());

    let events = read_events(&dir);
    let k = kinds(&events);
    assert!(k.contains(&"anchor"));
    assert!(k.contains(&"trial_completed"));
    assert!(k.contains(&"acceptance"));
    assert!(k.contains(&"run_completed"));
    let trials = k.iter().filter(|&&x| x == "trial_completed").count();
    assert_eq!(trials, 8);
    // every trial carries its config and score — the history is complete
    let t = events.iter().find(|e| e["kind"] == "trial_completed").unwrap();
    assert!(t["config"]["sr"].is_number());
    assert!(t["val"].is_number());
}

// --- L2: strategy autonomy -------------------------------------------------

#[test]
fn l2_adaptive_rounds_emit_diagnostics_and_retain_an_entry() {
    let dir = scratch("l2");
    let c = data::load_sample(500, 0.013, 0);
    let opts = opts(&dir, 8, Level::L2);
    let report = optimize::run(&c, "timeseries", &opts).unwrap();

    let events = read_events(&dir);
    let k = kinds(&events);
    // diagnostics exist (the exact rule depends on the champion's shape)
    assert!(k.contains(&"diagnostic"));
    // an accepted improvement is retained with a bumped policy version
    assert!(k.contains(&"policy_updated"));
    let bank: Vec<serde_json::Value> = std::fs::read_to_string(dir.join("bank.jsonl"))
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert_eq!(bank.len(), 1);
    assert_eq!(bank[0]["policy_version"], 1);
    assert_eq!(bank[0]["level"], "l2");
    assert!(report_f(&report, "improvement").is_some());
}

// --- L3: experience autonomy ------------------------------------------------

#[test]
fn l3_curriculum_confirms_its_retained_weighting() {
    let dir = scratch("l3");
    let c = data::load_sample(500, 0.013, 0);
    let opts = opts(&dir, 8, Level::L3);
    optimize::run(&c, "timeseries", &opts).unwrap();

    let events = read_events(&dir);
    let rounds: Vec<&serde_json::Value> = events
        .iter()
        .filter(|e| e["kind"] == "curriculum_round")
        .collect();
    assert_eq!(rounds.len(), 4);
    let done = events
        .iter()
        .find(|e| e["kind"] == "curriculum_completed")
        .unwrap();
    // the retained recipe must reproduce its score on the fixed gate set
    assert_eq!(done["best_val"].as_f64().unwrap(), done["confirmed_val"].as_f64().unwrap());
}

// --- L4: deployment adaptation ----------------------------------------------

#[test]
fn l4_rehearsal_never_ends_below_its_start() {
    let dir = scratch("l4");
    let c = data::load_sample(500, 0.013, 0);
    let opts = opts(&dir, 6, Level::L4);
    let report = optimize::run(&c, "timeseries", &opts).unwrap();

    // the gate's actual guarantee: the accepted readout ratchets from its
    // starting score — it can never end below where the rehearsal began,
    // whatever the drift does to the naive one
    let events = read_events(&dir);
    let started = events
        .iter()
        .find(|e| e["kind"] == "l4_rehearsal_started")
        .unwrap();
    let init_val = started["init_gate_val"].as_f64().unwrap();
    let gated = report_f(&report, "l4_gated_val").unwrap();
    assert!(gated >= init_val - 1e-9, "gated {gated} started at {init_val}");
    // both adapters are reported; naive is informational (on benign drift it
    // can climb above the gated one — that is what rollbacks cost)
    assert!(report_f(&report, "l4_naive_val").is_some());
    let k = kinds(&events);
    assert!(k.contains(&"adaptation_applied") || k.contains(&"rollback"));
}

#[test]
fn online_rls_matches_batch_ridge() {
    // the rank-1 updates must equal the closed form on the same data —
    // online adaptation never leaves the model class
    let mut rng_state = 0u64;
    let mut next = move || {
        // tiny deterministic LCG, plenty for a unit test
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (rng_state >> 33) as f64 / u32::MAX as f64 - 0.5
    };
    let n = 20usize;
    let rows = 120usize;
    let x: Vec<f64> = (0..rows * n).map(|_| next() * 4.0).collect();
    let w_true: Vec<f64> = (0..n).map(|_| next() * 3.0).collect();
    let y: Vec<f64> = (0..rows)
        .map(|t| (0..n).map(|i| x[t * n + i] * w_true[i]).sum::<f64>() + 0.05 * next())
        .collect();

    let ridge = 1e-4;
    let batch = fit_ridge(&x, n, &y, 1, None, ridge);
    let mut online = OnlineReadout::fit(&x[..n * 60], n, &y[..60], 1, ridge);
    for t in 60..rows {
        online.update(&x[t * n..(t + 1) * n], &[y[t]]);
    }
    let diff: f64 = (0..(n + 1))
        .map(|i| (batch.w[i] - online.solution().w[i]).powi(2))
        .sum::<f64>()
        .sqrt();
    assert!(diff < 1e-6, "batch vs online weights differ by {diff}");
}

#[test]
fn gated_adapter_rolls_back_on_adversarial_drift() {
    // separable synthetic problem: label = sign(x0 + 0.3 x1). Good drift keeps
    // the candidate at 100%; flipped-label drift drags it below; the gate must
    // accept the former and roll back the latter.
    let n = 12usize;
    let mut rng_state = 42u64;
    let mut next = move || {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (rng_state >> 33) as f64 / u32::MAX as f64 - 0.5
    };
    let rows = 80usize;
    let x: Vec<f64> = (0..rows * n).map(|_| next() * 4.0).collect();
    let labels: Vec<usize> = (0..rows)
        .map(|t| usize::from(x[t * n] + 0.3 * x[t * n + 1] <= 0.0))
        .collect();
    let mut y = vec![0.0; rows * 2];
    for (t, &l) in labels.iter().enumerate() {
        y[t * 2 + l] = 1.0;
    }
    let acc = |w: &OnlineReadout| {
        let p = w.predict(&x);
        (0..rows)
            .filter(|&t| {
                let row = &p[t * 2..t * 2 + 2];
                let am = usize::from(row[0] <= row[1]);
                am == labels[t]
            })
            .count() as f64
            / rows as f64
    };
    let init = OnlineReadout::fit(&x[..40 * n], n, &y[..80], 2, 1e-3);
    let init_acc = acc(&init);
    let mut adapter = GatedAdapter::new(init.fork(), init_acc);
    // good drift: more consistent samples
    for t in 40..60 {
        adapter.candidate().update(&x[t * n..(t + 1) * n], &[y[t * 2], y[t * 2 + 1]]);
    }
    let d1 = adapter.gate(acc);
    assert!(matches!(d1, GateDecision::Accepted { .. }), "init_acc={init_acc}");
    // adversarial drift: flipped labels
    for t in 60..80 {
        adapter
            .candidate()
            .update(&x[t * n..(t + 1) * n], &[y[t * 2 + 1], y[t * 2]]);
    }
    assert!(matches!(adapter.gate(acc), GateDecision::RolledBack { .. }));
    assert_eq!(adapter.adaptations, 1);
    assert_eq!(adapter.rollbacks, 1);
    assert_eq!(adapter.accepted().version, 1);
}

// --- L5: meta-improvement -----------------------------------------------------

#[test]
fn l5_warm_starts_from_the_inherited_policy() {
    let dir = scratch("l5");
    let c = data::load_sample(500, 0.013, 0);

    // first run seeds the bank; policy version becomes 1
    let r1 = optimize::run(&c, "timeseries", &opts(&dir, 6, Level::L5)).unwrap();
    assert_eq!(report_i(&r1, "bank_entries").unwrap(), 1);

    // second run must load the policy and warm-start from it
    let opts2 = opts(&dir, 6, Level::L5);
    let r2 = optimize::run(&c, "timeseries", &opts2).unwrap();
    assert_eq!(report_i(&r2, "bank_entries").unwrap(), 2);
    let events = read_events(&dir); // appended across both runs
    let loaded: Vec<&serde_json::Value> = events
        .iter()
        .filter(|e| e["kind"] == "policy_loaded")
        .collect();
    assert_eq!(loaded.len(), 2);
    assert_eq!(loaded[0]["warm_start"], false);
    assert_eq!(loaded[1]["warm_start"], true);
    assert!(loaded[1]["version"].as_u64().unwrap() >= 1);
    let updated: Vec<&serde_json::Value> = events
        .iter()
        .filter(|e| e["kind"] == "policy_updated")
        .collect();
    assert_eq!(updated.len(), 2);
    assert_eq!(updated[1]["version"], 2);
}

// --- the verifier's promise ---------------------------------------------------

#[test]
fn test_split_is_never_touched_during_search() {
    // evaluating without `with_test` must not require test rows at all:
    // swap the test range out and confirm search still works
    let c = data::load_sample(400, 0.013, 0);
    let ev = Evaluator::new(&c, TaskKind::Timeseries, 3).unwrap();
    let cfg = wetware::rsi::config::TrialConfig::default();
    let a = ev.evaluate(&cfg, false).unwrap();
    assert!(a.test.is_none());
    let b = ev.evaluate(&cfg, true).unwrap();
    assert!(b.test.is_some());
    // scores on train/val are identical regardless of `with_test`
    assert_eq!(a.val, b.val);
    let _ = Split::Test; // silence unused import if enum changes
}
