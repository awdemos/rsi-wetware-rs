//! The improvement loop: L1 execution autonomy → L5 meta-improvement.
//!
//! The connectome is the fixed substrate; everything this loop changes is the
//! *harness* around it. Each level internalizes one more improvement decision
//! (per the survey's autonomy ladder):
//!
//! | level | what's automated | human still decides |
//! |---|---|---|
//! | L1 | execute N random search trials, keep the best | the search space, the objective, the budget |
//! | L2 | *which* region to search next (elite-refined distribution + diagnostics) | objective, budget, acceptance |
//! | L3 | *what to learn from* (confusion-driven reweighting of the train pool) | the reweighting rule exists, its targets are chosen by the loop |
//! | L4 | whether a deployment adaptation persists (gated online RLS rehearsal) | the gate set, the gate rule |
//! | L5 | where the *next run* starts (policy derived from the experience bank) | the bank path, the derivation rule |
//!
//! Verification discipline is fixed infrastructure, not a level: validation
//! splits only during search, the test set touched once, and acceptance by
//! *median* validation over several seeds (the README's ±1% RNG wobble is the
//! seed-cherry-picking hazard in miniature).

use std::path::PathBuf;

use clap::ValueEnum;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use serde_json::json;

use crate::Result;
use crate::data::Connectome;
use crate::{Metric, Report};
use crate::rsi::bank::{Bank, BankEntry};
use crate::rsi::config::{SearchSpace, TrialConfig, task_default};
use crate::rsi::evaluate::{Evaluator, Split, TaskKind};
use crate::rsi::learn::{GatedAdapter, OnlineReadout, fit_ridge};
use crate::rsi::observe::Observer;

/// How much of the improvement loop is internalized. CLI: `--level l1`…`l5`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
pub enum Level {
    /// L1: humans specify the procedure; AI executes random search over the harness
    L1,
    /// L2: + AI picks where to search next (elite refinement + diagnostics)
    L2,
    /// L3: + AI picks what to learn from (curriculum from confusions)
    L3,
    /// L4: + deployment rehearsal with gated online adaptation (rollback on regression)
    L4,
    /// L5: + the experience bank warm-starts and is updated by this run
    L5,
}

impl Level {
    /// `"l2"` etc., stored in the bank
    pub fn code(&self) -> &'static str {
        match self {
            Level::L1 => "l1",
            Level::L2 => "l2",
            Level::L3 => "l3",
            Level::L4 => "l4",
            Level::L5 => "l5",
        }
    }
}

/// Knobs for one `optimize` run.
pub struct OptimizeOpts {
    /// number of L1 random-search trials
    pub budget: usize,
    /// highest autonomy level to exercise
    pub level: Level,
    /// master seed (splits, sampling, drift construction)
    pub seed: u64,
    /// seeds per acceptance check (median is the gate)
    pub verify_seeds: usize,
    /// experience-bank path (default: the shared cache)
    pub bank_path: Option<PathBuf>,
    /// JSONL event-log path (default: events only on stderr)
    pub log_path: Option<PathBuf>,
    /// human-readable progress lines on stderr
    pub human: bool,
}

/// One scored trial.
#[derive(Debug, Clone, Copy)]
struct Trial {
    cfg: TrialConfig,
    train: f64,
    val: f64,
}

/// Run the loop on `task_name` ("digits" | "timeseries") and return the report.
pub fn run(conn: &Connectome, task_name: &str, opts: &OptimizeOpts) -> Result<Report> {
    let mut obs = Observer::new(opts.log_path.as_deref(), opts.human)?;
    let task = match task_name {
        "digits" => TaskKind::Digits,
        _ => TaskKind::Timeseries,
    };
    let space = SearchSpace::default();
    let mut rng = StdRng::seed_from_u64(opts.seed);
    let ev = Evaluator::new(conn, task, opts.seed)?;
    let out_dim = ev.out_dim();
    let n = conn.n;

    obs.event(
        "run_started",
        json!({
            "task": task.name(),
            "level": opts.level.code(),
            "budget": opts.budget,
            "seed": opts.seed,
            "brain_neurons": n,
        }),
    );

    // --- L5a: inherit the policy (nothing to inherit on a fresh bank) --------
    let bank_default = crate::data::cache_dir()?.join("rsi-bank.jsonl");
    let bank_path = opts.bank_path.clone().unwrap_or(bank_default);
    let mut bank = Bank::load(&bank_path)?;
    obs.event(
        "bank_loaded",
        json!({
            "path": bank_path.display().to_string(),
            "entries": bank.entries.len(),
        }),
    );
    let warm = if opts.level == Level::L5 {
        bank.policy(task.name())
    } else {
        None
    };
    match &warm {
        Some(p) => obs.event(
            "policy_loaded",
            json!({
                "version": p.version,
                "source_entries": p.source_entries,
                "center": json!({"sr": p.center.spectral_radius, "leak": p.center.leak,
                                 "ridge": p.center.ridge}),
                "warm_start": true,
            }),
        ),
        None => obs.event(
            "policy_loaded",
            json!({"version": 0, "warm_start": false, "reason": "empty bank or level < l5"}),
        ),
    }

    // --- anchor: the status quo this run must beat ---------------------------
    let default_cfg = task_default(task.name());
    let anchor = ev.evaluate(&default_cfg, false)?;
    obs.event(
        "anchor",
        json!({
            "config": config_json(&default_cfg),
            "val": anchor.val,
            "train": anchor.train,
        }),
    );

    // --- L1: execute the search humans specified -----------------------------
    let mut history: Vec<Trial> = Vec::new();
    let t0 = std::time::Instant::now();
    for i in 0..opts.budget {
        let cfg = match &warm {
            // with a policy, half the L1 budget starts near inherited knowledge
            Some(p) if i % 2 == 0 => space.sample_around(&mut rng, &p.center, &p.spread),
            _ => space.sample(&mut rng),
        };
        let o = ev.evaluate(&cfg, false)?;
        history.push(Trial {
            cfg,
            train: o.train,
            val: o.val,
        });
        let best = history.iter().map(|t| t.val).fold(f64::MIN, f64::max);
        obs.event(
            "trial_completed",
            json!({
                "trial": i + 1,
                "phase": "l1_random",
                "config": config_json(&cfg),
                "val": o.val,
                "train": o.train,
                "best_val": best,
                "elapsed_ms": t0.elapsed().as_millis() as u64,
            }),
        );
    }
    obs.event(
        "level_completed",
        json!({"level": "l1", "trials": history.len(),
               "best_val": best_of(&history).map(|t| t.val).unwrap_or(f64::NAN)}),
    );

    // --- L2: choose where to search next --------------------------------------
    if opts.level >= Level::L2 {
        adaptive_rounds(&ev, &space, &mut rng, &mut history, &mut obs)?;
    }

    // --- L3: choose what to learn from ----------------------------------------
    // safe inheritance at the loop level: if the search never beat the anchor,
    // the champion IS the anchor — a flat round is retained as a flat record,
    // never as a regression (rollback-unless-improved, like ASPIRE's gate)
    let mut champion = *best_of(&history).expect("at least one trial");
    if champion.val < anchor.val {
        obs.event(
            "champion_fallback",
            json!({
                "reason": "search did not beat the anchor; retaining the status quo",
                "search_best_val": champion.val,
                "anchor_val": anchor.val,
            }),
        );
        champion = Trial {
            cfg: default_cfg,
            train: anchor.train,
            val: anchor.val,
        };
    }
    let mut curriculum_val = f64::NAN;
    if opts.level >= Level::L3 {
        curriculum_val = curriculum(&ev, &champion.cfg, out_dim, &mut obs)?;
        // the curriculum's retained state is the weighting recipe; the
        // champion config itself is unchanged, but its recipe improves
        obs.event(
            "level_completed",
            json!({"level": "l3", "champion_val": champion.val,
                   "curriculum_val": curriculum_val,
                   "curriculum_gain": curriculum_val - champion.val}),
        );
    }

    // --- L4: rehearsal — should a deployment adaptation persist? --------------
    let mut rehearsal: Option<serde_json::Value> = None;
    if opts.level >= Level::L4 {
        let r = rehearsal_l4(&ev, &champion.cfg, &mut obs)?;
        rehearsal = Some(r);
        obs.event("level_completed", json!({"level": "l4"}));
    }

    // --- acceptance: median over seeds, status quo as the bar -----------------
    let (champ_med, champ_spread) = ev.median_val(&champion.cfg, opts.verify_seeds)?;
    let (default_med, _) = ev.median_val(&default_cfg, opts.verify_seeds)?;
    let passed = champ_med >= default_med;
    obs.event(
        "acceptance",
        json!({
            "champion_median_val": champ_med,
            "seed_spread": champ_spread,
            "default_median_val": default_med,
            "passed": passed,
        }),
    );

    // the test set, touched exactly once per side
    let final_outcome = ev.evaluate(&champion.cfg, true)?;
    let default_outcome = ev.evaluate(&default_cfg, true)?;
    obs.event(
        "test_measured",
        json!({
            "champion_test": final_outcome.test,
            "default_test": default_outcome.test,
        }),
    );

    // --- L5b: retain this run for successors (from L2 up: once the loop
    // selects, the selection is worth recording; L5 additionally inherits) ---
    if opts.level >= Level::L2 {
        let entry = BankEntry {
            run: 0,
            task: task.name().to_string(),
            level: opts.level.code().to_string(),
            config: champion.cfg,
            val: champ_med,
            test: final_outcome.test,
            default_test: default_outcome.test,
            policy_version: 0,
        };
        bank.append(entry, &mut obs)?;
    }

    let improvement = match (final_outcome.test, default_outcome.test) {
        (Some(c), Some(d)) => c - d,
        _ => f64::NAN,
    };
    let mut report: Report = vec![
        ("task".into(), Metric::S(task.name().into())),
        ("level".into(), Metric::S(opts.level.code().into())),
        ("trials".into(), Metric::I(history.len() as i64)),
        (
            "best_config".into(),
            Metric::S(format_config(&champion.cfg)),
        ),
        ("median_val".into(), Metric::F(champ_med)),
        ("seed_spread".into(), Metric::F(champ_spread)),
        ("default_median_val".into(), Metric::F(default_med)),
        (
            "test".into(),
            Metric::F(final_outcome.test.unwrap_or(f64::NAN)),
        ),
        (
            "default_test".into(),
            Metric::F(default_outcome.test.unwrap_or(f64::NAN)),
        ),
        ("improvement".into(), Metric::F(improvement)),
        ("curriculum_gain".into(), Metric::F(curriculum_val - champion.val)),
        ("bank".into(), Metric::S(bank_path.display().to_string())),
        (
            "bank_entries".into(),
            Metric::I(bank.entries.len() as i64),
        ),
        ("events".into(), Metric::I(obs.count as i64)),
    ];
    if let Some(r) = rehearsal {
        if let (Some(naive), Some(gated)) = (r.get("naive_val"), r.get("gated_val")) {
            report.push((
                "l4_naive_val".into(),
                Metric::F(naive.as_f64().unwrap_or(f64::NAN)),
            ));
            report.push((
                "l4_gated_val".into(),
                Metric::F(gated.as_f64().unwrap_or(f64::NAN)),
            ));
            report.push((
                "l4_rollbacks".into(),
                Metric::I(r.get("rollbacks").and_then(|v| v.as_u64()).unwrap_or(0) as i64),
            ));
        }
    }
    let n_events = obs.count;
    obs.finish(json!({
        "task": task.name(),
        "level": opts.level.code(),
        "improvement": improvement,
        "champion_test": final_outcome.test,
        "default_test": default_outcome.test,
        "events": n_events,
    }));
    Ok(report)
}

fn best_of(history: &[Trial]) -> Option<&Trial> {
    history.iter().max_by(|a, b| a.val.total_cmp(&b.val))
}

fn config_json(cfg: &TrialConfig) -> serde_json::Value {
    json!({
        "sr": cfg.spectral_radius,
        "leak": cfg.leak,
        "ridge": cfg.ridge,
        "in_scale": cfg.input_scale,
        "inh": cfg.inhibitory_fraction,
        "seed": cfg.seed,
    })
}

fn format_config(cfg: &TrialConfig) -> String {
    format!(
        "sr={:.3} leak={:.3} ridge={:.4} in_scale={:.3} inh={:.3} seed={}",
        cfg.spectral_radius,
        cfg.leak,
        cfg.ridge,
        cfg.input_scale,
        cfg.inhibitory_fraction,
        cfg.seed
    )
}

/// L2: elite-refined rounds. The search distribution for round r+1 is the
/// top quartile of rounds ≤ r — strategy selection is now the loop's job.
/// Diagnostics inspect the champion each round and bias the next distribution
/// (ridge up on a generalization gap, dynamics up on underfitting, uniform
/// re-exploration on stagnation).
fn adaptive_rounds(
    ev: &Evaluator,
    space: &SearchSpace,
    rng: &mut StdRng,
    history: &mut Vec<Trial>,
    obs: &mut Observer,
) -> Result<()> {
    let per_round = ((history.len() as f64 / 4.0).round() as usize).max(2);
    let rounds = 3;
    let mut trial_no = history.len();
    for round in 0..rounds {
        // top quartile (at least 2) defines center + spread
        let mut sorted: Vec<&Trial> = history.iter().collect();
        sorted.sort_by(|a, b| b.val.total_cmp(&a.val));
        let k = sorted.len().max(4) / 4;
        let elites: Vec<&Trial> = sorted.into_iter().take(k.max(2)).collect();
        let (mut center, spread) = elite_stats(&elites);

        // diagnostics on the champion
        let champ = elites[0];
        let diags = diagnose(champ, &center);
        let mut reexplore = false;
        for (rule, action) in &diags {
            match rule.as_str() {
                "generalization_gap" => {
                    center.ridge = (center.ridge * 3.0).clamp(space.ridge.0, space.ridge.1)
                }
                "underfit" => {
                    center.spectral_radius =
                        (center.spectral_radius + 0.15).clamp(space.spectral_radius.0, space.spectral_radius.1);
                    center.input_scale =
                        (center.input_scale * 1.4).clamp(space.input_scale.0, space.input_scale.1);
                }
                "stagnation" => reexplore = true,
                _ => {}
            }
            obs.event(
                "diagnostic",
                json!({
                    "round": round + 1,
                    "rule": rule,
                    "action": action,
                    "champion_val": champ.val,
                    "train_val_gap": champ.train - champ.val,
                }),
            );
        }

        for _ in 0..per_round {
            trial_no += 1;
            let cfg = if reexplore {
                space.sample(rng)
            } else {
                space.sample_around(rng, &center, &spread)
            };
            let o = ev.evaluate(&cfg, false)?;
            obs.event(
                "trial_completed",
                json!({
                    "trial": trial_no,
                    "phase": "l2_adaptive",
                    "round": round + 1,
                    "reexplore": reexplore,
                    "config": config_json(&cfg),
                    "val": o.val,
                    "train": o.train,
                    "best_val": history.iter().map(|t| t.val).fold(f64::MIN, f64::max).max(o.val),
                }),
            );
            history.push(Trial {
                cfg,
                train: o.train,
                val: o.val,
            });
        }
    }
    obs.event(
        "level_completed",
        json!({"level": "l2", "trials": history.len(),
               "best_val": best_of(history).map(|t| t.val).unwrap_or(f64::NAN)}),
    );
    Ok(())
}

/// Mean/std of the elites per knob (ridge in log-space).
fn elite_stats(elites: &[&Trial]) -> (TrialConfig, TrialConfig) {
    let m = |f: fn(&TrialConfig) -> f64| {
        elites.iter().map(|e| f(&e.cfg)).sum::<f64>() / elites.len() as f64
    };
    let s = |f: fn(&TrialConfig) -> f64, mean: f64| {
        (elites.iter().map(|e| (f(&e.cfg) - mean).powi(2)).sum::<f64>() / elites.len() as f64)
            .sqrt()
    };
    let center = TrialConfig {
        spectral_radius: m(|c| c.spectral_radius),
        leak: m(|c| c.leak),
        ridge: m(|c| c.ridge),
        input_scale: m(|c| c.input_scale),
        inhibitory_fraction: m(|c| c.inhibitory_fraction),
        seed: 0,
    };
    let spread = TrialConfig {
        spectral_radius: s(|c| c.spectral_radius, center.spectral_radius).max(0.02),
        leak: s(|c| c.leak, center.leak).max(0.04),
        ridge: s(|c| c.ridge, center.ridge) / center.ridge.max(1e-12),
        input_scale: s(|c| c.input_scale, center.input_scale).max(0.08),
        inhibitory_fraction: s(|c| c.inhibitory_fraction, center.inhibitory_fraction).max(0.02),
        seed: 0,
    };
    (center, spread)
}

/// Read the champion and say what the next round should do about it. The
/// first entry is always an assessment of the champion's shape; conditional
/// rules follow it.
fn diagnose(champ: &Trial, center: &TrialConfig) -> Vec<(String, String)> {
    let mut out = Vec::new();
    let gap = champ.train - champ.val;
    let rel = gap / champ.train.abs().max(0.3);
    out.push((
        "assessment".into(),
        format!(
            "champion val={:.3} train={:.3} gap={:.3} (rel {:.2}); searching around ridge={:.3}",
            champ.val, champ.train, gap, rel, center.ridge
        ),
    ));
    if rel > 0.15 {
        out.push((
            "generalization_gap".into(),
            format!("train-val gap {gap:.3} (rel {rel:.2}); ridge {e:.3} -> {n:.3}", e = center.ridge, n = (center.ridge * 3.0).min(10.0)),
        ));
    }
    if champ.train < 0.6 && champ.val < 0.6 {
        out.push((
            "underfit".into(),
            "dynamics not expressive enough; raise spectral radius + input scale".into(),
        ));
    }
    if gap.abs() < 0.01 && champ.val < 0.98 {
        out.push((
            "stagnation".into(),
            "train≈val but low: model family saturated at this config; re-explore uniformly".into(),
        ));
    }
    out
}

/// L3: learner-conditioned experience acquisition. Fit on the train pool,
/// find what's misclassified (train) and which classes fail on validation,
/// upweight exactly those samples, refit. Retain the best-scoring weighting.
fn curriculum(
    ev: &Evaluator,
    cfg: &TrialConfig,
    out_dim: usize,
    obs: &mut Observer,
) -> Result<f64> {
    let train_rows = ev.rows(Split::Train);
    let val_rows = ev.rows(Split::Val);
    let mut rows = train_rows.clone();
    rows.extend_from_slice(&val_rows);
    let feats = ev.features(cfg, &rows);
    let (n_train, n_val, _) = ev.splits;
    let dim = feats.len() / rows.len(); // reservoir width
    let f_train = &feats[..n_train * dim];
    let f_val = &feats[n_train * dim..(n_train + n_val) * dim];
    let t_train = ev.targets(&train_rows, out_dim);

    let mut weights = vec![1.0f64; n_train];
    let mut best_val = f64::MIN;
    let mut best_weights = weights.clone();
    let rounds = 4;
    for round in 0..rounds {
        let sol = fit_ridge(f_train, dim, &t_train, out_dim, Some(&weights), cfg.ridge);
        let val_pred = sol.predict(f_val);
        let val = ev.score(&val_pred, &val_rows);
        if val > best_val {
            best_val = val;
            best_weights = weights.clone();
        }
        // what is the learner getting wrong?
        let tr_pred = sol.predict(f_train);
        let miss = ev.miss_mask(&tr_pred, &train_rows);
        let miss_n = miss.iter().filter(|&&m| m).count();
        // per-bucket validation error rates drive the upweighting
        let val_err = ev.bucket_errors(&val_pred, &val_rows, out_dim);
        for r in 0..n_train {
            let c = ev.bucket_of(train_rows[r]);
            weights[r] = (1.0 + 2.0 * usize::from(miss[r]) as f64 + 4.0 * val_err[c]).clamp(1.0, 8.0);
        }
        let focus: Vec<serde_json::Value> = (0..val_err.len())
            .filter(|&c| val_err[c] > 0.05)
            .map(|c| json!({"bucket": c, "val_err": val_err[c]}))
            .collect();
        obs.event(
            "curriculum_round",
            json!({
                "round": round + 1,
                "val": val,
                "train_misses": miss_n,
                "upweighted": weights.iter().filter(|&&w| w > 1.0).count(),
                "focus": focus,
            }),
        );
    }
    // re-validate the retained weighting: the recipe is only inherited if it
    // reproduces its score on the fixed validation split
    let sol = fit_ridge(f_train, dim, &t_train, out_dim, Some(&best_weights), cfg.ridge);
    let confirmed = ev.score(&sol.predict(f_val), &val_rows);
    obs.event(
        "curriculum_completed",
        json!({"rounds": rounds, "best_val": best_val, "confirmed_val": confirmed}),
    );
    Ok(confirmed)
}

/// L4 rehearsal: a deployment stream with distribution drift, adapted two
/// ways — naively (every update persists) and gated (a candidate must beat
/// the accepted readout on the fixed gate set). The gap between the two is
/// the measured value of safe inheritance.
pub fn rehearsal_l4(
    ev: &Evaluator,
    cfg: &TrialConfig,
    obs: &mut Observer,
) -> Result<serde_json::Value> {
    let train_rows = ev.rows(Split::Train);
    let val_rows = ev.rows(Split::Val);
    let dim = ev.features_dim();

    // the "deployed" readout is trained on a random pre-drift subset; the
    // drifted stream is what arrives after deployment — seed rows are held
    // out of the stream so the rehearsal measures adaptation, not re-fitting
    let mut shuffled = train_rows.clone();
    shuffled.shuffle(&mut StdRng::seed_from_u64(0x5eed));
    let seed_n = (shuffled.len() / 4).max(1);
    let (seed_rows, stream) = {
        let (a, b) = shuffled.split_at(seed_n);
        (a.to_vec(), drift_order(ev, b))
    };

    // features for stream, gate set, and the seed batch — one extraction each
    let stream_feats = ev.features(cfg, &stream);
    let val_feats = ev.features(cfg, &val_rows);
    let seed_feats = ev.features(cfg, &seed_rows);
    let gate = |or: &OnlineReadout| ev.score(&or.predict(&val_feats), &val_rows);

    let seed_targets = stream_targets(ev, &seed_rows);
    let init = OnlineReadout::fit(
        &seed_feats,
        dim,
        &seed_targets,
        ev.out_dim(),
        cfg.ridge,
    );
    let init_val = gate(&init);
    obs.event(
        "l4_rehearsal_started",
        json!({
            "stream_len": stream.len(),
            "seed_len": seed_n,
            "init_gate_val": init_val,
        }),
    );

    // naive: every sample updates the one readout, nothing is ever rolled back
    let mut naive = init.fork();
    for i in 0..stream.len() {
        let x = &stream_feats[i * dim..(i + 1) * dim];
        let y = one_target(ev, stream[i]);
        naive.update(x, &y);
    }
    let naive_val = gate(&naive);

    // gated: candidate absorbs the stream; the gate decides what persists
    let mut adapter = GatedAdapter::new(init.fork(), init_val);
    let gate_every = (stream.len() / 8).max(8);
    for i in 0..stream.len() {
        let x = &stream_feats[i * dim..(i + 1) * dim];
        let y = one_target(ev, stream[i]);
        adapter.candidate().update(x, &y);
        if (i + 1) % gate_every == 0 {
            match adapter.gate(gate) {
                crate::rsi::learn::GateDecision::Accepted {
                    new_val,
                    candidate_val,
                } => obs.event(
                    "adaptation_applied",
                    json!({
                        "step": i,
                        "version": adapter.accepted().version,
                        "gate_val": new_val,
                        "candidate_val": candidate_val,
                        "updates": adapter.accepted().updates,
                    }),
                ),
                crate::rsi::learn::GateDecision::RolledBack {
                    candidate_val,
                    kept_val,
                } => obs.event(
                    "rollback",
                    json!({
                        "step": i,
                        "candidate_val": candidate_val,
                        "kept_val": kept_val,
                    }),
                ),
            }
        }
    }
    let gated_val = adapter.accepted_val();
    obs.event(
        "l4_rehearsal_completed",
        json!({
            "naive_val": naive_val,
            "gated_val": gated_val,
            "adaptations": adapter.adaptations,
            "rollbacks": adapter.rollbacks,
            "gate_interval": gate_every,
        }),
    );
    Ok(json!({
        "naive_val": naive_val,
        "gated_val": gated_val,
        "init_val": init_val,
        "adaptations": adapter.adaptations,
        "rollbacks": adapter.rollbacks,
    }))
}

/// Deployment order with drift: digits streams classes 0–4 first, then 5–9;
/// timeseries streams its second half before its first (level shift).
fn drift_order(ev: &Evaluator, train_rows: &[usize]) -> Vec<usize> {
    match ev.task_kind() {
        TaskKind::Digits => {
            let mut first = Vec::new();
            let mut second = Vec::new();
            for &r in train_rows {
                let c = ev.label_of(r);
                if c < 5 {
                    first.push(r);
                } else {
                    second.push(r);
                }
            }
            first.extend(second);
            first
        }
        TaskKind::Timeseries => {
            let half = train_rows.len() / 2;
            let mut v: Vec<usize> = train_rows[half..].to_vec();
            v.extend_from_slice(&train_rows[..half]);
            v
        }
    }
}

fn stream_targets(ev: &Evaluator, rows: &[usize]) -> Vec<f64> {
    ev.targets(rows, ev.out_dim())
}

fn one_target(ev: &Evaluator, row: usize) -> Vec<f64> {
    let mut y = vec![0.0; ev.out_dim()];
    match ev.task_kind() {
        TaskKind::Digits => y[ev.label_of(row)] = 1.0,
        TaskKind::Timeseries => y[0] = ev.target_value(row),
    }
    y
}
