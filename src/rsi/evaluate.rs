//! The verifier: how a [`TrialConfig`] is scored.
//!
//! Two protocol rules from the survey are enforced here, not by convention:
//!
//! 1. **The test set is a protected anchor.** Trials are scored on a
//!    validation split carved out of the training pool; `test` is only
//!    computed when `with_test` is set, which happens once — at final
//!    acceptance of a champion. A search loop that could query the test set
//!    would turn it into part of the optimization surface.
//! 2. **Splits are seeded shuffles, not file order.** The demo tasks split by
//!    row order; the digits CSV is class-grouped, so a search loop could
//!    silently exploit that. Shuffling with a dedicated split seed makes
//!    evaluations comparable across trials and runs.
//!
//! Scores are higher-is-better for every task: accuracy for digits,
//! `1 - nrmse` for timeseries.

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

use crate::Result;
use crate::data::Connectome;
use crate::readout::Readout;
use crate::reservoir::{Params as ReservoirParams, Reservoir};
use crate::rsi::config::TrialConfig;
use crate::tasks::{digits, timeseries};

/// Which demo task is being improved.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TaskKind {
    /// handwritten-digit reading (classification accuracy)
    Digits,
    /// NARMA-style series prediction (`1 - nrmse`)
    Timeseries,
}

impl TaskKind {
    /// the task name used on the CLI and in the bank
    pub fn name(&self) -> &'static str {
        match self {
            TaskKind::Digits => "digits",
            TaskKind::Timeseries => "timeseries",
        }
    }
}

/// Scores for one evaluated config. `test` is `None` unless explicitly allowed.
#[derive(Debug, Clone, Copy)]
pub struct Outcome {
    /// score on the training split
    pub train: f64,
    /// score on the validation split — the selection signal
    pub val: f64,
    /// score on the held-out test split (protected)
    pub test: Option<f64>,
    /// validation split size
    pub n_val: usize,
    /// test split size
    pub n_test: usize,
}

/// A task's data, split once per run and reused by every trial.
pub struct Evaluator<'a> {
    conn: &'a Connectome,
    task: TaskKind,
    /// digits: shuffled dataset row indices (split order)
    idx: Vec<usize>,
    /// digits: (images, labels)
    images: Vec<f64>,
    labels: Vec<usize>,
    /// timeseries: (inputs, targets) after washout
    series: (Vec<f64>, Vec<f64>),
    /// row counts of (train, val, test)
    pub splits: (usize, usize, usize),
}

impl<'a> Evaluator<'a> {
    /// Build the evaluator: load the task data and carve the splits with
    /// `split_seed` (independent of any trial seed).
    pub fn new(conn: &'a Connectome, task: TaskKind, split_seed: u64) -> Result<Self> {
        match task {
            TaskKind::Digits => {
                let (images, labels) = digits::load_digits()?;
                let n = labels.len();
                let mut idx: Vec<usize> = (0..n).collect();
                // dedicated RNG so the split never moves between trials/runs
                idx.shuffle(&mut StdRng::seed_from_u64(split_seed));
                let (tr, va, te) = split3(n, 0.60, 0.20);
                Ok(Self {
                    conn,
                    task,
                    idx,
                    images,
                    labels,
                    series: (Vec::new(), Vec::new()),
                    splits: (tr, va, te),
                })
            }
            TaskKind::Timeseries => {
                // one fixed series (same as the demo); states are recomputed
                // per config inside `features`
                let (u, y) = timeseries::make(1500, 1);
                let washout = 100;
                let series = (u[washout..].to_vec(), y[washout..].to_vec());
                let rows = series.1.len();
                let (tr, va, te) = (800usize, 300usize, rows - 1100);
                Ok(Self {
                    conn,
                    task,
                    idx: Vec::new(),
                    images: Vec::new(),
                    labels: Vec::new(),
                    series,
                    splits: (tr, va, te),
                })
            }
        }
    }

    /// Reservoir features for `rows` under `cfg`. This is the expensive step —
    /// the brain must actually be driven — so L3/L4 call it once per champion
    /// and reuse the matrix.
    pub fn features(&self, cfg: &TrialConfig, rows: &[usize]) -> Vec<f64> {
        match self.task {
            TaskKind::Digits => {
                let mut imgs = vec![0.0f64; rows.len() * 64];
                for (r, &i) in rows.iter().enumerate() {
                    imgs[r * 64..(r + 1) * 64].copy_from_slice(&self.images[i * 64..i * 64 + 64]);
                }
                digits::extract_features(
                    self.conn,
                    &imgs,
                    &digits::FeatParams {
                        spectral_radius: cfg.spectral_radius,
                        leak: cfg.leak,
                        input_scale: cfg.input_scale,
                        inhibitory_fraction: cfg.inhibitory_fraction,
                        seed: cfg.seed,
                    },
                )
            }
            TaskKind::Timeseries => {
                let (u, _) = &self.series;
                let mut res = Reservoir::new(
                    &self.conn.w,
                    self.conn.n,
                    &ReservoirParams {
                        spectral_radius: cfg.spectral_radius,
                        leak: cfg.leak,
                        input_scale: cfg.input_scale,
                        inhibitory_fraction: cfg.inhibitory_fraction,
                        seed: cfg.seed,
                    },
                );
                // the series is one recurrent trajectory — it must be run in
                // full — but the contract is the same as digits: exactly one
                // feature row per requested row, in order
                let all = res.run(u, u.len(), 1, 0);
                let n = self.conn.n;
                let mut out = Vec::with_capacity(rows.len() * n);
                for &r in rows {
                    out.extend_from_slice(&all[r * n..(r + 1) * n]);
                }
                out
            }
        }
    }

    /// Row indices of a split, in canonical order. For timeseries these index
    /// into the post-washout series.
    pub fn rows(&self, split: Split) -> Vec<usize> {
        let (tr, va, te) = self.splits;
        match (self.task, split) {
            (TaskKind::Digits, Split::Train) => self.idx[..tr].to_vec(),
            (TaskKind::Digits, Split::Val) => self.idx[tr..tr + va].to_vec(),
            (TaskKind::Digits, Split::Test) => self.idx[tr + va..].to_vec(),
            (TaskKind::Timeseries, Split::Train) => (0..tr).collect(),
            (TaskKind::Timeseries, Split::Val) => (tr..tr + va).collect(),
            (TaskKind::Timeseries, Split::Test) => (tr + va..tr + va + te).collect(),
        }
    }
    /// Targets for a set of rows (one-hot for digits, scalar series otherwise).
    pub fn targets(&self, rows: &[usize], out_dim: usize) -> Vec<f64> {
        match self.task {
            TaskKind::Digits => {
                let mut t = vec![0.0f64; rows.len() * out_dim];
                for (r, &i) in rows.iter().enumerate() {
                    t[r * out_dim + self.labels[i]] = 1.0;
                }
                t
            }
            TaskKind::Timeseries => rows.iter().map(|&i| self.series.1[i]).collect(),
        }
    }

    /// output width of the readout for this task
    pub fn out_dim(&self) -> usize {
        match self.task {
            TaskKind::Digits => 10,
            TaskKind::Timeseries => 1,
        }
    }

    /// which task this evaluator scores
    pub fn task_kind(&self) -> TaskKind {
        self.task
    }

    /// Per-row miss mask for a prediction: classification error for digits;
    /// residual larger than mean + 0.5·std of |residual| for timeseries.
    pub fn miss_mask(&self, pred: &[f64], rows: &[usize]) -> Vec<bool> {
        match self.task {
            TaskKind::Digits => rows
                .iter()
                .enumerate()
                .map(|(r, &i)| argmax(&pred[r * 10..r * 10 + 10]) != self.labels[i])
                .collect(),
            TaskKind::Timeseries => {
                let res: Vec<f64> = rows
                    .iter()
                    .enumerate()
                    .map(|(r, &i)| (pred[r] - self.series.1[i]).abs())
                    .collect();
                let m = res.iter().sum::<f64>() / res.len() as f64;
                let sd = (res.iter().map(|v| (v - m).powi(2)).sum::<f64>() / res.len() as f64)
                    .sqrt();
                res.iter().map(|&e| e > m + 0.5 * sd).collect()
            }
        }
    }

    /// Per-bucket error rates on `rows`: per-class miss rates for digits
    /// (index = class); a single bucket with the mean normalized residual for
    /// timeseries (index 0).
    pub fn bucket_errors(&self, pred: &[f64], rows: &[usize], out_dim: usize) -> Vec<f64> {
        match self.task {
            TaskKind::Digits => {
                let mut err = vec![0.0f64; out_dim];
                let mut cnt = vec![0.0f64; out_dim];
                for (r, &i) in rows.iter().enumerate() {
                    let c = self.labels[i];
                    cnt[c] += 1.0;
                    if argmax(&pred[r * out_dim..r * out_dim + out_dim]) != c {
                        err[c] += 1.0;
                    }
                }
                for c in 0..out_dim {
                    if cnt[c] > 0.0 {
                        err[c] /= cnt[c];
                    }
                }
                err
            }
            TaskKind::Timeseries => {
                let m = rows
                    .iter()
                    .enumerate()
                    .map(|(r, &i)| (pred[r] - self.series.1[i]).powi(2))
                    .sum::<f64>()
                    / rows.len() as f64;
                let scale = self
                    .series
                    .1
                    .iter()
                    .copied()
                    .fold(0.0f64, |a, b| a.max(b.abs()))
                    .max(1e-9);
                vec![m.sqrt() / scale]
            }
        }
    }

    /// The bucket a row belongs to (class index, or 0 for timeseries).
    pub fn bucket_of(&self, row: usize) -> usize {
        match self.task {
            TaskKind::Digits => self.labels[row],
            TaskKind::Timeseries => 0,
        }
    }

    /// reservoir width — feature rows are this long
    pub fn features_dim(&self) -> usize {
        self.conn.n
    }

    /// the true label of a dataset row (digits only)
    pub fn label_of(&self, row: usize) -> usize {
        self.labels[row]
    }

    /// the target value of a series row (timeseries only)
    pub fn target_value(&self, row: usize) -> f64 {
        self.series.1[row]
    }

    /// Score predictions against the truth of `rows`.
    pub fn score(&self, pred: &[f64], rows: &[usize]) -> f64 {
        match self.task {
            TaskKind::Digits => {
                let mut hits = 0usize;
                for (r, i) in rows.iter().enumerate() {
                    let row = &pred[r * 10..r * 10 + 10];
                    let argmax = row
                        .iter()
                        .enumerate()
                        .max_by(|a, b| a.1.total_cmp(b.1))
                        .map(|(k, _)| k)
                        .unwrap_or(0);
                    if argmax == self.labels[*i] {
                        hits += 1;
                    }
                }
                hits as f64 / rows.len() as f64
            }
            TaskKind::Timeseries => {
                let truth: Vec<f64> = rows.iter().map(|&i| self.series.1[i]).collect();
                1.0 - timeseries::nrmse(pred, &truth)
            }
        }
    }

    /// Full evaluation of a config: features → ridge fit on train → score.
    ///
    /// `with_test` must stay `false` during search; the driver sets it once
    /// for the accepted champion.
    pub fn evaluate(&self, cfg: &TrialConfig, with_test: bool) -> Result<Outcome> {
        let out_dim = self.out_dim();
        let train_rows = self.rows(Split::Train);
        let val_rows = self.rows(Split::Val);
        let test_rows = if with_test { self.rows(Split::Test) } else { Vec::new() };

        // one feature extraction covering train|val|test rows, then sliced
        let mut all = train_rows.clone();
        all.extend_from_slice(&val_rows);
        all.extend_from_slice(&test_rows);
        let feats = self.features(cfg, &all);
        let n = self.conn.n;

        let take = |from: usize, len: usize| feats[from * n..(from + len) * n].to_vec();
        let f_train = take(0, train_rows.len());
        let f_val = take(train_rows.len(), val_rows.len());
        let f_test = take(train_rows.len() + val_rows.len(), test_rows.len());

        let mut ro = Readout::new(cfg.ridge);
        ro.fit(&f_train, n, &self.targets(&train_rows, out_dim), out_dim);
        let val_pred = ro.predict(&f_val, n)?;
        let val = self.score(&val_pred, &val_rows);
        let train_pred = ro.predict(&f_train, n)?;
        let train = self.score(&train_pred, &train_rows);
        let test = if with_test {
            let p = ro.predict(&f_test, n)?;
            Some(self.score(&p, &test_rows))
        } else {
            None
        };
        Ok(Outcome {
            train,
            val,
            test,
            n_val: val_rows.len(),
            n_test: test_rows.len(),
        })
    }

    /// Acceptance check against the seed-cherry-picking hazard: re-score the
    /// config under `n` seeds and return the median validation score plus the
    /// max-min spread across seeds.
    pub fn median_val(&self, cfg: &TrialConfig, n: usize) -> Result<(f64, f64)> {
        let mut vals = Vec::new();
        for k in 0..n.max(1) {
            let mut c = *cfg;
            c.seed = (cfg.seed + k as u64) % 64;
            vals.push(self.evaluate(&c, false)?.val);
        }
        vals.sort_by(f64::total_cmp);
        let med = vals[vals.len() / 2];
        let spread = vals.last().copied().unwrap_or(med) - vals.first().copied().unwrap_or(med);
        Ok((med, spread))
    }
}

/// The three splits.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Split {
    /// training rows (also the curriculum/stream pool)
    Train,
    /// validation rows — the selection signal
    Val,
    /// held-out test rows — protected, touched once
    Test,
}

fn split3(n: usize, f_tr: f64, f_va: f64) -> (usize, usize, usize) {
    let tr = (n as f64 * f_tr) as usize;
    let va = (n as f64 * f_va) as usize;
    (tr, va, n - tr - va)
}

/// argmax of a score row
fn argmax(row: &[f64]) -> usize {
    row.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(k, _)| k)
        .unwrap_or(0)
}
