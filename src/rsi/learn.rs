//! Learning machinery for the improvement loop.
//!
//! Three pieces, each used by a different level:
//!
//! * [`fit_ridge`] — weighted ridge regression (L3 curriculum reweights the
//!   training pool by what's been misclassified; the stock [`crate::readout`]
//!   has no weights, so the weighted normal equations live here).
//! * [`OnlineReadout`] — the same readout updated one sample at a time by
//!   recursive least squares (Sherman–Morrison on `(XᵀX + λI)⁻¹`), the L4
//!   deployment-adaptation mechanism.
//! * [`GatedAdapter`] — L4's safety inheritance: a candidate adaptation is
//!   validated against a fixed gate set before it may replace the accepted
//!   readout, otherwise the change is rolled back.
//!
//! Nothing here touches the connectome; all learning is still "a thin linear
//! layer on top of the fixed brain", exactly as the demo intends.

use faer::Mat;
use faer::linalg::solvers::Solve;

/// A fitted linear readout: `w` of shape `(n + 1) * out_dim` (bias last row).
#[derive(Debug, Clone)]
pub struct LinearSolution {
    /// flattened `(n + 1) x out_dim` weights, bias row last
    pub w: Vec<f64>,
    /// reservoir width
    pub n: usize,
    /// output width
    pub out_dim: usize,
}

/// Weighted ridge fit: minimize `Σ_i w_i ‖x̃_iᵀW − y_i‖² + λ‖W‖²`, where
/// `x̃` is the state with a bias column appended. `weights == None` is the
/// plain unweighted case. Panics on malformed shapes, like `Readout::fit`.
pub fn fit_ridge(
    states: &[f64],
    n: usize,
    targets: &[f64],
    out_dim: usize,
    weights: Option<&[f64]>,
    ridge: f64,
) -> LinearSolution {
    assert_eq!(states.len() % n, 0, "states length must be a multiple of n");
    let rows = states.len() / n;
    assert_eq!(
        targets.len(),
        rows * out_dim,
        "targets must be rows x out_dim ({} x {})",
        rows,
        out_dim
    );
    let nf = n + 1;
    if let Some(w) = weights {
        assert_eq!(w.len(), rows, "weights must have one entry per row");
    }

    // scale rows by sqrt(weight) so the normal equations become XᵀWX
    let x = Mat::<f64>::from_fn(rows, nf, |t, i| {
        let v = if i < n { states[t * n + i] } else { 1.0 };
        match weights {
            Some(w) => v * w[t].sqrt(),
            None => v,
        }
    });
    let y = Mat::<f64>::from_fn(rows, out_dim, |t, j| {
        let v = targets[t * out_dim + j];
        match weights {
            Some(w) => v * w[t].sqrt(),
            None => v,
        }
    });
    let mut a = x.as_ref().adjoint() * x.as_ref();
    for i in 0..nf {
        a[(i, i)] += ridge;
    }
    let b = x.as_ref().adjoint() * y.as_ref();
    use faer::linalg::solvers::Llt;
    let w = match Llt::new(a.as_ref(), faer::Side::Lower) {
        Ok(llt) => llt.solve(b.as_ref()),
        Err(_) => a.partial_piv_lu().solve(b.as_ref()),
    };
    let mut flat = vec![0.0; nf * out_dim];
    for i in 0..nf {
        for j in 0..out_dim {
            flat[i * out_dim + j] = w[(i, j)];
        }
    }
    LinearSolution {
        w: flat,
        n,
        out_dim,
    }
}

impl LinearSolution {
    /// Predict all rows of `states` (`rows x n`) → `rows * out_dim`.
    pub fn predict(&self, states: &[f64]) -> Vec<f64> {
        let rows = states.len() / self.n;
        let n = self.n;
        let mut out = vec![0.0f64; rows * self.out_dim];
        for r in 0..rows {
            for j in 0..self.out_dim {
                let mut s = self.w[n * self.out_dim + j]; // bias row
                for i in 0..n {
                    s += states[r * n + i] * self.w[i * self.out_dim + j];
                }
                out[r * self.out_dim + j] = s;
            }
        }
        out
    }
}

/// A readout that adapts online by recursive least squares.
///
/// State is the pair `(W, P)` with `P = (XᵀX + λI)⁻¹`. Batch init from the
/// closed form, then each new sample `(x, y)` applies the rank-1 update
///
/// ```text
/// k   = P x / (1 + xᵀ P x)
/// W  += k (y − xᵀW)ᵀ
/// P  -= (P x)(P x)ᵀ / (1 + xᵀ P x)
/// ```
///
/// which is exactly batch ridge on everything seen so far — the algebra is
/// identical, so online adaptation never leaves the model class.
pub struct OnlineReadout {
    sol: LinearSolution,
    /// inverse of the normal-equations matrix, `(n+1)²`
    p: Vec<f64>,
    /// number of rank-1 updates applied
    pub updates: u64,
    /// version bumps on every accepted gate (see GatedAdapter)
    pub version: u64,
}

impl OnlineReadout {
    /// Closed-form init on a batch, including `P = (XᵀX + λI)⁻¹`.
    pub fn fit(states: &[f64], n: usize, targets: &[f64], out_dim: usize, ridge: f64) -> Self {
        let sol = fit_ridge(states, n, targets, out_dim, None, ridge);
        let rows = states.len() / n;
        let nf = n + 1;
        let x = Mat::<f64>::from_fn(rows, nf, |t, i| {
            if i < n {
                states[t * n + i]
            } else {
                1.0
            }
        });
        let mut a = x.as_ref().adjoint() * x.as_ref();
        for i in 0..nf {
            a[(i, i)] += ridge;
        }
        use faer::linalg::solvers::Llt;
        let eye = Mat::<f64>::identity(nf, nf);
        let p = match Llt::new(a.as_ref(), faer::Side::Lower) {
            Ok(llt) => llt.solve(eye.as_ref()),
            Err(_) => a.partial_piv_lu().solve(eye.as_ref()),
        };
        let mut flat = vec![0.0; nf * nf];
        for i in 0..nf {
            for j in 0..nf {
                flat[i * nf + j] = p[(i, j)];
            }
        }
        Self {
            sol,
            p: flat,
            updates: 0,
            version: 0,
        }
    }

    /// A copy of the current (W, P) state — used to seed candidates and
    /// reconstruct them after a rollback.
    pub fn fork(&self) -> Self {
        Self {
            sol: self.sol.clone(),
            p: self.p.clone(),
            updates: self.updates,
            version: self.version,
        }
    }

    /// One rank-1 RLS update. `x` is a raw reservoir state (length `n`); the
    /// bias is appended internally.
    pub fn update(&mut self, x: &[f64], y: &[f64]) {
        let n = self.sol.n;
        let nf = n + 1;
        debug_assert_eq!(x.len(), n);
        debug_assert_eq!(y.len(), self.sol.out_dim);
        // xb = [x; 1]
        let mut xb = vec![0.0f64; nf];
        xb[..n].copy_from_slice(x);
        xb[n] = 1.0;
        // px = P xb
        let mut px = vec![0.0f64; nf];
        for i in 0..nf {
            let mut acc = 0.0;
            for j in 0..nf {
                acc += self.p[i * nf + j] * xb[j];
            }
            px[i] = acc;
        }
        let denom: f64 = 1.0 + (0..nf).map(|i| xb[i] * px[i]).sum::<f64>();
        let k: Vec<f64> = px.iter().map(|v| v / denom).collect();
        // err = y - xᵀW
        for j in 0..self.sol.out_dim {
            let mut pred = self.sol.w[n * self.sol.out_dim + j]; // bias
            for i in 0..n {
                pred += x[i] * self.sol.w[i * self.sol.out_dim + j];
            }
            let e = y[j] - pred;
            for i in 0..nf {
                self.sol.w[i * self.sol.out_dim + j] += k[i] * e;
            }
        }
        // P -= (P x)(P x)ᵀ / denom  (symmetric; write both halves)
        for i in 0..nf {
            for j in 0..nf {
                self.p[i * nf + j] -= px[i] * px[j] / denom;
            }
        }
        self.updates += 1;
    }

    /// Current weights as a solution (for scoring/prediction).
    pub fn solution(&self) -> &LinearSolution {
        &self.sol
    }

    /// Predict all rows of `states`.
    pub fn predict(&self, states: &[f64]) -> Vec<f64> {
        self.sol.predict(states)
    }
}

/// What the gate decided at a checkpoint.
pub enum GateDecision {
    /// candidate validated better; it becomes the accepted readout
    Accepted {
        /// gate score of the now-accepted readout
        new_val: f64,
        /// gate score the candidate had
        candidate_val: f64,
    },
    /// candidate regressed; accepted readout kept, candidate reset
    RolledBack {
        /// gate score of the rejected candidate
        candidate_val: f64,
        /// gate score of the retained readout
        kept_val: f64,
    },
}

/// L4's safety inheritance: an accepted readout plus a candidate being
/// evaluated, gated on a fixed validation set before any change persists.
pub struct GatedAdapter {
    accepted: OnlineReadout,
    accepted_val: f64,
    candidate: OnlineReadout,
    /// number of accepted adaptations so far
    pub adaptations: u64,
    /// number of rolled-back adaptations so far
    pub rollbacks: u64,
}

impl GatedAdapter {
    /// Init from a batch-fitted readout whose gate score is `accepted_val`.
    pub fn new(accepted: OnlineReadout, accepted_val: f64) -> Self {
        let candidate = OnlineReadout {
            sol: accepted.solution().clone(),
            p: accepted.p.clone(),
            updates: accepted.updates,
            version: accepted.version,
        };
        Self {
            accepted,
            accepted_val,
            candidate,
            adaptations: 0,
            rollbacks: 0,
        }
    }

    /// Mutable access to the candidate: stream updates into it, never into
    /// the accepted readout.
    pub fn candidate(&mut self) -> &mut OnlineReadout {
        &mut self.candidate
    }

    /// The accepted readout (what production would use).
    pub fn accepted(&self) -> &OnlineReadout {
        &self.accepted
    }

    /// Gate score of the currently accepted readout.
    pub fn accepted_val(&self) -> f64 {
        self.accepted_val
    }

    /// Validate the candidate against the gate set; keep it only if it beats
    /// the accepted readout. Either way the candidate resets to the (possibly
    /// new) accepted state for the next window.
    pub fn gate(&mut self, score: impl Fn(&OnlineReadout) -> f64) -> GateDecision {
        let candidate_val = score(&self.candidate);
        if candidate_val >= self.accepted_val {
            self.candidate.version = self.accepted.version + 1;
            self.accepted_val = candidate_val;
            std::mem::swap(&mut self.accepted, &mut self.candidate);
            self.adaptations += 1;
            GateDecision::Accepted {
                new_val: self.accepted_val,
                candidate_val,
            }
        } else {
            self.candidate = OnlineReadout {
                sol: self.accepted.solution().clone(),
                p: self.accepted.p.clone(),
                updates: self.accepted.updates,
                version: self.accepted.version,
            };
            self.rollbacks += 1;
            GateDecision::RolledBack {
                candidate_val,
                kept_val: self.accepted_val,
            }
        }
    }
}
