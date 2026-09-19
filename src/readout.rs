use faer::Mat;
use faer::linalg::solvers::Solve;

use crate::{Result, WetwareError};

/// The readout: the only thing that learns.
///
/// Given the brain's activity (the reservoir states) and the answers you want, fit
/// a single linear layer by ridge regression — a closed-form least-squares solve,
/// no backprop, no epochs. This is the whole "training": the brain stays fixed,
/// and a matrix of weights on top learns to read its activity as a decision.
///
/// `Wout = (XᵀX + λI)⁻¹ XᵀY`
///
/// That's it. If the reservoir is any good, a linear readout of it can do the task.
pub struct Readout {
    /// ridge regularization λ
    pub ridge: f64,
    w: Option<Vec<f64>>,
    n: usize,
    out_dim: usize,
}

impl Default for Readout {
    fn default() -> Self {
        Self::new(1e-2)
    }
}

/// Design matrix entry: reservoir state with a bias column appended.
#[inline]
fn design(states: &[f64], n: usize, t: usize, i: usize) -> f64 {
    if i < n { states[t * n + i] } else { 1.0 }
}

impl Readout {
    /// A readout with the given ridge regularization.
    pub fn new(ridge: f64) -> Self {
        Self {
            ridge,
            w: None,
            n: 0,
            out_dim: 0,
        }
    }

    /// Fit on reservoir states (`rows x n`, row-major) and targets
    /// (`rows x out_dim`, row-major). Returns `self` for chaining.
    ///
    /// Panics on malformed inputs (wrong `targets` length, or `states` not a
    /// multiple of `n`) — these are programmer errors, like the `ValueError`
    /// the Python original raised.
    pub fn fit(&mut self, states: &[f64], n: usize, targets: &[f64], out_dim: usize) -> &mut Self {
        assert_eq!(states.len() % n, 0, "states length must be a multiple of n");
        let rows = states.len() / n;
        assert_eq!(
            targets.len(),
            rows * out_dim,
            "targets must be rows x out_dim ({} x {})",
            rows,
            out_dim
        );
        let n_feat = n + 1;
        let ridge = self.ridge;

        // Design matrix X (rows x n_feat) with a bias column, built once; A and B
        // are then single (multithreaded) gemms: A = XᵀX + λI, B = XᵀY.
        let x = Mat::<f64>::from_fn(rows, n_feat, |t, i| design(states, n, t, i));
        let mut a = x.as_ref().adjoint() * x.as_ref();
        for i in 0..n_feat {
            a[(i, i)] += ridge;
        }
        let y = Mat::<f64>::from_fn(rows, out_dim, |t, j| targets[t * out_dim + j]);
        let b = x.as_ref().adjoint() * y.as_ref();

        // A is symmetric positive definite (ridge λ > 0), so Cholesky; LU as a
        // safety net if it ever fails.
        use faer::linalg::solvers::Llt;
        let w = match Llt::new(a.as_ref(), faer::Side::Lower) {
            Ok(llt) => llt.solve(b.as_ref()),
            Err(_) => a.partial_piv_lu().solve(b.as_ref()),
        };

        let mut flat = vec![0.0; n_feat * out_dim];
        for i in 0..n_feat {
            for j in 0..out_dim {
                flat[i * out_dim + j] = w[(i, j)];
            }
        }
        self.w = Some(flat);
        self.n = n;
        self.out_dim = out_dim;
        self
    }

    fn check_input(&self, states: &[f64], n: usize) -> Result<usize> {
        let w = self.w.as_ref().ok_or(WetwareError::Untrained)?;
        debug_assert_eq!(w.len(), (self.n + 1) * self.out_dim);
        if n != self.n {
            return Err(WetwareError::Other(format!(
                "states have n={n} but this readout was fitted with n={}",
                self.n
            )));
        }
        if !states.len().is_multiple_of(n) {
            return Err(WetwareError::Other(format!(
                "states length {} is not a multiple of n={n}",
                states.len()
            )));
        }
        Ok(states.len() / n)
    }

    /// Predict targets for reservoir states (`rows x n`).
    pub fn predict(&self, states: &[f64], n: usize) -> Result<Vec<f64>> {
        let rows = self.check_input(states, n)?;
        let w = self.w.as_ref().ok_or(WetwareError::Untrained)?;
        let mut out = vec![0.0; rows * self.out_dim];
        for r in 0..rows {
            for j in 0..self.out_dim {
                // bias row is the extra row of the design matrix
                let mut s = w[n * self.out_dim + j];
                for i in 0..n {
                    s += states[r * n + i] * w[i * self.out_dim + j];
                }
                out[r * self.out_dim + j] = s;
            }
        }
        Ok(out)
    }

    /// For one-hot targets: the argmax class per row.
    pub fn classify(&self, states: &[f64], n: usize) -> Result<Vec<usize>> {
        let p = self.predict(states, n)?;
        Ok(p.chunks(self.out_dim)
            .map(|row| {
                row.iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .map(|(i, _)| i)
                    .expect("readout has zero outputs")
            })
            .collect())
    }
}
