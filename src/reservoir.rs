use faer::linalg::solvers::Eigen;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};

/// Compressed-sparse-row matrix used for the recurrence hot loop.
///
/// The connectome is sparse (~1% of pairs connect), so the state update runs as
/// a sparse matrix-vector product, like the SciPy path of the Python original.
#[derive(Debug, Clone)]
pub struct Csr {
    nrows: usize,
    ncols: usize,
    indptr: Vec<usize>,
    indices: Vec<usize>,
    data: Vec<f64>,
}

impl Csr {
    /// Build a CSR matrix from a dense row-major matrix, keeping nonzero entries.
    pub fn from_dense(w: &[f64], n: usize) -> Self {
        let mut indptr = Vec::with_capacity(n + 1);
        let mut indices = Vec::new();
        let mut data = Vec::new();
        indptr.push(0);
        for i in 0..n {
            for (j, &v) in w[i * n..(i + 1) * n].iter().enumerate() {
                if v != 0.0 {
                    indices.push(j);
                    data.push(v);
                }
            }
            indptr.push(indices.len());
        }
        Self {
            nrows: n,
            ncols: n,
            indptr,
            indices,
            data,
        }
    }

    /// `out = W x`
    pub fn matvec(&self, x: &[f64], out: &mut [f64]) {
        debug_assert_eq!(self.ncols, x.len());
        debug_assert_eq!(self.nrows, out.len());
        for (i, o) in out.iter_mut().enumerate() {
            let mut acc = 0.0;
            for k in self.indptr[i]..self.indptr[i + 1] {
                acc += self.data[k] * x[self.indices[k]];
            }
            *o = acc;
        }
    }

    /// Expand back to dense row-major.
    pub fn to_dense(&self) -> Vec<f64> {
        let mut d = vec![0.0; self.nrows * self.ncols];
        for i in 0..self.nrows {
            for k in self.indptr[i]..self.indptr[i + 1] {
                d[i * self.ncols + self.indices[k]] = self.data[k];
            }
        }
        d
    }

    /// Number of stored (nonzero) entries.
    pub fn nnz(&self) -> usize {
        self.data.len()
    }
}

/// The reservoir: the fly's brain, run as a fixed dynamical system.
///
/// The connectome gives a wiring matrix W — who connects to whom, weighted by real
/// synapse counts. We do not train it. It is the animal's brain; you don't get to
/// edit it. Instead we drive it with input and read the pattern of activity it
/// produces, and train only a thin readout on top (see [`crate::readout`]). That
/// split — fixed biological recurrence, trained linear readout — is reservoir
/// computing, and using a connectome as the reservoir is the "Biological
/// Processing Unit" idea from the recent literature.
///
/// One knob matters for it to work at all: the spectral radius. W is scaled so its
/// largest eigenvalue magnitude sits just below ~1 and the network has the
/// echo-state property — activity fades rather than blows up or dies. Everything
/// else is the brain as nature wired it.
pub struct Reservoir {
    /// number of neurons
    pub n: usize,
    /// leak rate of the state update
    pub leak: f64,
    w: Csr,
    win: Vec<f64>,
    win_dim: usize,
    input_scale: f64,
    rng: StdRng,
}

/// Construction parameters for a [`Reservoir`].
#[derive(Debug, Clone)]
pub struct Params {
    /// target spectral radius (the echo-state condition)
    pub spectral_radius: f64,
    /// leak rate: `x <- (1-leak) x + leak tanh(W x + Win u)`
    pub leak: f64,
    /// scale of the random input weights
    pub input_scale: f64,
    /// fraction of neurons assigned inhibitory (negative out-weights)
    pub inhibitory_fraction: f64,
    /// rng seed for signs and input weights
    pub seed: u64,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            spectral_radius: 0.95,
            leak: 0.3,
            input_scale: 1.0,
            inhibitory_fraction: 0.2,
            seed: 0,
        }
    }
}

impl Reservoir {
    /// Build a reservoir from a dense row-major wiring matrix with `n` neurons.
    ///
    /// Assigns Dale-style inhibitory signs to a fraction of neurons, then scales W
    /// to the target spectral radius. The wiring itself is never trained.
    pub fn new(w: &[f64], n: usize, p: &Params) -> Self {
        let mut rng = StdRng::seed_from_u64(p.seed);
        let mut w = w.to_vec();

        // Dale-ish signs: a fraction of neurons are inhibitory (negative out-weights).
        // The synapse-count matrix is unsigned; real circuits are E/I, and giving the
        // reservoir a sign structure makes its dynamics rich enough to compute with.
        let mut order: Vec<usize> = (0..n).collect();
        order.shuffle(&mut rng);
        let n_inh = (p.inhibitory_fraction * n as f64) as usize;
        for &i in &order[..n_inh.min(n)] {
            for v in &mut w[i * n..(i + 1) * n] {
                *v = -*v;
            }
        }

        // scale to the target spectral radius (the echo-state condition)
        let radius = spectral_radius(&w, n);
        if radius > 0.0 {
            let s = p.spectral_radius / radius;
            for v in &mut w {
                *v *= s;
            }
        }

        Self {
            n,
            leak: p.leak,
            w: Csr::from_dense(&w, n),
            win: Vec::new(),
            win_dim: 0,
            input_scale: p.input_scale,
            rng,
        }
    }

    /// Largest eigenvalue magnitude of a dense row-major matrix.
    pub fn spectral_radius_of(w: &[f64], n: usize) -> f64 {
        spectral_radius(w, n)
    }

    /// The (signed, scaled) wiring matrix as a sparse matrix.
    pub fn w(&self) -> &Csr {
        &self.w
    }

    /// The wiring matrix expanded to dense row-major (what the Python tests call
    /// `res.W.toarray()`). Useful for verification; the hot loop uses [`Csr`].
    pub fn w_dense(&self) -> Vec<f64> {
        self.w.to_dense()
    }

    /// Lazily build (and cache) random input weights of shape `n x in_dim`.
    pub fn ensure_input_weights(&mut self, in_dim: usize) {
        if self.win_dim != in_dim {
            self.win = (0..self.n * in_dim)
                .map(|_| self.rng.random_range(-1.0..1.0) * self.input_scale)
                .collect();
            self.win_dim = in_dim;
        }
    }

    /// Drive the brain with a sequence of inputs; return the state at each step.
    ///
    /// `inputs` is `t * in_dim` row-major (one `in_dim` row per timestep).
    /// Returns `(t - washout) * n` — the activity of every neuron over time, which
    /// is what the readout learns to decode. `washout >= t` yields an empty Vec,
    /// like Python's `states[washout:]`.
    pub fn run(&mut self, inputs: &[f64], t: usize, in_dim: usize, washout: usize) -> Vec<f64> {
        assert_eq!(inputs.len(), t * in_dim, "inputs must be t * in_dim");
        self.ensure_input_weights(in_dim);
        let mut x = vec![0.0; self.n];
        let mut pre = vec![0.0; self.n];
        let mut states = Vec::with_capacity(t.saturating_sub(washout) * self.n);
        for step in 0..t {
            self.w.matvec(&x, &mut pre);
            for i in 0..self.n {
                let mut p = pre[i];
                for (k, &u) in self.win[i * in_dim..(i + 1) * in_dim].iter().enumerate() {
                    p += u * inputs[step * in_dim + k];
                }
                x[i] = (1.0 - self.leak) * x[i] + self.leak * p.tanh();
            }
            if step >= washout {
                states.extend_from_slice(&x);
            }
        }
        states
    }

    /// One state update `x <- (1-leak) x + leak tanh(W x + Win u)` using the
    /// cached input weights. Call [`Reservoir::ensure_input_weights`] first.
    pub fn step(&self, x: &mut [f64], input: &[f64]) {
        let in_dim = self.win_dim;
        assert_eq!(
            input.len(),
            in_dim,
            "input width does not match the cached input weights"
        );
        let mut pre = vec![0.0; self.n];
        self.w.matvec(x, &mut pre);
        for i in 0..self.n {
            let mut p = pre[i];
            for (&w, &u) in self.win[i * in_dim..(i + 1) * in_dim]
                .iter()
                .zip(input.iter())
            {
                p += w * u;
            }
            x[i] = (1.0 - self.leak) * x[i] + self.leak * p.tanh();
        }
    }
}

/// Largest eigenvalue magnitude |λ_max|.
///
/// Cheap path: power iteration — for `x_{k+1} = A x_k / ‖x_k‖`, the ratio
/// `‖A x_k‖ / ‖x_k‖` converges to |λ_max| whenever the dominant eigenvalue is
/// unique in magnitude (a sign flip or complex rotation of the dominant pair
/// does not matter, the norm still converges to |λ|). A full QR
/// eigendecomposition is the fallback for the rare cases where the iteration
/// fails to converge (equal-magnitude dominant pair, pathological start
/// vector); that is what the NumPy-only path of the Python original always did.
fn spectral_radius(w: &[f64], n: usize) -> f64 {
    if let Some(r) = power_radius(&Csr::from_dense(w, n), n) {
        return r;
    }
    let a = faer::Mat::<f64>::from_fn(n, n, |i, j| w[i * n + j]);
    let evd = Eigen::new_from_real(a.as_ref()).expect("eigendecomposition did not converge");
    let s = evd.S();
    let mut r = 0.0f64;
    for j in 0..n {
        let z = s[j];
        r = r.max(z.re * z.re + z.im * z.im);
    }
    r.sqrt()
}

const POWER_ITERS: usize = 5000;

fn power_radius(a: &Csr, n: usize) -> Option<f64> {
    let mut x = vec![1.0 / (n as f64).sqrt(); n]; // deterministic start
    let mut y = vec![0.0; n];
    let mut prev: Option<f64> = None;
    let mut stable = 0;
    for _ in 0..POWER_ITERS {
        a.matvec(&x, &mut y);
        let r = y.iter().map(|v| v * v).sum::<f64>().sqrt();
        if r == 0.0 {
            return Some(0.0); // zero matrix
        }
        let converged = match prev {
            Some(p) => (r - p).abs() / p.max(1.0) < 1e-9,
            None => false,
        };
        if converged {
            stable += 1;
            if stable >= 3 {
                return Some(r);
            }
        } else {
            stable = 0;
        }
        prev = Some(r);
        for i in 0..n {
            x[i] = y[i] / r;
        }
    }
    None
}
