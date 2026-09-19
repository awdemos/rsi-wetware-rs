//! The editable surface of the RSI loop.
//!
//! The connectome is fixed — it is the animal's brain and never enters the
//! search space. Everything the loop is allowed to change about how that brain
//! is *used* is a [`TrialConfig`]: the reservoir hyperparameters, the readout
//! regularization, and the RNG seed. In the survey's terms this is the
//! improvement target; the search machinery in [`crate::rsi::optimize`] is the
//! improver.

use rand::Rng;
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};

/// Standard normal draw by Box–Muller (the project pins rand 0.9 while
/// rand_distr 0.6 pulls rand 0.10; hand-rolling avoids the version skew).
fn standard_normal(rng: &mut StdRng) -> f64 {
    let u1 = rng.random::<f64>().max(1e-300);
    let u2 = rng.random::<f64>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

fn gauss(rng: &mut StdRng, mean: f64, std: f64) -> f64 {
    if std <= 0.0 {
        mean
    } else {
        mean + std * standard_normal(rng)
    }
}

/// One point in the harness search space.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TrialConfig {
    /// target spectral radius (echo-state scaling of W)
    pub spectral_radius: f64,
    /// leak rate of the state update
    pub leak: f64,
    /// readout ridge λ (sampled in log-space)
    pub ridge: f64,
    /// scale of the random input weights
    pub input_scale: f64,
    /// fraction of neurons assigned inhibitory sign
    pub inhibitory_fraction: f64,
    /// rng seed for E/I signs and input weights
    pub seed: u64,
}

/// The neutral defaults, matching `Reservoir::Params::default` + readout default.
impl Default for TrialConfig {
    fn default() -> Self {
        Self {
            spectral_radius: 0.95,
            leak: 0.3,
            ridge: 1e-2,
            input_scale: 1.0,
            inhibitory_fraction: 0.2,
            seed: 0,
        }
    }
}

/// Per-task anchor: the config each task's `demo` ships with. Trial zero of
/// every search evaluates this, so the loop always reports improvement over
/// the status quo rather than an absolute number.
pub fn task_default(task: &str) -> TrialConfig {
    match task {
        // src/tasks/digits.rs `Params::default`
        "digits" => TrialConfig {
            spectral_radius: 1.1,
            leak: 0.5,
            ridge: 1.0,
            ..Default::default()
        },
        // src/tasks/timeseries.rs `Params::default`
        _ => TrialConfig::default(),
    }
}

/// Box bounds for every continuous knob, and the seed range.
#[derive(Debug, Clone, Copy)]
pub struct SearchSpace {
    /// (lo, hi) for the spectral radius
    pub spectral_radius: (f64, f64),
    /// (lo, hi) for the leak rate
    pub leak: (f64, f64),
    /// (lo, hi) for ridge λ, sampled in log-space
    pub ridge: (f64, f64),
    /// (lo, hi) for the input weight scale
    pub input_scale: (f64, f64),
    /// (lo, hi) for the inhibitory fraction
    pub inhibitory_fraction: (f64, f64),
    /// seeds are uniform in `0..seed_range`
    pub seed_range: u64,
}

impl Default for SearchSpace {
    fn default() -> Self {
        Self {
            spectral_radius: (0.6, 1.3),
            leak: (0.05, 0.95),
            ridge: (1e-4, 10.0),
            input_scale: (0.1, 3.0),
            inhibitory_fraction: (0.05, 0.45),
            seed_range: 64,
        }
    }
}

impl SearchSpace {
    /// Uniform sample over the space (`ridge` log-uniform).
    pub fn sample(&self, rng: &mut StdRng) -> TrialConfig {
        TrialConfig {
            spectral_radius: rng.random_range(self.spectral_radius.0..self.spectral_radius.1),
            leak: rng.random_range(self.leak.0..self.leak.1),
            ridge: (rng.random_range(self.ridge.0.ln()..self.ridge.1.ln())).exp(),
            input_scale: rng.random_range(self.input_scale.0..self.input_scale.1),
            inhibitory_fraction: rng
                .random_range(self.inhibitory_fraction.0..self.inhibitory_fraction.1),
            seed: rng.random_range(0..self.seed_range),
        }
    }

    /// Sample around `center` with per-dimension `std` (Gaussian; `ridge` in
    /// log-space, `seed` uniform). Used by the L2 adaptive rounds: the search
    /// distribution itself is what gets revised between rounds.
    pub fn sample_around(&self, rng: &mut StdRng, center: &TrialConfig, std: &TrialConfig) -> TrialConfig {
        let cfg = TrialConfig {
            spectral_radius: gauss(rng, center.spectral_radius, std.spectral_radius),
            leak: gauss(rng, center.leak, std.leak),
            ridge: gauss(rng, center.ridge.ln(), std.ridge.ln()).exp(),
            input_scale: gauss(rng, center.input_scale, std.input_scale),
            inhibitory_fraction: gauss(
                rng,
                center.inhibitory_fraction,
                std.inhibitory_fraction,
            ),
            seed: rng.random_range(0..self.seed_range),
        };
        self.clamp(cfg)
    }

    /// Pull every field back inside the box.
    pub fn clamp(&self, mut cfg: TrialConfig) -> TrialConfig {
        let cl = |v: f64, (lo, hi): (f64, f64)| v.clamp(lo, hi);
        cfg.spectral_radius = cl(cfg.spectral_radius, self.spectral_radius);
        cfg.leak = cl(cfg.leak, self.leak);
        cfg.ridge = cl(cfg.ridge, self.ridge);
        cfg.input_scale = cl(cfg.input_scale, self.input_scale);
        cfg.inhibitory_fraction = cl(cfg.inhibitory_fraction, self.inhibitory_fraction);
        cfg.seed %= self.seed_range.max(1);
        cfg
    }
}
