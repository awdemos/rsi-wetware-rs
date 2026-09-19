//! Task: predict a nonlinear time-series.
//!
//! Drive the brain with a random signal; ask the readout to output a value that
//! depends nonlinearly on several *past* inputs (a NARMA-style target). Getting it
//! right needs both memory and nonlinearity — exactly what a recurrent network of
//! neurons provides and a memoryless model can't fake. This is the classic reservoir
//! benchmark, run on a real connectome.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::data::Connectome;
use crate::readout::Readout;
use crate::reservoir::{Params as ReservoirParams, Reservoir};
use crate::{Metric, Report, Result};

/// NARMA-style input/target series: `u` is `t` samples, `y` the `t` targets.
pub fn make(t: usize, seed: u64) -> (Vec<f64>, Vec<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let u: Vec<f64> = (0..t).map(|_| rng.random_range(0.0..0.5)).collect();
    let mut y = vec![0.0; t];
    for t in 3..t {
        y[t] = 0.4 * y[t - 1] + 0.4 * u[t - 1] * u[t - 2] + 0.6 * u[t - 3].powi(2) + 0.1;
    }
    (u, y)
}

/// Parameters for [`demo`].
#[derive(Debug, Clone)]
pub struct Params {
    /// series length
    pub t: usize,
    /// initial steps dropped from the readout fit
    pub washout: usize,
    /// number of steps used for training
    pub train: usize,
    /// reservoir spectral radius
    pub spectral_radius: f64,
    /// reservoir leak rate
    pub leak: f64,
    /// readout ridge λ
    pub ridge: f64,
    /// reservoir seed
    pub seed: u64,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            t: 1500,
            washout: 100,
            train: 1000,
            spectral_radius: 0.95,
            leak: 0.3,
            ridge: 1e-2,
            seed: 0,
        }
    }
}

/// Run the benchmark on the given brain and report metrics.
pub fn demo(conn: &Connectome, p: &Params) -> Result<Report> {
    let (u, y) = make(p.t, 1);
    let mut res = Reservoir::new(
        &conn.w,
        conn.n,
        &ReservoirParams {
            spectral_radius: p.spectral_radius,
            leak: p.leak,
            ..Default::default()
        },
    );
    let x = res.run(&u, p.t, 1, p.washout);
    let rows = p.t - p.washout;
    debug_assert_eq!(x.len(), rows * conn.n);
    let yw = &y[p.washout..];

    let mut ro = Readout::new(p.ridge);
    ro.fit(&x[..p.train * conn.n], conn.n, &yw[..p.train], 1);
    let pred = ro.predict(&x[p.train * conn.n..], conn.n)?;
    let true_ = &yw[p.train..];

    let train_mean = yw[..p.train].iter().sum::<f64>() / p.train as f64;
    let baseline = vec![train_mean; true_.len()]; // predict-the-mean

    Ok(vec![
        ("task".into(), Metric::S("timeseries".into())),
        ("nrmse".into(), Metric::F(nrmse(&pred, true_))),
        ("baseline_nrmse".into(), Metric::F(nrmse(&baseline, true_))),
        ("neurons".into(), Metric::I(conn.n as i64)),
    ])
}

/// Normalized root-mean-square error against the standard deviation of the truth.
pub(crate) fn nrmse(pred: &[f64], true_: &[f64]) -> f64 {
    let mse = pred
        .iter()
        .zip(true_)
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f64>()
        / true_.len() as f64;
    let mean = true_.iter().sum::<f64>() / true_.len() as f64;
    let var = true_.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / true_.len() as f64;
    mse.sqrt() / (var.sqrt() + 1e-9)
}
