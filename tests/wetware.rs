//! Tests run on the offline synthetic sample — no download, no network.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use wetware::data;
use wetware::readout::Readout;
use wetware::reservoir::{Params as ReservoirParams, Reservoir};
use wetware::tasks::timeseries;

fn standard_normal(rng: &mut StdRng) -> f64 {
    let u1 = rng.random::<f64>().max(1e-300);
    let u2 = rng.random::<f64>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

fn report_f(r: &wetware::Report, key: &str) -> Option<f64> {
    r.iter().find(|(k, _)| k == key).and_then(|(_, v)| match v {
        wetware::Metric::F(x) => Some(*x),
        _ => None,
    })
}

// --- connectome sample ------------------------------------------------------

#[test]
fn sample_shape_and_sparsity() {
    let c = data::load_sample(400, 0.013, 0);
    assert_eq!(c.w.len(), 400 * 400);
    assert_eq!(c.ids.len(), 400);
    let density = c.w.iter().filter(|&&v| v != 0.0).count() as f64 / c.w.len() as f64;
    assert!(density > 0.0 && density < 0.15); // sparse, like a real connectome
}

// --- reservoir --------------------------------------------------------------

#[test]
fn reservoir_is_stable_echo_state() {
    let c = data::load_sample(300, 0.013, 0);
    let mut rng = StdRng::seed_from_u64(0);
    let inputs: Vec<f64> = (0..200).map(|_| rng.random_range(0.0..0.5)).collect();
    let mut res = Reservoir::new(
        &c.w,
        c.n,
        &ReservoirParams {
            spectral_radius: 0.9,
            ..Default::default()
        },
    );
    let states = res.run(&inputs, 200, 1, 0);
    assert_eq!(states.len(), 200 * 300);
    assert!(states.iter().all(|s| s.is_finite()));
    assert!(states.iter().all(|s| s.abs() <= 1.0 + 1e-6)); // tanh-bounded, not blowing up
}

#[test]
fn reservoir_washout_trims() {
    let c = data::load_sample(200, 0.013, 0);
    let mut res = Reservoir::new(&c.w, c.n, &Default::default());
    let states = res.run(&[0.0; 100], 100, 1, 20);
    assert_eq!(states.len() / c.n, 80);
}

#[test]
fn spectral_radius_is_scaled() {
    // after scaling, the reservoir's largest eigenvalue magnitude ~ target
    let c = data::load_sample(250, 0.013, 0);
    let res = Reservoir::new(
        &c.w,
        c.n,
        &ReservoirParams {
            spectral_radius: 0.8,
            ..Default::default()
        },
    );
    let radius = Reservoir::spectral_radius_of(&res.w_dense(), c.n);
    assert!((radius - 0.8).abs() < 0.05);
}

// --- readout ----------------------------------------------------------------

#[test]
fn readout_fits_a_linear_map() {
    let mut rng = StdRng::seed_from_u64(0);
    let x: Vec<f64> = (0..200 * 10).map(|_| standard_normal(&mut rng)).collect();
    let true_w: Vec<f64> = (0..10).map(|_| standard_normal(&mut rng)).collect();
    let y: Vec<f64> = (0..200)
        .map(|t| (0..10).map(|i| x[t * 10 + i] * true_w[i]).sum())
        .collect();

    let mut ro = Readout::new(1e-6);
    ro.fit(&x, 10, &y, 1);
    let pred = ro.predict(&x, 10).unwrap();
    let corr = corr(pred.iter().copied(), y.iter().copied());
    assert!(corr > 0.99);
}

#[test]
fn readout_untrained_errors() {
    assert!(Readout::new(1e-2).predict(&[0.0; 15], 3).is_err());
}

// --- it actually learns (the whole point) -----------------------------------

#[test]
fn timeseries_beats_the_baseline_on_the_sample() {
    let c = data::load_sample(500, 0.013, 0);
    let r = timeseries::demo(&c, &Default::default()).unwrap();
    // the brain-reservoir must predict the nonlinear series better than predict-the-mean
    assert!(report_f(&r, "nrmse").unwrap() < report_f(&r, "baseline_nrmse").unwrap());
}

fn corr(mut a: impl Iterator<Item = f64> + Clone, mut b: impl Iterator<Item = f64> + Clone) -> f64 {
    let n = a.clone().count() as f64;
    let ma = a.clone().sum::<f64>() / n;
    let mb = b.clone().sum::<f64>() / n;
    let mut num = 0.0;
    let mut da = 0.0;
    let mut db = 0.0;
    while let (Some(x), Some(y)) = (a.next(), b.next()) {
        num += (x - ma) * (y - mb);
        da += (x - ma) * (x - ma);
        db += (y - mb) * (y - mb);
    }
    num / (da.sqrt() * db.sqrt())
}

// --- regression tests for the analysis findings ------------------------------

#[test]
fn reservoir_run_washout_beyond_t_is_empty() {
    // Python returns states[washout:] == empty; we must not underflow usize
    let c = data::load_sample(50, 0.013, 0);
    let mut res = Reservoir::new(&c.w, c.n, &Default::default());
    let states = res.run(&[0.0; 10], 10, 1, 100);
    assert!(states.is_empty());
}

#[test]
fn readout_rejects_shape_mismatches() {
    let mut rng = StdRng::seed_from_u64(0);
    let x: Vec<f64> = (0..5 * 4).map(|_| standard_normal(&mut rng)).collect();
    let y: Vec<f64> = (0..5).map(|t| x[t * 4]).collect();
    let mut ro = Readout::new(1e-6);
    ro.fit(&x, 4, &y, 1);
    assert!(ro.predict(&x, 4).is_ok());
    // fitted with n=4; predict with a different n or a ragged input must error
    assert!(ro.predict(&[0.0; 10], 5).is_err());
    assert!(ro.predict(&[0.0; 7], 4).is_err());
}

#[test]
#[should_panic(expected = "targets must be rows x out_dim")]
fn readout_fit_rejects_bad_targets() {
    let mut ro = Readout::new(1e-6);
    ro.fit(&[0.0; 10], 2, &[0.0; 4], 1); // 5 rows x out_dim 1 wants 5 targets
}

#[test]
fn power_iteration_agrees_with_full_eig() {
    // the spectral radius fast path is custom math; pin it against a full
    // eigendecomposition on random sparse signed matrices
    use faer::linalg::solvers::Eigen;
    for (n, seed) in [(20usize, 1u64), (77, 2), (151, 3)] {
        let mut rng = StdRng::seed_from_u64(seed);
        let w: Vec<f64> = (0..n * n)
            .map(|_| {
                if rng.random::<f64>() < 0.05 {
                    rng.random_range(0.5..2.0) * if rng.random::<bool>() { 1.0 } else { -1.0 }
                } else {
                    0.0
                }
            })
            .collect();
        let r_power = Reservoir::spectral_radius_of(&w, n);

        let a = faer::Mat::<f64>::from_fn(n, n, |i, j| w[i * n + j]);
        let evd = Eigen::new_from_real(a.as_ref()).unwrap();
        let s = evd.S();
        let r_eig = (0..n)
            .map(|j| {
                let z = s[j];
                z.re * z.re + z.im * z.im
            })
            .fold(0.0f64, f64::max)
            .sqrt();

        assert!(
            (r_power - r_eig).abs() <= 1e-8 * r_eig.max(1.0),
            "n={n}: power={r_power} eig={r_eig}"
        );
    }
}

#[test]
fn power_iteration_agrees_with_full_eig_on_sample() {
    use faer::linalg::solvers::Eigen;
    let c = data::load_sample(300, 0.013, 0);
    let r_power = Reservoir::spectral_radius_of(&c.w, c.n);

    let a = faer::Mat::<f64>::from_fn(c.n, c.n, |i, j| c.w[i * c.n + j]);
    let evd = Eigen::new_from_real(a.as_ref()).unwrap();
    let s = evd.S();
    let r_eig = (0..c.n)
        .map(|j| {
            let z = s[j];
            z.re * z.re + z.im * z.im
        })
        .fold(0.0f64, f64::max)
        .sqrt();

    assert!((r_power - r_eig).abs() <= 1e-8 * r_eig.max(1.0));
}
