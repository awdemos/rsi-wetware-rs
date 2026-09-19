//! Task: read handwritten digits.
//!
//! The flagship. Present each 8x8 handwritten digit to the brain one column at a
//! time, let the activity settle, and read the final state. Train a linear readout
//! to name the digit. The wiring never changes — a real animal's brain, used as the
//! feature extractor, and a one-layer readout that learns to see.
//!
//! The digits are the same dataset scikit-learn bundles (`digits.csv.gz`), fetched
//! once and cached.

use std::io::Read;

use flate2::read::GzDecoder;

use crate::data::{Connectome, cache_dir};
use crate::readout::Readout;
use crate::reservoir::{Params as ReservoirParams, Reservoir};
use crate::{Metric, Report, Result, WetwareError, fetch_url};

const DIGITS_URL: &str = "https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/sklearn/datasets/data/digits.csv.gz";

/// Parameters for [`demo`].
#[derive(Debug, Clone)]
pub struct Params {
    /// number of digits used for training (the rest are the test set)
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
            train: 1200,
            spectral_radius: 1.1,
            leak: 0.5,
            ridge: 1.0,
            seed: 0,
        }
    }
}

/// Load the digits dataset (downloading and caching it once), as
/// `(images, labels)` — `images` is `n * 64` in `[0, 1]`, row-major 8x8.
pub fn load_digits() -> Result<(Vec<f64>, Vec<usize>)> {
    let path = cache_dir()?.join("digits.csv.gz");
    if !path.exists() {
        println!("downloading the handwritten-digits dataset...");
        let bytes = fetch(DIGITS_URL)?;
        crate::write_atomic(&path, &bytes)?;
    }
    let mut s = String::new();
    GzDecoder::new(std::fs::File::open(&path)?).read_to_string(&mut s)?;

    let mut images = Vec::new();
    let mut labels = Vec::new();
    for line in s.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let vals: Vec<f64> = line
            .split(',')
            .filter_map(|x| x.trim().parse().ok())
            .collect();
        if vals.len() < 65 {
            continue;
        }
        let label = vals[64] as usize;
        if label >= 10 {
            continue; // digits are 0-9; also skips the 0..=64 index header row
        }
        images.extend(vals[..64].iter().map(|v| v / 16.0)); // (N, 8, 8) in [0,1]
        labels.push(label);
    }
    if labels.is_empty() {
        return Err(WetwareError::Other(
            "no digits parsed from the dataset".into(),
        ));
    }
    Ok((images, labels))
}

fn fetch(url: &str) -> Result<Vec<u8>> {
    fetch_url(url)
}

/// Reservoir knobs that control feature extraction for the digits task.
///
/// This is the same construction [`demo`] uses, but with the full knob set
/// exposed — the RSI loop varies `input_scale` and `inhibitory_fraction` too,
/// which the fixed demo leaves at their defaults.
#[derive(Debug, Clone, Copy)]
pub struct FeatParams {
    /// target spectral radius
    pub spectral_radius: f64,
    /// leak rate
    pub leak: f64,
    /// scale of the random input weights
    pub input_scale: f64,
    /// fraction of neurons assigned inhibitory sign
    pub inhibitory_fraction: f64,
    /// rng seed for signs and input weights
    pub seed: u64,
}

/// Drive the brain with every image (column-by-column, state reset between
/// images) and return the final reservoir state per image: `n_img * n`.
pub fn extract_features(conn: &Connectome, images: &[f64], p: &FeatParams) -> Vec<f64> {
    let n_img = images.len() / 64;
    let mut res = Reservoir::new(
        &conn.w,
        conn.n,
        &ReservoirParams {
            spectral_radius: p.spectral_radius,
            leak: p.leak,
            input_scale: p.input_scale,
            inhibitory_fraction: p.inhibitory_fraction,
            seed: p.seed,
        },
    );
    res.ensure_input_weights(8);

    let mut feats = vec![0.0f64; n_img * conn.n];
    for i in 0..n_img {
        let mut x = vec![0.0f64; conn.n];
        for col in 0..8 {
            let input: Vec<f64> = (0..8).map(|r| images[i * 64 + r * 8 + col]).collect();
            res.step(&mut x, &input); // 8 timesteps of 8 pixels
        }
        feats[i * conn.n..(i + 1) * conn.n].copy_from_slice(&x);
    }
    feats
}

/// Run the digit-reading demo on the given brain and report metrics.
pub fn demo(conn: &Connectome, p: &Params) -> Result<Report> {
    let (images, labels) = load_digits()?;
    let n_img = labels.len();

    let feats = extract_features(
        conn,
        &images,
        &FeatParams {
            spectral_radius: p.spectral_radius,
            leak: p.leak,
            input_scale: 1.0,
            inhibitory_fraction: 0.2,
            seed: p.seed,
        },
    );

    let train = p.train.min(n_img);
    let mut targets = vec![0.0f64; train * 10];
    for (t, &l) in labels[..train].iter().enumerate() {
        targets[t * 10 + l] = 1.0;
    }
    let mut ro = Readout::new(p.ridge);
    ro.fit(&feats[..train * conn.n], conn.n, &targets, 10);

    let pred = ro.classify(&feats[train * conn.n..], conn.n)?;
    let true_ = &labels[train..];
    let acc = pred.iter().zip(true_).filter(|(a, b)| a == b).count() as f64 / true_.len() as f64;

    // baseline: majority class of the training labels
    let mut counts = [0usize; 10];
    for &l in &labels[..train] {
        counts[l] += 1;
    }
    let majority = counts
        .iter()
        .position(|&x| x == counts.iter().copied().max().unwrap_or(0))
        .unwrap_or(0);
    let baseline = true_.iter().filter(|&&l| l == majority).count() as f64 / true_.len() as f64;

    Ok(vec![
        ("task".into(), Metric::S("digits".into())),
        ("accuracy".into(), Metric::F(acc)),
        ("baseline_accuracy".into(), Metric::F(baseline)),
        ("test_n".into(), Metric::I(true_.len() as i64)),
        ("neurons".into(), Metric::I(conn.n as i64)),
    ])
}
