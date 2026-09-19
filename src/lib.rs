//! wetware — compute on a real brain.
//!
//! The complete larval Drosophila connectome (the first synapse-resolution wiring
//! diagram of an entire animal brain) run as a fixed reservoir. The wiring never
//! changes — it's the animal's brain — and a one-layer readout learns to read its
//! activity into a decision. Reservoir computing on real biological hardware.
//!
//! ```no_run
//! let brain = wetware::load()?;            // the real fly brain, 2952 neurons
//! let report = wetware::tasks::digits::demo(&brain, &Default::default())?;
//! println!("{report:?}");
//! # Ok::<(), wetware::WetwareError>(())
//! ```

pub mod data;
pub mod readout;
pub mod reservoir;
pub mod rsi;
pub mod tasks;

use std::io::Read;

pub use data::{Connectome, download, load, load_sample};
pub use readout::Readout;
pub use reservoir::Reservoir;

/// Crate version, matching the original Python package.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// A single metric in a task report.
#[derive(Debug, Clone)]
pub enum Metric {
    /// a floating-point metric
    F(f64),
    /// an integer metric
    I(i64),
    /// a string metric
    S(String),
}

/// Ordered task metrics, printed as JSON by the CLI.
pub type Report = Vec<(String, Metric)>;

/// Errors returned by wetware.
#[derive(Debug)]
pub enum WetwareError {
    /// `predict`/`classify` called before `fit`.
    Untrained,
    /// Anything else: io, network, archive, or parse failures.
    Other(String),
}

impl std::fmt::Display for WetwareError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WetwareError::Untrained => write!(f, "readout is not trained"),
            WetwareError::Other(msg) => write!(f, "{msg}"),
        }
    }
}

impl std::error::Error for WetwareError {}

impl From<std::io::Error> for WetwareError {
    fn from(e: std::io::Error) -> Self {
        WetwareError::Other(e.to_string())
    }
}

impl From<zip::result::ZipError> for WetwareError {
    fn from(e: zip::result::ZipError) -> Self {
        WetwareError::Other(e.to_string())
    }
}

/// Convenient result alias.
pub type Result<T> = std::result::Result<T, WetwareError>;

/// Download `url` with a 120 s timeout, mirroring the Python `urllib` calls.
pub(crate) fn fetch_url(url: &str) -> Result<Vec<u8>> {
    let agent = ureq::Agent::new_with_config(
        ureq::config::Config::builder()
            .timeout_global(Some(std::time::Duration::from_secs(120)))
            .build(),
    );
    let mut buf = Vec::new();
    agent
        .get(url)
        .call()
        .map_err(|e| WetwareError::Other(format!("download failed: {e}")))?
        .into_body()
        .into_reader()
        .read_to_end(&mut buf)?;
    Ok(buf)
}

/// Write `bytes` to `path` atomically (temp file + rename) so a crash, or a
/// concurrent reader, never sees a partial cache file.
pub(crate) fn write_atomic(path: &std::path::Path, bytes: &[u8]) -> std::io::Result<()> {
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, bytes)?;
    std::fs::rename(&tmp, path)
}
