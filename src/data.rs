//! Loading the brain.
//!
//! [`load`] gives you the wiring matrix W of the *complete larval Drosophila
//! connectome* — the first and only synapse-resolution wiring diagram of an entire
//! animal brain (Winding et al., Science 2023). 2952 neurons, ~110k weighted
//! directed connections. The first call downloads it (~1 MB) and caches a compact
//! `.npz`; after that it's instant.
//!
//! [`load_sample`] builds a small synthetic small-world network with the same
//! shape of statistics, clearly labelled as *not* the real brain. It exists so the
//! tests and a first look run offline with nothing downloaded. Anything you
//! publish should run on [`load`] — the real thing is the whole point.

use std::env;
use std::fs::File;
use std::io::{Cursor, Read, Seek, Write};
use std::path::PathBuf;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::{Result, WetwareError};

const URL: &str =
    "https://codeload.github.com/brain-networks/larval-drosophila-connectome/zip/refs/heads/main";
const INNER_ZIP: &str = "Supplementary-Data-S1.zip";
const INNER_CSV: &str = "Supplementary-Data-S1/all-all_connectivity_matrix.csv";

/// The wiring matrix of a brain: `w` is `n x n` row-major, `ids` the neuron ids.
pub struct Connectome {
    /// weighted wiring matrix, row-major (`w[i*n + j]` is i -> j)
    pub w: Vec<f64>,
    /// number of neurons (`w` has `n * n` entries)
    pub n: usize,
    /// neuron ids from the connectome
    pub ids: Vec<i64>,
}

fn home() -> Result<PathBuf> {
    env::var_os("HOME")
        .map(PathBuf::from)
        .ok_or_else(|| WetwareError::Other("HOME is not set".into()))
}

/// Cache directory: `$WETWARE_CACHE`, or `~/.cache/wetware`. Created if missing.
pub fn cache_dir() -> Result<PathBuf> {
    let d = env::var_os("WETWARE_CACHE")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            home()
                .expect("HOME is not set")
                .join(".cache")
                .join("wetware")
        });
    std::fs::create_dir_all(&d)?;
    Ok(d)
}

/// Fetch the real connectome, parse it, cache a compact npz. Returns the path.
pub fn download(force: bool) -> Result<PathBuf> {
    let out = cache_dir()?.join("larval_connectome.npz");
    if out.exists() && !force {
        return Ok(out);
    }

    println!("downloading the larval Drosophila connectome (~1 MB)...");
    let raw = crate::fetch_url(URL)?;

    // zip of zips: the repo archive contains the supplementary zip, which holds
    // the all-to-all connectivity CSV.
    let mut outer = zip::ZipArchive::new(Cursor::new(raw))?;
    let mut inner_zip = None;
    for i in 0..outer.len() {
        let mut f = outer.by_index(i)?;
        if f.name().ends_with(INNER_ZIP) {
            let mut buf = Vec::new();
            f.read_to_end(&mut buf)?;
            inner_zip = Some(buf);
            break;
        }
    }
    let inner_zip = inner_zip.ok_or_else(|| {
        WetwareError::Other(format!("{INNER_ZIP} not found in the downloaded archive"))
    })?;

    let mut inner = zip::ZipArchive::new(Cursor::new(inner_zip))?;
    let mut csv = None;
    for i in 0..inner.len() {
        let mut f = inner.by_index(i)?;
        if f.name().ends_with(INNER_CSV) {
            let mut s = String::new();
            f.read_to_string(&mut s)?;
            csv = Some(s);
            break;
        }
    }
    let csv =
        csv.ok_or_else(|| WetwareError::Other(format!("{INNER_CSV} not found in {INNER_ZIP}")))?;

    println!("parsing the connectivity matrix...");
    let (w, n, ids) = parse_connectome_csv(&csv)?;

    // cache as npz (zip of .npy), the same layout Python's np.savez_compressed
    // writes, so the two implementations can share a cache directory. Written
    // atomically so a crash never leaves a corrupt cache behind.
    let tmp = out.with_extension("tmp");
    let file = File::create(&tmp)?;
    let mut zw = zip::ZipWriter::new(file);
    let opts = zip::write::SimpleFileOptions::default()
        .compression_method(zip::CompressionMethod::Deflated);
    zw.start_file("W.npy", opts)?;
    zw.write_all(&npy_f32(&format!("({n}, {n})"), &w))?;
    zw.start_file("ids.npy", opts)?;
    zw.write_all(&npy_i64(&format!("({n},)"), &ids))?;
    zw.finish()?;
    std::fs::rename(&tmp, &out)?;

    println!("cached {n} neurons -> {}", out.display());
    Ok(out)
}

pub fn load() -> Result<Connectome> {
    let path = download(false)?;
    parse_connectome_npz(File::open(&path)?)
}

/// Parse a cached npz into a [`Connectome`].
///
/// Errors if the cache is corrupt (truncated, wrong dtypes or shapes) — delete
/// it or re-run `wetware download --force`.
fn parse_connectome_npz(file: impl Read + Seek) -> Result<Connectome> {
    let mut za = zip::ZipArchive::new(file)?;

    let (w32, n) = {
        let mut e = za.by_name("W.npy")?;
        let mut buf = Vec::new();
        e.read_to_end(&mut buf)?;
        let npy = read_npy(&buf)?;
        if !npy.descr.starts_with("<f4") {
            return Err(WetwareError::Other(format!(
                "unexpected W dtype {} (want little-endian f4)",
                npy.descr
            )));
        }
        let n = match npy.shape.as_slice() {
            [a, b] if a == b => *a,
            other => {
                return Err(WetwareError::Other(format!(
                    "corrupt cache: W has shape {other:?}, want (n, n)"
                )));
            }
        };
        if npy.data.len() != n * n * 4 {
            return Err(WetwareError::Other(format!(
                "corrupt cache: W is {} bytes, want {} for a ({n}, {n}) f4 matrix",
                npy.data.len(),
                n * n * 4
            )));
        }
        let w: Vec<f32> = npy
            .data
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();
        (w, n)
    };
    let ids = {
        let mut e = za.by_name("ids.npy")?;
        let mut buf = Vec::new();
        e.read_to_end(&mut buf)?;
        let npy = read_npy(&buf)?;
        if !npy.descr.starts_with("<i8") {
            return Err(WetwareError::Other(format!(
                "unexpected ids dtype {} (want little-endian i8)",
                npy.descr
            )));
        }
        npy.data
            .as_chunks::<8>()
            .0
            .iter()
            .map(|b| i64::from_le_bytes(*b))
            .collect::<Vec<i64>>()
    };
    if ids.len() != n {
        return Err(WetwareError::Other(format!(
            "corrupt cache: {} ids for a ({n}, {n}) matrix",
            ids.len()
        )));
    }

    let w = w32.into_iter().map(f64::from).collect();
    Ok(Connectome { w, n, ids })
}

/// A synthetic small-world stand-in — NOT the real brain. Offline/tests only.
///
/// Matches the connectome's rough sparsity and heavy-tailed weights so the engine
/// behaves similarly, but it is generated, not measured. Never present output from
/// this as 'the fruit-fly brain'.
pub fn load_sample(n: usize, density: f64, seed: u64) -> Connectome {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut w = vec![0.0f64; n * n];
    let k = ((density * n as f64) as usize).max(1);
    for i in 0..n {
        // local ring + a few long-range shortcuts (Watts-Strogatz flavour)
        for j in 1..=k {
            let t = (i + j) % n;
            if t != i {
                w[i * n + t] = gamma(&mut rng, 1.5, 2.0); // heavy-tailed, like synapse counts
            }
        }
        for _ in 0..k {
            let t = rng.random_range(0..n);
            if t != i {
                w[i * n + t] = gamma(&mut rng, 1.5, 2.0);
            }
        }
    }
    Connectome {
        w,
        n,
        ids: (0..n as i64).collect(),
    }
}

// --- connectome csv ---------------------------------------------------------

fn parse_connectome_csv(text: &str) -> Result<(Vec<f32>, usize, Vec<i64>)> {
    let mut lines = text.lines();
    let header = lines
        .next()
        .ok_or_else(|| WetwareError::Other("connectome csv is empty".into()))?;
    let ids: Vec<i64> = header
        .split(',')
        .skip(1) // leading row-label column
        .filter_map(|s| s.trim().parse::<i64>().ok())
        .collect();
    let n = ids.len();
    if n == 0 {
        return Err(WetwareError::Other(
            "connectome csv has no neuron columns".into(),
        ));
    }

    let mut w = vec![0f32; n * n];
    let mut i = 0;
    for line in lines {
        if i >= n {
            break;
        }
        if line.trim().is_empty() {
            continue;
        }
        let mut parts = line.split(',').skip(1);
        for j in 0..n {
            let s = parts.next().ok_or_else(|| {
                WetwareError::Other(format!("connectome csv row {i} has only {j} columns"))
            })?;
            w[i * n + j] = s.trim().parse::<f32>().map_err(|_| {
                WetwareError::Other(format!(
                    "connectome csv: bad value {:?} at row {i}, col {j}",
                    s.trim()
                ))
            })?;
        }
        i += 1;
    }
    if i < n {
        return Err(WetwareError::Other(format!(
            "connectome csv has {i} rows, expected {n}"
        )));
    }
    Ok((w, n, ids))
}

// --- minimal .npy codec -----------------------------------------------------

struct Npy {
    descr: String,
    shape: Vec<usize>,
    data: Vec<u8>,
}

/// Serialize one array in .npy v1.0 format (little-endian), matching what
/// `numpy.lib.format` writes: the dict header is padded with spaces so the whole
/// preamble is a multiple of 64 bytes, terminated by '\n'.
fn npy_bytes(descr: &str, shape: &str, data: &[u8]) -> Vec<u8> {
    let dict = format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': {shape}, }}");
    let pad = (64 - ((10 + dict.len() + 1) % 64)) % 64;
    let header = format!("{}{}\n", dict, " ".repeat(pad));
    let mut out = Vec::with_capacity(10 + header.len() + data.len());
    out.extend_from_slice(b"\x93NUMPY");
    out.extend_from_slice(&[1, 0]); // version 1.0
    out.extend_from_slice(&(header.len() as u16).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    out.extend_from_slice(data);
    out
}

fn npy_f32(shape: &str, w: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(w.len() * 4);
    for v in w {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    npy_bytes("<f4", shape, &bytes)
}

fn npy_i64(shape: &str, ids: &[i64]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(ids.len() * 8);
    for v in ids {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    npy_bytes("<i8", shape, &bytes)
}

fn read_npy(buf: &[u8]) -> Result<Npy> {
    if buf.len() < 10 || &buf[0..6] != b"\x93NUMPY" {
        return Err(WetwareError::Other("not an .npy file".into()));
    }
    let (hlen, hoff) = match buf[6] {
        1 => (u16::from_le_bytes([buf[8], buf[9]]) as usize, 10usize),
        major => {
            if major != 2 {
                return Err(WetwareError::Other(format!(
                    "unsupported .npy version {major}"
                )));
            }
            (
                u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]) as usize,
                12usize,
            )
        }
    };
    let header = std::str::from_utf8(
        buf.get(hoff..hoff + hlen)
            .ok_or_else(|| WetwareError::Other("truncated .npy header".into()))?,
    )
    .map_err(|e| WetwareError::Other(e.to_string()))?;

    let descr = quoted_after(header, "'descr':")?;
    if !header.contains("'fortran_order': False") {
        return Err(WetwareError::Other(
            "fortran_order=True arrays are not supported".into(),
        ));
    }
    // shape is the parenthesized tuple after the key, e.g. (2952, 2952)
    let pos = header
        .find("'shape':")
        .ok_or_else(|| WetwareError::Other("'shape': missing from .npy header".into()))?;
    let rest = &header[pos + "'shape':".len()..];
    let inner = rest
        .trim_start()
        .strip_prefix('(')
        .and_then(|s| s.split_once(')').map(|(t, _)| t))
        .ok_or_else(|| WetwareError::Other("malformed 'shape': in .npy header".into()))?;
    let shape: Vec<usize> = inner
        .split(',')
        .filter_map(|s| s.trim().parse::<usize>().ok())
        .collect();

    let data = buf[hoff + hlen..].to_vec();
    Ok(Npy {
        descr: descr.to_string(),
        shape,
        data,
    })
}

/// Value of the quoted string after `key` in an npy header dict.
fn quoted_after<'a>(header: &'a str, key: &str) -> Result<&'a str> {
    let pos = header
        .find(key)
        .ok_or_else(|| WetwareError::Other(format!("{key} missing from .npy header")))?;
    let rest = &header[pos + key.len()..];
    let start = rest
        .find('\'')
        .ok_or_else(|| WetwareError::Other(format!("malformed {key} in .npy header")))?;
    let quoted = &rest[start + 1..];
    let end = quoted
        .find('\'')
        .ok_or_else(|| WetwareError::Other(format!("malformed {key} in .npy header")))?;
    Ok(&quoted[..end])
}

// --- sampling ---------------------------------------------------------------

/// Standard normal via Box-Muller.
fn standard_normal(rng: &mut StdRng) -> f64 {
    let u1 = rng.random::<f64>().max(1e-300);
    let u2 = rng.random::<f64>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// Gamma(shape, scale) via the Marsaglia–Tsang method (shape >= 1).
fn gamma(rng: &mut StdRng, shape: f64, scale: f64) -> f64 {
    debug_assert!(shape >= 1.0, "gamma sampler assumes shape >= 1");
    let d = shape - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    loop {
        let x = standard_normal(rng);
        let v = (1.0 + c * x).powi(3);
        if v <= 0.0 {
            continue;
        }
        let u: f64 = rng.random();
        if u.ln() < 0.5 * x * x + d - d * v + d * v.ln() {
            return scale * d * v;
        }
    }
}

/// Build an in-memory npz with the two standard entries (for tests).
#[cfg(test)]
fn npz_bytes(w_npy: &[u8], ids_npy: &[u8]) -> Vec<u8> {
    let mut buf = Vec::new();
    {
        let cursor = Cursor::new(&mut buf);
        let mut zw = zip::ZipWriter::new(cursor);
        let opts = zip::write::SimpleFileOptions::default();
        zw.start_file("W.npy", opts).unwrap();
        zw.write_all(w_npy).unwrap();
        zw.start_file("ids.npy", opts).unwrap();
        zw.write_all(ids_npy).unwrap();
        zw.finish().unwrap();
    }
    buf
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn npy_roundtrip_f32_and_i64() {
        let w: Vec<f32> = vec![1.5, -2.25, 3.0, 4.125];
        let ids: Vec<i64> = vec![7, -3, 42];

        let back = read_npy(&npy_f32("(2, 2)", &w)).unwrap();
        assert_eq!(back.descr, "<f4");
        assert_eq!(back.shape, vec![2, 2]);
        let w2: Vec<f32> = back
            .data
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();
        assert_eq!(w, w2);

        let back = read_npy(&npy_i64("(3,)", &ids)).unwrap();
        assert_eq!(back.descr, "<i8");
        assert_eq!(back.shape, vec![3]);
        let ids2: Vec<i64> = back
            .data
            .as_chunks::<8>()
            .0
            .iter()
            .map(|b| i64::from_le_bytes(*b))
            .collect();
        assert_eq!(ids, ids2);
    }

    #[test]
    fn npy_reads_numpy_style_header() {
        // header exactly as numpy.lib.format writes it: dict padded with spaces
        // so magic+version+len+header is a multiple of 64 bytes, then '\n'
        let dict = "{'descr': '<f4', 'fortran_order': False, 'shape': (5, 5), }";
        let pad = (64 - ((10 + dict.len() + 1) % 64)) % 64;
        let header = format!("{}{}\n", dict, " ".repeat(pad));
        let mut buf = b"\x93NUMPY\x01\x00".to_vec();
        buf.extend_from_slice(&(header.len() as u16).to_le_bytes());
        buf.extend_from_slice(header.as_bytes());
        buf.extend_from_slice(&[0u8; 5 * 5 * 4]);

        let npy = read_npy(&buf).unwrap();
        assert_eq!(npy.descr, "<f4");
        assert_eq!(npy.shape, vec![5, 5]);
        assert_eq!(npy.data.len(), 100);
    }

    #[test]
    fn parse_npz_accepts_valid_cache() {
        let w: Vec<f32> = vec![0.0; 25];
        let mut w0 = w.clone();
        w0[0] = 2.5;
        let ids: Vec<i64> = (0..5).collect();
        let buf = npz_bytes(&npy_f32("(5, 5)", &w0), &npy_i64("(5,)", &ids));
        let c = parse_connectome_npz(Cursor::new(buf)).unwrap();
        assert_eq!(c.n, 5);
        assert_eq!(c.w.len(), 25);
        assert_eq!(c.w[0], 2.5f64);
        assert_eq!(c.ids, ids);
    }

    #[test]
    fn parse_npz_rejects_corrupt_cache() {
        // W claims (7, 7) but holds 3 f32 values; must error, not panic later
        let w = vec![1.0f32, 2.0, 3.0];
        let ids: Vec<i64> = (0..7).collect();
        let buf = npz_bytes(&npy_f32("(7, 7)", &w), &npy_i64("(7,)", &ids));
        assert!(parse_connectome_npz(Cursor::new(buf)).is_err());

        // ids length disagrees with W
        let w = vec![1.0f32; 25];
        let ids: Vec<i64> = (0..3).collect();
        let buf = npz_bytes(&npy_f32("(5, 5)", &w), &npy_i64("(3,)", &ids));
        assert!(parse_connectome_npz(Cursor::new(buf)).is_err());
    }
}
