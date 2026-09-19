//! The experience bank: persistent memory across improvement runs.
//!
//! Every accepted champion (L2 and above) is appended as one JSONL record:
//! task, level, config, validation and test scores, and the policy version in
//! force at the time. Two things make this the L5 substrate rather than a log:
//!
//! 1. the *policy is derived from the bank* — the top quartile of past entries
//!    defines a warm-start distribution (center + spread per knob) that shapes
//!    where the next run's search begins, and
//! 2. the policy is versioned — each append bumps the version, so any run can
//!    say exactly which inherited state it built on and which successors
//!    inherited from it.
//!
//! The bank is append-only and human-inspectable (`wetware bank`).

use std::fs::OpenOptions;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::Result;
use crate::rsi::config::TrialConfig;
use crate::rsi::observe::Observer;

/// One accepted improvement, retained for later rounds.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BankEntry {
    /// 1-based run id within this bank file
    pub run: u64,
    /// task the entry was produced on
    pub task: String,
    /// highest RSI level exercised, e.g. "l4"
    pub level: String,
    /// champion config
    pub config: TrialConfig,
    /// validation score (higher is better; see evaluate.rs)
    pub val: f64,
    /// test score, measured once at final acceptance
    pub test: Option<f64>,
    /// test score of the task's default config, as an anchor
    pub default_test: Option<f64>,
    /// policy version in force when this entry was written
    pub policy_version: u64,
}

/// Derived warm-start state, computed from the bank's top performers.
#[derive(Debug, Clone)]
pub struct Policy {
    /// bank version this policy was derived from (entries.len())
    pub version: u64,
    /// number of entries the policy was derived from
    pub source_entries: usize,
    /// per-task or global center of the top quartile
    pub center: TrialConfig,
    /// per-dimension spread (std) of the top quartile, floored
    pub spread: TrialConfig,
}

impl Policy {
    fn from_entries(entries: &[BankEntry], task: &str) -> Option<Self> {
        // prefer same-task experience; fall back to the global population
        let mut relevant: Vec<&BankEntry> = entries.iter().filter(|e| e.task == task).collect();
        if relevant.is_empty() {
            relevant = entries.iter().collect();
        }
        if relevant.is_empty() {
            return None;
        }
        let mut vals: Vec<f64> = relevant.iter().map(|e| e.val).collect();
        vals.sort_by(f64::total_cmp);
        let cutoff = vals[(vals.len() * 3 / 4).min(vals.len() - 1)];
        let top: Vec<&BankEntry> = relevant.iter().copied().filter(|e| e.val >= cutoff).collect();
        if top.is_empty() {
            return None;
        }
        let mean = |f: fn(&TrialConfig) -> f64| top.iter().map(|e| f(&e.config)).sum::<f64>() / top.len() as f64;
        let std = |f: fn(&TrialConfig) -> f64, m: f64| {
            (top.iter().map(|e| (f(&e.config) - m).powi(2)).sum::<f64>() / top.len() as f64)
                .sqrt()
        };
        let center = TrialConfig {
            spectral_radius: mean(|c| c.spectral_radius),
            leak: mean(|c| c.leak),
            ridge: mean(|c| c.ridge),
            input_scale: mean(|c| c.input_scale),
            inhibitory_fraction: mean(|c| c.inhibitory_fraction),
            seed: 0,
        };
        let spread = TrialConfig {
            spectral_radius: std(|c| c.spectral_radius, center.spectral_radius).max(0.03),
            leak: std(|c| c.leak, center.leak).max(0.05),
            // ridge spread lives in log-space (sample_around logs it)
            ridge: std(|c| c.ridge, center.ridge) / center.ridge.max(1e-12),
            input_scale: std(|c| c.input_scale, center.input_scale).max(0.1),
            inhibitory_fraction: std(|c| c.inhibitory_fraction, center.inhibitory_fraction)
                .max(0.03),
            seed: 0,
        };
        Some(Policy {
            version: entries.len() as u64,
            source_entries: top.len(),
            center,
            spread,
        })
    }
}

/// The append-only store of accepted improvements and its derived policy.
pub struct Bank {
    /// where the bank lives on disk
    pub path: PathBuf,
    /// every entry ever appended, in order
    pub entries: Vec<BankEntry>,
}

impl Bank {
    /// Load the bank at `path`; a missing file is an empty bank (version 0),
    /// not an error — the first run inherits nothing.
    pub fn load(path: &Path) -> Result<Self> {
        let mut entries = Vec::new();
        if path.exists() {
            let f = std::fs::File::open(path)
                .map_err(|e| crate::WetwareError::Other(format!("cannot open {}: {e}", path.display())))?;
            for line in BufReader::new(f).lines() {
                let line = line.map_err(|e| crate::WetwareError::Other(e.to_string()))?;
                if line.trim().is_empty() {
                    continue;
                }
                if let Ok(e) = serde_json::from_str::<BankEntry>(&line) {
                    entries.push(e);
                }
                // a corrupt line must not kill the loop; skip it
            }
        }
        Ok(Self {
            path: path.to_path_buf(),
            entries,
        })
    }

    /// The policy in force right now (derived from everything retained so far).
    pub fn policy(&self, task: &str) -> Option<Policy> {
        Policy::from_entries(&self.entries, task)
    }

    /// Append an accepted improvement; bumps the policy version. Emits a
    /// `policy_updated` event so the inheritance chain is observable.
    pub fn append(&mut self, mut entry: BankEntry, obs: &mut Observer) -> Result<u64> {
        entry.run = self.entries.len() as u64 + 1;
        entry.policy_version = entry.run; // version N = bank with N entries
        let mut f = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .map_err(|e| crate::WetwareError::Other(format!("cannot open {}: {e}", self.path.display())))?;
        writeln!(f, "{}", serde_json::to_string(&entry).unwrap_or_default())
            .map_err(|e| crate::WetwareError::Other(e.to_string()))?;
        self.entries.push(entry);
        let version = self.entries.len() as u64;
        obs.event(
            "policy_updated",
            json!({
                "version": version,
                "entries": self.entries.len(),
                "bank": self.path.display().to_string(),
            }),
        );
        Ok(version)
    }

    /// Human-readable summary for `wetware bank`.
    pub fn describe(&self) -> String {
        let mut s = format!(
            "bank: {} ({} entries)\n",
            self.path.display(),
            self.entries.len()
        );
        for e in &self.entries {
            s.push_str(&format!(
                "  #{:<3} {:<10} {:<3} val={:.4} test={} policy_v={}\n",
                e.run,
                e.task,
                e.level,
                e.val,
                e.test.map(|t| format!("{t:.4}")).unwrap_or("-".into()),
                e.policy_version,
            ));
        }
        if let Some(p) = self.policy("digits") {
            s.push_str(&format!(
                "policy v{} (from {} top entries): center sr={:.3} leak={:.3} ridge={:.4} in_scale={:.3} inh={:.3}\n",
                p.version, p.source_entries,
                p.center.spectral_radius, p.center.leak, p.center.ridge,
                p.center.input_scale, p.center.inhibitory_fraction,
            ));
        }
        s
    }
}
