//! Observability for the improvement loop.
//!
//! Every decision the loop makes — every trial, every diagnosis, every
//! curriculum shift, every adaptation and rollback — is an event. Events go to
//! two places: a human-readable line on stderr, and (with `--log`) a JSONL file
//! with millisecond timestamps, suitable for replaying the trajectory of an
//! improvement run later. The survey's evaluation guidance (match budgets,
//! freeze the verifier, record rejected updates too) is only enforceable if
//! the loop's history is complete, so nothing here is opt-in.

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::{Map, Value};

use crate::Result;

/// Timestamp in milliseconds since the epoch, as a JSON number.
fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Sinks for improvement-loop events.
pub struct Observer {
    file: Option<File>,
    human: bool,
    /// events emitted so far, for the final summary
    pub count: u64,
}

impl Observer {
    /// An observer writing JSONL to `path` (if given) and human lines to
    /// stderr (if `human`).
    pub fn new(path: Option<&Path>, human: bool) -> Result<Self> {
        let file = match path {
            Some(p) => Some(
                OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(p)
                    .map_err(|e| crate::WetwareError::Other(format!("cannot open {}: {e}", p.display())))?,
            ),
            None => None,
        };
        Ok(Self {
            file,
            human,
            count: 0,
        })
    }

    /// Record one event. `fields` becomes the JSON body alongside `ts`/`kind`.
    pub fn event(&mut self, kind: &str, fields: Value) {
        self.count += 1;
        let mut line = Map::new();
        line.insert("ts".into(), now_ms().into());
        line.insert("kind".into(), kind.into());
        if let Value::Object(m) = fields {
            for (k, v) in m {
                line.insert(k, v);
            }
        }
        let v = Value::Object(line);
        if let Some(f) = &mut self.file {
            let _ = writeln!(f, "{}", serde_json::to_string(&v).unwrap_or_default());
            let _ = f.flush();
        }
        if self.human {
            eprintln!("[rsi] {}", human_line(&v));
        }
    }

    /// A `run_completed` summary with the headline numbers.
    pub fn finish(mut self, fields: Value) {
        self.event("run_completed", fields);
    }
}

/// Render `{ts, kind, a: 1, b: "x"}` as `kind a=1 b="x"` for stderr.
fn human_line(v: &Value) -> String {
    let obj = v.as_object().cloned().unwrap_or_default();
    let kind = obj.get("kind").and_then(Value::as_str).unwrap_or("?");
    let mut parts: Vec<String> = Vec::new();
    for (k, val) in &obj {
        if k == "ts" || k == "kind" {
            continue;
        }
        parts.push(format!("{k}={}", compact(val)));
    }
    if parts.is_empty() {
        kind.to_string()
    } else {
        format!("{kind} {}", parts.join(" "))
    }
}

fn compact(v: &Value) -> String {
    match v {
        Value::String(s) => {
            if s.contains(' ') {
                format!("\"{s}\"")
            } else {
                s.clone()
            }
        }
        Value::Number(n) => n.to_string(),
        Value::Bool(b) => b.to_string(),
        Value::Null => "null".into(),
        // nested objects/arrays: compact JSON
        other => serde_json::to_string(other).unwrap_or_default(),
    }
}
