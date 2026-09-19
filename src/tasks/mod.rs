//! Demo tasks. Each exposes `demo(conn, params) -> Result<Report>`.

pub mod digits;
pub mod timeseries;

/// Task names in the same order as the Python `TASKS` dict.
pub const TASKS: [&str; 2] = ["timeseries", "digits"];
