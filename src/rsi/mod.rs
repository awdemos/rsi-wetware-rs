//! Recursive self-improvement on a fixed brain.
//!
//! The connectome never changes — the whole improvement loop lives in the
//! harness around it: hyperparameters (L1–L2), curricula (L3), gated online
//! adaptation (L4), and a persistent experience bank whose derived policy
//! warm-starts successors (L5). See [`optimize`] for the level ladder and
//! [`observe`] for the event stream that makes every decision inspectable.

pub mod adapt;
pub mod bank;
pub mod config;
pub mod evaluate;
pub mod learn;
pub mod observe;
pub mod optimize;
