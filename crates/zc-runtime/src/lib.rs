//! Talking to installed inference runtimes, and turning what they report into
//! calibration data.
//!
//! This crate is what converts every constant in `zc-model` from a guess into
//! a measurement. Nothing else in the project can do that.

pub mod calibrate;
pub mod http;

pub mod ollama;
