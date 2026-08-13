//! Micro-benchmarks that establish what this machine can actually do.
//!
//! Everything here measures rather than assumes. Spec-sheet lookups are wrong
//! for exactly our target hardware: single-channel RAM, DRAM-less SSDs, iGPU
//! carve-outs, thermal throttling and VM overhead are all invisible on paper
//! and all caught here.

pub mod compute;
pub mod disk;
pub mod ram;
