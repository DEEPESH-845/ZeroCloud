//! One result, rendered many ways.
//!
//! Every surface — the terminal, `--json`, and later the HTTP server and the
//! TUI — builds the same [`Report`] and hands it to a renderer here. That is
//! the point: the moment a second surface computes its own view of a
//! prediction, the two drift, and a user comparing `zc check` against
//! `zc serve` finds two different answers with no way to tell which is real.
//!
//! [`Report`] borrows rather than owns. Copying the machine facts into a
//! parallel struct would mean listing every field twice and keeping the copies
//! in sync forever, which is the same drift problem one layer down.

pub mod json;
pub mod text;

use zc_bench::{compute::ComputeResult, disk::DiskResult, ram::RamResult};
use zc_model::{Backend, Prediction, Quant, Verdict};
use zc_probe::{cpu::Cpu, env::Env, memory::Memory, storage::Storage};

/// One prediction, plus the identity of what was predicted.
pub struct Row<'a> {
    pub model_id: &'a str,
    pub quant: &'a Quant,
    pub prediction: Prediction,
}

/// What the predictions assumed. Stated explicitly because a tok/s figure
/// without its context length and KV precision is not a claim anyone can check.
pub struct Assumptions {
    pub prompt_tokens: u32,
    pub ubatch: u32,
    pub kv_precision: &'static str,
    /// True when the budget used was the idle-machine figure rather than what
    /// is free right now.
    pub idle_machine: bool,
    /// Set when no calibration data backs any coefficient yet.
    pub uncalibrated: bool,
    /// True when no prefill measurement exists for this backend, so TTFT is
    /// reported as unknown rather than derived.
    pub prefill_unmeasured: bool,
}

pub struct Report<'a> {
    pub cpu: &'a Cpu,
    pub mem: &'a Memory,
    pub env: &'a Env,
    pub storage: &'a Storage,
    pub ram: &'a RamResult,
    pub compute: &'a ComputeResult,
    /// `None` when the disk measurement failed. Never substituted with a
    /// default: streaming speed is predicted from it, so a stand-in would
    /// produce a confident number with nothing behind it.
    pub disk: Option<&'a DiskResult>,
    pub backend: Backend,
    /// Bandwidth figure actually used for prediction — the performance-core
    /// number on CPU, the all-core peak on unified memory.
    pub ram_bw_gbs: f64,
    pub disk_gbs: f64,
    pub budget_idle: u64,
    pub budget_now: u64,
    pub assumptions: Assumptions,
    pub models: Vec<Row<'a>>,
}

/// Stable machine-readable tag for a verdict.
///
/// Separate from `Debug` on purpose: `Debug` output is free to change, and
/// anything a script parses is a promise we have to keep.
pub fn verdict_tag(v: Verdict) -> &'static str {
    match v {
        Verdict::Good => "good",
        Verdict::Usable => "usable",
        Verdict::Slow => "slow",
        Verdict::WontFit => "wont_fit",
    }
}

pub fn backend_tag(b: Backend) -> &'static str {
    match b {
        Backend::Cpu => "cpu",
        Backend::Metal => "metal",
        Backend::Discrete => "discrete",
    }
}
