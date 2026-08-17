//! `zc doctor` — one paste that makes a machine we do not own debuggable.
//!
//! `VERIFICATION.md` lists the Linux and Windows probe paths as never executed,
//! and `cpu.rs` still marks the Windows topology walk UNVALIDATED. Those get
//! fixed by someone running this on their hardware and pasting the result, so
//! the report carries the raw readings *and* what we concluded from them: a
//! disagreement between the two sections is the bug, and neither half alone
//! shows it.
//!
//! Redaction is handled in `zc_report::markdown` — nothing here collects an
//! identifier, and paths are rewritten against `$HOME`.

use crate::machine::Machine;
use zc_model::{Fit, KvPrecision};

pub fn run(m: &Machine, fit: &Fit, kv: KvPrecision) -> i32 {
    // No model rows: this is a hardware and calibration report, and a table of
    // predictions is what `zc check` is for.
    let report = crate::check::report(m, fit, kv, Vec::new(), 0);
    print!(
        "{}",
        zc_report::markdown::render(&report, &crate::fit_cmd::summary_text(&crate::fit_cmd::path()))
    );
    0
}
