//! Hardware detection for ZeroCloud.
//!
//! Everything here is read-only observation of the host. No writes, no
//! network, no persistence — that belongs to callers.

pub mod cpu;
pub mod env;
pub mod memory;
pub mod storage;

/// Format bytes as GiB/MiB. Binary units, because that's what RAM is sold in
/// and what every OS reports.
pub fn human(bytes: u64) -> String {
    const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
    const MIB: f64 = 1024.0 * 1024.0;
    let b = bytes as f64;
    if b >= GIB {
        format!("{:.2} GiB", b / GIB)
    } else {
        format!("{:.0} MiB", b / MIB)
    }
}
