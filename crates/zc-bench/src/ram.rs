//! DRAM read bandwidth.
//!
//! Decode streams weights read-only, so we measure *reads* rather than the
//! classic STREAM triad. Triad is 2 reads + 1 write and its write leg incurs
//! read-for-ownership traffic, which makes it the wrong shape by ~20%.
//!
//! Two things this must get right:
//!
//!   * Working set must exceed last-level cache several times over, or we
//!     measure cache bandwidth. 256 MiB clears any consumer LLC (and Apple's
//!     System Level Cache, which reports no size at all).
//!   * The summation must not serialise. A single accumulator creates a
//!     dependency chain and the loop becomes latency-bound rather than
//!     bandwidth-bound. Eight independent accumulators give the out-of-order
//!     engine enough to keep loads in flight.

use std::hint::black_box;
use std::time::Instant;

/// 256 MiB. Large enough to defeat any last-level cache we will meet.
const WORKING_SET: usize = 256 << 20;
const REPEATS: usize = 5;

#[derive(Debug, Clone)]
pub struct RamResult {
    /// Best bandwidth observed across all thread counts, GB/s (decimal).
    pub peak_gbs: f64,
    /// Thread count that achieved `peak_gbs`.
    pub peak_threads: usize,
    /// Every (threads, GB/s) pair we measured.
    pub by_threads: Vec<(usize, f64)>,
    pub working_set: usize,
}

/// Sum a slice with eight independent accumulators.
///
/// `chunks_exact(8)` gives LLVM a fixed-width inner body it can unroll and
/// vectorise into wide loads; the eight accumulators keep several loads in
/// flight at once. `wrapping_add` because overflow is meaningless here and we
/// do not want a panic check in the hot loop.
#[inline]
fn sum_ilp(s: &[u64]) -> u64 {
    let mut acc = [0u64; 8];
    let mut it = s.chunks_exact(8);
    for c in &mut it {
        for k in 0..8 {
            acc[k] = acc[k].wrapping_add(c[k]);
        }
    }
    let mut total = it.remainder().iter().fold(0u64, |a, &b| a.wrapping_add(b));
    for a in acc {
        total = total.wrapping_add(a);
    }
    total
}

fn run_once(buf: &[u64], threads: usize) -> f64 {
    let chunk = buf.len().div_ceil(threads);
    let t0 = Instant::now();
    let total: u64 = std::thread::scope(|s| {
        let handles: Vec<_> = buf
            .chunks(chunk)
            .map(|c| s.spawn(move || sum_ilp(black_box(c))))
            .collect();
        handles.into_iter().map(|h| h.join().unwrap_or(0)).sum()
    });
    let dt = t0.elapsed().as_secs_f64();
    black_box(total);
    (buf.len() * 8) as f64 / dt / 1e9
}

/// Measure read bandwidth at each requested thread count.
///
/// `thread_counts` should span 1 up to the machine's performance-core count;
/// bandwidth typically saturates well before core count and we want to report
/// the plateau, not the single-thread figure.
pub fn measure(thread_counts: &[usize]) -> RamResult {
    // vec! writes every element, which first-touches and faults in the whole
    // allocation. Without this the first timed pass would measure page faults.
    let buf = vec![1u64; WORKING_SET / 8];

    // Warm up: pull the pages through once, untimed.
    black_box(sum_ilp(&buf));

    let mut by_threads = Vec::new();
    for &t in thread_counts {
        // Best of N. Interference from other processes only ever makes a
        // sample slower, so the maximum is the least-contaminated estimate.
        let best = (0..REPEATS)
            .map(|_| run_once(&buf, t))
            .fold(0.0f64, f64::max);
        by_threads.push((t, best));
    }

    let (peak_threads, peak_gbs) = by_threads
        .iter()
        .copied()
        .fold((1, 0.0), |acc, (t, g)| if g > acc.1 { (t, g) } else { acc });

    RamResult {
        peak_gbs,
        peak_threads,
        by_threads,
        working_set: WORKING_SET,
    }
}
