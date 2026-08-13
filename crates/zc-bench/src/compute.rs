//! Compute throughput.
//!
//! Only prefill (prompt processing) is compute-bound; decode is memory-bound.
//! So these numbers set time-to-first-token, not tokens/sec.
//!
//! Two primitives, because quantised inference uses two very different paths:
//!
//!   f32 FMA   the classic peak-FLOPS figure. Relevant for f16/f32 models and
//!             as a stable cross-platform reference point.
//!   int8 dot  what quantised prefill actually runs. llama.cpp's Q4_K prefill
//!             goes through `vec_dot_q4_K_q8_K`, an int8 dot product.
//!
//! Neither figure is a usable basis for predicting prefill, and we measured why
//! rather than assuming: stable Rust cannot emit ARM SDOT (see `dot_i8`), and
//! nothing here runs on the GPU at all. Prefill is therefore *fitted from real
//! runs* rather than derived from these numbers. They remain useful as relative
//! hardware signals and as the denominator the fit is expressed against.
//!
//! Both use many independent accumulator chains. With one chain you measure
//! instruction *latency* (~4 cycles) rather than throughput, which is a real
//! mistake this file made once: 16 f32 lanes gave 4 vector chains and reported
//! 25% of peak.

use std::hint::black_box;
use std::time::Instant;

/// 64 f32 = 16 vector registers = 16 independent chains. Must exceed
/// `fp_pipes * fma_latency` (roughly 4 x 4) or the loop is latency-bound.
/// ARM has 32 vector registers and x86-64 AVX2 has 16, so this fits both.
const F32_LANES: usize = 64;
const F32_ITERS: u64 = 1_000_000;

/// Working set for the int8 dot. Small enough to sit in L1/L2 so we measure
/// arithmetic rather than memory — prefill GEMM is tiled to do exactly that.
const I8_LEN: usize = 16 << 10;
const I8_ROUNDS: u64 = 20_000;

#[derive(Debug, Clone)]
pub struct ComputeResult {
    pub gflops_1t: f64,
    pub gflops_nt: f64,
    /// Billions of int8 multiply-accumulate operations per second, all threads.
    /// Each MAC counts as 2 ops, matching the FLOPS convention.
    pub int8_gops_nt: f64,
    /// int8 throughput relative to f32, as measured by the portable kernel.
    /// A weak signal only — see the limitation documented on `dot_i8`.
    pub int8_ratio: f64,
    pub threads: usize,
}

#[inline(never)]
fn fma_kernel(iters: u64) -> f32 {
    let mut a = [0.0f32; F32_LANES];
    for (k, v) in a.iter_mut().enumerate() {
        *v = k as f32 * 1e-3;
    }
    let b = black_box(1.000_000_1f32);
    let c = black_box(1e-7f32);
    for _ in 0..iters {
        for v in a.iter_mut() {
            // mul_add lowers to a single FMA (FMLA on ARM, VFMADD on x86).
            // `v * b + c` would be two instructions, since Rust does not
            // enable fast-math contraction.
            *v = v.mul_add(b, c);
        }
    }
    a.iter().sum()
}

/// int8 dot product with four independent i32 accumulators.
///
/// MEASURED LIMITATION: this does not reach the hardware's real int8 peak, and
/// cannot on stable Rust.
///
/// `vdotq_s32` (ARM SDOT) is gated behind the unstable `stdarch_neon_dotprod`
/// feature, and LLVM will not auto-lower to it even with
/// `-C target-feature=+dotprod`. Benchmarked on an M5, 4 threads:
///
///   separate-lane widening MAC (this)     363 GOPS
///   sdot-shaped (4 products -> 1 lane)    108 GOPS
///
/// The sdot *shape* is 3.4x slower here because its inner reduction forms a
/// dependency chain, so we use the shape that actually vectorises. Real
/// runtimes reach SDOT through intrinsics and go several times faster than
/// either. Treat this figure as a relative signal, never as achievable
/// quantised-prefill throughput — which is why prefill is fitted from
/// measurement rather than derived from it (see zc-model::fit).
#[inline(never)]
fn dot_i8(a: &[i8], b: &[i8]) -> i32 {
    let mut acc = [0i32; 4];
    for (x, y) in a.chunks_exact(4).zip(b.chunks_exact(4)) {
        for k in 0..4 {
            acc[k] = acc[k].wrapping_add(x[k] as i32 * y[k] as i32);
        }
    }
    acc.iter().fold(0i32, |s, &v| s.wrapping_add(v))
}

/// Returns (seconds, one result kept alive so the work cannot be optimised
/// away). `None` if every worker panicked — a benchmark must never take the
/// process down with it.
fn run_threads<T, F>(threads: usize, f: F) -> (f64, Option<T>)
where
    F: Fn() -> T + Sync,
    T: Send,
{
    let t0 = Instant::now();
    let mut out = std::thread::scope(|s| {
        let hs: Vec<_> = (0..threads).map(|_| s.spawn(&f)).collect();
        hs.into_iter().filter_map(|h| h.join().ok()).collect::<Vec<_>>()
    });
    (t0.elapsed().as_secs_f64(), out.pop())
}

fn gflops(threads: usize) -> f64 {
    let (dt, last) = run_threads(threads, || fma_kernel(F32_ITERS));
    if last.is_none() || dt <= 0.0 {
        return 0.0;
    }
    black_box(last);
    // Each mul_add is one multiply plus one add = 2 flops.
    (F32_ITERS * F32_LANES as u64 * 2 * threads as u64) as f64 / dt / 1e9
}

fn int8_gops(threads: usize) -> f64 {
    // Non-trivial, non-uniform data so nothing can be constant-folded.
    let a: Vec<i8> = (0..I8_LEN).map(|i| (i as i32 % 127 - 63) as i8).collect();
    let b: Vec<i8> = (0..I8_LEN).map(|i| (i as i32 % 91 - 45) as i8).collect();

    let (dt, last) = run_threads(threads, || {
        let mut s = 0i32;
        for _ in 0..I8_ROUNDS {
            s = s.wrapping_add(dot_i8(black_box(&a), black_box(&b)));
        }
        s
    });
    if last.is_none() || dt <= 0.0 {
        return 0.0;
    }
    black_box(last);
    // One multiply + one add per element, matching the FLOPS convention.
    (I8_ROUNDS * I8_LEN as u64 * 2 * threads as u64) as f64 / dt / 1e9
}

pub fn measure(threads: usize) -> ComputeResult {
    black_box(fma_kernel(100_000)); // settle clocks

    // Best of 3: interference only ever makes a sample slower.
    let gflops_1t = (0..3).map(|_| gflops(1)).fold(0.0, f64::max);
    let gflops_nt = (0..3).map(|_| gflops(threads)).fold(0.0, f64::max);
    let int8_gops_nt = (0..3).map(|_| int8_gops(threads)).fold(0.0, f64::max);

    ComputeResult {
        gflops_1t,
        gflops_nt,
        int8_gops_nt,
        int8_ratio: if gflops_nt > 0.0 {
            int8_gops_nt / gflops_nt
        } else {
            1.0
        },
        threads,
    }
}
