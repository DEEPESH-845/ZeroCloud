//! Uncached disk read bandwidth and IOPS.
//!
//! The whole difficulty is bypassing the OS page cache. If you do not, a
//! second run reports 10+ GB/s on a drive that physically cannot exceed 5,
//! and every downstream prediction is nonsense.
//!
//!   macOS   fcntl(F_NOCACHE) — reads bypass the buffer cache. No alignment
//!           requirement, which makes this the easy platform.
//!   Linux   O_DIRECT — genuinely bypasses, but demands that buffer address,
//!           file offset and length are all block-aligned, or read() fails
//!           with EINVAL.
//!
//! We also disable readahead. Sequential prefetch would otherwise hide the
//! per-request latency we are trying to observe.

use std::fs::File;
use std::io;
use std::path::Path;
use std::time::{Duration, Instant};

const BLOCK_LARGE: usize = 128 << 10; // matches NVMe optimal transfer size
const BLOCK_SMALL: usize = 4 << 10; // exposes DRAM-less controllers
const ALIGN: usize = 4096;
const SAMPLE: Duration = Duration::from_millis(400);

#[derive(Debug, Clone)]
pub struct DiskResult {
    pub seq_qd1_gbs: f64,
    pub rand_128k_qd1_gbs: f64,
    pub rand_128k_qdn_gbs: f64,
    pub rand_4k_qdn_iops: f64,
    pub queue_depth: usize,
    /// False means we could not bypass the cache and the numbers are inflated.
    pub uncached: bool,
    pub file_bytes: u64,
    /// True if we had to create a scratch file (and therefore wrote to disk).
    pub created_file: bool,
}

/// A heap buffer aligned to `ALIGN`, required by O_DIRECT on Linux.
///
/// `Vec<u8>` only guarantees alignment of 1, so we allocate manually. The
/// Drop impl is the entire reason this is a struct rather than a function.
struct AlignedBuf {
    ptr: *mut u8,
    len: usize,
}

impl AlignedBuf {
    fn new(len: usize) -> Self {
        let layout = std::alloc::Layout::from_size_align(len, ALIGN).expect("valid layout");
        // SAFETY: len is non-zero and the layout is valid.
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        assert!(!ptr.is_null(), "allocation failed");
        Self { ptr, len }
    }
    fn as_mut(&mut self) -> &mut [u8] {
        // SAFETY: ptr is a valid allocation of exactly len bytes, and &mut self
        // guarantees we hold exclusive access.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl Drop for AlignedBuf {
    fn drop(&mut self) {
        let layout = std::alloc::Layout::from_size_align(self.len, ALIGN).unwrap();
        // SAFETY: same layout used to allocate.
        unsafe { std::alloc::dealloc(self.ptr, layout) };
    }
}

// AlignedBuf owns its allocation exclusively, so moving it between threads is
// sound. The raw pointer is what makes this opt-in rather than automatic.
unsafe impl Send for AlignedBuf {}

/// xorshift64*. We need offsets that defeat readahead, not cryptographic
/// randomness, and this avoids a dependency.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
}

/// Positional read. Does not touch the file's shared offset, so many threads
/// may safely share one handle.
///
/// Unix and Windows spell this differently (`pread` vs `ReadFile` with an
/// explicit offset) but the contract is identical.
fn read_at(f: &File, buf: &mut [u8], offset: u64) -> io::Result<usize> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::FileExt;
        f.read_at(buf, offset)
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::FileExt;
        f.seek_read(buf, offset)
    }
}

fn write_at(f: &File, buf: &[u8], offset: u64) -> io::Result<usize> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::FileExt;
        f.write_at(buf, offset)
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::FileExt;
        f.seek_write(buf, offset)
    }
}

/// Open with caching disabled. Returns (file, actually_uncached).
fn open_uncached(path: &Path) -> io::Result<(File, bool)> {
    #[cfg(target_os = "linux")]
    {
        use std::os::unix::fs::OpenOptionsExt;
        // O_DIRECT is refused by some filesystems (tmpfs, some network FS),
        // so fall back rather than failing the whole benchmark.
        match File::options()
            .read(true)
            .custom_flags(libc::O_DIRECT)
            .open(path)
        {
            Ok(f) => return Ok((f, true)),
            Err(_) => return Ok((File::open(path)?, false)),
        }
    }
    #[cfg(target_os = "macos")]
    {
        use std::os::unix::io::AsRawFd;
        let f = File::open(path)?;
        let fd = f.as_raw_fd();
        // SAFETY: fd is valid and owned by `f` for the duration of the calls.
        let nocache = unsafe { libc::fcntl(fd, libc::F_NOCACHE, 1) } == 0;
        unsafe { libc::fcntl(fd, libc::F_RDAHEAD, 0) };
        Ok((f, nocache))
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        // FILE_FLAG_NO_BUFFERING bypasses the system cache entirely. It demands
        // that buffer address, file offset and length all be sector-aligned;
        // AlignedBuf gives 4096-byte alignment and every offset here is
        // block-aligned by construction, so the contract holds.
        //
        // UNVALIDATED on real hardware. Known limitation: without
        // FILE_FLAG_OVERLAPPED, concurrent reads on one handle may serialise
        // inside the kernel, which would understate queue-depth bandwidth.
        // Opening a handle per thread is the fix if measurement shows it.
        const FILE_FLAG_NO_BUFFERING: u32 = 0x2000_0000;
        match File::options()
            .read(true)
            .custom_flags(FILE_FLAG_NO_BUFFERING)
            .open(path)
        {
            Ok(f) => return Ok((f, true)),
            // Some filesystems refuse it; fall back rather than fail the probe.
            Err(_) => return Ok((File::open(path)?, false)),
        }
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos", windows)))]
    {
        Ok((File::open(path)?, false))
    }
}

/// Time-boxed read loop. Returns (bytes_read, ops, seconds).
///
/// Time-boxed rather than count-boxed so a slow drive does not stall the whole
/// probe for minutes.
fn read_loop(
    file: &File,
    span: u64,
    block: usize,
    sequential: bool,
    seed: u64,
    budget: Duration,
) -> (u64, u64, f64) {
    let mut buf = AlignedBuf::new(block);
    let mut rng = Rng(seed | 1);
    let blocks = (span / block as u64).max(1);
    let (mut bytes, mut ops, mut cursor) = (0u64, 0u64, 0u64);
    let t0 = Instant::now();
    while t0.elapsed() < budget {
        // Batch between clock reads; Instant::now() is a syscall-ish cost on
        // some platforms and would otherwise pollute the measurement.
        for _ in 0..32 {
            let idx = if sequential {
                cursor = (cursor + 1) % blocks;
                cursor
            } else {
                rng.next() % blocks
            };
            // Offset is block-aligned by construction, satisfying O_DIRECT.
            let off = idx * block as u64;
            match read_at(file, buf.as_mut(), off) {
                Ok(n) if n > 0 => {
                    bytes += n as u64;
                    ops += 1;
                }
                _ => break,
            }
        }
    }
    (bytes, ops, t0.elapsed().as_secs_f64())
}

/// Same loop across `threads` workers to build queue depth.
///
/// `pread` does not touch the shared file offset, so all threads can safely
/// share one `&File`.
fn read_loop_parallel(
    file: &File,
    span: u64,
    block: usize,
    threads: usize,
    budget: Duration,
) -> (u64, u64, f64) {
    let t0 = Instant::now();
    let results: Vec<(u64, u64, f64)> = std::thread::scope(|s| {
        let hs: Vec<_> = (0..threads)
            .map(|i| {
                s.spawn(move || {
                    read_loop(file, span, block, false, 0x9E37_79B9_7F4A_7C15 ^ (i as u64) << 17, budget)
                })
            })
            .collect();
        hs.into_iter().filter_map(|h| h.join().ok()).collect()
    });
    let wall = t0.elapsed().as_secs_f64();
    (
        results.iter().map(|r| r.0).sum(),
        results.iter().map(|r| r.1).sum(),
        wall,
    )
}

/// Benchmark the volume that `target` lives on.
///
/// `target` should be an existing large file so that we write nothing. If it
/// is `None`, a scratch file is created and removed, and `created_file` is set
/// so the caller can disclose the write.
pub fn measure(target: Option<&Path>, scratch_dir: &Path, threads: usize) -> io::Result<DiskResult> {
    const MIN_SPAN: u64 = 512 << 20;

    let (path, created_file) = match target {
        Some(p) if std::fs::metadata(p).map(|m| m.len()).unwrap_or(0) >= MIN_SPAN => {
            (p.to_path_buf(), false)
        }
        _ => {
            let p = scratch_dir.join(".zc-bench-scratch.tmp");
            let f = File::create(&p)?;
            f.set_len(MIN_SPAN)?;
            // set_len allocates sparsely; write real blocks so reads hit media
            // rather than the filesystem's hole-punching fast path.
            let chunk = vec![0xA5u8; 1 << 20];
            for i in 0..(MIN_SPAN >> 20) {
                write_at(&f, &chunk, i << 20)?;
            }
            f.sync_all()?;
            (p, true)
        }
    };

    let span = std::fs::metadata(&path)?.len();
    let (file, uncached) = open_uncached(&path)?;

    let (b1, _, t1) = read_loop(&file, span, BLOCK_LARGE, true, 1, SAMPLE);
    let (b2, _, t2) = read_loop(&file, span, BLOCK_LARGE, false, 2, SAMPLE);
    let (b3, _, t3) = read_loop_parallel(&file, span, BLOCK_LARGE, threads, SAMPLE);
    let (_, o4, t4) = read_loop_parallel(&file, span, BLOCK_SMALL, threads, SAMPLE);

    if created_file {
        let _ = std::fs::remove_file(&path);
    }

    Ok(DiskResult {
        seq_qd1_gbs: b1 as f64 / t1 / 1e9,
        rand_128k_qd1_gbs: b2 as f64 / t2 / 1e9,
        rand_128k_qdn_gbs: b3 as f64 / t3 / 1e9,
        rand_4k_qdn_iops: o4 as f64 / t4,
        queue_depth: threads,
        uncached,
        file_bytes: span,
        created_file,
    })
}
