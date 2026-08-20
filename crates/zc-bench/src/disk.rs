//! Uncached disk read bandwidth and IOPS.
//!
//! The whole difficulty is bypassing the OS page cache. If you do not, a
//! second run reports 10+ GB/s on a drive that physically cannot exceed 5,
//! and every downstream prediction is nonsense.
//!
//!   macOS   fcntl(F_NOCACHE) — no alignment requirement, but it only stops
//!           *future* caching. Pages already resident are still served from
//!           the unified buffer cache, so a model file a runtime currently has
//!           mapped reads at 68 GB/s on a 2 GB/s drive. They have to be
//!           invalidated first — see `drop_cached_pages`.
//!   Linux   O_DIRECT — genuinely bypasses, but demands that buffer address,
//!           file offset and length are all block-aligned, or read() fails
//!           with EINVAL.
//!
//! We also disable readahead. Sequential prefetch would otherwise hide the
//! per-request latency we are trying to observe.

use std::fs::File;
use std::io;
use std::path::{Path, PathBuf};
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
            Ok(f) => Ok((f, true)),
            Err(_) => Ok((File::open(path)?, false)),
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
        let len = f.metadata()?.len();
        // F_NOCACHE alone measured this machine's SSD at 20.6 GB/s while it was
        // physically doing 2.05 GB/s, because the file was already in the cache.
        // Evicting first is the difference between a measurement and a memcpy.
        let evicted = drop_cached_pages(fd, len);
        Ok((f, nocache && evicted))
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
            Ok(f) => Ok((f, true)),
            // Some filesystems refuse it; fall back rather than fail the probe.
            Err(_) => Ok((File::open(path)?, false)),
        }
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos", windows)))]
    {
        Ok((File::open(path)?, false))
    }
}

/// Evict this file's resident pages from the unified buffer cache.
///
/// `msync(MS_INVALIDATE)` over a shared read-only mapping is the only portable-
/// on-macOS way to do it without root: `purge(8)` needs privileges and drops
/// the whole system's cache, which is both rude and slower.
///
/// Returns false if the eviction did not happen, which makes `uncached` false
/// and gets the number labelled `[CACHED - inflated]` rather than published as
/// a disk speed.
///
/// ponytail: another process may fault the same pages back in while we measure
/// — a runtime holding the model mapped will. One eviction before the loop is
/// what we can do without fighting for the cache; the upgrade, if it ever
/// matters, is to prefer a scratch file over a file someone else has open.
#[cfg(target_os = "macos")]
fn drop_cached_pages(fd: std::os::unix::io::RawFd, len: u64) -> bool {
    if len == 0 {
        return false;
    }
    let len = len as usize;
    // SAFETY: a fresh MAP_SHARED PROT_READ mapping of a file we hold open;
    // msync and munmap operate only on the address range mmap just returned.
    unsafe {
        let addr = libc::mmap(std::ptr::null_mut(), len, libc::PROT_READ, libc::MAP_SHARED, fd, 0);
        if addr == libc::MAP_FAILED {
            return false;
        }
        let ok = libc::msync(addr, len, libc::MS_INVALIDATE) == 0;
        libc::munmap(addr, len);
        ok
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

/// Deletes the scratch file on *every* exit path.
///
/// Five `?` returns used to sit between creating a 512 MiB scratch file and the
/// `remove_file` at the end of `measure`, so a failed write left the whole thing
/// orphaned in the user's home directory. The way that write fails is a full
/// disk, which is squarely the hardware this project targets -- the machine
/// least able to spare half a gigabyte was the one guaranteed to keep it.
struct Scratch(Option<PathBuf>);

impl Drop for Scratch {
    fn drop(&mut self) {
        if let Some(p) = self.0.take() {
            let _ = std::fs::remove_file(p);
            crate::cleanup::forget_file();
        }
    }
}

/// Benchmark the volume that `target` lives on.
///
/// `target` should be an existing large file so that we write nothing. If it
/// is `None`, a scratch file is created and removed, and `created_file` is set
/// so the caller can disclose the write.
pub fn measure(target: Option<&Path>, scratch_dir: &Path, threads: usize) -> io::Result<DiskResult> {
    const MIN_SPAN: u64 = 512 << 20;

    // Declared before the file exists so the guard covers the creation itself.
    let mut scratch = Scratch(None);
    let (path, created_file) = match target {
        Some(p) if std::fs::metadata(p).map(|m| m.len()).unwrap_or(0) >= MIN_SPAN => {
            (p.to_path_buf(), false)
        }
        _ => {
            let p = scratch_dir.join(".zc-bench-scratch.tmp");
            let f = File::create(&p)?;
            scratch.0 = Some(p.clone());
            // Drop covers every `?` below, but not a signal -- and this file
            // exists through the slowest phase of the run, which is when an
            // impatient user interrupts. See `cleanup`.
            crate::cleanup::remove_on_signal(&p);
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

    // `scratch` removes it on drop, including on every `?` above.
    drop(scratch);

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

#[cfg(test)]
mod tests {
    /// The scratch file must not survive an early return.
    ///
    /// Five `?` operators sit between creating it and the end of `measure`, and
    /// the way the write fails is a full disk -- the machine least able to spare
    /// 512 MiB was the one guaranteed to be left holding it.
    #[test]
    fn a_scratch_file_does_not_survive_an_early_return() {
        let path = std::env::temp_dir().join("zc-bench-scratch-guard-test.bin");
        std::fs::write(&path, b"x").expect("write");
        assert!(path.exists());
        {
            let _guard = super::Scratch(Some(path.clone()));
        }
        assert!(!path.exists(), "the guard must delete it on every exit path");
    }

    /// A scratch directory that cannot be written to must produce an error and
    /// no file, rather than a panic or a half-written half-gigabyte.
    #[test]
    fn an_unwritable_scratch_directory_is_an_error_not_a_mess() {
        let nowhere = std::path::Path::new("/dev/null/definitely-not-a-directory");
        assert!(super::measure(None, nowhere, 2).is_err());
    }

    /// The eviction has to actually succeed, or every macOS disk number is a
    /// page-cache read wearing a disk number's clothes. Constants and argument
    /// order are the whole risk here, and both fail loudly as a false return.
    #[cfg(target_os = "macos")]
    #[test]
    fn cached_pages_can_be_dropped() {
        use std::io::Write;
        use std::os::unix::io::AsRawFd;

        let path = std::env::temp_dir().join("zc-bench-evict-test.bin");
        let mut f = std::fs::File::create(&path).expect("create");
        f.write_all(&vec![0x5Au8; 1 << 20]).expect("write");
        f.sync_all().expect("sync");
        drop(f);

        let f = std::fs::File::open(&path).expect("open");
        assert!(super::drop_cached_pages(f.as_raw_fd(), 1 << 20));
        // A zero-length file has no pages, and claiming success would let an
        // empty scratch file report itself as honestly measured.
        assert!(!super::drop_cached_pages(f.as_raw_fd(), 0));

        let _ = std::fs::remove_file(&path);
    }
}
