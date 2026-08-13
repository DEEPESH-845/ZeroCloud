//! CPU topology and ISA features.
//!
//! The output that matters is `recommended_threads`. Inference splits a matmul
//! across threads and joins at the end, so throughput is set by the *slowest*
//! participant. Two consequences:
//!
//!   * SMT/hyperthread siblings share load/store units and L1/L2. Decode is
//!     memory-bound, so siblings starve each other. Net effect ~0 to -10%.
//!   * Efficiency cores are several times slower than performance cores.
//!     Including them means every fast thread waits at the join barrier.
//!
//! So: physical performance cores, not `num_cpus::get()`.

#[derive(Debug, Clone, Default)]
pub struct IsaFeatures {
    pub avx2: bool,
    pub avx512f: bool,
    pub f16c: bool,
    /// ARM SDOT/UDOT — large speedup for int8 quantized matmul.
    pub neon_dotprod: bool,
    /// ARM int8 matrix multiply — larger still, where kernels use it.
    pub neon_i8mm: bool,
}

#[derive(Debug, Clone)]
pub struct Cpu {
    pub brand: String,
    pub physical: u32,
    pub logical: u32,
    /// Performance cores. Equals `physical` on homogeneous CPUs.
    pub p_cores: u32,
    /// Efficiency cores. Zero on homogeneous CPUs.
    pub e_cores: u32,
    pub smt: bool,
    /// Last-level cache in bytes. Best effort — used only to size benchmark
    /// working sets, and we oversize those anyway.
    pub llc_bytes: u64,
    pub cache_line: u64,
    pub features: IsaFeatures,
    /// What we will actually tell the user to run inference with.
    pub recommended_threads: u32,
}

impl Cpu {
    fn finish(mut self) -> Self {
        self.smt = self.logical > self.physical;
        self.recommended_threads = if self.p_cores > 0 && self.e_cores > 0 {
            self.p_cores // heterogeneous: P-cores only, avoid straggler stalls
        } else {
            self.physical // homogeneous: physical, never logical
        }
        .max(1);
        self
    }
}

// ---------------------------------------------------------------- macOS ----

#[cfg(target_os = "macos")]
mod imp {
    use super::{Cpu, IsaFeatures};

    fn num(name: &str) -> Option<u64> {
        let cname = std::ffi::CString::new(name).ok()?;
        let mut val: u64 = 0;
        let mut len = std::mem::size_of::<u64>();
        let rc = unsafe {
            libc::sysctlbyname(
                cname.as_ptr(),
                &mut val as *mut u64 as *mut libc::c_void,
                &mut len,
                std::ptr::null_mut(),
                0,
            )
        };
        (rc == 0).then_some(val)
    }

    fn flag(name: &str) -> bool {
        num(name).unwrap_or(0) != 0
    }

    fn string(name: &str) -> Option<String> {
        let cname = std::ffi::CString::new(name).ok()?;
        // Two-call idiom: first with a null buffer to learn the length,
        // then again to actually read it.
        let mut len: usize = 0;
        unsafe {
            if libc::sysctlbyname(
                cname.as_ptr(),
                std::ptr::null_mut(),
                &mut len,
                std::ptr::null_mut(),
                0,
            ) != 0
            {
                return None;
            }
            let mut buf = vec![0u8; len];
            if libc::sysctlbyname(
                cname.as_ptr(),
                buf.as_mut_ptr() as *mut libc::c_void,
                &mut len,
                std::ptr::null_mut(),
                0,
            ) != 0
            {
                return None;
            }
            buf.truncate(len.saturating_sub(1)); // drop trailing NUL
            String::from_utf8(buf).ok()
        }
    }

    pub fn probe() -> Cpu {
        let physical = num("hw.physicalcpu").unwrap_or(1) as u32;
        let logical = num("hw.logicalcpu").unwrap_or(physical as u64) as u32;

        // hw.nperflevels > 1 means a heterogeneous CPU. perflevel0 is always
        // the fastest tier, perflevel1 the efficiency tier.
        let hetero = num("hw.nperflevels").unwrap_or(1) > 1;
        let (p_cores, e_cores) = if hetero {
            (
                num("hw.perflevel0.physicalcpu").unwrap_or(0) as u32,
                num("hw.perflevel1.physicalcpu").unwrap_or(0) as u32,
            )
        } else {
            (physical, 0)
        };

        // Apple Silicon reports no hw.l3cachesize — it has a System Level
        // Cache instead. Fall back to L2.
        let llc_bytes = num("hw.l3cachesize")
            .filter(|&v| v > 0)
            .or_else(|| num("hw.l2cachesize"))
            .unwrap_or(8 << 20);

        Cpu {
            brand: string("machdep.cpu.brand_string").unwrap_or_else(|| "unknown".into()),
            physical,
            logical,
            p_cores,
            e_cores,
            smt: false,
            llc_bytes,
            cache_line: num("hw.cachelinesize").unwrap_or(64),
            features: IsaFeatures {
                avx2: flag("hw.optional.avx2_0"),
                avx512f: flag("hw.optional.avx512f"),
                f16c: flag("hw.optional.f16c"),
                // NEON itself is mandatory on arm64, so there is no flag for
                // it — only for the extensions that actually change kernel
                // selection.
                neon_dotprod: flag("hw.optional.arm.FEAT_DotProd"),
                neon_i8mm: flag("hw.optional.arm.FEAT_I8MM"),
            },
            recommended_threads: 0,
        }
        .finish()
    }
}

// ---------------------------------------------------------------- Linux ----

#[cfg(target_os = "linux")]
mod imp {
    use super::{Cpu, IsaFeatures};
    use std::collections::HashSet;

    fn read(p: &str) -> Option<String> {
        std::fs::read_to_string(p).ok().map(|s| s.trim().to_string())
    }

    /// Parse a kernel cpulist ("0-7,16-23") into a count.
    fn cpulist_len(s: &str) -> u32 {
        s.split(',')
            .filter_map(|part| match part.split_once('-') {
                Some((a, b)) => Some(b.trim().parse::<u32>().ok()? - a.trim().parse::<u32>().ok()? + 1),
                None => part.trim().parse::<u32>().ok().map(|_| 1),
            })
            .sum()
    }

    pub fn probe() -> Cpu {
        let cpuinfo = read("/proc/cpuinfo").unwrap_or_default();

        let brand = cpuinfo
            .lines()
            .find(|l| l.starts_with("model name"))
            .and_then(|l| l.split_once(':'))
            .map(|(_, v)| v.trim().to_string())
            .unwrap_or_else(|| "unknown".into());

        let logical = cpuinfo.lines().filter(|l| l.starts_with("processor")).count() as u32;

        // Physical cores = distinct (physical id, core id) pairs. Anything
        // beyond that is an SMT sibling.
        let mut seen = HashSet::new();
        let (mut pkg, mut core) = (None, None);
        for line in cpuinfo.lines() {
            if let Some((k, v)) = line.split_once(':') {
                match k.trim() {
                    "physical id" => pkg = v.trim().parse::<u32>().ok(),
                    "core id" => core = v.trim().parse::<u32>().ok(),
                    _ => {}
                }
            }
            if line.trim().is_empty() {
                if let (Some(p), Some(c)) = (pkg, core) {
                    seen.insert((p, c));
                }
                pkg = None;
                core = None;
            }
        }
        if let (Some(p), Some(c)) = (pkg, core) {
            seen.insert((p, c));
        }
        let physical = if seen.is_empty() { logical } else { seen.len() as u32 };

        // Hybrid Intel (Alder Lake and later) exposes two PMU devices.
        // ARM big.LITTLE instead exposes per-cpu `cpu_capacity`.
        let (p_cores, e_cores) = if let (Some(pc), Some(ec)) = (
            read("/sys/devices/cpu_core/cpus"),
            read("/sys/devices/cpu_atom/cpus"),
        ) {
            (cpulist_len(&pc), cpulist_len(&ec))
        } else {
            let caps: Vec<u32> = (0..logical)
                .filter_map(|i| {
                    read(&format!(
                        "/sys/devices/system/cpu/cpu{i}/cpu_capacity"
                    ))?
                    .parse()
                    .ok()
                })
                .collect();
            match caps.iter().max() {
                Some(&hi) if caps.iter().any(|&c| c < hi) => (
                    caps.iter().filter(|&&c| c == hi).count() as u32,
                    caps.iter().filter(|&&c| c < hi).count() as u32,
                ),
                _ => (physical, 0),
            }
        };

        // Largest cache level advertised by cpu0.
        let llc_bytes = (0..8)
            .filter_map(|i| {
                let base = format!("/sys/devices/system/cpu/cpu0/cache/index{i}");
                let size = read(&format!("{base}/size"))?;
                let kb: u64 = size.trim_end_matches(['K', 'k']).parse().ok()?;
                Some(kb * 1024)
            })
            .max()
            .unwrap_or(8 << 20);

        let flags: HashSet<&str> = cpuinfo
            .lines()
            .find(|l| l.starts_with("flags") || l.starts_with("Features"))
            .and_then(|l| l.split_once(':'))
            .map(|(_, v)| v.split_whitespace().collect())
            .unwrap_or_default();

        Cpu {
            brand,
            physical,
            logical,
            p_cores,
            e_cores,
            smt: false,
            llc_bytes,
            cache_line: read("/sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size")
                .and_then(|s| s.parse().ok())
                .unwrap_or(64),
            features: IsaFeatures {
                avx2: flags.contains("avx2"),
                avx512f: flags.contains("avx512f"),
                f16c: flags.contains("f16c"),
                neon_dotprod: flags.contains("asimddp"),
                neon_i8mm: flags.contains("i8mm"),
            },
            recommended_threads: 0,
        }
        .finish()
    }
}

// -------------------------------------------------------------- Windows ----
//
// UNVALIDATED: written from the API contract, not yet run on real hardware.
// GetLogicalProcessorInformationEx returns a packed stream of variable-length
// records, so we walk it by each record's own Size field.

#[cfg(target_os = "windows")]
mod imp {
    use super::{Cpu, IsaFeatures};
    use windows_sys::Win32::System::SystemInformation::{
        GetLogicalProcessorInformationEx, RelationProcessorCore,
        SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX,
    };

    pub fn probe() -> Cpu {
        let mut len: u32 = 0;
        unsafe {
            // First call fails with the required buffer size.
            GetLogicalProcessorInformationEx(RelationProcessorCore, std::ptr::null_mut(), &mut len);
        }
        let mut buf = vec![0u8; len as usize];
        let ok = unsafe {
            GetLogicalProcessorInformationEx(
                RelationProcessorCore,
                buf.as_mut_ptr() as *mut SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX,
                &mut len,
            ) != 0
        };

        let (mut physical, mut logical, mut p_cores, mut e_cores) = (0u32, 0u32, 0u32, 0u32);
        if ok {
            let mut off = 0usize;
            while off + std::mem::size_of::<u32>() * 2 <= len as usize {
                let rec = unsafe {
                    &*(buf.as_ptr().add(off) as *const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)
                };
                let size = rec.Size as usize;
                if size == 0 {
                    break;
                }
                physical += 1;
                let p = unsafe { &rec.Anonymous.Processor };
                // Each GroupMask bit is one logical processor on this core.
                logical += p.GroupMask[0].Mask.count_ones();
                // EfficiencyClass: higher is faster. 0 = slowest tier.
                if p.EfficiencyClass > 0 {
                    p_cores += 1;
                } else {
                    e_cores += 1;
                }
                off += size;
            }
            // Homogeneous CPUs report EfficiencyClass 0 everywhere, which the
            // loop above would misread as "all efficiency cores".
            if p_cores == 0 {
                p_cores = physical;
                e_cores = 0;
            }
        }

        Cpu {
            brand: std::env::var("PROCESSOR_IDENTIFIER").unwrap_or_else(|_| "unknown".into()),
            physical: physical.max(1),
            logical: logical.max(1),
            p_cores,
            e_cores,
            smt: false,
            llc_bytes: 8 << 20, // TODO: RelationCache pass
            cache_line: 64,
            features: IsaFeatures::default(), // TODO: __cpuid
            recommended_threads: 0,
        }
        .finish()
    }
}

pub fn probe() -> Cpu {
    imp::probe()
}
