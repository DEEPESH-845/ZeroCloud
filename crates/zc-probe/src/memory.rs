//! Physical memory detection.
//!
//! The number that matters is `available` — free memory *plus* memory the
//! kernel can reclaim without swapping. Every OS computes it differently.

#[derive(Debug, Clone)]
pub struct Memory {
    /// Physical RAM visible to the OS, in bytes.
    pub total: u64,
    /// Reclaimable without swapping, in bytes. This is what we budget from.
    pub available: u64,
    pub page_size: u64,
    /// RAM the firmware carved out before the OS booted — almost always the
    /// integrated GPU's aperture. `None` when we can't determine it.
    pub firmware_reserved: Option<u64>,
    /// macOS caps how much unified memory the GPU may wire down.
    /// `None` on non-Apple platforms.
    pub gpu_wired_limit: Option<u64>,
    /// True when CPU and GPU share one physical pool (Apple Silicon).
    /// Changes the budget math: VRAM is not additional memory.
    pub unified: bool,
}

// ---------------------------------------------------------------- macOS ----

#[cfg(target_os = "macos")]
mod imp {
    use super::Memory;

    /// Read an integer sysctl by name.
    ///
    /// sysctl writes either 4 or 8 bytes depending on the key. We zero-init a
    /// u64 and let it write into the low bytes — correct on little-endian,
    /// which covers every Mac ever shipped (x86_64 and arm64).
    fn sysctl_u64(name: &str) -> Option<u64> {
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

    pub fn probe() -> Memory {
        let page_size = sysctl_u64("hw.pagesize").unwrap_or(16384);
        let total = sysctl_u64("hw.memsize").unwrap_or(0);

        // Mach exposes VM state as page counters. There is no "available"
        // counter, so we reconstruct Activity Monitor's "Memory Used" and
        // subtract:
        //
        //   internal    - anonymous pages owned by processes (app memory)
        //   purgeable   - subset of internal the kernel may discard on demand
        //   wire        - pinned, never pageable (kernel, GPU, mlock'd)
        //   compressor  - pages the memory compressor is holding
        //
        // used = (internal - purgeable) + wire + compressor
        //
        // Everything else (free, and file-backed pages) is reclaimable.
        let mut used_pages: u64 = 0;
        let mut ok = false;
        unsafe {
            let mut s: libc::vm_statistics64 = std::mem::zeroed();
            let mut count = (std::mem::size_of::<libc::vm_statistics64>()
                / std::mem::size_of::<libc::integer_t>())
                as libc::mach_msg_type_number_t;
            // mach_host_self is deprecated in `libc` in favour of the `mach2`
            // crate. Not worth a dependency for one call; swap if it's removed.
            #[allow(deprecated)]
            let host = libc::mach_host_self();
            let rc = libc::host_statistics64(
                host,
                libc::HOST_VM_INFO64,
                &mut s as *mut _ as *mut libc::integer_t,
                &mut count,
            );
            if rc == 0 {
                used_pages = (s.internal_page_count as u64)
                    .saturating_sub(s.purgeable_count as u64)
                    + s.wire_count as u64
                    + s.compressor_page_count as u64;
                ok = true;
            }
        }

        let available = if ok {
            total.saturating_sub(used_pages * page_size)
        } else {
            // Degrade rather than fail. 70% is a conservative stand-in.
            total * 7 / 10
        };

        // 0 means "kernel default" (~75% of RAM on machines up to 36 GB).
        let gpu_wired_limit = match sysctl_u64("iogpu.wired_limit_mb") {
            Some(0) | None => None,
            Some(mb) => Some(mb * 1024 * 1024),
        };

        Memory {
            total,
            available,
            page_size,
            // Apple Silicon has no firmware carve-out: the GPU shares the
            // same pool dynamically rather than reserving it at boot.
            firmware_reserved: Some(0),
            gpu_wired_limit,
            unified: cfg!(target_arch = "aarch64"),
        }
    }
}

// ---------------------------------------------------------------- Linux ----

#[cfg(target_os = "linux")]
mod imp {
    use super::Memory;

    /// /proc/meminfo is `Key:   value kB` lines. MemAvailable has been kernel-
    /// computed since 3.14 and already accounts for reclaimable page cache and
    /// slab, so we never have to estimate it ourselves.
    fn meminfo(key: &str) -> Option<u64> {
        let text = std::fs::read_to_string("/proc/meminfo").ok()?;
        for line in text.lines() {
            let (k, v) = line.split_once(':')?;
            if k == key {
                let kb: u64 = v.split_whitespace().next()?.parse().ok()?;
                return Some(kb * 1024);
            }
        }
        None
    }

    pub fn probe() -> Memory {
        let total = meminfo("MemTotal").unwrap_or(0);
        let available = meminfo("MemAvailable")
            // Pre-3.14 fallback; deliberately crude, and old enough to be rare.
            .or_else(|| Some(meminfo("MemFree")? + meminfo("Cached")?))
            .unwrap_or(total * 7 / 10);

        Memory {
            total,
            available,
            page_size: unsafe { libc::sysconf(libc::_SC_PAGESIZE) as u64 },
            // Comparing SMBIOS installed capacity against MemTotal reveals the
            // firmware carve-out, but reading DMI usually needs root. Deferred.
            firmware_reserved: None,
            gpu_wired_limit: None,
            unified: false,
        }
    }
}

// -------------------------------------------------------------- Windows ----

#[cfg(target_os = "windows")]
mod imp {
    use super::Memory;
    use windows_sys::Win32::System::SystemInformation::{
        GlobalMemoryStatusEx, MEMORYSTATUSEX,
    };

    pub fn probe() -> Memory {
        let mut st: MEMORYSTATUSEX = unsafe { std::mem::zeroed() };
        st.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
        let ok = unsafe { GlobalMemoryStatusEx(&mut st) } != 0;

        let total = if ok { st.ullTotalPhys } else { 0 };
        Memory {
            total,
            // ullAvailPhys is already "available", not "free" — it counts
            // standby (cached) pages the memory manager can hand back.
            available: if ok { st.ullAvailPhys } else { total * 7 / 10 },
            page_size: 4096,
            // Sum of Win32_PhysicalMemory.Capacity minus ullTotalPhys gives the
            // iGPU carve-out. Needs WMI; deferred to the WMI step.
            firmware_reserved: None,
            gpu_wired_limit: None,
            unified: false,
        }
    }
}

pub fn probe() -> Memory {
    imp::probe()
}
