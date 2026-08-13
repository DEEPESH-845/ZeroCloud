//! Execution environment: are we on bare metal, in a VM, or in a container?
//!
//! This matters because virtualised environments lie about memory:
//!
//!   * WSL2 defaults to 50% of host RAM and reports that as MemTotal.
//!   * Containers have a cgroup ceiling that /proc/meminfo never mentions.
//!
//! Both produce a budget that is silently wrong, so we detect and say so.

#[derive(Debug, Clone, PartialEq)]
pub enum Virt {
    /// Windows Subsystem for Linux v2 — a real VM with a memory cap.
    Wsl2,
    /// Any OCI-style container.
    Container,
    /// Some hypervisor; the string is the vendor when we can name it.
    Hypervisor(String),
}

#[derive(Debug, Clone)]
pub struct Env {
    pub os: &'static str,
    pub arch: &'static str,
    pub virt: Option<Virt>,
    /// A cgroup/VM ceiling below physical RAM. When present this, not
    /// MemTotal, is the real limit.
    pub memory_ceiling: Option<u64>,
    /// Human-readable advice when the environment will distort results.
    pub warning: Option<String>,
}

#[cfg(target_os = "linux")]
fn detect() -> (Option<Virt>, Option<u64>) {
    let read = |p: &str| std::fs::read_to_string(p).ok();

    // cgroup v2 first, then v1. v1 signals "unlimited" with a huge sentinel.
    let ceiling = read("/sys/fs/cgroup/memory.max")
        .and_then(|s| s.trim().parse::<u64>().ok())
        .or_else(|| {
            read("/sys/fs/cgroup/memory/memory.limit_in_bytes")?
                .trim()
                .parse::<u64>()
                .ok()
                .filter(|&v| v < (1 << 62))
        });

    let version = read("/proc/version").unwrap_or_default().to_lowercase();
    if version.contains("microsoft") || version.contains("wsl") {
        return (Some(Virt::Wsl2), ceiling);
    }
    if std::path::Path::new("/.dockerenv").exists() {
        return (Some(Virt::Container), ceiling);
    }
    // DMI names the hypervisor on most VMs: QEMU, VMware, Microsoft
    // Corporation (Hyper-V), innotek GmbH (VirtualBox), Xen.
    if let Some(vendor) = read("/sys/class/dmi/id/sys_vendor") {
        let v = vendor.trim();
        if ["QEMU", "VMware", "Xen", "innotek GmbH", "Microsoft Corporation"]
            .iter()
            .any(|k| v.contains(k))
        {
            return (Some(Virt::Hypervisor(v.to_string())), ceiling);
        }
    }
    (None, ceiling)
}

#[cfg(target_os = "macos")]
fn detect() -> (Option<Virt>, Option<u64>) {
    // The kernel tells us directly whether it is running under a hypervisor.
    let mut val: u32 = 0;
    let mut len = std::mem::size_of::<u32>();
    let name = c"kern.hv_vmm_present";
    let present = unsafe {
        libc::sysctlbyname(
            name.as_ptr(),
            &mut val as *mut u32 as *mut libc::c_void,
            &mut len,
            std::ptr::null_mut(),
            0,
        ) == 0
            && val != 0
    };
    (present.then(|| Virt::Hypervisor("unknown".into())), None)
}

#[cfg(target_os = "windows")]
fn detect() -> (Option<Virt>, Option<u64>) {
    // CPUID leaf 1, ECX bit 31 is the "hypervisor present" bit. It is set by
    // every mainstream hypervisor and never by bare metal.
    #[cfg(target_arch = "x86_64")]
    {
        // __cpuid is safe on x86_64: the leaves we read are architecturally
        // defined and always present.
        let r = std::arch::x86_64::__cpuid(1);
        if r.ecx & (1 << 31) != 0 {
            // Leaf 0x4000_0000 returns a 12-byte vendor string in EBX:ECX:EDX.
            let v = std::arch::x86_64::__cpuid(0x4000_0000);
            let mut s = Vec::new();
            for reg in [v.ebx, v.ecx, v.edx] {
                s.extend_from_slice(&reg.to_le_bytes());
            }
            let name = String::from_utf8_lossy(&s).trim_end_matches('\0').to_string();
            return (Some(Virt::Hypervisor(name)), None);
        }
    }
    (None, None)
}

pub fn probe(physical_total: u64) -> Env {
    let (virt, memory_ceiling) = detect();

    let warning = match (&virt, memory_ceiling) {
        (Some(Virt::Wsl2), _) => Some(
            "WSL2 caps memory at ~50% of host RAM by default. Raise it in \
             %UserProfile%\\.wslconfig ([wsl2] memory=12GB) or run natively \
             on Windows for accurate results."
                .into(),
        ),
        (Some(Virt::Container), Some(c)) if c < physical_total => Some(format!(
            "Container memory limit is {:.1} GiB, below physical RAM. \
             Predictions apply to the container, not the host.",
            c as f64 / 1073741824.0
        )),
        (Some(Virt::Hypervisor(name)), _) => Some(format!(
            "Running under a hypervisor ({name}). Disk and memory bandwidth \
             are typically 10-30% below bare metal."
        )),
        _ => None,
    };

    Env {
        os: std::env::consts::OS,
        arch: std::env::consts::ARCH,
        virt,
        memory_ceiling,
        warning,
    }
}
