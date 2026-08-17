//! GPU discovery.
//!
//! Without this, a machine with a discrete GPU is predicted as a CPU machine,
//! which is wrong by roughly an order of magnitude. That is the largest single
//! correctness gap in the tool.
//!
//! Two rules govern everything here:
//!
//! **Only dedicated VRAM counts.** Windows reports "shared system memory" as
//! adapter memory and an integrated GPU's aperture as if it were VRAM. Neither
//! is extra memory — on an 8 GB laptop an iGPU *subtracts* memory rather than
//! adding it. Every integrated part is therefore recorded with `vram_bytes: 0`.
//!
//! **Bandwidth here is looked up, not measured.** ZeroCloud measures RAM, disk
//! and FMA throughput on the real machine; there is no dependency-free way to
//! measure GPU bandwidth without a driver toolchain. So [`bandwidth_gbs`] is a
//! table, it is small and family-level rather than an exhaustive card list, and
//! callers must carry the fact that it was looked up all the way into the
//! published confidence — see `Hardware::vram_bw_measured`. A lookup table
//! feeding predictions without being labelled as one is exactly what this
//! project exists not to do.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Vendor {
    Nvidia,
    Amd,
    Intel,
    Apple,
    Other,
}

impl Vendor {
    pub fn tag(self) -> &'static str {
        match self {
            Vendor::Nvidia => "nvidia",
            Vendor::Amd => "amd",
            Vendor::Intel => "intel",
            Vendor::Apple => "apple",
            Vendor::Other => "other",
        }
    }
}

#[derive(Debug, Clone)]
pub struct Gpu {
    pub name: String,
    pub vendor: Vendor,
    /// Dedicated VRAM of **one** card, in bytes. Always 0 for integrated parts.
    ///
    /// Never summed across cards at this layer: two 8 GB cards are not a 16 GB
    /// card, and treating them as one is how a model gets predicted to fit when
    /// it cannot.
    pub vram_bytes: u64,
    pub integrated: bool,
    /// Identical adapters collapse into one entry with a count.
    pub count: u32,
    /// Which detector produced this entry. Kept for `zc doctor` and bug
    /// reports, where "what did you read, and from where" is the whole value.
    pub source: &'static str,
    /// Peak memory bandwidth, GB/s. **Looked up** from [`bandwidth_gbs`].
    pub bw_gbs: f64,
    /// Shader core count, when the platform reports one. `None` means unknown,
    /// never zero.
    ///
    /// Load-bearing on macOS, where it is the only signal separating a real
    /// Apple GPU from a VM's paravirtual adapter: both identify as
    /// `Vendor: Apple (0x106b)` with Metal support, and only the real one
    /// reports how many cores it has. See [`Gpu::usable_for_compute`].
    pub cores: Option<u32>,
}

impl Gpu {
    /// Build an entry from a detector's raw name and memory figure, applying
    /// the dedicated-VRAM-only rule in one place so no detector can forget it.
    pub fn new(name: String, vram: u64, source: &'static str) -> Self {
        let vendor = vendor_of(&name);
        let integrated = is_integrated(&name, vram);
        Gpu {
            bw_gbs: bandwidth_gbs(&name),
            vendor,
            // An integrated GPU has no memory of its own. Recording the
            // aperture (or Windows' shared-memory figure) as VRAM would invent
            // capacity that does not exist.
            vram_bytes: if integrated { 0 } else { vram },
            integrated,
            count: 1,
            source,
            name,
            cores: None,
        }
    }

    /// Whether this adapter can be relied on to actually run inference.
    ///
    /// Only meaningful for Apple parts today, and deliberately conservative:
    /// a macOS VM presents an "Apple Paravirtual device" that advertises Metal
    /// but has no shader cores behind it. Trusting the advertisement predicted
    /// 27-45 tok/s on a GitHub macOS runner that measured 3.9 — an 822% error,
    /// the worst single miss in the calibration set (record `cf62d6b5590582da`).
    ///
    /// A core count is reported by real silicon and by nothing else, so it is
    /// the gate. Unknown means unusable: claiming GPU speed we cannot
    /// substantiate is the one direction this codebase never errs in.
    pub fn usable_for_compute(&self) -> bool {
        match self.vendor {
            Vendor::Apple => self.cores.is_some(),
            // Every other vendor is discovered through a driver interface that
            // would not have answered at all if the card were not real.
            _ => true,
        }
    }
}

/// Every adapter we can see, integrated ones included.
pub fn probe() -> Vec<Gpu> {
    group(imp::detect())
}

// ------------------------------------------------------------ bandwidth ----

/// Bandwidth assumed for a discrete GPU we do not recognise, GB/s.
///
/// Deliberately the *floor* of the table rather than its middle: 192 GB/s is
/// the slowest modern discrete part (GTX 1650/1660 class). Understating speed
/// makes a model look slower than it is, which is the direction this codebase
/// always errs in — the failure that destroys trust is the optimistic one.
pub const UNMATCHED_GBS: f64 = 192.0;

/// Peak memory bandwidth by card family, GB/s.
///
/// Family-level on purpose. llmfit carries ~80 exact card names; that table is
/// mostly maintenance, and the machines this product targets are precisely the
/// ones missing from it. A family figure that is 10% off feeds a coefficient
/// that calibration corrects; a missing entry falls to [`UNMATCHED_GBS`].
pub fn bandwidth_gbs(name: &str) -> f64 {
    const TABLE: &[(&str, f64)] = &[
        // NVIDIA consumer, newest first so "5090" cannot be shadowed.
        ("5090", 1792.0),
        ("5080", 960.0),
        ("5070", 672.0),
        ("5060", 448.0),
        ("4090", 1008.0),
        ("4080", 717.0),
        ("4070", 504.0),
        ("4060", 272.0),
        ("3090", 936.0),
        ("3080", 760.0),
        ("3070", 448.0),
        ("3060", 360.0),
        ("2080", 448.0),
        ("2070", 448.0),
        ("2060", 336.0),
        ("1660", 192.0),
        ("1650", 192.0),
        // NVIDIA datacenter.
        ("h100", 3350.0),
        ("a100", 1555.0),
        ("l40", 864.0),
        ("a6000", 768.0),
        // AMD RDNA2/3.
        ("7900", 960.0),
        ("7800", 624.0),
        ("7700", 432.0),
        ("7600", 288.0),
        ("6900", 512.0),
        ("6800", 512.0),
        ("6700", 384.0),
        ("6600", 224.0),
        // Intel Arc.
        ("a770", 560.0),
        ("a750", 512.0),
        ("b580", 456.0),
    ];
    let n = name.to_lowercase();
    for (key, gbs) in TABLE {
        if n.contains(key) {
            return *gbs;
        }
    }
    UNMATCHED_GBS
}

// ------------------------------------------------------- classification ----

fn vendor_of(name: &str) -> Vendor {
    let n = name.to_lowercase();
    if n.contains("nvidia") || n.contains("geforce") || n.contains("quadro") || n.contains("tesla")
    {
        Vendor::Nvidia
    } else if n.contains("amd") || n.contains("radeon") || n.contains("ati ") {
        Vendor::Amd
    } else if n.contains("intel") {
        Vendor::Intel
    } else if n.contains("apple") {
        Vendor::Apple
    } else {
        Vendor::Other
    }
}

fn tokens(lower: &str) -> impl Iterator<Item = &str> {
    lower
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|t| !t.is_empty())
}

/// AMD's mobile iGPUs are named `760M`, `780M`, `890M` — three digits then M.
fn has_mobile_igpu_token(lower: &str) -> bool {
    tokens(lower).any(|t| {
        t.len() == 4 && t.ends_with('m') && t[..3].bytes().all(|b| b.is_ascii_digit())
    })
}

/// Intel names its Meteor Lake *integrated* graphics "Intel Arc Graphics", so
/// "arc" alone does not mean discrete. The discrete parts carry a model token:
/// `A770`, `B580`.
fn has_arc_model_token(lower: &str) -> bool {
    tokens(lower).any(|t| {
        t.len() == 4
            && matches!(t.as_bytes()[0], b'a' | b'b')
            && t[1..].bytes().all(|b| b.is_ascii_digit())
    })
}

/// Integrated or discrete? Subtler than it looks, and every wrong answer is a
/// prediction off by an order of magnitude in one direction or the other.
pub fn is_integrated(name: &str, vram: u64) -> bool {
    let n = name.to_lowercase();

    match vendor_of(name) {
        Vendor::Apple => true,
        // NVIDIA ships one family of unified-memory parts: Jetson/Tegra and the
        // Grace-Blackwell desktops. Everything else is discrete.
        Vendor::Nvidia => {
            n.contains("nvgpu")
                || n.contains("tegra")
                || n.contains("orin")
                || n.contains("gb10")
                || n.contains("gb20")
        }
        Vendor::Intel => !has_arc_model_token(&n),
        Vendor::Amd => {
            // Strix Halo: one pool, marketed as an APU.
            if n.contains("ryzen ai max") {
                return true;
            }
            if has_mobile_igpu_token(&n) {
                return true;
            }
            if tokens(&n).any(|t| t == "rx") {
                return false;
            }
            // Datacenter MI50/MI60 report the generic family name "AMD Radeon
            // Graphics" with a large pool. A generic name plus real VRAM is a
            // discrete card; a generic name with a small aperture is an APU.
            if n.contains("graphics") {
                return vram < 8 * (1 << 30);
            }
            false
        }
        Vendor::Other => vram == 0,
    }
}

/// Collapse identical adapters into one entry with a count. VRAM stays
/// per-card.
fn group(gpus: Vec<Gpu>) -> Vec<Gpu> {
    let mut out: Vec<Gpu> = Vec::new();
    for g in gpus {
        match out.iter_mut().find(|o| o.name == g.name && o.vram_bytes == g.vram_bytes) {
            Some(o) => o.count += 1,
            None => out.push(g),
        }
    }
    out
}

// -------------------------------------------------------------- process ----

/// Seconds before we give up on a detection command.
///
/// nvidia-smi routinely hangs for tens of seconds against a wedged driver, and
/// a hardware prober that never returns is worse than one that reports no GPU.
#[cfg(any(target_os = "linux", target_os = "windows", target_os = "macos"))]
const CMD_TIMEOUT_S: u64 = 4;

/// Run a command, or `None` if it is missing, fails, or takes too long.
///
/// ponytail: on timeout the worker thread stays blocked on the child until the
/// child exits on its own — `std::process` has no portable kill-on-timeout and
/// `zc` is about to exit anyway. Revisit if `zc serve` ever polls this in a
/// long-lived process.
#[cfg(any(target_os = "linux", target_os = "windows", target_os = "macos"))]
fn run(prog: &str, args: &[&str]) -> Option<String> {
    use std::process::{Command, Stdio};
    let (tx, rx) = std::sync::mpsc::channel();
    let prog = prog.to_string();
    let args: Vec<String> = args.iter().map(|s| s.to_string()).collect();
    std::thread::spawn(move || {
        let out = Command::new(&prog)
            .args(&args)
            .stdin(Stdio::null())
            .stderr(Stdio::null())
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| String::from_utf8_lossy(&o.stdout).into_owned());
        let _ = tx.send(out);
    });
    rx.recv_timeout(std::time::Duration::from_secs(CMD_TIMEOUT_S))
        .ok()
        .flatten()
}

// --------------------------------------------------------------- NVIDIA ----

/// Parse `nvidia-smi --query-gpu=memory.total,name --format=csv,noheader,nounits`.
///
/// One line per *card*, so two identical cards produce two lines and are
/// grouped later. Memory is MiB.
#[cfg_attr(target_os = "macos", allow(dead_code))]
fn parse_nvidia_smi(out: &str) -> Vec<Gpu> {
    let mut gpus = Vec::new();
    for line in out.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let mut f = line.split(',');
        let Some(mib) = f.next().and_then(|m| m.trim().parse::<u64>().ok()) else {
            continue;
        };
        let name = f.next().unwrap_or("").trim();
        if name.is_empty() {
            continue;
        }
        gpus.push(Gpu::new(
            format!("NVIDIA {name}"),
            mib * 1024 * 1024,
            "nvidia-smi",
        ));
    }
    gpus
}

#[cfg(any(target_os = "linux", target_os = "windows"))]
fn nvidia() -> Vec<Gpu> {
    let out = run(
        "nvidia-smi",
        &[
            "--query-gpu=memory.total,name",
            "--format=csv,noheader,nounits",
        ],
    );
    out.as_deref().map(parse_nvidia_smi).unwrap_or_default()
}

// ---------------------------------------------------------------- macOS ----

#[cfg(target_os = "macos")]
mod imp {
    use super::{run, Gpu};

    /// This used to return nothing, on the reasoning that `system_profiler`
    /// "costs 1-2 seconds to learn nothing the prediction uses". A calibration
    /// record disproved that sentence: with no GPU probe, `machine.rs` selected
    /// the Metal backend from `mem.unified` alone — and `unified` is
    /// `cfg!(target_arch = "aarch64")`, a compile-time constant. Every
    /// virtualized Apple Silicon machine was therefore predicted as a GPU
    /// machine. On a GitHub macOS runner that meant 27-45 tok/s predicted
    /// against 3.9 measured.
    ///
    /// 1-2 seconds against a ~20 s benchmark is noise; an 822% error is not.
    ///
    /// Intel Macs with a discrete AMD card are still not detected: on those
    /// `unified` is false, so they already take the CPU path and this changes
    /// nothing for them.
    pub fn detect() -> Vec<Gpu> {
        match run("system_profiler", &["SPDisplaysDataType"]) {
            Some(out) => parse(&out),
            None => Vec::new(),
        }
    }

    /// Pull one entry per `Chipset Model:` out of `system_profiler` text.
    ///
    /// Text rather than `-json`: `zc-probe` is dependency-free by policy, the
    /// rest of this module already parses `nvidia-smi` and `lspci` output the
    /// same way, and the two fields needed are on their own lines.
    pub(super) fn parse(out: &str) -> Vec<Gpu> {
        let mut gpus: Vec<Gpu> = Vec::new();
        for line in out.lines() {
            let line = line.trim();
            if let Some(name) = line.strip_prefix("Chipset Model:") {
                // Apple parts share the system pool, so 0 VRAM is correct here
                // and `Gpu::new` enforces it for anything integrated anyway.
                gpus.push(Gpu::new(name.trim().to_string(), 0, "system_profiler"));
            } else if let Some(n) = line.strip_prefix("Total Number of Cores:") {
                // Belongs to the most recent chipset block. Absent entirely on
                // a paravirtual adapter, which is the whole point.
                if let (Some(g), Ok(n)) = (gpus.last_mut(), n.trim().parse::<u32>()) {
                    g.cores = Some(n);
                }
            }
        }
        gpus
    }
}

// ---------------------------------------------------------------- Linux ----

#[cfg(target_os = "linux")]
mod imp {
    use super::{run, Gpu};
    use std::collections::HashMap;
    use std::path::Path;

    pub fn detect() -> Vec<Gpu> {
        let mut gpus = super::nvidia();
        gpus.extend(sysfs());
        gpus
    }

    fn read_u64(p: &Path) -> Option<u64> {
        let s = std::fs::read_to_string(p).ok()?;
        let s = s.trim();
        // sysfs writes some IDs in hex with an 0x prefix and sizes in decimal.
        match s.strip_prefix("0x") {
            Some(hex) => u64::from_str_radix(hex, 16).ok(),
            None => s.parse().ok(),
        }
    }

    /// `lspci -nnD` → {pci address: human name}. Best effort: sysfs knows the
    /// VRAM but not the marketing name, and lspci is the only portable place to
    /// get one without shipping a PCI ID table.
    fn lspci_names() -> HashMap<String, String> {
        let mut map = HashMap::new();
        let Some(out) = run("lspci", &["-nnD"]) else {
            return map;
        };
        for line in out.lines() {
            let Some((addr, rest)) = line.split_once(' ') else {
                continue;
            };
            // "VGA compatible controller [0300]: NVIDIA Corporation GA106 [10de:2504] (rev a1)"
            let Some((_class, tail)) = rest.split_once(": ") else {
                continue;
            };
            let name = tail.rsplit_once(" [").map_or(tail, |(n, _)| n);
            map.insert(addr.to_string(), name.trim().to_string());
        }
        map
    }

    fn sysfs() -> Vec<Gpu> {
        let names = lspci_names();
        let mut gpus = Vec::new();
        let Ok(entries) = std::fs::read_dir("/sys/class/drm") else {
            return gpus;
        };
        for e in entries.flatten() {
            let file = e.file_name();
            let Some(card) = file.to_str() else { continue };
            // "card0" is a device; "card0-DP-1" is a connector on it.
            if !card.starts_with("card") || card.contains('-') {
                continue;
            }
            let dev = e.path().join("device");
            let Some(vendor) = read_u64(&dev.join("vendor")) else {
                continue;
            };

            let vram = match vendor {
                // amdgpu. `mem_info_vram_total` is amdgpu-only — reading it on
                // an Intel card silently yields nothing, which is why the
                // vendor check comes first.
                0x1002 => read_u64(&dev.join("mem_info_vram_total")),
                // Intel discrete. The two drivers disagree on the path: xe
                // exposes it per tile, i915 on the card node.
                0x8086 => read_u64(&dev.join("tile0/physical_vram_size_bytes"))
                    .or_else(|| read_u64(&e.path().join("lmem_total_bytes"))),
                // NVIDIA VRAM is not in sysfs; nvidia-smi already covered it.
                _ => continue,
            };

            let addr = std::fs::read_link(&dev)
                .ok()
                .and_then(|p| p.file_name().map(|s| s.to_string_lossy().into_owned()))
                .unwrap_or_default();
            let name = names.get(&addr).cloned().unwrap_or_else(|| {
                match vendor {
                    0x1002 => "AMD Radeon Graphics".to_string(),
                    _ => "Intel Graphics".to_string(),
                }
            });
            gpus.push(Gpu::new(name, vram.unwrap_or(0), "sysfs"));
        }
        gpus
    }
}

// -------------------------------------------------------------- Windows ----

#[cfg(target_os = "windows")]
mod imp {
    use super::{parse_windows, run, Gpu};

    /// One PowerShell invocation, because starting PowerShell costs about a
    /// second and doing it twice would double that for every `zc check`.
    ///
    /// `Win32_VideoController.AdapterRAM` is a 32-bit field and saturates at
    /// ~4 GB, so an 8 GB card reports 4 GB and a 24 GB card reports 4 GB. The
    /// real figure lives in the display-class registry key as
    /// `HardwareInformation.qwMemorySize`. Undocumented, and the only way to
    /// get a true VRAM figure out of Windows without a vendor SDK.
    const PS: &str = "\
$ErrorActionPreference='SilentlyContinue';\
$reg=Get-ItemProperty 'HKLM:\\SYSTEM\\CurrentControlSet\\Control\\Class\\{4d36e968-e325-11ce-bfc1-08002be10318}\\*';\
foreach($v in Get-CimInstance Win32_VideoController){\
$q=0;\
foreach($r in $reg){if($r.DriverDesc -eq $v.Name -and $r.'HardwareInformation.qwMemorySize'){$q=$r.'HardwareInformation.qwMemorySize'}};\
\"$($v.Name)|$($v.AdapterRAM)|$q\"}";

    pub fn detect() -> Vec<Gpu> {
        // nvidia-smi ships with the driver on Windows too, and it is the only
        // source that is right about VRAM without the registry detour.
        let mut gpus = super::nvidia();
        if !gpus.is_empty() {
            return gpus;
        }
        if let Some(out) = run(
            "powershell",
            &["-NoProfile", "-NonInteractive", "-Command", PS],
        ) {
            gpus = parse_windows(&out);
        }
        gpus
    }
}

/// Parse the `name|adapterram|qwmemorysize` lines emitted by the PowerShell
/// probe. Split out so it is testable on every platform, not only Windows.
#[cfg_attr(not(target_os = "windows"), allow(dead_code))]
fn parse_windows(out: &str) -> Vec<Gpu> {
    let mut gpus = Vec::new();
    for line in out.lines() {
        let mut f = line.trim().split('|');
        let name = f.next().unwrap_or("").trim();
        if name.is_empty() {
            continue;
        }
        let adapter_ram: u64 = f.next().and_then(|v| v.trim().parse().ok()).unwrap_or(0);
        let qw: u64 = f.next().and_then(|v| v.trim().parse().ok()).unwrap_or(0);
        // Prefer the registry figure; AdapterRAM is only a floor.
        gpus.push(Gpu::new(name.to_string(), qw.max(adapter_ram), "wmi"));
    }
    gpus
}

// Platforms with no detector yet: report nothing rather than guessing.
#[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
mod imp {
    use super::Gpu;
    pub fn detect() -> Vec<Gpu> {
        Vec::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verbatim `system_profiler SPDisplaysDataType` from a real M5 laptop.
    #[cfg(target_os = "macos")]
    const REAL_M5: &str = "\
Graphics/Displays:

    Apple M5:

      Chipset Model: Apple M5
      Type: GPU
      Bus: Built-In
      Total Number of Cores: 10
      Vendor: Apple (0x106b)
      Metal Support: Metal 4
";

    /// A macOS VM's adapter. Note it claims the same Apple vendor id and
    /// advertises Metal — only the core count is missing.
    #[cfg(target_os = "macos")]
    const PARAVIRTUAL: &str = "\
Graphics/Displays:

    Apple Paravirtual device:

      Chipset Model: Apple Paravirtual device
      Type: GPU
      Bus: Built-In
      Vendor: Apple (0x106b)
      Metal Support: Metal 3
";

    /// Real silicon reports its shader cores, so it may drive the Metal path.
    #[cfg(target_os = "macos")]
    #[test]
    fn a_real_apple_gpu_reports_cores_and_is_usable() {
        let g = super::imp::parse(REAL_M5);
        assert_eq!(g.len(), 1);
        assert_eq!(g[0].name, "Apple M5");
        assert_eq!(g[0].vendor, Vendor::Apple);
        assert_eq!(g[0].cores, Some(10));
        assert!(g[0].usable_for_compute());
    }

    /// Locks in calibration record `cf62d6b5590582da`: a GitHub macOS runner
    /// predicted 27-45 tok/s and measured 3.9, an 822% error, because nothing
    /// probed the GPU and `mem.unified` is a compile-time constant on aarch64.
    ///
    /// The paravirtual adapter is still *listed* — `zc doctor` should show what
    /// was seen — but it must not be usable for compute.
    #[cfg(target_os = "macos")]
    #[test]
    fn a_paravirtual_adapter_advertises_metal_but_is_not_usable() {
        let g = super::imp::parse(PARAVIRTUAL);
        assert_eq!(g.len(), 1);
        assert_eq!(g[0].vendor, Vendor::Apple, "it does claim to be Apple");
        assert_eq!(g[0].cores, None, "and reports no shader cores");
        assert!(
            !g[0].usable_for_compute(),
            "an adapter with no cores must not select the Metal backend"
        );
    }

    /// A machine where `system_profiler` fails or returns nothing must yield no
    /// adapters rather than a default one.
    #[cfg(target_os = "macos")]
    #[test]
    fn empty_output_yields_no_adapters() {
        assert!(super::imp::parse("").is_empty());
        assert!(super::imp::parse("Graphics/Displays:\n").is_empty());
    }

    /// The classification that decides whether a machine is predicted at GPU
    /// speed or CPU speed. Each of these names is one that a naive rule gets
    /// wrong.
    #[test]
    fn integrated_classification_handles_the_awkward_names() {
        let gib8 = 8 * (1 << 30);
        // Plain discrete cards.
        assert!(!is_integrated("NVIDIA GeForce RTX 3060", 12 << 30));
        assert!(!is_integrated("AMD Radeon RX 7900 XTX", 24 << 30));
        // AMD mobile iGPU: "Radeon 780M Graphics" has no RX and a NNNm token.
        assert!(is_integrated("AMD Radeon 780M Graphics", 2 << 30));
        assert!(is_integrated("AMD Radeon 890M Graphics", 2 << 30));
        // Generic family name: an APU below 8 GB, a datacenter MI-series above.
        assert!(is_integrated("AMD Radeon Graphics", 512 << 20));
        assert!(!is_integrated("AMD Radeon Graphics", gib8));
        // Intel calls its Meteor Lake *integrated* graphics "Arc" as well;
        // only the discrete parts carry a model token.
        assert!(is_integrated("Intel(R) Arc(TM) Graphics", 0));
        assert!(is_integrated("Intel(R) UHD Graphics 620", 0));
        assert!(!is_integrated("Intel(R) Arc(TM) A770 Graphics", gib8));
        // NVIDIA unified-memory parts.
        assert!(is_integrated("NVIDIA Tegra Orin (nvgpu)", 0));
        assert!(!is_integrated("NVIDIA RTX A6000", 48 << 30));
    }

    /// An integrated GPU must never contribute VRAM. Windows reports its
    /// aperture and its "shared system memory" as adapter memory; counting
    /// either would invent capacity that does not exist, on exactly the 8 GB
    /// machines this tool exists for.
    #[test]
    fn an_integrated_gpu_contributes_no_vram() {
        let g = Gpu::new("Intel(R) UHD Graphics 620".into(), 2 << 30, "test");
        assert!(g.integrated);
        assert_eq!(g.vram_bytes, 0);
    }

    #[test]
    fn nvidia_smi_rows_parse_and_identical_cards_group() {
        let out = "12288, NVIDIA GeForce RTX 3060\n12288, NVIDIA GeForce RTX 3060\n";
        let gpus = group(parse_nvidia_smi(out));
        assert_eq!(gpus.len(), 1);
        assert_eq!(gpus[0].count, 2);
        // VRAM stays per-card: two 12 GB cards are not one 24 GB card.
        assert_eq!(gpus[0].vram_bytes, 12288 * 1024 * 1024);
        assert_eq!(gpus[0].vendor, Vendor::Nvidia);
        assert!((gpus[0].bw_gbs - 360.0).abs() < 1e-9);
    }

    /// AdapterRAM is a 32-bit field that saturates near 4 GB. Trusting it on a
    /// 24 GB card would understate VRAM by 6x and push the prediction back onto
    /// the host RAM tier.
    #[test]
    fn windows_registry_size_beats_the_saturated_adapter_ram() {
        let out = "NVIDIA GeForce RTX 4090|4293918720|25757220864\n\
                   Intel(R) UHD Graphics 770|1073741824|0\n";
        let gpus = parse_windows(out);
        assert_eq!(gpus[0].vram_bytes, 25_757_220_864);
        assert!(!gpus[0].integrated);
        assert!(gpus[1].integrated && gpus[1].vram_bytes == 0);
    }

    /// An unrecognised card falls to the floor of the table, not its middle:
    /// too slow is recoverable, too fast is a broken promise.
    #[test]
    fn unknown_cards_get_the_floor_not_a_guess() {
        assert_eq!(bandwidth_gbs("NVIDIA RTX 4000 Ada Generation"), UNMATCHED_GBS);
        assert!(bandwidth_gbs("NVIDIA GeForce RTX 4090") > UNMATCHED_GBS);
    }
}
