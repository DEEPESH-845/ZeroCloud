//! Storage detection.
//!
//! We do not care about "the disk". We care about the specific volume the
//! model files live on, because that is what inference will stream from.
//! Those differ constantly: models on an external SSD, a second NVMe, a
//! network share, or — worst and most common — a cloud-sync folder.

use std::path::{Path, PathBuf};

#[derive(Debug, Clone, PartialEq)]
pub enum Medium {
    Nvme,
    Ssd,
    Rotational,
    Network,
    Unknown,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Hazard {
    /// OneDrive / iCloud / Dropbox / Google Drive. Files may be stubs that
    /// re-download on read; throughput is unpredictable and writes sync.
    CloudSync(String),
    /// SMB/NFS/AFP. Bandwidth is the network, not the disk.
    NetworkMount(String),
    /// Removable. Often USB 2/3 and far slower than internal.
    Removable,
    /// Spinning disk. Random reads collapse to ~1% of an SSD.
    Rotational,
    /// Not enough free space to hold a useful model.
    LowSpace(u64),
}

#[derive(Debug, Clone)]
pub struct Storage {
    /// Directory whose volume we profiled.
    pub model_dir: PathBuf,
    /// Mount point of that volume.
    pub mount: PathBuf,
    pub fstype: String,
    pub medium: Medium,
    pub total: u64,
    pub free: u64,
    pub hazards: Vec<Hazard>,
    /// An existing large file we can benchmark against, so we write nothing.
    pub bench_file: Option<PathBuf>,
    /// True when `bench_file` is an actual model file rather than an unrelated
    /// large file we borrowed. Bandwidth is the same either way.
    pub bench_is_weight: bool,
    /// Where the model dir came from, for the report.
    pub source: &'static str,
}

/// Directories real runtimes use, in the order we prefer them.
fn candidate_dirs() -> Vec<(PathBuf, &'static str)> {
    let home = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .map(PathBuf::from)
        .unwrap_or_default();
    let mut v = Vec::new();

    if let Ok(p) = std::env::var("ZC_MODEL_DIR") {
        v.push((PathBuf::from(p), "ZC_MODEL_DIR"));
    }
    if let Ok(p) = std::env::var("OLLAMA_MODELS") {
        v.push((PathBuf::from(p), "OLLAMA_MODELS"));
    }
    for (rel, src) in [
        (".ollama/models", "ollama"),
        (".lmstudio/models", "lm studio"),
        (".cache/lm-studio/models", "lm studio"),
        (".cache/huggingface/hub", "huggingface"),
        (".local/share/models", "local"),
    ] {
        v.push((home.join(rel), src));
    }
    v.push((home.clone(), "home (no runtime found)"));
    v
}

/// A candidate benchmark file and its size.
type Candidate = Option<(PathBuf, u64)>;

/// Smallest file we will benchmark against. Random offsets need enough span
/// that the drive cannot serve everything from its own cache.
const MIN_BENCH_BYTES: u64 = 512 << 20;

fn is_weight_file(path: &Path) -> bool {
    // Ollama stores blobs as `sha256-<hex>` with no extension, so name and
    // extension are both signals and neither alone is sufficient.
    let ext = matches!(
        path.extension().and_then(|e| e.to_str()),
        Some("gguf" | "safetensors")
    );
    let blob = path
        .file_name()
        .and_then(|n| n.to_str())
        .is_some_and(|n| n.starts_with("sha256"));
    ext || blob
}

/// Walk a tree (depth-capped) for a file to benchmark against. Prefers a real
/// weight file; falls back to any large file on the same volume, which is
/// equally valid for measuring bandwidth. Depth cap stops a stray symlink or a
/// large home directory from becoming a full-disk scan.
fn find_bench_target(root: &Path, depth: u32) -> (Candidate, Candidate) {
    let (mut weight, mut any): (Candidate, Candidate) = (None, None);
    if depth == 0 {
        return (weight, any);
    }
    let Ok(dir) = std::fs::read_dir(root) else {
        return (weight, any);
    };
    for entry in dir.flatten() {
        let path = entry.path();
        let Ok(md) = entry.metadata() else { continue };
        if md.is_dir() && !md.is_symlink() {
            let (w, a) = find_bench_target(&path, depth - 1);
            if let Some(f) = w
                && weight.as_ref().is_none_or(|b| f.1 > b.1)
            {
                weight = Some(f);
            }
            if let Some(f) = a
                && any.as_ref().is_none_or(|b| f.1 > b.1)
            {
                any = Some(f);
            }
        } else if md.is_file() && md.len() >= MIN_BENCH_BYTES {
            if is_weight_file(&path)
                && weight.as_ref().is_none_or(|b| md.len() > b.1)
            {
                weight = Some((path.clone(), md.len()));
            }
            if any.as_ref().is_none_or(|b| md.len() > b.1) {
                any = Some((path, md.len()));
            }
        }
    }
    (weight, any)
}

fn sync_provider(path: &Path) -> Option<String> {
    let s = path.to_string_lossy();
    for needle in [
        "OneDrive",
        "Dropbox",
        "Google Drive",
        "GoogleDrive",
        "Mobile Documents", // iCloud Drive's on-disk name
        "iCloud",
        "pCloud",
        "Nextcloud",
        "Sync.com",
    ] {
        if s.contains(needle) {
            return Some(needle.replace("Mobile Documents", "iCloud Drive"));
        }
    }
    None
}

// ------------------------------------------------------------ unix impl ----

#[cfg(unix)]
fn volume_info(path: &Path) -> Option<(PathBuf, String, u64, u64, bool)> {
    use std::os::unix::ffi::OsStrExt;
    let cpath = std::ffi::CString::new(path.as_os_str().as_bytes()).ok()?;

    // statvfs gives portable space accounting.
    let mut vfs: libc::statvfs = unsafe { std::mem::zeroed() };
    if unsafe { libc::statvfs(cpath.as_ptr(), &mut vfs) } != 0 {
        return None;
    }
    let frsize = vfs.f_frsize as u64;
    let total = vfs.f_blocks as u64 * frsize;
    // bavail = blocks available to an unprivileged user, which is what we
    // actually get to use. bfree includes root's reserve.
    let free = vfs.f_bavail as u64 * frsize;

    // statfs (BSD/macOS/Linux, non-POSIX) gives the mount point and fs type.
    #[cfg(target_os = "macos")]
    {
        let mut s: libc::statfs = unsafe { std::mem::zeroed() };
        if unsafe { libc::statfs(cpath.as_ptr(), &mut s) } != 0 {
            return None;
        }
        let cstr = |a: &[libc::c_char]| unsafe {
            std::ffi::CStr::from_ptr(a.as_ptr()).to_string_lossy().to_string()
        };
        const MNT_LOCAL: u32 = 0x0000_1000;
        let local = s.f_flags & MNT_LOCAL != 0;
        Some((
            PathBuf::from(cstr(&s.f_mntonname)),
            cstr(&s.f_fstypename),
            total,
            free,
            local,
        ))
    }
    #[cfg(target_os = "linux")]
    {
        // Longest mount-point prefix of our path wins.
        let mounts = std::fs::read_to_string("/proc/mounts").ok()?;
        let mut best: Option<(PathBuf, String)> = None;
        for line in mounts.lines() {
            let mut f = line.split_whitespace();
            let (_dev, mp, fs) = (f.next()?, f.next()?, f.next()?);
            let mp = PathBuf::from(mp.replace("\\040", " "));
            if path.starts_with(&mp)
                && best
                    .as_ref()
                    .is_none_or(|b| mp.as_os_str().len() > b.0.as_os_str().len())
            {
                best = Some((mp, fs.to_string()));
            }
        }
        let (mount, fstype) = best?;
        let local = !matches!(
            fstype.as_str(),
            "nfs" | "nfs4" | "cifs" | "smb3" | "fuse.sshfs" | "afs" | "9p"
        );
        Some((mount, fstype, total, free, local))
    }
}

#[cfg(windows)]
fn volume_info(path: &Path) -> Option<(PathBuf, String, u64, u64, bool)> {
    use std::os::windows::ffi::OsStrExt;
    use windows_sys::Win32::Storage::FileSystem::{
        GetDiskFreeSpaceExW, GetDriveTypeW, GetVolumePathNameW,
    };
    let wide: Vec<u16> = path
        .as_os_str()
        .encode_wide()
        .chain(std::iter::once(0))
        .collect();
    let mut root = [0u16; 260];
    if unsafe { GetVolumePathNameW(wide.as_ptr(), root.as_mut_ptr(), 260) } == 0 {
        return None;
    }
    let (mut avail, mut total, mut free) = (0u64, 0u64, 0u64);
    unsafe { GetDiskFreeSpaceExW(root.as_ptr(), &mut avail, &mut total, &mut free) };
    let dtype = unsafe { GetDriveTypeW(root.as_ptr()) };
    let len = root.iter().position(|&c| c == 0).unwrap_or(0);
    Some((
        PathBuf::from(String::from_utf16_lossy(&root[..len])),
        match dtype {
            2 => "removable".into(),
            3 => "fixed".into(),
            4 => "remote".into(),
            _ => "unknown".into(),
        },
        total,
        avail,
        dtype != 4,
    ))
}

/// Classify the medium. Best effort; the benchmark is the real authority and
/// will correct us in the report.
fn classify(mount: &Path, fstype: &str, local: bool) -> Medium {
    if !local || matches!(fstype, "nfs" | "nfs4" | "cifs" | "smb3" | "remote" | "afpfs") {
        return Medium::Network;
    }
    #[cfg(target_os = "linux")]
    {
        // Resolve mount -> block device -> queue/rotational.
        if let Ok(mounts) = std::fs::read_to_string("/proc/mounts") {
            for line in mounts.lines() {
                let mut f = line.split_whitespace();
                let (Some(dev), Some(mp)) = (f.next(), f.next()) else { continue };
                if Path::new(&mp.replace("\\040", " ")) != mount {
                    continue;
                }
                let base = dev.trim_start_matches("/dev/");
                if base.starts_with("nvme") {
                    return Medium::Nvme;
                }
                let stem: String = base.trim_end_matches(char::is_numeric).to_string();
                if let Ok(rot) =
                    std::fs::read_to_string(format!("/sys/block/{stem}/queue/rotational"))
                {
                    return if rot.trim() == "1" { Medium::Rotational } else { Medium::Ssd };
                }
            }
        }
        Medium::Unknown
    }
    #[cfg(target_os = "macos")]
    {
        let _ = mount;
        // Every Mac that can run this has soldered NVMe internally. External
        // volumes are reclassified by the benchmark.
        Medium::Nvme
    }
    #[cfg(windows)]
    {
        let _ = mount;
        if fstype == "removable" { Medium::Unknown } else { Medium::Ssd }
    }
}

pub fn probe() -> Storage {
    let (model_dir, source) = candidate_dirs()
        .into_iter()
        .find(|(p, _)| p.is_dir())
        .unwrap_or_else(|| (std::env::temp_dir(), "temp dir (fallback)"));

    let (mount, fstype, total, free, local) = volume_info(&model_dir)
        .unwrap_or_else(|| (model_dir.clone(), "unknown".into(), 0, 0, true));

    let medium = classify(&mount, &fstype, local);

    let mut hazards = Vec::new();
    if let Some(p) = sync_provider(&model_dir) {
        hazards.push(Hazard::CloudSync(p));
    }
    if medium == Medium::Network {
        hazards.push(Hazard::NetworkMount(fstype.clone()));
    }
    if medium == Medium::Rotational {
        hazards.push(Hazard::Rotational);
    }
    if free < 8 << 30 {
        hazards.push(Hazard::LowSpace(free));
    }

    // Prefer an existing file so the benchmark writes nothing at all.
    let (weight, any) = find_bench_target(&model_dir, 4);
    let is_real_weight = weight.is_some();
    let bench_file = weight.or(any).map(|(p, _)| p);

    Storage {
        model_dir,
        mount,
        fstype,
        medium,
        total,
        free,
        hazards,
        bench_file,
        bench_is_weight: is_real_weight,
        source,
    }
}
