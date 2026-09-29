//! Power source and thermal state.
//!
//! Neither changes what a machine *is*, and both change what it measures. A
//! laptop on battery or in low-power mode caps its clocks, and one that is
//! thermally throttled mid-benchmark reports a bandwidth it will beat once it
//! cools. The benchmark is still the truth about *this run* -- so nothing here
//! adjusts a number. It says which conditions the numbers were taken under, so
//! a reader can tell a slow machine from a machine measured slowly.
//!
//! Every field is `Option`: `None` means the platform did not say, which is
//! different from "no" and is never rendered as one.

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Power {
    /// `Some(true)` when running from a battery. `None` on a desktop with no
    /// power-supply information, or when the platform will not say.
    pub on_battery: Option<bool>,
    /// The OS power-saving mode (macOS Low Power Mode, Windows Battery Saver,
    /// Linux `platform_profile = low-power`).
    pub low_power: Option<bool>,
    /// Whether the CPU was held below its rated speed at any point sampled
    /// around the benchmark. See [`Throttle`].
    pub throttled: Option<bool>,
}

impl Power {
    /// Prose for the report, one entry per condition that distorts the numbers.
    /// Empty when nothing was detected, including when nothing was detectable.
    pub fn warnings(&self) -> Vec<String> {
        let mut w = Vec::new();
        if self.throttled == Some(true) {
            w.push(
                "The CPU was throttled during the benchmark (heat or a power limit). \
                 The measured speeds understate this machine when it is cool; let it \
                 cool and re-run for representative numbers."
                    .into(),
            );
        }
        if self.on_battery == Some(true) {
            w.push(
                "Measured on battery. Laptops usually cap CPU and memory clocks on \
                 battery, so these numbers can understate the machine plugged in."
                    .into(),
            );
        }
        if self.low_power == Some(true) {
            w.push(
                "Low-power mode is on, which caps clocks. Turn it off and re-run for \
                 numbers that describe the hardware rather than the power setting."
                    .into(),
            );
        }
        w
    }
}

/// A throttle counter or level, sampled before the benchmark and again after.
///
/// Sampling twice is the point on Linux, where the kernel exposes a cumulative
/// count since boot: only the *difference* says whether this run was affected.
/// Elsewhere the reading is a level, and the worse of the two is kept.
#[derive(Debug, Clone, Copy)]
pub struct Throttle(Option<u64>);

impl Throttle {
    pub fn sample() -> Self {
        Throttle(imp::throttle_reading())
    }

    /// Whether the run between `self` and `after` was throttled.
    pub fn during(self, after: Throttle) -> Option<bool> {
        let (a, b) = (self.0?, after.0?);
        Some(if imp::CUMULATIVE { b > a } else { a.max(b) >= imp::THROTTLED_AT })
    }
}

/// Everything except the throttle verdict, which needs two samples.
pub fn probe() -> Power {
    let (on_battery, low_power) = imp::source();
    Power {
        on_battery,
        low_power,
        throttled: None,
    }
}

// ---------------------------------------------------------------- macOS ----

/// `pmset -g batt`'s first line names the source in quotes.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
fn parse_pmset_batt(out: &str) -> Option<bool> {
    let line = out.lines().find(|l| l.contains("drawing from"))?;
    if line.contains("'Battery Power'") {
        Some(true)
    } else if line.contains("'AC Power'") || line.contains("'UPS Power'") {
        Some(false)
    } else {
        None
    }
}

/// `pmset -g` prints `lowpowermode 1` on most Macs, and `powermode 1` on those
/// that also offer a high-power mode (0 automatic, 1 low, 2 high).
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
fn parse_pmset_low_power(out: &str) -> Option<bool> {
    out.lines().find_map(|l| {
        let mut f = l.split_whitespace();
        match (f.next()?, f.next()?) {
            ("lowpowermode" | "powermode", v) => Some(v == "1"),
            _ => None,
        }
    })
}

#[cfg(target_os = "macos")]
mod imp {
    /// The thermal pressure level is a level, not a counter.
    pub const CUMULATIVE: bool = false;
    /// `kOSThermalPressureLevelHeavy`. Moderate (1) is "slightly elevated"
    /// and does not by itself mean clocks came down; Heavy is where
    /// `ProcessInfo.thermalState` turns `serious`, documented as reducing
    /// performance.
    pub const THROTTLED_AT: u64 = 2;

    unsafe extern "C" {
        fn notify_register_check(name: *const libc::c_char, token: *mut libc::c_int) -> u32;
        fn notify_get_state(token: libc::c_int, state: *mut u64) -> u32;
        fn notify_cancel(token: libc::c_int) -> u32;
    }

    /// `kOSThermalNotificationPressureLevelName` through libSystem's notify
    /// API: the same signal `ProcessInfo.thermalState` reads, on Intel and
    /// Apple Silicon alike, without spawning anything.
    pub fn throttle_reading() -> Option<u64> {
        let mut token = 0;
        let mut level = 0u64;
        unsafe {
            if notify_register_check(c"com.apple.system.thermalpressurelevel".as_ptr(), &mut token) != 0 {
                return None;
            }
            let ok = notify_get_state(token, &mut level) == 0;
            notify_cancel(token);
            ok.then_some(level)
        }
    }

    pub fn source() -> (Option<bool>, Option<bool>) {
        let batt = crate::gpu::run("pmset", &["-g", "batt"]);
        let all = crate::gpu::run("pmset", &["-g"]);
        (
            batt.as_deref().and_then(super::parse_pmset_batt),
            all.as_deref().and_then(super::parse_pmset_low_power),
        )
    }
}

// ---------------------------------------------------------------- Linux ----

/// `/sys/class/power_supply/*` as (type, online, status) triples.
///
/// A discharging battery means battery; any mains supply online, or a battery
/// that is not discharging, means plugged in. No supplies at all is a desktop
/// or a VM, and says nothing.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
fn linux_on_battery(supplies: &[(String, Option<bool>, String)]) -> Option<bool> {
    let discharging = supplies
        .iter()
        .any(|(t, _, s)| t == "Battery" && s == "Discharging");
    if discharging {
        return Some(true);
    }
    let mains = supplies
        .iter()
        .any(|(t, online, _)| t != "Battery" && *online == Some(true));
    let battery = supplies.iter().any(|(t, _, _)| t == "Battery");
    (mains || battery).then_some(false)
}

#[cfg(target_os = "linux")]
mod imp {
    use std::fs;

    /// `thermal_throttle/*_throttle_count` counts events since boot.
    pub const CUMULATIVE: bool = true;
    pub const THROTTLED_AT: u64 = 1;

    fn read(p: &std::path::Path) -> Option<String> {
        fs::read_to_string(p).ok().map(|s| s.trim().to_string())
    }

    /// Sum of core and package throttle events over every CPU. Intel exposes
    /// these; AMD and ARM usually do not, which yields `None` rather than 0 --
    /// "no counter" must not read as "never throttled".
    pub fn throttle_reading() -> Option<u64> {
        let mut total = None;
        for e in fs::read_dir("/sys/devices/system/cpu").ok()?.flatten() {
            let dir = e.path().join("thermal_throttle");
            for f in ["core_throttle_count", "package_throttle_count"] {
                if let Some(n) = read(&dir.join(f)).and_then(|s| s.parse::<u64>().ok()) {
                    total = Some(total.unwrap_or(0) + n);
                }
            }
        }
        total
    }

    pub fn source() -> (Option<bool>, Option<bool>) {
        let mut supplies = Vec::new();
        if let Ok(entries) = fs::read_dir("/sys/class/power_supply") {
            for e in entries.flatten() {
                let p = e.path();
                supplies.push((
                    read(&p.join("type")).unwrap_or_default(),
                    read(&p.join("online")).map(|s| s == "1"),
                    read(&p.join("status")).unwrap_or_default(),
                ));
            }
        }
        let low_power = read("/sys/firmware/acpi/platform_profile".as_ref()).map(|s| s == "low-power");
        (super::linux_on_battery(&supplies), low_power)
    }
}

// -------------------------------------------------------------- Windows ----

/// Any processor whose clock ceiling sits below its rated maximum. Windows
/// lowers `MhzLimit` for heat and for power policy alike; both cap the run.
#[cfg_attr(not(target_os = "windows"), allow(dead_code))]
fn windows_limited(procs: &[(u32, u32)]) -> Option<bool> {
    let real: Vec<_> = procs.iter().filter(|(max, _)| *max > 0).collect();
    (!real.is_empty()).then(|| real.iter().any(|(max, limit)| limit < max))
}

#[cfg(target_os = "windows")]
mod imp {
    use windows_sys::Win32::System::Power::{
        CallNtPowerInformation, GetSystemPowerStatus, ProcessorInformation,
        PROCESSOR_POWER_INFORMATION, SYSTEM_POWER_STATUS,
    };

    pub const CUMULATIVE: bool = false;
    pub const THROTTLED_AT: u64 = 1;

    /// 1 when any processor's `MhzLimit` is below its `MaxMhz`, else 0.
    pub fn throttle_reading() -> Option<u64> {
        let n = std::thread::available_parallelism().map_or(64, |n| n.get());
        let mut buf: Vec<PROCESSOR_POWER_INFORMATION> = vec![unsafe { std::mem::zeroed() }; n];
        let bytes = (n * std::mem::size_of::<PROCESSOR_POWER_INFORMATION>()) as u32;
        let status = unsafe {
            CallNtPowerInformation(
                ProcessorInformation,
                std::ptr::null(),
                0,
                buf.as_mut_ptr().cast(),
                bytes,
            )
        };
        if status != 0 {
            return None;
        }
        let procs: Vec<(u32, u32)> = buf.iter().map(|p| (p.MaxMhz, p.MhzLimit)).collect();
        super::windows_limited(&procs).map(u64::from)
    }

    pub fn source() -> (Option<bool>, Option<bool>) {
        let mut s: SYSTEM_POWER_STATUS = unsafe { std::mem::zeroed() };
        if unsafe { GetSystemPowerStatus(&mut s) } == 0 {
            return (None, None);
        }
        // 0 offline, 1 online, 255 unknown. BatteryFlag 128 is "no battery".
        let on_battery = match s.ACLineStatus {
            0 => Some(true),
            1 => Some(false),
            _ => None,
        };
        // SystemStatusFlag 1 is Battery Saver on.
        (on_battery, Some(s.SystemStatusFlag == 1))
    }
}

#[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
mod imp {
    pub const CUMULATIVE: bool = false;
    pub const THROTTLED_AT: u64 = 1;
    pub fn throttle_reading() -> Option<u64> {
        None
    }
    pub fn source() -> (Option<bool>, Option<bool>) {
        (None, None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verbatim from a MacBook, plugged in and then unplugged.
    #[test]
    fn pmset_names_the_source() {
        let ac = "Now drawing from 'AC Power'\n -InternalBattery-0 (id=22544483)\t80%; AC attached; not charging present: true\n";
        let batt = "Now drawing from 'Battery Power'\n -InternalBattery-0 (id=22544483)\t79%; discharging; 5:12 remaining present: true\n";
        assert_eq!(parse_pmset_batt(ac), Some(false));
        assert_eq!(parse_pmset_batt(batt), Some(true));
        // A desktop Mac has no battery lines but still names AC.
        assert_eq!(parse_pmset_batt("Now drawing from 'AC Power'\n"), Some(false));
        assert_eq!(parse_pmset_batt(""), None);
    }

    #[test]
    fn pmset_low_power_reads_both_spellings() {
        assert_eq!(parse_pmset_low_power(" lowpowermode         0\n"), Some(false));
        assert_eq!(parse_pmset_low_power(" lowpowermode         1\n"), Some(true));
        assert_eq!(parse_pmset_low_power(" powermode            1\n"), Some(true));
        // High-power mode is not low-power mode.
        assert_eq!(parse_pmset_low_power(" powermode            2\n"), Some(false));
        assert_eq!(parse_pmset_low_power(" sleep 1\n"), None);
    }

    #[test]
    fn linux_supplies_decide_the_source() {
        let s = |t: &str, on: Option<bool>, st: &str| (t.to_string(), on, st.to_string());
        assert_eq!(linux_on_battery(&[s("Mains", Some(false), ""), s("Battery", None, "Discharging")]), Some(true));
        assert_eq!(linux_on_battery(&[s("Mains", Some(true), ""), s("Battery", None, "Charging")]), Some(false));
        // A full battery on AC reports "Full" or "Not charging", not "Charging".
        assert_eq!(linux_on_battery(&[s("Battery", None, "Not charging")]), Some(false));
        // No supplies: a desktop or a VM. Unknown, not "on AC".
        assert_eq!(linux_on_battery(&[]), None);
    }

    #[test]
    fn windows_limit_below_max_is_throttled() {
        assert_eq!(windows_limited(&[(3000, 3000), (3000, 3000)]), Some(false));
        assert_eq!(windows_limited(&[(3000, 3000), (3000, 2100)]), Some(true));
        // A zeroed buffer (the call filled fewer entries) is not evidence.
        assert_eq!(windows_limited(&[(0, 0)]), None);
    }

    /// Linux counts since boot, so only the difference means anything: a
    /// laptop that throttled yesterday must not taint today's run.
    #[test]
    fn a_counter_only_counts_what_happened_during_the_run() {
        if imp::CUMULATIVE {
            assert_eq!(Throttle(Some(40)).during(Throttle(Some(40))), Some(false));
            assert_eq!(Throttle(Some(40)).during(Throttle(Some(41))), Some(true));
        } else {
            assert_eq!(Throttle(Some(0)).during(Throttle(Some(imp::THROTTLED_AT))), Some(true));
            assert_eq!(Throttle(Some(0)).during(Throttle(Some(0))), Some(false));
        }
        assert_eq!(Throttle(None).during(Throttle(Some(5))), None);
    }

    #[test]
    fn nothing_detected_means_no_warnings() {
        assert!(Power::default().warnings().is_empty());
        let p = Power { on_battery: Some(true), low_power: Some(true), throttled: Some(true) };
        assert_eq!(p.warnings().len(), 3);
    }
}
