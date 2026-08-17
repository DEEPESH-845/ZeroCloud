//! Probe + benchmark the host once, and hand the result to any subcommand.
//!
//! Both `check` and `verify` need identical hardware facts. Measuring twice
//! would waste ~20 seconds and, worse, could produce two different answers if
//! thermal state shifted between them — which would make a calibration record
//! disagree with the prediction it is supposed to be calibrating.

use zc_bench::{compute, disk, ram};
use zc_model::{predict, Backend, Hardware};
use zc_probe::{cpu, env, gpu, memory, storage};

pub struct Machine {
    pub mem: memory::Memory,
    pub cpu: cpu::Cpu,
    pub env: env::Env,
    pub storage: storage::Storage,
    /// Every adapter found, integrated ones included. Only the largest
    /// discrete one feeds the prediction.
    pub gpus: Vec<gpu::Gpu>,
    pub ram: ram::RamResult,
    pub compute: compute::ComputeResult,
    pub disk: Option<disk::DiskResult>,
    pub hw: Hardware,
    pub backend: Backend,
    /// Budget when nothing else is running — the basis for predictions.
    pub budget_idle: u64,
    /// Budget right now, for the report.
    pub budget_now: u64,
}

impl Machine {
    /// Stable, non-identifying description used to group calibration records.
    ///
    /// The discrete card is part of the machine's identity: two laptops with
    /// the same CPU and RAM but different GPUs decode at completely different
    /// speeds, and `zc gate` medians *per machine* — so sharing a bucket would
    /// let one card's error cancel another's and hide a real miss.
    ///
    /// Appended only when a card is present, so every existing GPU-less record
    /// keeps the fingerprint it was written with.
    pub fn profile(&self) -> String {
        let base = format!(
            "{}|{}P{}E|{}|{}|{:?}",
            self.cpu.brand,
            self.cpu.p_cores,
            self.cpu.e_cores,
            self.mem.total,
            self.env.os,
            self.storage.medium
        );
        match self.gpus.iter().find(|g| !g.integrated && g.vram_bytes > 0) {
            Some(g) => format!("{base}|{}|{}", g.name, g.vram_bytes),
            None => base,
        }
    }
}

/// Which backend the runtime will actually use.
///
/// Apple Silicon runtimes default to Metal, and on unified memory the GPU
/// reaches close to the all-core peak — well above what the CPU cluster alone
/// sustains, so predicting Metal from a CPU figure understates badly.
///
/// But unified memory alone is not evidence of a GPU. `mem.unified` is
/// `cfg!(target_arch = "aarch64")` — a compile-time constant — so every
/// virtualized Apple Silicon machine claimed it, and with no macOS GPU probe
/// there was nothing to contradict them. A GitHub macOS runner was predicted at
/// 27-45 tok/s and measured 3.9: an 822% error, the worst in the calibration
/// set. Metal now requires an adapter that reports shader cores.
///
/// Erring toward `Cpu` is the safe direction. Predicting CPU speed on a machine
/// that turns out to have a GPU understates it, which calibration corrects; the
/// reverse promises speed that never arrives.
fn select_backend(has_discrete_card: bool, unified: bool, gpus: &[gpu::Gpu]) -> Backend {
    if has_discrete_card {
        return Backend::Discrete;
    }
    let apple_gpu = gpus
        .iter()
        .any(|g| g.vendor == gpu::Vendor::Apple && g.usable_for_compute());
    if unified && apple_gpu {
        Backend::Metal
    } else {
        Backend::Cpu
    }
}

pub fn probe() -> Machine {
    let mem = memory::probe();
    let cpu = cpu::probe();
    let env = env::probe(mem.total);
    let storage = storage::probe();
    let gpus = gpu::probe();
    let p_threads = cpu.p_cores.max(1) as usize;

    let mut counts = vec![1usize, 2, p_threads, cpu.physical as usize];
    counts.sort_unstable();
    counts.dedup();
    let ram = ram::measure(&counts);
    let compute = compute::measure(p_threads);
    let disk = disk::measure(storage.bench_file.as_deref(), &storage.model_dir, 16).ok();

    let reserved = mem.firmware_reserved.unwrap_or(0);
    let budget_idle = predict::potential_budget(mem.total, env.memory_ceiling, reserved);
    let budget_now = predict::current_budget(mem.total, mem.available, env.memory_ceiling, reserved);

    // A card too small to hold anything useful is not worth switching backend
    // for: the runtime will offload a handful of layers at best, and the
    // prediction is better served by the CPU path it will mostly take.
    const MIN_DISCRETE_VRAM: u64 = 2 << 30;
    let card = gpus
        .iter()
        .filter(|g| !g.integrated && g.vram_bytes >= MIN_DISCRETE_VRAM)
        .max_by_key(|g| g.vram_bytes);

    let backend = select_backend(card.is_some(), mem.unified, &gpus);
    let ram_infer = ram
        .by_threads
        .iter()
        .find(|(t, _)| *t == p_threads)
        .map(|(_, g)| *g)
        .unwrap_or(ram.peak_gbs);
    let bw = match backend {
        Backend::Metal => ram.peak_gbs,
        _ => ram_infer,
    };

    let hw = Hardware {
        backend,
        ram_bw_gbs: bw,
        ram_bw_peak_gbs: ram.peak_gbs,
        // Fall back low rather than high: overstating disk speed would make an
        // unusable model look viable.
        disk_gbs: disk.as_ref().map_or(0.5, |d| d.rand_128k_qdn_gbs),
        gflops: compute.gflops_nt,
        usable_bytes: budget_idle,
        threads: cpu.recommended_threads,
        vram_bytes: card.map_or(0, |g| g.vram_bytes),
        vram_bw_gbs: card.map_or(0.0, |g| g.bw_gbs),
        // ponytail: no GPU bandwidth benchmark exists yet, so this is always a
        // table lookup and `predict` caps confidence accordingly. Flip it in
        // this one place when zc-bench grows a device-side read probe.
        vram_bw_measured: false,
    };

    Machine {
        mem,
        cpu,
        env,
        storage,
        gpus,
        ram,
        compute,
        disk,
        hw,
        backend,
        budget_idle,
        budget_now,
    }
}

#[cfg(test)]
mod tests {
    use super::select_backend;
    use zc_model::Backend;
    use zc_probe::gpu::Gpu;

    fn apple(cores: Option<u32>) -> Gpu {
        let mut g = Gpu::new("Apple M5".into(), 0, "test");
        g.cores = cores;
        g
    }

    /// Real silicon: unified memory plus an adapter reporting shader cores.
    #[test]
    fn a_real_apple_gpu_selects_metal() {
        assert_eq!(
            select_backend(false, true, &[apple(Some(10))]),
            Backend::Metal
        );
    }

    /// Regression for calibration record `cf62d6b5590582da`.
    ///
    /// A virtualized Mac reports unified memory — `mem.unified` is
    /// `cfg!(target_arch = "aarch64")`, so it cannot report anything else — and
    /// presents a paravirtual adapter with no shader cores. Predicting Metal
    /// there gave 27-45 tok/s against 3.9 measured, an 822% error.
    #[test]
    fn unified_memory_without_a_usable_gpu_is_a_cpu_machine() {
        assert_eq!(select_backend(false, true, &[apple(None)]), Backend::Cpu);
    }

    /// The probe failing entirely must not be read as "GPU present".
    #[test]
    fn no_adapters_at_all_is_a_cpu_machine() {
        assert_eq!(select_backend(false, true, &[]), Backend::Cpu);
    }

    /// A discrete card outranks everything: it was found through a driver that
    /// would not have answered if the card were not real.
    #[test]
    fn a_discrete_card_wins_regardless() {
        assert_eq!(select_backend(true, true, &[apple(None)]), Backend::Discrete);
        assert_eq!(select_backend(true, false, &[]), Backend::Discrete);
    }
}
