//! Probe + benchmark the host once, and hand the result to any subcommand.
//!
//! Both `check` and `verify` need identical hardware facts. Measuring twice
//! would waste ~20 seconds and, worse, could produce two different answers if
//! thermal state shifted between them — which would make a calibration record
//! disagree with the prediction it is supposed to be calibrating.

use zc_bench::{compute, disk, ram};
use zc_model::{predict, Backend, Hardware};
use zc_probe::{cpu, env, memory, storage};

pub struct Machine {
    pub mem: memory::Memory,
    pub cpu: cpu::Cpu,
    pub env: env::Env,
    pub storage: storage::Storage,
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
    pub fn profile(&self) -> String {
        format!(
            "{}|{}P{}E|{}|{}|{:?}",
            self.cpu.brand,
            self.cpu.p_cores,
            self.cpu.e_cores,
            self.mem.total,
            self.env.os,
            self.storage.medium
        )
    }
}

pub fn probe() -> Machine {
    let mem = memory::probe();
    let cpu = cpu::probe();
    let env = env::probe(mem.total);
    let storage = storage::probe();
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

    // Apple Silicon runtimes default to Metal, and on unified memory the GPU
    // reaches close to the all-core peak — well above what the CPU cluster
    // alone sustains. Predicting Metal from a CPU figure understates badly.
    let backend = if mem.unified { Backend::Metal } else { Backend::Cpu };
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
    };

    Machine {
        mem,
        cpu,
        env,
        storage,
        ram,
        compute,
        disk,
        hw,
        backend,
        budget_idle,
        budget_now,
    }
}
