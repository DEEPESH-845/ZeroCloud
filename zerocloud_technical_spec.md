# ZeroCloud Core: Technical Architecture & Implementation Spec

This document serves as the comprehensive technical blueprint for implementing ZeroCloud Core. It is designed to be fed into an implementation agent (like Claude) to dictate the exact architecture, tech stack, and safety constraints for building a local LLM engine for low-RAM devices.

## 1. System Architecture

The architecture bypasses standard "load into RAM" paradigms and acts more like a high-performance database streaming weights.

### Component Breakdown
1. **Zero-Copy Paging Engine:** The core memory manager. It reads model weights from the NVMe drive using OS-level memory mapping (`mmap`) with aggressive, manual page-eviction (`madvise` with `MADV_DONTNEED`) to ensure RAM stays clear.
2. **Compute Dispatcher:** Routes matrix multiplication tasks. If a layer is small enough, it routes to the GPU/Metal. If it's too large, it routes to CPU/SIMD.
3. **Speculative Draft Engine:** Keeps a tiny 0.5B-1B model pinned in RAM permanently to generate draft tokens, reducing the frequency of SSD fetches for the main model.
4. **Hardware Safety Governor (Critical):** A daemon running alongside the inference engine that polls hardware metrics and throttles the engine to prevent thermal or physical damage to the host machine.

---

## 2. Hardware Safety & Edge Case Handling (The "Do No Harm" Policy)

Running 70B models on low-end laptops can melt CPUs or destroy SSDs if done naively. ZeroCloud implements strict hardware guardrails:

### A. SSD Wear & Tear Protection (TBW Guard)
* **The Danger:** If the OS runs out of RAM, it will aggressively use "Swap" space. Swapping writes gigabytes of data to the SSD constantly, which can destroy an SSD's Terabytes Written (TBW) lifespan in weeks.
* **The Solution:** The weights file must be mapped strictly as **READ-ONLY** (`PROT_READ`). ZeroCloud will utilize `mlock()` on the KV cache to lock it in RAM and prevent the OS from writing it to swap. ZeroCloud reads heavily from the SSD but **writes almost zero data**, preserving the drive's lifespan.

### B. Thermal Throttling & Battery Drain
* **The Danger:** Pushing a low-end laptop CPU/GPU to 100% utilization continuously will cause it to overheat, spin fans at max, and kill the battery in 30 minutes.
* **The Solution:** ZeroCloud introduces **Dynamic Duty Cycling**. 
  * It polls OS thermal sensors (e.g., Apple SMC for Mac, `lm-sensors` for Linux). 
  * If the CPU hits > 80°C, the engine automatically injects micro-sleeps (e.g., 5ms pauses between matrix multiplications). 
  * **Power Profiles:** Users can select "Eco Mode" (low heat, silent fans, 5 tokens/sec) vs "Unleashed Mode" (max speed, hot).

### C. OS Freezing (OOM Killer)
* **The Danger:** Allocating too much RAM freezes the laptop.
* **The Solution:** The engine queries the OS for *Available* (not just Free) physical memory on startup. It sets a hard ceiling for itself (e.g., Host RAM - 2GB for OS). If it hits the ceiling, it drops the oldest KV cache tokens rather than allocating more RAM.

---

## 3. Technology Stack Recommendations

To achieve the memory safety and bare-metal performance required, the following stack is recommended for implementation:

1. **Core Language:** **Rust**. Rust prevents memory leaks and data races out of the box, which is vital when doing aggressive manual memory paging.
2. **Tensor/Math Backend:** **Candle** (by HuggingFace, written in Rust) OR a custom **C/C++** backend utilizing **GGML** (the backend for llama.cpp). GGML is highly optimized for Apple Silicon (Metal) and AVX2/AVX-512 (Intel/AMD). 
3. **Hardware Polling Libraries:** `sysinfo` (Rust crate) for memory limits and thermal sensor readings.

---

## 4. Implementation Plan & Milestones

For the AI/Developer implementing this, follow these strict phases:

### Phase 1: Foundation & Safety (Weeks 1-2)
* Setup the Rust project structure.
* Implement the OS memory polling and Thermal Governor. (Do not implement inference yet).
* Create a dummy load that stresses the CPU, and prove the Thermal Governor successfully throttles it before 85°C.

### Phase 2: The Paging Memory Manager (Weeks 2-3)
* Implement the custom `mmap` allocator.
* Load a quantized model file (.gguf) but do not run inference. Prove that the engine can sequentially read layers from SSD to RAM, flush them (`MADV_DONTNEED`), and keep overall RAM utilization under a strict 4GB cap.

### Phase 3: Core Inference & Math (Weeks 3-4)
* Hook the Memory Manager up to the Tensor backend (Candle/GGML).
* Implement the forward pass for Llama 3 architecture. 
* *Edge Case test:* Disconnect the laptop from power, run inference, and ensure the Eco Mode properly throttles down.

### Phase 4: Optimization (Weeks 5+)
* Implement KV Cache ring-buffers (dropping old context automatically).
* Add Speculative Decoding with a smaller draft model to boost tokens/sec.
* Build the OpenAI-compatible HTTP server layer.

---

## 5. Potential Edge Cases & Mitigations

| Edge Case | Consequence | ZeroCloud Mitigation |
| :--- | :--- | :--- |
| **User pulls out power cord during inference** | Laptop drops to low-power state, inference stutters or crashes. | Listen for ACPI power events; instantly switch to "Eco Mode" batch sizes. |
| **Context Window grows too large** | Out of Memory crash. | Sparse Attention: Evict middle tokens, keeping system prompt and recent chat. |
| **Slow HDD (Spinning disk) instead of NVMe** | Extremely slow generation (0.1 tokens/sec). | Startup hardware check: Warn user if disk read speed is < 1000MB/s and refuse to run massive models. |
