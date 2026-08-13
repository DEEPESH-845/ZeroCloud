//! Model description. Every field here comes from the model's own
//! `config.json` — none of it is inferred, because inference is where wrong
//! numbers come from.

/// How the model stores attention state. This determines KV cache size, and
/// the four variants differ by more than an order of magnitude.
#[derive(Debug, Clone, PartialEq)]
pub enum Attention {
    /// Multi-head / grouped-query. `n_kv_heads == n_heads` is plain MHA.
    /// Stores a separate K and V per layer, hence the factor of 2.
    Gqa { n_kv_heads: u32, head_dim: u32 },

    /// Multi-head latent attention (DeepSeek family). Stores one *compressed*
    /// latent vector per layer instead of full K and V — so there is no factor
    /// of 2, and the width is the LoRA rank rather than n_heads * head_dim.
    /// Roughly an order of magnitude smaller than the GQA equivalent.
    Mla { kv_lora_rank: u32, rope_head_dim: u32 },

    /// Sliding-window attention (Gemma, Mistral family). Local layers keep at
    /// most `window` tokens, so their KV stops growing. A minority of layers
    /// are global and grow normally.
    Swa {
        n_kv_heads: u32,
        head_dim: u32,
        window: u32,
        /// Every Nth layer is full-attention. Gemma 3 uses 6.
        global_every: u32,
    },

    /// Hybrid transformer + state-space (Mamba-style). SSM layers hold a
    /// fixed-size state that does not grow with context at all; only the
    /// attention layers accumulate KV.
    Hybrid {
        attn_layers: u32,
        n_kv_heads: u32,
        head_dim: u32,
        /// Per-layer SSM state, constant regardless of context length.
        ssm_state_bytes: u64,
    },
}

/// Mixture-of-experts geometry. Absent for dense models.
#[derive(Debug, Clone, PartialEq)]
pub struct Moe {
    pub n_expert: u32,
    /// Experts routed to per token, per layer (top-k).
    pub n_active: u32,
    /// Parameters in one routed expert.
    pub expert_params: u64,
    /// Parameters that fire on every token regardless of routing: embeddings,
    /// attention, routers, shared experts. This is the "hot core" that should
    /// be pinned in RAM.
    pub shared_params: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantFamily {
    /// Q4_0, Q8_0 — simplest dequant, fastest per byte.
    Legacy,
    /// Q4_K_M etc. — superblock scales, slightly more dequant work.
    KQuant,
    /// IQ2/IQ3 — codebook lookup, noticeably slower per byte on CPU.
    IQuant,
    /// MXFP4 and friends — hardware-friendly block float.
    BlockFloat,
    /// F16 / BF16 — no dequant at all.
    Float,
}

impl QuantFamily {
    /// From a catalog file's `family` tag.
    ///
    /// k_quant is the default because its dequant cost sits between legacy and
    /// i-quant, so a mislabelled entry errs small in both directions.
    pub fn from_data_tag(s: &str) -> Self {
        match s {
            "legacy" => Self::Legacy,
            "i_quant" => Self::IQuant,
            "block_float" => Self::BlockFloat,
            "float" => Self::Float,
            _ => Self::KQuant,
        }
    }

    /// From a GGUF quantisation label as reported by a runtime (`Q4_K_M`,
    /// `IQ3_XXS`, `MXFP4`, ...).
    ///
    /// Order matters: `IQ4_NL` starts with IQ and must not fall through to the
    /// `_K` branch, and `F16` must be caught before the legacy default.
    pub fn from_gguf_label(label: &str) -> Self {
        let u = label.to_uppercase();
        if u.starts_with("IQ") {
            Self::IQuant
        } else if u.contains("MXFP") || u.contains("NF4") {
            Self::BlockFloat
        } else if u.starts_with("F16") || u.starts_with("BF16") || u.starts_with("F32") {
            Self::Float
        } else if u.contains("_K") {
            Self::KQuant
        } else {
            // Q4_0, Q5_1, Q8_0 and friends.
            Self::Legacy
        }
    }

    /// Stable key for grouping calibration records.
    pub fn tag(self) -> &'static str {
        match self {
            Self::Legacy => "legacy",
            Self::KQuant => "k_quant",
            Self::IQuant => "i_quant",
            Self::BlockFloat => "block_float",
            Self::Float => "float",
        }
    }

    /// Relative decode throughput per byte moved, versus f16.
    ///
    /// Dequantisation costs cycles that partly overlap the memory stall, so
    /// these are damping factors, not raw compute ratios. Hand-set priors —
    /// the calibration dataset replaces them.
    pub fn efficiency(self) -> f64 {
        match self {
            QuantFamily::Float => 1.00,
            QuantFamily::Legacy => 0.95,
            QuantFamily::BlockFloat => 0.92,
            QuantFamily::KQuant => 0.88,
            QuantFamily::IQuant => 0.72,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Quant {
    pub name: String,
    /// On-disk size of the whole model at this quantisation.
    pub bytes: u64,
    pub family: QuantFamily,
}

#[derive(Debug, Clone)]
pub struct ModelSpec {
    pub id: String,
    pub n_layers: u32,
    pub n_embd: u32,
    pub n_vocab: u32,
    /// Total parameter count, dense or MoE.
    pub params: u64,
    pub attention: Attention,
    pub moe: Option<Moe>,
    pub quants: Vec<Quant>,
}

impl ModelSpec {
    /// Parameters actually read per token.
    ///
    /// For a dense model that is all of them. For MoE it is the shared core
    /// plus `n_active` experts per layer — typically 5-20% of the total, which
    /// is the entire reason MoE is viable on constrained hardware.
    pub fn active_params(&self) -> u64 {
        match &self.moe {
            None => self.params,
            Some(m) => {
                m.shared_params + m.expert_params * m.n_active as u64 * self.n_layers as u64
            }
        }
    }

    /// Bytes read per token at a given quantisation.
    ///
    /// Scales the model's on-disk size by the active fraction, which assumes
    /// uniform bits-per-weight across the model. Good enough while quants are
    /// applied uniformly; a per-tensor byte map would supersede it.
    pub fn active_bytes(&self, q: &Quant) -> u64 {
        let frac = self.active_params() as f64 / self.params as f64;
        (q.bytes as f64 * frac) as u64
    }

    /// Bytes that should be pinned in RAM and never streamed: for MoE the
    /// shared core, for dense models there is no such split.
    pub fn hot_core_bytes(&self, q: &Quant) -> u64 {
        match &self.moe {
            None => 0,
            Some(m) => {
                let frac = m.shared_params as f64 / self.params as f64;
                (q.bytes as f64 * frac) as u64
            }
        }
    }
}
