#!/usr/bin/env python3
"""Generate catalog entries from Hugging Face model configs.

The catalog is the width of the product: six models today, and a machine that
can run a model we have never heard of is told nothing. This turns a list of
repos into `data/models/*.json`.

The gate is the whole point. Every field is *derived from a reading* or the
entry is rejected with a logged reason:

  * attention kind must follow from config.json's own keys, never from the
    architecture name. A wrong KV formula is wrong by an order of magnitude and
    the error always runs one way — the model looks bigger than it is, so it is
    marked unrunnable on a machine that could run it.
  * parameter counts come from the safetensors index the API reports, not from
    a bits-per-weight estimate.
  * quantisation sizes come from the actual GGUF blobs, summed across shards.
  * MoE geometry must reproduce the advertised active-parameter figure when the
    model's own name states one ("A3B").

Rejection is the safe outcome: a missing model is recoverable, a confidently
wrong one is not.

Standard library only, to match the rest of the repo — this runs in CI on a
plain python image with no install step.

Usage:
    ingest_hf.py --self-test           check the derivation against known models
    ingest_hf.py --dry-run             fetch and report, write nothing
    ingest_hf.py                       write data/models/*.json
    ingest_hf.py --manifest FILE       use a different repo list
"""

import argparse
import datetime
import json
import os
import re
import sys
import urllib.error
import urllib.request

HF = "https://huggingface.co"
UA = {"User-Agent": "zerocloud-ingest/0.1 (+https://github.com/zerocloud/zerocloud)"}
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MANIFEST = os.path.join(ROOT, "scripts", "models.txt")
OUT_DIR = os.path.join(ROOT, "data", "models")

# Quantisations worth carrying, with llama.cpp's published bits-per-weight.
#
# The low end of this list is what actually decides whether an 8 GB laptop can
# run something, which is the whole audience. The bpw figure is not used to
# *compute* anything — sizes are read from real files — it is the sanity check
# that catches a misread listing, and it has already earned its keep once (see
# `plausible`).
WANTED_QUANTS = {
    "Q4_K_M": 4.85,
    "Q5_K_M": 5.69,
    "Q6_K": 6.56,
    "Q8_0": 8.50,
    "IQ2_XXS": 2.06,
    "IQ3_XXS": 3.06,
    "IQ4_XS": 4.25,
}


class Reject(Exception):
    """A model we decline to publish, with the reason a human needs."""


# ---------------------------------------------------------------- helpers ----


def get_json(url):
    headers = dict(UA)
    # Meta and Google gate their base repos behind a licence click, so
    # config.json is a 401 without a token. CI passes one; a local run without
    # one simply reports those models as gated and moves on.
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())


def plausible(label, size_bytes, params):
    """Does this file size make sense for this quantisation?

    Sizes are read, not computed — but a listing can be *misread*, and this is
    the arithmetic that catches it. Qwen publishes both a sharded set and a
    merged single file for the same quantisation; summing across both doubled
    every figure, and a doubled Q4_K_M is a model predicted to need twice the
    memory it does, on exactly the machines where that decides the answer.

    The band is asymmetric on purpose. Small models legitimately run high —
    embedding and output tensors are kept at higher precision and dominate a
    360M file — while nothing legitimately comes in far under its own format.
    """
    if not params:
        return True
    bpw = size_bytes * 8 / params
    expected = WANTED_QUANTS[label]
    return 0.55 * expected <= bpw <= 1.7 * expected


def quant_family(label):
    """Mirror of `QuantFamily::from_gguf_label` in zc-model/src/spec.rs.

    Order matters: IQ4_NL starts with IQ and must not fall through to the _K
    branch, and F16 must be caught before the legacy default.
    """
    u = label.upper()
    if u.startswith("IQ"):
        return "i_quant"
    if "MXFP" in u or "NF4" in u:
        return "block_float"
    if u.startswith(("F16", "BF16", "F32")):
        return "float"
    if "_K" in u:
        return "k_quant"
    return "legacy"


def model_id(repo, override=None):
    """`meta-llama/Llama-3.1-8B-Instruct` -> `llama-3.1-8b`."""
    if override:
        return override
    name = repo.split("/")[-1].lower()
    for suffix in ("-instruct", "-it", "-chat"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name


def advertised_active_b(repo):
    """The `A3B` in `Qwen3-30B-A3B`, in billions. None when the name is silent."""
    m = re.search(r"[-_]a(\d+(?:\.\d+)?)b\b", repo.lower())
    return float(m.group(1)) if m else None


# ------------------------------------------------------------- derivation ----


def derive_attention(cfg, n_layers):
    """Attention geometry, or `Reject`.

    Every branch keys off a field that is only present when that mechanism is
    in use. Nothing is inferred from the architecture string, because a new
    model in a known family is exactly the case that would break.
    """
    hidden = cfg.get("hidden_size")
    n_heads = cfg.get("num_attention_heads")
    n_kv = cfg.get("num_key_value_heads", n_heads)

    head_dim = cfg.get("head_dim")
    if not head_dim and hidden and n_heads:
        head_dim = hidden // n_heads

    # MLA — DeepSeek. The latent rank *is* the stored width, and there is no
    # separate K and V, so this must be caught before the GQA branch.
    if cfg.get("kv_lora_rank"):
        rope = cfg.get("qk_rope_head_dim")
        if not rope:
            raise Reject("kv_lora_rank present but qk_rope_head_dim missing")
        return {
            "kind": "mla",
            "kv_lora_rank": int(cfg["kv_lora_rank"]),
            "rope_head_dim": int(rope),
        }

    # State-space hybrids. The layer-by-layer attention/SSM split and the SSM
    # state width are not stated in config.json in any portable form, and
    # guessing which layers are attention would misplace the entire KV curve.
    if any(k in cfg for k in ("ssm_cfg", "ssm_d_state", "mamba_d_state", "linear_attn_config")):
        raise Reject("hybrid SSM: per-layer attention/SSM split not derivable from config.json")

    if not (n_kv and head_dim):
        raise Reject("no num_key_value_heads / head_dim to derive KV width from")

    # SWA — a window only counts when the model actually uses it. Qwen2 ships
    # `sliding_window` alongside `use_sliding_window: false`, and believing it
    # would understate KV several-fold on the models most likely to fit.
    window = cfg.get("sliding_window")
    if window and cfg.get("use_sliding_window", True):
        pattern = cfg.get("sliding_window_pattern") or cfg.get("layer_types_pattern")
        if isinstance(pattern, int) and pattern > 0:
            global_every = pattern
        else:
            # Every layer is local (Mistral-style). `global_every` counts global
            # layers as `n_layers // global_every`, so a value above the layer
            # count is how "none of them" is spelled.
            global_every = n_layers + 1
        return {
            "kind": "swa",
            "n_kv_heads": int(n_kv),
            "head_dim": int(head_dim),
            "window": int(window),
            "global_every": int(global_every),
        }

    return {"kind": "gqa", "n_kv_heads": int(n_kv), "head_dim": int(head_dim)}


def derive_moe(cfg, n_layers, params):
    """MoE geometry, or None for a dense model. Raises `Reject` if partial."""
    n_expert = cfg.get("num_local_experts") or cfg.get("n_routed_experts") or cfg.get("num_experts")
    if not n_expert:
        return None

    n_active = cfg.get("num_experts_per_tok") or cfg.get("n_experts_per_tok")
    ff = cfg.get("moe_intermediate_size") or cfg.get("expert_intermediate_size")
    hidden = cfg.get("hidden_size")
    if not (n_active and ff and hidden):
        raise Reject(
            "MoE without full geometry (need num_experts_per_tok and "
            "moe_intermediate_size); active parameters would be a guess"
        )

    # gate + up + down projections, each hidden x ff.
    expert_params = 3 * int(hidden) * int(ff)
    routed_total = expert_params * int(n_expert) * n_layers
    if routed_total >= params:
        raise Reject(
            f"routed experts ({routed_total/1e9:.1f}B) exceed the model's own "
            f"parameter count ({params/1e9:.1f}B) - geometry cannot be right"
        )
    return {
        "n_expert": int(n_expert),
        "n_active": int(n_active),
        "expert_params": expert_params,
        "shared_params": params - routed_total,
    }


def active_params(spec):
    """Mirror of `ModelSpec::active_params`."""
    moe = spec.get("moe")
    if not moe:
        return spec["params"]
    return moe["shared_params"] + moe["expert_params"] * moe["n_active"] * spec["n_layers"]


def build_spec(repo, cfg, params, n_vocab_override=None):
    """Turn a fetched config into a catalog entry, or raise `Reject`."""
    n_layers = cfg.get("num_hidden_layers")
    hidden = cfg.get("hidden_size")
    n_vocab = n_vocab_override or cfg.get("vocab_size")
    # Multimodal repos nest the language model one level down.
    if not n_layers and isinstance(cfg.get("text_config"), dict):
        return build_spec(repo, cfg["text_config"], params, n_vocab)
    if not (n_layers and hidden and n_vocab):
        raise Reject("config.json lacks num_hidden_layers / hidden_size / vocab_size")
    if not params:
        raise Reject("no parameter count reported by the API (no safetensors index)")

    spec = {
        "n_layers": int(n_layers),
        "n_embd": int(hidden),
        "n_vocab": int(n_vocab),
        "params": int(params),
        "attention": derive_attention(cfg, int(n_layers)),
    }
    # The context the model was trained for. Without it a sliding-window model
    # on a large machine reports a context it was never trained to handle:
    # Ministral's 32K window came out at 1024K, which is not a usable number.
    trained = cfg.get("max_position_embeddings")
    if trained:
        spec["n_ctx_train"] = int(trained)
    moe = derive_moe(cfg, int(n_layers), int(params))
    if moe:
        spec["moe"] = moe

    # The model's own name is a claim about its active parameters. If our
    # derived geometry disagrees with it, the geometry is wrong.
    want = advertised_active_b(repo)
    if want is not None:
        got = active_params(spec) / 1e9
        if abs(got - want) / want > 0.25:
            raise Reject(
                f"derived {got:.2f}B active parameters but the name claims {want}B"
            )
    return spec


# ----------------------------------------------------------------- remote ----


def fetch_params(repo):
    """Total parameters, from the API's safetensors index. None if absent."""
    try:
        info = get_json(f"{HF}/api/models/{repo}")
    except urllib.error.HTTPError as e:
        raise Reject(f"model info returned HTTP {e.code}")
    total = (info.get("safetensors") or {}).get("total")
    return int(total) if total else None


def fetch_config(repo):
    try:
        return get_json(f"{HF}/{repo}/resolve/main/config.json")
    except urllib.error.HTTPError as e:
        raise Reject(f"config.json returned HTTP {e.code}")


SHARD = re.compile(r"-\d{5}-of-\d{5}\.gguf$", re.I)


def quant_of(stem):
    """Which wanted quantisation a file stem names, if any."""
    u = stem.upper()
    for q in WANTED_QUANTS:
        if u.endswith(q) or f"-{q}-" in u or f".{q}." in u:
            return q
    return None


def fetch_quants(repo, params):
    """Real GGUF sizes, one complete copy per quantisation.

    Two traps, both of which silently produce a wrong weight size:

      * A large model is published in shards (`-00001-of-00003.gguf`) and
        counting one shard understates the weights several-fold.
      * The same quantisation is often published *both* ways — Qwen ships
        `...-q4_k_m.gguf` alongside `...-q4_k_m-00001-of-00002.gguf` — and
        adding the shards to the merged file doubles it.

    So shards are summed into their own total and the merged file is preferred
    when it exists; either way the result is one copy. `plausible` then checks
    the answer against the format's own bits-per-weight.
    """
    try:
        tree = get_json(f"{HF}/api/models/{repo}/tree/main?recursive=true")
    except urllib.error.HTTPError as e:
        raise Reject(f"file tree returned HTTP {e.code}")

    single, sharded = {}, {}
    for entry in tree:
        path = entry.get("path", "")
        if not path.lower().endswith(".gguf"):
            continue
        # The LFS record holds the real size; `size` is the pointer file.
        size = int((entry.get("lfs") or {}).get("size") or entry.get("size") or 0)
        name = os.path.basename(path)
        is_shard = bool(SHARD.search(name))
        stem = SHARD.sub(".gguf", name)[: -len(".gguf")]
        q = quant_of(stem)
        if not q:
            continue
        if is_shard:
            sharded[q] = sharded.get(q, 0) + size
        else:
            single[q] = size

    quants = []
    for q in WANTED_QUANTS:
        size = single.get(q) or sharded.get(q, 0)
        if size <= 0:
            continue
        if not plausible(q, size, params):
            print(f"          skip {q}: {size*8/params:.2f} bits/weight is not {q}")
            continue
        quants.append({"name": q, "bytes": size, "family": quant_family(q)})
    if not quants:
        raise Reject(f"no recognised GGUF quantisations in {repo}")
    return quants


# ------------------------------------------------------------------- main ----


def read_manifest(path):
    """`base_repo  gguf_repo  [id]` per line; `#` comments."""
    rows = []
    with open(path) as f:
        for raw in f:
            line = raw.split("#")[0].strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                print(f"  SKIP  {line}: needs a base repo and a GGUF repo")
                continue
            rows.append((parts[0], parts[1], parts[2] if len(parts) > 2 else None))
    return rows


def ingest(base_repo, gguf_repo, override, today):
    cfg = fetch_config(base_repo)
    params = fetch_params(base_repo)
    spec = build_spec(base_repo, cfg, params)
    quants = fetch_quants(gguf_repo, params)

    entry = {
        "id": model_id(base_repo, override),
        "arch": (cfg.get("model_type") or "unknown"),
        "source": f"hf:{base_repo}",
        "verified": today,
        "note": f"Generated by scripts/ingest_hf.py. Quant sizes from hf:{gguf_repo}.",
    }
    entry.update(spec)
    entry["quants"] = quants
    return entry


def self_test():
    """Derivation checks against models whose geometry is hand-verified.

    These configs are the real published ones, trimmed to the keys that matter.
    Each expected value is cross-checked against `data/models/*.json`, which
    `cargo test -p zc-model` in turn checks against hand-computed KV bytes — so
    a break here means the ingest disagrees with arithmetic done by hand.
    """
    llama = {
        "model_type": "llama", "num_hidden_layers": 32, "hidden_size": 4096,
        "num_attention_heads": 32, "num_key_value_heads": 8, "vocab_size": 128256,
    }
    a = derive_attention(llama, 32)
    assert a == {"kind": "gqa", "n_kv_heads": 8, "head_dim": 128}, a

    gemma = {
        "model_type": "gemma3", "num_hidden_layers": 48, "hidden_size": 3840,
        "num_attention_heads": 16, "num_key_value_heads": 8, "head_dim": 256,
        "vocab_size": 262144, "sliding_window": 1024, "sliding_window_pattern": 6,
    }
    a = derive_attention(gemma, 48)
    assert a["kind"] == "swa" and a["window"] == 1024 and a["global_every"] == 6, a
    assert a["head_dim"] == 256, "explicit head_dim must win over hidden/heads"

    # Qwen2 ships a window it does not use. Believing it understates KV ~4x.
    qwen2 = {
        "num_hidden_layers": 28, "hidden_size": 3584, "num_attention_heads": 28,
        "num_key_value_heads": 4, "vocab_size": 152064,
        "sliding_window": 131072, "use_sliding_window": False,
    }
    assert derive_attention(qwen2, 28)["kind"] == "gqa"

    # Mistral: every layer local, spelled as "no global layers".
    mistral = {
        "num_hidden_layers": 32, "hidden_size": 4096, "num_attention_heads": 32,
        "num_key_value_heads": 8, "vocab_size": 32768, "sliding_window": 4096,
    }
    a = derive_attention(mistral, 32)
    assert a["kind"] == "swa" and a["global_every"] == 33, a

    dsv3 = {
        "model_type": "deepseek_v3", "num_hidden_layers": 61, "hidden_size": 7168,
        "num_attention_heads": 128, "num_key_value_heads": 128, "vocab_size": 129280,
        "kv_lora_rank": 512, "qk_rope_head_dim": 64,
        "n_routed_experts": 256, "num_experts_per_tok": 8, "moe_intermediate_size": 2048,
    }
    a = derive_attention(dsv3, 61)
    assert a == {"kind": "mla", "kv_lora_rank": 512, "rope_head_dim": 64}, a

    # Qwen3-30B-A3B: the geometry has to reproduce the 3B the name claims.
    q3moe = {
        "model_type": "qwen3_moe", "num_hidden_layers": 48, "hidden_size": 2048,
        "num_attention_heads": 32, "num_key_value_heads": 4, "head_dim": 128,
        "vocab_size": 151936, "num_experts": 128, "num_experts_per_tok": 8,
        "moe_intermediate_size": 768,
    }
    ctx = build_spec("X/Y", {**llama, "max_position_embeddings": 131072}, 8_030_261_248)
    assert ctx["n_ctx_train"] == 131072
    assert "n_ctx_train" not in build_spec("X/Y", llama, 8_030_261_248)

    spec = build_spec("Qwen/Qwen3-30B-A3B", q3moe, 30_532_122_624)
    got = active_params(spec) / 1e9
    assert 3.0 <= got <= 3.7, f"active {got}B, expected ~3.3B"
    assert spec["moe"]["expert_params"] == 3 * 2048 * 768

    # And a name whose claim the geometry cannot meet must be refused.
    try:
        build_spec("Fake/Model-30B-A20B", q3moe, 30_532_122_624)
        raise AssertionError("should have rejected a 20B claim on 3.3B of geometry")
    except Reject:
        pass

    # An SSM hybrid is rejected rather than billed as full attention.
    try:
        derive_attention({**llama, "ssm_d_state": 128}, 32)
        raise AssertionError("hybrid should be rejected")
    except Reject:
        pass

    # A MoE missing its expert width is rejected, not averaged.
    try:
        derive_moe({"num_experts": 8, "hidden_size": 4096}, 32, 47_000_000_000)
        raise AssertionError("partial MoE should be rejected")
    except Reject:
        pass

    # The double-count that shipped wrong sizes until the listing was read
    # properly: 7.6B params at Q4_K_M is ~4.7 GB, and 9.4 GB is not Q4_K_M.
    assert plausible("Q4_K_M", 4_680_000_000, 7_615_616_512)
    assert not plausible("Q4_K_M", 9_360_000_000, 7_615_616_512)
    # A 360M model's Q4_K_M legitimately runs high: its embedding table is a
    # large share of the file and is not quantised as far.
    assert plausible("Q4_K_M", 300_000_000, 361_821_120)
    assert quant_of("qwen2.5-coder-7b-instruct-q4_k_m") == "Q4_K_M"
    assert quant_of("Qwen3-1.7B-Q8_0") == "Q8_0"
    assert quant_of("qwen2.5-coder-7b-instruct-fp16") is None

    assert quant_family("Q4_K_M") == "k_quant"
    assert quant_family("IQ4_NL") == "i_quant"
    assert quant_family("Q8_0") == "legacy"
    assert model_id("meta-llama/Llama-3.1-8B-Instruct") == "llama-3.1-8b"
    assert model_id("google/gemma-3-12b-it") == "gemma-3-12b"
    assert model_id("Qwen/Qwen3-30B-A3B") == "qwen3-30b-a3b"
    assert SHARD.sub(".gguf", "m-Q4_K_M-00001-of-00003.gguf") == "m-Q4_K_M.gguf"
    print("self-test OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--dry-run", action="store_true", help="fetch and report, write nothing")
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="replace catalog files that already exist (they may be hand-verified)",
    )
    ap.add_argument("--self-test", action="store_true", help="check derivation, no network")
    ap.add_argument("repos", nargs="*", help="base_repo:gguf_repo pairs, overriding the manifest")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return 0

    if args.repos:
        rows = [(r.split(":")[0], r.split(":")[1], None) for r in args.repos]
    else:
        rows = read_manifest(args.manifest)

    today = datetime.date.today().isoformat()
    accepted, rejected = [], []
    for base, gguf, override in rows:
        try:
            entry = ingest(base, gguf, override, today)
        except Reject as e:
            rejected.append((base, str(e)))
            print(f"  REJECT  {base}: {e}")
            continue
        except Exception as e:  # network, malformed JSON, anything unforeseen
            rejected.append((base, f"{type(e).__name__}: {e}"))
            print(f"  ERROR   {base}: {type(e).__name__}: {e}")
            continue

        accepted.append(entry)
        quants = ", ".join(f"{q['name']} {q['bytes']/1e9:.1f}GB" for q in entry["quants"])
        print(f"  ACCEPT  {entry['id']:<24} {entry['attention']['kind']:<6} {quants}")

        if not args.dry_run:
            os.makedirs(OUT_DIR, exist_ok=True)
            path = os.path.join(OUT_DIR, f"{entry['id']}.json")
            # Existing entries were verified by hand and are covered by tests
            # that assert hand-computed KV bytes. A weekly robot does not get to
            # overwrite those without someone asking it to.
            if os.path.exists(path) and not args.overwrite:
                print(f"          keeping existing {entry['id']}.json (--overwrite to replace)")
                continue
            with open(path, "w") as f:
                json.dump(entry, f, indent=2)
                f.write("\n")

    print(f"\n{len(accepted)} accepted, {len(rejected)} rejected")
    # Rejections are the expected outcome for models we cannot model correctly,
    # so they are not a failure. An empty run is, because it means the manifest
    # or the API changed under us.
    return 0 if accepted else 1


if __name__ == "__main__":
    sys.exit(main())
