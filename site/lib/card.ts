// Decoding a result card from the URL fragment. The format is written by
// crates/zc-report/src/card.rs; change them together.
//
// A card link is input anyone can craft, so every field is checked and
// bounded here, and anything unexpected is refused rather than guessed at.
// React escapes what is rendered, so the risk is a misleading card, not
// script execution -- which is why a malformed one says so instead of
// rendering whatever parsed.

export type Verdict = "good" | "usable" | "slow" | "wont_fit";

export type CardRow = {
  id: string;
  quant: string;
  verdict: Verdict;
  low: number | null;
  high: number | null;
  ctx: number;
  confidence: string;
};

export type Card = {
  zc: string;
  kv: string;
  calibrated: boolean;
  hw: {
    cpu: string;
    cores: string;
    ram: number;
    gpu: string | null;
    backend: string;
    bwGbs: number | null;
    budget: number;
  };
  rows: CardRow[];
};

export type Decoded =
  | { ok: true; card: Card }
  | { ok: false; reason: "empty" | "version" | "malformed" };

const MAX_ROWS = 8;
const VERDICTS: readonly string[] = ["good", "usable", "slow", "wont_fit"];

const str = (v: unknown, max = 80): string | null =>
  typeof v === "string" && v.length > 0 && v.length <= max ? v : null;
const num = (v: unknown): number | null =>
  typeof v === "number" && Number.isFinite(v) && v >= 0 ? v : null;

function base64url(s: string): Uint8Array {
  const b64 = s.replace(/-/g, "+").replace(/_/g, "/") + "=".repeat((4 - (s.length % 4)) % 4);
  return Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
}

function row(v: unknown): CardRow | null {
  if (!Array.isArray(v) || v.length !== 7) return null;
  const [id, quant, verdict, low, high, ctx, confidence] = v;
  const r = {
    id: str(id),
    quant: str(quant, 24),
    verdict: typeof verdict === "string" && VERDICTS.includes(verdict) ? (verdict as Verdict) : null,
    low: low === null ? null : num(low),
    high: high === null ? null : num(high),
    ctx: num(ctx),
    confidence: str(confidence, 16),
  };
  if (!r.id || !r.quant || !r.verdict || r.ctx === null || !r.confidence) return null;
  if ((low !== null && r.low === null) || (high !== null && r.high === null)) return null;
  return { ...r, id: r.id, quant: r.quant, verdict: r.verdict, ctx: r.ctx, confidence: r.confidence };
}

export function decodeCard(hash: string): Decoded {
  const frag = hash.replace(/^#/, "");
  if (!frag) return { ok: false, reason: "empty" };
  const dot = frag.indexOf(".");
  if (dot < 0 || frag.slice(0, dot) !== "v1") return { ok: false, reason: "version" };
  let d: Record<string, unknown>;
  try {
    const text = new TextDecoder("utf-8", { fatal: true }).decode(base64url(frag.slice(dot + 1)));
    d = JSON.parse(text);
  } catch {
    return { ok: false, reason: "malformed" };
  }
  if (typeof d !== "object" || d === null || d.v !== 1) return { ok: false, reason: "version" };
  const hw = d.hw as Record<string, unknown> | undefined;
  if (typeof hw !== "object" || hw === null || !Array.isArray(d.rows)) return { ok: false, reason: "malformed" };
  const rows = d.rows.slice(0, MAX_ROWS).map(row);
  const card: Card = {
    zc: str(d.zc, 24) ?? "",
    kv: str(d.kv, 8) ?? "",
    calibrated: d.calibrated === true,
    hw: {
      cpu: str(hw.cpu) ?? "",
      cores: str(hw.cores, 16) ?? "",
      ram: num(hw.ram) ?? 0,
      gpu: hw.gpu === null ? null : str(hw.gpu),
      backend: str(hw.backend, 16) ?? "",
      bwGbs: num(hw.bw_gbs),
      budget: num(hw.budget) ?? 0,
    },
    rows: rows.filter((r): r is CardRow => r !== null),
  };
  if (!card.zc || !card.hw.cpu || !card.hw.ram || rows.some((r) => r === null) || card.rows.length === 0) {
    return { ok: false, reason: "malformed" };
  }
  return { ok: true, card };
}

export const gib = (bytes: number) => `${(bytes / 2 ** 30).toFixed(1).replace(/\.0$/, "")} GiB`;

export const ctxLabel = (n: number) => (n >= 1024 ? `${Math.round(n / 1024)}K` : String(n));
