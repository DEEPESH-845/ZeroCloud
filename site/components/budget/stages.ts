// Every number is from the README capture `zc plan qwen3-8b --context 32K`
// (budget 12.80 GiB measured idle, 131 GB/s measured, Metal, target 10 tok/s,
// KV F16 at 32K = 4.50 GiB). Weight sizes match crates/zc-model/data/models/qwen3-8b.json.
export const BUDGET = 12.8;
export const MODEL = "qwen3-8b";

export type Stage = { name: string; weights: number; kv: number; total: string; needs: string; verdict: string; over?: boolean };

export const STAGES: Stage[] = [
  { name: "empty", weights: 0, kv: 0, total: "-", needs: "-", verdict: "12.80 GiB budget, measured on an idle machine" },
  { name: "Q4_K_M", weights: 4.68, kv: 4.5, total: "9.40", needs: "57", verdict: "fits, 17-28 t/s" },
  { name: "Q5_K_M", weights: 5.45, kv: 4.5, total: "10.17", needs: "67", verdict: "fits, 15-24 t/s" },
  { name: "Q6_K", weights: 6.26, kv: 4.5, total: "10.98", needs: "77", verdict: "fits, 13-21 t/s" },
  { name: "Q8_0", weights: 8.11, kv: 4.5, total: "12.83", needs: "92", verdict: "over by 0.03 GiB", over: true },
];

export const FOOTNOTE =
  "GiB is memory, however it is provided -- RAM, VRAM or unified. 'needs' is the bandwidth for 10 tok/s at eta 0.875, low confidence.";

// Scroll progress 0..1 → stage. Boundaries are where the readout switches;
// centers are where each stage is fully formed and held.
const BOUNDS = [0.1, 0.325, 0.55, 0.775];
export const CENTERS = [0.05, 0.2125, 0.4375, 0.6625, 0.8875];

export function stageAt(p: number): number {
  return BOUNDS.filter((b) => p >= b).length;
}

// Piecewise between stage centers, eased so each stage holds near its center.
export function lengthsAt(p: number): { w: number; kv: number } {
  if (p <= CENTERS[0]) return { w: STAGES[0].weights, kv: STAGES[0].kv };
  const last = CENTERS.length - 1;
  if (p >= CENTERS[last]) return { w: STAGES[last].weights, kv: STAGES[last].kv };
  let i = 0;
  while (p >= CENTERS[i + 1]) i++;
  let t = (p - CENTERS[i]) / (CENTERS[i + 1] - CENTERS[i]);
  t = t * t * (3 - 2 * t);
  const a = STAGES[i], b = STAGES[i + 1];
  return { w: a.weights + (b.weights - a.weights) * t, kv: a.kv + (b.kv - a.kv) * t };
}
