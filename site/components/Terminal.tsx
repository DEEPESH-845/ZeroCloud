"use client";
import { useRef, useState, type ReactNode } from "react";
import { gsap, useGSAP } from "@/lib/gsap";

// Real capture from the README. `at` is the timeline position in seconds:
// ~2.1s of measuring before the predictions table appears, as a user sees it.
type Line = { at: number; node: ReactNode; row?: boolean };

const K = (s: string) => <span className="k">{s}</span>;
const D = (s: string) => <span className="d">{s}</span>;
const N = (s: string) => <span className="n">{s}</span>;
const row = (m: string, q: string, t: string, c: string, ttft: string) => (
  <>
    {"  "}
    <span className="ok">OK</span>
    {"   " + m.padEnd(13) + q + "    " + t.padStart(11) + "   " + c.padStart(3) + "  " + ttft + " low"}
  </>
);

const RAM_STEPS: [string, number][] = [["1t:75", 0.18], ["2t:80", 0.18], ["4t:87", 0.2], ["10t:125", 0.22]];
const RAM_AT = 0.4;
const RAM_DONE = RAM_AT + RAM_STEPS.reduce((a, [, ms]) => a + ms, 0); // 1.18

const LINES: Line[] = [
  { at: 0, node: K("== hardware ==") },
  { at: 0.06, node: "  Apple M5   4P+6E   16.00 GiB total / 3.93 GiB available   unified" },
  { at: 0.1, node: "  /System/Volumes/Data on apfs (NVMe)" },
  { at: 0.14, node: "  Apple M5   integrated (shares system memory)" },
  { at: 0.34, node: "" },
  { at: 0.34, node: K("== measured ==") },
  { at: RAM_DONE, node: <>{"  ram          "}{N("125 GB/s")}{" peak @10t   "}{D("[1t:75  2t:80  4t:87  10t:125]")}</> },
  { at: RAM_DONE + 0.42, node: <>{"  compute      "}{N("427 GFLOPS")}{" f32 @4t   409 GOPS int8 (0.96x)"}</> },
  { at: RAM_DONE + 0.9, node: <>{"  disk         "}{N("5.04 GB/s")}{" random 128K @QD16   164K IOPS 4K"}</> },
  { at: RAM_DONE + 1.02, node: <>{"  budget       "}{N("12.80 GiB")}{" on an idle machine   (2.33 GiB free right now)"}</> },
  { at: RAM_DONE + 1.24, node: "" },
  { at: RAM_DONE + 1.24, node: <>{K("== predictions ==")}{"  (Metal backend, 125 GB/s, KV at F16, 2048-token prompt)"}</> },
  { at: RAM_DONE + 1.24, node: D("  assumes an otherwise-idle machine") },
  { at: RAM_DONE + 1.36, node: K("       model        quant    decode t/s   ctx  TTFT conf   %RAM") },
  ...[
    ["smollm2-360m", "Q8_0", "230.0-383.3", "8K", "0.1s"],
    ["qwen3-0.6b", "Q8_0", "139.0-231.6", "40K", "0.3s"],
    ["smollm2-1.7b", "Q8_0", "48.8-81.4", "8K", "0.6s"],
    ["qwen3-1.7b", "Q8_0", "48.4-80.7", "40K", "0.7s"],
    ["qwen2.5-3b", "Q8_0", "24.6-41.0", "32K", "1.1s"],
    ["phi-3.5-mini", "Q8_0", "21.9-36.5", "23K", "1.4s"],
    ["phi-4-mini", "Q8_0", "21.8-36.3", "70K", "1.4s"],
    ["qwen3-4b", "Q8_0", "20.8-34.6", "40K", "1.5s"],
  ].map(([m, q, t, c, f], i) => ({ at: RAM_DONE + 1.43 + i * 0.07, node: row(m, q, t, c, f), row: true })),
  { at: RAM_DONE + 2.15, node: "" },
  { at: RAM_DONE + 2.15, node: D("  showing 8 of 26 - ranked by verdict, then speed, then context") },
  { at: RAM_DONE + 2.15, node: D("  --all for every quantisation, --top N to change the cut") },
];
const END = RAM_DONE + 2.3;
const RAM_INDEX = LINES.findIndex((l) => l.at === RAM_DONE);

export default function Terminal() {
  const root = useRef<HTMLDivElement>(null);
  const [shown, setShown] = useState(0); // lines visible
  const [ramStep, setRamStep] = useState(-1); // -1 idle, 0..3 measuring
  const [clock, setClock] = useState("0.00s");
  const [running, setRunning] = useState(false);
  const tl = useRef<gsap.core.Timeline | null>(null);

  useGSAP(
    () => {
      const mm = gsap.matchMedia();
      mm.add("(prefers-reduced-motion: reduce)", () => {
        setShown(LINES.length);
        setClock("2.13s");
      });
      mm.add("(prefers-reduced-motion: no-preference)", () => {
        const t = gsap.timeline({
          paused: true,
          scrollTrigger: { trigger: root.current, start: "top 80%", once: true },
          onStart: () => setRunning(true),
          onUpdate: () => setClock(t.time().toFixed(2) + "s"),
          onComplete: () => setRunning(false),
        });
        // ponytail: timeline of callbacks; React state paints, GSAP keeps time.
        LINES.forEach((l, i) => t.call(() => setShown(i + 1), [], l.at + 0.001));
        RAM_STEPS.forEach(([, d], i) =>
          t.call(() => setRamStep(i), [], RAM_AT + RAM_STEPS.slice(0, i).reduce((a, [, x]) => a + x, 0)),
        );
        t.call(() => setRamStep(-1), [], RAM_DONE);
        t.call(() => setShown(LINES.length), [], END);
        tl.current = t;
      });
    },
    { scope: root },
  );

  const replay = () => {
    const t = tl.current;
    if (!t) return;
    setShown(0);
    setRamStep(-1);
    t.restart(true);
  };

  return (
    <>
      <div className="term" id="term" aria-label="A recorded run of zc check" ref={root}>
        <div className="term-bar">
          <i /><i /><i /> zc check --top 8 <b>elapsed <em>{clock}</em></b>
        </div>
        <pre aria-live="polite" className={running ? "" : "done"}>
          {D("$")} zc check --top 8{"\n\n"}
          {LINES.slice(0, shown).map((l, i) =>
            l.row ? (
              <span key={i} className="row" ref={(el) => { if (el) requestAnimationFrame(() => el.classList.add("in")); }}>{l.node}{"\n"}</span>
            ) : (
              <span key={i}>{l.node}{"\n"}</span>
            ),
          )}
          {ramStep >= 0 && shown === RAM_INDEX && (
            <>{"  ram          "}{N("measuring")}{" "}{D("[" + RAM_STEPS.slice(0, ramStep + 1).map(([s]) => s).join("  ") + "]")}</>
          )}
          {running && <span className="cur" />}
        </pre>
      </div>
      <div className="term-foot">
        <span>real capture · Apple M5, 16 GiB · predictions are ranges, never points</span>
        <button type="button" onClick={replay}>replay ↺</button>
      </div>
    </>
  );
}
