"use client";
import dynamic from "next/dynamic";
import { useEffect, useRef, useState } from "react";
import { motion } from "motion/react";
import { gsap, ScrollTrigger, useGSAP } from "@/lib/gsap";
import { CENTERS, FOOTNOTE, MODEL, STAGES, stageAt } from "./stages";
import type { Pointer, Progress } from "./BudgetScene";
import s from "./Budget.module.css";

const Scene = dynamic(() => import("./BudgetScene"), {
  ssr: false,
  loading: () => <div className={s.fallback}>12.80 GiB budget</div>,
});

type Mode = "pin" | "tween" | "instant";

export default function BudgetSection() {
  const root = useRef<HTMLElement>(null);
  const progress = useRef<Progress>({ v: 0 }).current;
  const pointer = useRef<Pointer>({ x: 0, y: 0 }).current;
  const invalidateRef = useRef<(() => void) | null>(null);
  const trigger = useRef<ScrollTrigger | null>(null);
  const mode = useRef<Mode>("instant");
  const lastStage = useRef(0);
  const [stage, setStage] = useState(0);
  const [active, setActive] = useState(false);
  const [reduced, setReduced] = useState(false);
  // The three.js chunk is ~236 KB compressed: fetch it one viewport before the
  // section is reached, not on page load.
  const [near, setNear] = useState(false);
  useEffect(() => {
    const el = root.current;
    if (!el) return;
    const io = new IntersectionObserver(
      (es) => { if (es.some((e) => e.isIntersecting)) { setNear(true); io.disconnect(); } },
      { rootMargin: "100% 0px" },
    );
    io.observe(el);
    return () => io.disconnect();
  }, []);

  const sync = () => {
    const i = stageAt(progress.v);
    if (i !== lastStage.current) { lastStage.current = i; setStage(i); }
    invalidateRef.current?.();
  };

  useGSAP(() => {
    const mm = gsap.matchMedia();
    mm.add(
      {
        pin: "(min-width: 820px) and (prefers-reduced-motion: no-preference)",
        tween: "(max-width: 819px) and (prefers-reduced-motion: no-preference)",
        reduce: "(prefers-reduced-motion: reduce)",
      },
      (ctx) => {
        const { pin, reduce } = ctx.conditions as Record<string, boolean>;
        if (reduce) {
          mode.current = "instant";
          setReduced(true);
          progress.v = CENTERS[1];
          sync();
          return;
        }
        setReduced(false);
        if (pin) {
          mode.current = "pin";
          const tween = gsap.to(progress, {
            v: 1,
            ease: "none",
            onUpdate: sync,
            scrollTrigger: {
              trigger: root.current,
              start: "top top",
              end: "+=320%",
              pin: true,
              scrub: 0.6,
              snap: { snapTo: CENTERS, directional: false, duration: { min: 0.2, max: 0.5 }, delay: 0.05, ease: "power1.inOut" },
              refreshPriority: 1,
              onToggle: (self) => setActive(self.isActive),
            },
          });
          trigger.current = tween.scrollTrigger ?? null;
          ScrollTrigger.refresh();
        } else {
          mode.current = "tween";
          ScrollTrigger.create({
            trigger: root.current,
            start: "top bottom",
            end: "bottom top",
            onToggle: (self) => setActive(self.isActive),
          });
        }
        return () => { trigger.current = null; };
      },
    );
    return () => mm.revert();
  }, { scope: root });

  function go(i: number) {
    const p = CENTERS[i];
    const st = trigger.current;
    if (mode.current === "pin" && st) {
      window.scrollTo({ top: st.start + p * (st.end - st.start), behavior: "smooth" });
    } else if (mode.current === "tween") {
      gsap.to(progress, { v: p, duration: 0.8, ease: "power2.inOut", onUpdate: sync });
    } else {
      progress.v = p;
      sync();
    }
  }

  const st = STAGES[stage];
  return (
    <section ref={root} id="budget" className={s.section} aria-labelledby="budget-h">
      <div className={s.stage}>
        <div className={s.left}>
          <p className="eyebrow">the arithmetic</p>
          <h2 id="budget-h">Same model, four quantisations, one measured budget.</h2>
          <p className={s.lede}>
            Decode is memory-bound: every token reads the active weights once, so the whole question is what fits and how fast it can be read.
          </p>

          <div className={s.tabs} role="group" aria-label="Quantisation">
            {STAGES.map((t, i) => (
              <motion.button
                key={t.name}
                type="button"
                className={s.tab}
                aria-pressed={i === stage}
                onClick={() => go(i)}
                whileTap={{ scale: 0.97 }}
              >
                {i === stage && (
                  <motion.span layoutId="budget-tab" className={s.tabActive} transition={{ type: "spring", stiffness: 500, damping: 40 }} />
                )}
                {t.name}
              </motion.button>
            ))}
          </div>

          <div className={s.readout} aria-live="polite">
            <header>
              <span><b>{MODEL}</b>{stage > 0 ? ` · ${st.name}` : ""}</span>
              <span>32K context · KV F16</span>
            </header>
            <div className={s.row}><span>weights</span><span>{stage ? st.weights.toFixed(2) : "-"}</span><span>GiB</span></div>
            <div className={s.row}><span>KV</span><span>{stage ? st.kv.toFixed(2) : "-"}</span><span>GiB</span></div>
            <div className={s.row}><span>total</span><span>{st.total}</span><span>of 12.80 GiB budget</span></div>
            <div className={s.row}><span>needs</span><span>{st.needs}</span><span>GB/s for 10 tok/s</span></div>
            <div className={s.row}><span>measured</span><span>131</span><span>GB/s on this machine</span></div>
            <div className={s.verdict}>
              <span>on this machine</span>
              <span className={st.over ? s.over : stage ? s.ok : undefined}>{st.verdict}</span>
            </div>
          </div>
          <p className={s.foot}>{FOOTNOTE}</p>
        </div>

        <div
          className={s.canvasBox}
          onPointerMove={(e) => {
            const r = e.currentTarget.getBoundingClientRect();
            pointer.x = ((e.clientX - r.left) / r.width) * 2 - 1;
            pointer.y = ((e.clientY - r.top) / r.height) * 2 - 1;
            invalidateRef.current?.();
          }}
          onPointerLeave={() => { pointer.x = 0; pointer.y = 0; invalidateRef.current?.(); }}
          aria-hidden="true"
        >
          {near ? (
            <Scene progress={progress} pointer={pointer} active={active} reduced={reduced} invalidateRef={invalidateRef} />
          ) : (
            <div className={s.fallback}>12.80 GiB budget</div>
          )}
        </div>
      </div>
    </section>
  );
}
