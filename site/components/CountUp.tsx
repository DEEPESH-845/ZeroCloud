"use client";
import { useRef } from "react";
import { gsap, useGSAP } from "@/lib/gsap";

// Server renders the final value; GSAP only rewrites the text while counting.
export default function CountUp({ value, decimals = 0, suffix }: { value: number; decimals?: number; suffix: string }) {
  const el = useRef<HTMLSpanElement>(null);
  useGSAP(() => {
    const mm = gsap.matchMedia();
    mm.add("(prefers-reduced-motion: no-preference)", () => {
      const o = { v: 0 };
      gsap.to(o, {
        v: value, duration: 1.2, ease: "power2.out",
        scrollTrigger: { trigger: el.current, start: "top 90%", once: true },
        onUpdate: () => { if (el.current) el.current.textContent = o.v.toFixed(decimals); },
      });
    });
    return () => mm.revert();
  });
  return (
    <b><span ref={el}>{value.toFixed(decimals)}</span><small>{suffix}</small></b>
  );
}
