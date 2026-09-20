"use client";
import { useRef, type ReactNode, type PointerEvent } from "react";
import { motion, useMotionValue, useSpring } from "motion/react";

const PULL = 6; // px, max displacement toward the cursor
const ZONE = 12; // px around the element that still pulls

// A link that leans toward the cursor. Mouse only; off under reduced motion.
export default function MagneticButton({ href, className, children }: { href: string; className?: string; children: ReactNode }) {
  const ref = useRef<HTMLAnchorElement>(null);
  const x = useMotionValue(0), y = useMotionValue(0);
  const sx = useSpring(x, { stiffness: 300, damping: 20 });
  const sy = useSpring(y, { stiffness: 300, damping: 20 });

  function move(e: PointerEvent<HTMLSpanElement>) {
    if (e.pointerType !== "mouse" || matchMedia("(prefers-reduced-motion: reduce)").matches || !ref.current) return;
    const r = ref.current.getBoundingClientRect();
    const dx = (e.clientX - (r.left + r.width / 2)) / (r.width / 2 + ZONE);
    const dy = (e.clientY - (r.top + r.height / 2)) / (r.height / 2 + ZONE);
    x.set(Math.max(-1, Math.min(1, dx)) * PULL);
    y.set(Math.max(-1, Math.min(1, dy)) * PULL);
  }
  function reset() { x.set(0); y.set(0); }

  return (
    <span className="mag" onPointerMove={move} onPointerLeave={reset}>
      {/* the anchor is the whole 40px hit area; the span only listens for the pull */}
      <motion.a ref={ref} href={href} className={className} style={{ x: sx, y: sy }}>
        {children}
      </motion.a>
    </span>
  );
}
