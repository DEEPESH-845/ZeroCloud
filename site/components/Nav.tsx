"use client";
import { useState } from "react";
import { motion, useScroll, useMotionValueEvent } from "motion/react";
import { useActiveSection } from "./useActiveSection";

// Page order: why → the arithmetic → the rule → what → accuracy
const LINKS = [
  ["why", "Why measure"],
  ["budget", "The arithmetic"],
  ["what", "What it does"],
  ["accuracy", "Accuracy"],
] as const;
const IDS = LINKS.map(([id]) => id);

export default function Nav() {
  const active = useActiveSection(IDS);
  const [condensed, setCondensed] = useState(false);
  const { scrollY } = useScroll();
  useMotionValueEvent(scrollY, "change", (y) => setCondensed(y > 80));

  return (
    <nav aria-label="Primary" className={condensed ? "condensed" : undefined}>
      <div className="wrap">
        <a className="brand" href="#top">
          zc <span>/ ZeroCloud</span>
        </a>
        <ul>
          {LINKS.map(([id, label]) => (
            <li key={id}>
              <a href={`#${id}`} aria-current={active === id ? "location" : undefined}>{label}</a>
              {active === id && <motion.span className="nav-active" layoutId="nav-active" />}
            </li>
          ))}
          <li><a href="https://github.com/DEEPESH-845/ZeroCloud">GitHub ↗</a></li>
        </ul>
      </div>
    </nav>
  );
}
