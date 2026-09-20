"use client";
import { useState } from "react";
import { motion, AnimatePresence } from "motion/react";

const MICRO = 0.16; // mirrors --dur-micro

export default function CopyButton({ text }: { text: string }) {
  const [label, setLabel] = useState("Copy");
  async function copy() {
    try {
      await navigator.clipboard.writeText(text);
      setLabel("Copied");
    } catch {
      setLabel("Select & copy");
    }
    setTimeout(() => setLabel("Copy"), 1600);
  }
  return (
    <motion.button type="button" onClick={copy} whileTap={{ scale: 0.97 }}>
      <span className="sr-only" aria-live="polite">{label === "Copy" ? "Copy install command" : label}</span>
      <AnimatePresence mode="popLayout" initial={false}>
        <motion.span
          aria-hidden="true"
          key={label}
          initial={{ y: 6, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          exit={{ y: -6, opacity: 0 }}
          transition={{ duration: MICRO, ease: [0.22, 1, 0.36, 1] }}
        >
          {label}
        </motion.span>
      </AnimatePresence>
    </motion.button>
  );
}
