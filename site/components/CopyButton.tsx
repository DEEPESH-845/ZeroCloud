"use client";
import { useState } from "react";

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
    <button type="button" onClick={copy} aria-label="Copy install command">
      {label}
    </button>
  );
}
