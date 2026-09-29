"use client";
import { useEffect, useState } from "react";
import Link from "next/link";
import CopyButton from "@/components/CopyButton";
import { INSTALL, Wrappable } from "@/components/Hero";
import { ctxLabel, decodeCard, gib, type Decoded, type Verdict } from "@/lib/card";

const GH = "https://github.com/DEEPESH-845/ZeroCloud";

// The words `zc check` prints, so a card and a terminal read the same.
const VERDICT: Record<Verdict, { mark: string; label: string }> = {
  good: { mark: "OK", label: "runs well" },
  usable: { mark: "ok", label: "usable" },
  slow: { mark: "SLOW", label: "slow" },
  wont_fit: { mark: "XX", label: "won't fit" },
};

const BACKEND: Record<string, string> = { cpu: "CPU", metal: "Metal", discrete: "GPU" };

function useHash(): string | null {
  // null until mounted: the fragment exists only in the browser, so the
  // static HTML renders a neutral state rather than a wrong one.
  const [hash, setHash] = useState<string | null>(null);
  useEffect(() => {
    const read = () => setHash(window.location.hash);
    read();
    window.addEventListener("hashchange", read);
    return () => window.removeEventListener("hashchange", read);
  }, []);
  return hash;
}

export default function CardView() {
  const hash = useHash();
  const decoded: Decoded | null = hash === null ? null : decodeCard(hash);

  return (
    <>
      <header className="card-top">
        <div className="wrap">
          <Link className="brand" href="/">zc <span>/ ZeroCloud</span></Link>
          <a href={GH}>GitHub ↗</a>
        </div>
      </header>
      <main className="wrap card-page">
        {decoded === null ? (
          <p className="eyebrow">Reading the card…</p>
        ) : decoded.ok ? (
          <Sheet card={decoded.card} />
        ) : (
          <Empty reason={decoded.reason} />
        )}
        <section className="card-cta" aria-labelledby="yours">
          <h2 id="yours">Measure yours</h2>
          <p className="lede">Two seconds of benchmarks, then a prediction for every model in the catalog. No account and no upload. <code>zc check --card</code> makes a link like this one.</p>
          <div className="install" style={{ marginTop: 20 }}>
            <code><Wrappable text={INSTALL} /></code>
            <CopyButton text={INSTALL} label="Copy install command" />
          </div>
        </section>
      </main>
    </>
  );
}

function Sheet({ card }: { card: import("@/lib/card").Card }) {
  const { hw } = card;
  const top = Math.max(...card.rows.map((r) => r.high ?? 0), 1);
  return (
    <article aria-labelledby="card-title">
      <p className="eyebrow">Result card · measured by zc {card.zc}</p>
      <h1 id="card-title" className="card-h">
        {hw.cpu}, {gib(hw.ram)}
      </h1>
      <p className="facts">
        <span>{hw.cores}</span>
        {hw.gpu && <span>{hw.gpu}</span>}
        <span>{BACKEND[hw.backend] ?? hw.backend} backend</span>
        {hw.bwGbs !== null && <span>{Math.round(hw.bwGbs)} GB/s measured</span>}
        <span>{gib(hw.budget)} budget, idle</span>
        <span>KV {card.kv.toUpperCase()}</span>
        <span>{card.calibrated ? "calibrated" : "uncalibrated priors"}</span>
      </p>

      <div className="card-table">
        <table>
          <caption className="sr-only">Predicted decode speed per model on this machine</caption>
          <thead>
            <tr><th scope="col">model</th><th scope="col">verdict</th><th scope="col">decode tok/s</th><th scope="col">context</th><th scope="col">confidence</th></tr>
          </thead>
          <tbody>
            {card.rows.map((r) => (
              <tr key={`${r.id}/${r.quant}`}>
                <td>{r.id}<small>{r.quant}</small></td>
                <td><b className={`v v-${r.verdict}`}>{VERDICT[r.verdict].mark}</b> {VERDICT[r.verdict].label}</td>
                <td>
                  {r.low !== null && r.high !== null ? (
                    <>
                      <span className="range-n">{r.low.toFixed(1)}–{r.high.toFixed(1)}</span>
                      <span className="range" aria-hidden="true">
                        <i style={{ left: `${(r.low / top) * 100}%`, width: `${Math.max(((r.high - r.low) / top) * 100, 1)}%` }} />
                      </span>
                    </>
                  ) : "–"}
                </td>
                <td>{ctxLabel(r.ctx)}</td>
                <td>{r.confidence}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="cap-label">
        Predictions, not benchmarks of the models: decode speed as a range, from memory bandwidth measured on this
        machine, with the confidence that calibration data behind it earns. <code>zc verify</code> runs a model for
        real and checks it.
      </p>
      <p className="card-privacy">
        This page read everything above from the link itself, after the <code>#</code>. Browsers never send that part
        to a server, so nothing about this machine was uploaded to show it.
      </p>
    </article>
  );
}

function Empty({ reason }: { reason: "empty" | "version" | "malformed" }) {
  const text = {
    empty: "This link has no card in it. A card link ends in #v1. followed by the result, and zc check --card prints one.",
    version: "This card was written in a format this page does not know — probably by a newer zc. Updating the page is on us, not you.",
    malformed: "This card link is damaged: it may have been cut off when it was pasted. Ask for the whole link, or run zc check --card again.",
  }[reason];
  return (
    <div>
      <p className="eyebrow">Result card</p>
      <h1 className="card-h">{reason === "empty" ? "No card here yet" : "Can't read this card"}</h1>
      <p className="lede" style={{ marginTop: 16 }}>{text}</p>
    </div>
  );
}
