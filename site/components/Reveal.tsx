"use client";
import { gsap, useGSAP } from "@/lib/gsap";

// Animates every .reveal on the page. Initial state is set here, not in CSS,
// so content stays visible if JS never runs.
export default function Reveal() {
  useGSAP(() => {
    const mm = gsap.matchMedia();
    mm.add("(prefers-reduced-motion: no-preference)", () => {
      gsap.utils.toArray<HTMLElement>(".reveal").forEach((el) => {
        gsap.fromTo(
          el,
          { opacity: 0, y: 14 },
          { opacity: 1, y: 0, duration: 0.6, ease: "power2.out", scrollTrigger: { trigger: el, start: "top 92%", once: true } },
        );
      });
    });
    return () => mm.revert();
  });
  return null;
}
