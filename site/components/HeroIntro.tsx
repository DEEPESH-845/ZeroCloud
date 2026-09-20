"use client";
import { gsap, ScrollTrigger, useGSAP } from "@/lib/gsap";

// One timeline: eyebrow → headline words → lede + ctas → terminal.
// Hidden states are set here, not in CSS, so the hero renders without JS.
export default function HeroIntro() {
  useGSAP(() => {
    const mm = gsap.matchMedia();
    mm.add("(prefers-reduced-motion: no-preference)", () => {
      // The term starts 24px lower during the intro, which moves the Terminal's
      // own ScrollTrigger start line; re-measure once everything is in place.
      const tl = gsap.timeline({ defaults: { ease: "power4.out" }, onComplete: () => ScrollTrigger.refresh() });
      tl.fromTo(".hero .eyebrow", { opacity: 0 }, { opacity: 1, duration: 0.4 })
        .fromTo(".hero h1 .wi", { yPercent: 110 }, { yPercent: 0, duration: 0.7, stagger: 0.04 }, "<")
        .fromTo(".hero .lede, .hero .ctas", { opacity: 0, y: 12 }, { opacity: 1, y: 0, duration: 0.5, stagger: 0.08 }, "-=0.3")
        .fromTo(".hero .term, .hero .term-foot", { opacity: 0, y: 24 }, { opacity: 1, y: 0, duration: 0.6 }, "-=0.2");
    });
    return () => mm.revert();
  });
  return null;
}
