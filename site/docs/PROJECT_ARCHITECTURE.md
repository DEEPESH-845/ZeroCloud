# PROJECT_ARCHITECTURE — site/

Recon for the 3D + motion upgrade. The Rust workspace at the repo root is a
separate product and is never touched by site work.

## Stack (as installed)
- Next.js 16.3 App Router, React 19.2, TypeScript, npm. No Tailwind. No ESLint.
- Fonts: `next/font/google` — Instrument Sans (`--font-sans`), JetBrains Mono (`--font-mono`).
- CSS: one global stylesheet `app/globals.css` with design tokens on `:root`
  (`--paper --ink --ink-2 --ink-3 --rule --term --term-ink --term-dim --ok --amber --link`).
  New per-component styles go in CSS Modules; tokens stay in globals.
- Animation: `gsap` 3.15 + `@gsap/react` (registered once in `lib/gsap.ts`: `gsap`, `ScrollTrigger`, `useGSAP`).
- 3D: `three` 0.182 (pinned below r183, where `Clock` is deprecated and R3F 9.7 still uses it), `@react-three/fiber` 9.7, `@react-three/drei` 10.7, `@types/three`.
- UI motion: `motion` 13 (`import { motion } from "motion/react"`).
- Deploy: Vercel, Root Directory `site`, no env vars, `vercel.json` sets framework.

## Page composition (`app/page.tsx`)
`Nav` → `Hero` (eyebrow, h1, lede, install CopyButton, `Terminal`) → `Sections`
(`Reveal` mount, `#why`, `#rule`, `#what`, `#accuracy`, `.final`, footer).

## Reusable pieces
- `components/Terminal.tsx` — GSAP timeline replaying a real `zc check` capture. Owns its own timeline; do not re-animate its children.
- `components/Reveal.tsx` — animates every `.reveal` element on scroll (GSAP, matchMedia for reduced motion). Add `.reveal` to any block to get entrance motion; do not write a second reveal system.
- `components/CopyButton.tsx` — clipboard with fallback label.
- CSS classes in globals: `.wrap .eyebrow .lede .two .sticky .cap .formula .stats .stat .note .steps .install .facts .term*`.

## Ownership model for animation (no property animated by two systems)
- **GSAP**: scroll timelines, pins/scrubs, cross-element choreography, camera/scene progress, number count-ups, the terminal replay.
- **motion/react**: component-level state motion — hover, press, layout indicators, label swaps, magnetic offsets.
- **R3F**: everything inside the single `<Canvas>`. Scene reads a progress ref that GSAP writes; it never sets React state per frame.

## Safe integration points
- New 3D section: `components/budget/*` mounted from `page.tsx` between `#why` and `#rule` (the narrative slot "how the arithmetic works").
- Nav/buttons: replace internals of `Nav.tsx`, `CopyButton.tsx`; keep class names so globals.css still applies.
- Motion tokens: add `--dur-micro/--dur-standard/--dur-dramatic/--ease-out/--ease-in-out` to `:root` in globals.css.

## Risky areas / rules
- `Canvas` must be client-only (`next/dynamic`, `ssr:false`) inside a box with a reserved height (no CLS).
- Pinned ScrollTriggers must be created before non-pinned ones or given `refreshPriority`; call `ScrollTrigger.refresh()` after the 3D section mounts.
- One WebGL canvas on the page. No postprocessing, no HDRI/network assets (drei `Environment` presets fetch from a CDN — do not use).
- No drei `Text` (fetches a font). Labels via drei `Html` or DOM.
- Every number shown must come from README captures or `crates/zc-model/data/models/*.json`.
- `prefers-reduced-motion`: pins/scrubs off, scene renders final state, everything remains readable.

## Do not rewrite
`Terminal.tsx` timeline logic, `Reveal.tsx`, `lib/gsap.ts`, `app/layout.tsx` fonts/metadata.
