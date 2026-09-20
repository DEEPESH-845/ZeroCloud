# PREMIUM_COMPONENT_SHORTLIST — site/

Research date 2026-09-13. Scores /10: Rel (product relevance) · Vis · Int · Cust · Perf · Orig; VALUE = Σ six − Diff. Threshold ≥ 45.
Rule for every row: **source-code-first** — read the source, rewrite as our own component with our tokens. Only motion deps are `gsap` (3.15, `lib/gsap.ts`) and `motion/react`. No new npm packages. Ownership per PROJECT_ARCHITECTURE: GSAP = scroll/timelines/count-ups, motion/react = hover/press/layout, R3F = inside the one Canvas.

Sites that yielded nothing adoptable: **21st.dev** (registry, per-author licence unclear, metered copies — do not lift code), **skiper-ui** (Pro; only Dynamic Island free, rejected below), **originui** (MIT, no motion; now coss/ui with AGPL dirs — irrelevant here), **cult-ui** (MIT; only AnimatedNumber relevant, folded into row 4). `motion-plus` (`splitText`, `AnimateNumber`, `Cursor`, `Ticker`) is paid — not used, not needed.

## Shortlist (VALUE ≥ 45)

| # | Pattern · source · licence | Deps | Lives in | Rel Vis Int Cust Perf Orig − Diff = VALUE | What changes to fit the datasheet |
|---|---|---|---|---|---|
| 1 | **Line-mask text reveal** — GSAP `SplitText` (`gsap.com/docs/v3/Plugins/SplitText`, free in gsap ≥3.13, GSAP Standard licence); Codrops "7 GSAP tips" #1 (`mask:"lines"`); cf. motion-primitives TextEffect (MIT), reactbits SplitText (MIT+CC) | gsap only (already installed) | Hero h1, section h2s | 9 9 8 9 9 6 − 2 = **48** | `type:"lines"`, `mask:"lines"`, `autoSplit:true`; `yPercent:110→0`, 800ms `--ease-out`, stagger 60ms; no blur, no scale, no per-char; `aria:"auto"`; reduced-motion → static |
| 2 | **Scroll-pinned instrument section** — GSAP ScrollTrigger pin+scrub (`gsap.com` docs); layout from Aceternity Sticky Scroll Reveal (MIT free tier); Codrops "Scroll-Driven 3D Gallery" (MIT demo) | gsap + R3F (installed) | `#budget` (between `#why` and `#rule`) | 10 9 9 9 7 7 − 6 = **45** | GSAP writes a `progress` ref, scene reads it (never setState/frame); `snap` to stage centres; `scrub:0.6`; pinned box has reserved height; `.two`/`.sticky` grid not a full-bleed hero; reduced-motion → final stage, no pin |
| 3 | **Technical/isometric 3D product drawing** — drei `OrthographicCamera` `Edges` `Line` `Html` `ContactShadows` (github.com/pmndrs/drei, MIT); Codrops "Sketching the Impossible" (procedural geometry, no models) | three/R3F/drei (installed) | `#budget` canvas | 9 9 8 8 8 9 − 6 = **45** | Flat ink/paper materials + 1px ink `Edges`; ortho camera, fixed isometric tilt, ≤3° pointer parallax; dimension lines via `Line` + `Html` labels in JetBrains Mono; `frameloop:"demand"`; no Environment/Text/HDRI/postfx |
| 4 | **Count-up number** — motion-primitives AnimatedNumber (MIT), magicui NumberTicker (MIT), reactbits CountUp (MIT+CC), cult-ui AnimatedNumber (MIT) | gsap (ownership rule) | `#accuracy` `.stat b` | 9 8 7 9 9 4 − 1 = **45** | Tween a plain object with GSAP, `power2.out`, 800ms, `once:true`; **no spring** (a spring overshoots 9.4% → reads as fabricated); `font-variant-numeric: tabular-nums`; SSR renders final value; every value from README captures |
| 5 | **Nav active indicator** — motion `layoutId` shared-layout (`motion.dev/docs/react-layout-animations`, MIT) + IntersectionObserver | motion/react | Nav | 8 7 8 9 9 5 − 1 = **45** | 1px `--ink` hairline under the link (not a pill/background), `layout` transition 360ms `--ease-out`; `aria-current="location"`; IO band, not scroll math; hide on ≤640px |
| 6 | **Terminal replay** — magicui Terminal (MIT; `sequence`, `startOnView`), Aceternity Typewriter (MIT), motion.dev Typewriter (free) | gsap | Hero `Terminal.tsx` — **do not rewrite** | 10 8 7 8 9 4 − 1 = **45** | Ours already replays a real capture, which beats a typewriter. Only borrow: start-on-view, replay control, reduced-motion → `pre.done`. No per-char typing of output (fake), no cursor on output lines |

Near misses (<45, not adopted on merit): Magnetic button — motion-primitives Magnetic / reactbits Magnet (MIT) 7 7 8 8 9 5 − 1 = **43**; already in tree as `MagneticButton.tsx` at 6px pull — acceptable to keep because it is nearly free, but cap pull ≤6px, mouse only, and cut it if the CTA ever reads as playful. Sliding/odometer number (motion-primitives) 37. Scroll-progress bar 36.

## How to adapt — top 5

**1. Line-mask headline (Hero).** `HeroIntro.tsx` already does a manual word split with `yPercent:110`. Replace the hand split with `SplitText.create(".hero h1", { type:"lines", mask:"lines", autoSplit:true, aria:"auto", onSplit: self => gsap.from(self.lines, { yPercent:110, duration:.8, ease:"power4.out", stagger:.06 }) })`. Lines, not words: a datasheet headline reads as sentences, and `autoSplit` fixes the re-wrap on font load/resize that a manual split cannot. Run inside the existing `matchMedia("(prefers-reduced-motion: no-preference)")`; call `revert()` in cleanup. Reuse for section h2s via `.reveal`? No — `Reveal.tsx` owns those; one system per element.

**2. Pinned instrument (#budget).** Keep the Aceternity layout (left: stage list; right: sticky instrument) but not its `useScroll` mapping — GSAP pins the section, `scrub:0.6`, `snap:{ snapTo: CENTERS, duration:.36 }`, and `onUpdate` writes `progress.v`; `BudgetScene` reads it in `useFrame`. Create this ScrollTrigger before `Reveal`'s (or give `refreshPriority:1`) and call `ScrollTrigger.refresh()` after the canvas mounts. Stage readout under the canvas swaps with the same 160ms label-swap already used in `CopyButton`. Reduced motion: no pin, render stage 4, list is plain.

**3. Isometric drawing (#budget canvas).** Materials are `meshBasicMaterial` in `--paper`/`--ink`/`--ok`/`--amber` with drei `<Edges color="#13161B" />` — the object should look printed, not lit. Ortho camera, fixed tilt; pointer parallax ≤3° via a ref, spring-smoothed in `useFrame`. Dimension callouts: drei `<Line>` ticks + `<Html transform={false}>` labels in `--mono` 12px (`Text` fetches a font — banned). `frameloop="demand"`, `invalidate()` on progress/pointer change; `ContactShadows` at ≤0.15 opacity is the only softness. No sweep shader brighter than 0.12 alpha.

**4. Count-up (#accuracy).** `CountUp.tsx` is already the GSAP-tween shape of motion-primitives' `useSpring`+`useTransform` — keep it, change duration to 0.8 (dramatic token) and add `font-variant-numeric: tabular-nums` on `.stat b` so width does not jitter. Never spring: overshoot on an accuracy figure is a credibility bug. `toFixed(decimals)` only; suffix stays static in `<small>`.

**5. Nav indicator (Nav).** `Nav.tsx` already has `layoutId="nav-active"` + `useActiveSection`. Style `.nav-active` as a 1px `--ink` hairline at the link's baseline, `transition={{ layout:{ duration:.36, ease:[.22,1,.36,1] } }}`; no background pill. Under reduced motion pass `transition={{ duration:0 }}`. Keep the 40% IO band; it beats scroll-math.

## REJECTED — would make this page read as generic AI/SaaS output

- **Aurora / wavy / beams / lamp backgrounds** (Aceternity) — gradient atmosphere on a page whose whole argument is "no atmosphere, measured numbers".
- **Spotlight / Magic Card / Card Spotlight / border-beam / shine-border / hover-border-gradient / moving-border** — blur-xl radial glows and animated gradient borders; the palette has no glow and the cards are ruled tables. Also GPU-costly.
- **Bento grids** — the page's IA is a datasheet's stacked fields and one table, not a tile mosaic.
- **Particle fields / sparkles / gravity mouse trails / image-cursor-trail** (skiper Pro) — decoration with zero information; cursor trails are irrelevant for a CLI benchmark.
- **Dynamic island nav** (cult-ui MIT, skiper free) — an iOS device metaphor; a morphing pill header contradicts a fixed spec-sheet header. Irrelevant.
- **Container Scroll / MacBook Scroll** — 3D-rotating a raster screenshot; the real instrument here is a live terminal and a vector 3D drawing.
- **Blur-in / scale-in text, per-character scatter/wavy, text scramble / decrypted text, flip-words / word-rotate, hyper-text** — the headline is one fixed question; blur filters on text are expensive and "hacker" scrambling is the opposite of understated.
- **Ticker / marquee / logo wall, animated-shiny-text, gradient headings, 3D tilt cards, evervault card** — SaaS social proof and gloss.
- **Tracing beam / timeline beam / scroll-progress bar** — ornament on lists that are already numbered.
- **Second reveal system** (motion `whileInView`) — `Reveal.tsx` exists; two systems fighting one property is the bug class the ownership model forbids.
