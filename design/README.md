# Predict performance — drop-in UI

A visual reference and interaction inventory for ConfigIQ's Predict performance
page. The page is now wired to the AISimulators API and its current implementation
lives in `app/predict/`.

## Files

| File | What it is |
|---|---|
| `Performance.tsx` | Main page composition (PatternFly v6 + the module CSS). |
| `Performance.module.css` | All styling — type scale, flip tiles, scenarios, constraints, drivers, memory bar. Uses `--gc-*` theme tokens with safe fallbacks. |
| `performanceHelpers.tsx` | `FlipTile`, `Sparkline`, `useCountUp`, and `Term` (the "?" glossary popovers). |
| `performance-prefill.ts` | Query-string handoff used by the embeddable widget. |
| `preview.html` | **Standalone visual reference** — open in any browser to see the exact target. No build needed. Use this as the source of truth for "does it look right". |
| `deploy-preview.html` | Same role for the Deploy page (`app/deploy/`): answer-first layout, hero metric, target chips, fit and failure states. |

## Install

1. The implementation is already in `app/predict/`.
2. Make sure these are available (they're standard in a PatternFly v6 app):
   - `@patternfly/react-core`
   - `@patternfly/react-icons`
3. Render `<Performance />` from your route. The component is `'use client'`.
4. The root layout loads Red Hat Display, Red Hat Text, and Red Hat Mono via
   `next/font/google`.

```tsx
// app/predict/page.tsx
import Performance from './Performance';
export default function Page() { return <Performance />; }
```

## Theme tokens it expects (with fallbacks)

If your `theme.css` defines these on `:root`, the page inherits them; otherwise the
fallbacks in the module render an equivalent clean look:

```
--gc-text #151515   --gc-text-2 #3c3f42   --gc-text-3 #54585c
--gc-bg #fff        --gc-bg-2 #f5f5f5     --gc-bg-3 #e0e0e0
--gc-border #d2d2d2 --gc-link #06c        --gc-brand-red #ee0000
--gc-success #3d7317  --gc-warn #f0ab00   --gc-danger #c9190b
--gc-chart-1..5  (memory bar + scenario accents)
--gc-font-display / --gc-font-sans / --gc-font-mono
```

## Design rules baked in (keep these when editing)

- **Legibility first.** No font-size below 11.5px. No text lighter than
  `--gc-text-2` for any label, value, or detail. `--gc-text-3` is reserved for
  one-word captions only.
- **Type:** Red Hat Display (titles + big numbers), Red Hat Text (body), and
  Red Hat Mono (numbers, labels, code).
- **Color:** red `#ee0000` is the logo/brand pop only. Blue `#0066cc` is the
  interactive/primary color. Status uses green/amber/red pips, used sparingly.
- **Default-first flow:** show three numbers immediately (weights, KV/req, GPUs)
  + the cost; everything else is collapsed and editable.

## Interaction inventory (the "polish" — don't lose these)

- **Flip tiles** — the 4 result tiles flip on click/Enter to reveal the formula.
  Robust opacity+rotate flip (not bare `backface-visibility`).
- **Count-up** — headline numbers animate from 0 on load (respects
  `prefers-reduced-motion`).
- **Sparkline** — GPUs-vs-concurrency line on the dark hero tile.
- **Glossary popovers** — every jargon term (`KV cache`, `max_num_seqs`,
  `tensor parallel`, `GQA`, `worst-case context`, `range drivers`, …) has a `?`
  with a plain-language explanation. See `GLOSSARY` in `performanceHelpers.tsx`.
- **Accordions** — six assumption sections, collapsed by default; **each closed
  row shows its current values** so you can read state without expanding. The
  warning strip's "Customize →" opens the Workload section.

## One number to keep honest when you wire the math

- **KV cache / request** is PER request (mock: 19 MB).
- The **KV cache scenario tiles are totals across all concurrent requests** (mock:
  87), which is why they're labeled "× 87 requests". Keep that distinction in the
  labels so users don't read a batch total as a per-request number.
