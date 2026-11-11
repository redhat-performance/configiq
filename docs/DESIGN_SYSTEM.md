# Design System Guidelines

This document defines the repeatable design patterns, component guidelines, and styling rules for ConfigIQ. Follow these standards for all new pages and components to maintain consistency.

## Typography System

### Fonts
- **Display** (headings, big numbers): "Red Hat Display"
- **Body** (descriptions, labels, UI text): "Red Hat Text"
- **Mono** (numbers in context, code, technical labels): "Red Hat Mono"

**Critical rule**: Red Hat Mono is for **numbers and technical labels only** — never for long-form text, descriptions, or explanatory copy.

### Type Scale

```css
/* Page structure */
--page-title: 26px / 500 / Red Hat Display
--card-title: 16px / 600 / Red Hat Display
--section-title: 16px / 600 / Red Hat Display

/* Metrics & data display */
--metric-large: 32px / 700 / Red Hat Display  /* Hero numbers */
--metric-medium: 26px / 700 / Red Hat Display /* Scenario values */
--metric-small: 13px / 600 / Red Hat Mono    /* Inline numbers */

/* Body text */
--body: 14px / 400 / 1.5 / Red Hat Text
--body-large: 15px / 400 / 1.5 / Red Hat Text
--caption: 11.5px / 400 / Red Hat Mono

/* Labels */
--label-mono: 12px / 500 / 0.06em / uppercase / Red Hat Mono
--label-sans: 13px / 500 / Red Hat Text

/* Minimum allowed */
--minimum: 11.5px  /* Absolute floor — nothing smaller */
```

### Hard Limits
- **No font-size below 11.5px** anywhere
- **No uppercase label lighter than #3c3f42** (--text-secondary)
- **Letter-spacing on uppercase labels ≤ 0.08em**
- All number displays: `font-variant-numeric: tabular-nums`

## Color System

### Text Colors
```css
--text-primary: #151515      /* Body copy, values, labels */
--text-secondary: #3c3f42    /* Descriptions, secondary labels */
--text-tertiary: #54585c     /* Captions only (lightest allowed) */
--text-link: #0066cc         /* Links, interactive elements */
```

**Contrast rule**: Body copy, descriptions, table/detail values, and constraint rows must be `--text-secondary` (#3c3f42) or darker. `--text-tertiary` (#54585c) is only for short uppercase captions.

### Surface Colors
```css
--bg-page: #ffffff
--bg-card: #ffffff
--bg-secondary: #f5f5f5      /* Subtle backgrounds */
--bg-tertiary: #e0e0e0       /* Disabled states */
```

### Border Colors
```css
--border-default: #d2d2d2
--border-strong: #b8bbbe
```

### Status Colors
```css
--success: #3d7317
--success-bg: color-mix(in srgb, #3d7317 12%, transparent)

--warning: #f0ab00
--warning-bg: #fdf7e7
--warning-text: #795600

--danger: #c9190b
--danger-bg: color-mix(in srgb, #c9190b 12%, transparent)

--info: #0066cc
--info-bg: color-mix(in srgb, #0066cc 12%, transparent)
```

### Chart/Data Colors
```css
--chart-1: #0066cc  /* Blue */
--chart-2: #009596  /* Cyan */
--chart-3: #5752d1  /* Purple */
--chart-4: #ec7a08  /* Orange */
```

## Spacing Scale

Use a consistent 4px-based scale:

```css
--space-1: 4px
--space-2: 8px
--space-3: 12px
--space-4: 16px
--space-5: 20px
--space-6: 24px
--space-8: 32px
--space-10: 40px
--space-12: 48px
--space-16: 64px
```

### Common Applications
- Card padding: `20px` (--space-5)
- Section margin-bottom: `24px` (--space-6)
- Grid gap: `16px` (--space-4)
- Input height: `42px`
- Border-radius (cards): `6px`
- Border-radius (buttons): `4px`

## Layout Patterns

### Card Structure
```tsx
<div className={styles.card}>
  <div className={styles.cardHead}>
    <h2 className={styles.cardTitle}>Card title</h2>
    <span className={styles.cardHint}>Optional hint text</span>
  </div>
  <div className={styles.cardBody}>
    {/* Content */}
  </div>
</div>
```

```css
.card {
  background: var(--bg-card);
  border: 1px solid var(--border-default);
  border-radius: 6px;
}

.cardHead {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 18px 20px 0;
  flex-wrap: wrap;
}

.cardTitle {
  font-size: 16px;
  font-weight: 600;
  font-family: "Red Hat Display";
}

.cardBody {
  padding: 20px;
}
```

### Grid Layouts

**Result Tiles Grid** (responsive 4-column)
```css
.tilesGrid {
  display: grid;
  grid-template-columns: 1.4fr 1fr 1fr 1fr;
  gap: 16px;
  margin-bottom: 24px;
}

@media (max-width: 1100px) {
  .tilesGrid { grid-template-columns: 1fr 1fr; }
}

@media (max-width: 600px) {
  .tilesGrid { grid-template-columns: 1fr; }
}
```

**Two-Column Split**
```css
.twoCol {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 16px;
  margin-bottom: 24px;
}

@media (max-width: 980px) {
  .twoCol { grid-template-columns: 1fr; }
}
```

**Input Row with Aligned Controls**
```css
.inputRow {
  display: grid;
  grid-template-columns: minmax(280px, 1.6fr) minmax(220px, 1fr) auto;
  grid-template-rows: auto auto auto;
  column-gap: 16px;
  row-gap: 6px;
  align-items: center;
}
```

## Component Patterns

### Flip Tiles (Interactive Result Cards)

Tiles that flip on click to reveal the math/formula behind a metric.

```tsx
<FlipTile
  dark={boolean}  // true for hero tiles with dark background
  front={<>/* Display value */}</>}
  back={<>/* Formula/explanation */}</>}
/>
```

**Front face structure:**
```tsx
<>
  <span className={styles.tileLabel}>
    <Icon /> Label text <Term k="glossaryKey" />
  </span>
  <span className={styles.tileValue}>
    {animatedNumber}
    <span className={styles.tileUnit}>unit</span>
  </span>
  <span className={styles.tileSub}>Supporting detail</span>
</>
```

**Back face structure:**
```tsx
<>
  <div className={styles.backTitle}>HOW WE GOT {value}</div>
  <div className={styles.formula}>
    step 1 = <span className={styles.em}>value</span><br />
    step 2 = <span className={styles.em}>value</span><br />
    result = <span className={styles.em}>final</span>
  </div>
</>
```

### Search with Live Filtering

```tsx
const [searchQuery, setSearchQuery] = useState('');
const [debouncedQuery, setDebouncedQuery] = useState('');

// Debounce search input
useEffect(() => {
  const timer = setTimeout(() => setDebouncedQuery(searchQuery), 120);
  return () => clearTimeout(timer);
}, [searchQuery]);

const matchesSearch = (keywords: string) => {
  if (!debouncedQuery.trim()) return true;
  return keywords.toLowerCase().includes(debouncedQuery.toLowerCase());
};

// Wrap sections with dimming
<div className={!matchesSearch('keywords for this section') ? styles.dimmed : ''}>
  {/* Content */}
</div>
```

```css
.dimmed {
  opacity: 0.35;
  transition: opacity 0.2s ease;
}

@media (prefers-reduced-motion: reduce) {
  .dimmed { transition: none; }
}
```

### Accordions with Inline Summaries

```tsx
<Accordion>
  <AccordionItem>
    <AccordionToggle onClick={() => toggle('id')} isExpanded={expanded.includes('id')}>
      <div className={styles.assumptionsHead}>
        <h3 className={styles.assumptionsTitle}>Section Title</h3>
        {!expanded.includes('id') && (
          <div className={styles.accSummary}>
            <span><span className={styles.k}>Key:</span> value</span>
            <span><span className={styles.k}>Key:</span> value</span>
          </div>
        )}
      </div>
    </AccordionToggle>
    <AccordionContent>
      {/* Detailed controls */}
    </AccordionContent>
  </AccordionItem>
</Accordion>
```

**Rule**: Closed accordions MUST show current values inline so users can read state without expanding.

### Warning/Info Strips

```tsx
<div className={styles.warn}>
  <ExclamationTriangleIcon style={{ color: 'var(--warning)', flexShrink: 0 }} />
  <span>Message text with context.</span>
  <button className={styles.warnLink} onClick={action}>Action →</button>
</div>
```

```css
.warn {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 12px 16px;
  background: var(--warning-bg);
  border: 1px solid var(--warning);
  border-left-width: 4px;
  border-radius: 6px;
  font-size: 14px;
}
```

### Status Pills/Badges

```tsx
<span className={`${styles.conPill} ${styles.pillOk}`}>OK</span>
<span className={`${styles.conPill} ${styles.pillWatch}`}>WATCH</span>
<span className={`${styles.conPill} ${styles.pillBottleneck}`}>BOTTLENECK</span>
```

```css
.conPill {
  font-family: "Red Hat Mono";
  font-size: 11.5px;
  font-weight: 600;
  letter-spacing: 0.05em;
  text-transform: uppercase;
  padding: 3px 10px;
  border-radius: 10px;
}

.pillOk {
  color: var(--success);
  background: var(--success-bg);
}
```

### Glossary Popovers (Term Tooltips)

```tsx
import { Term } from './helpers';

<label>
  KV cache / request <Term k="kvPerReq" />
</label>
```

Every technical term should have a popover explanation. Add new terms to `GLOSSARY` in helpers file.

## Animation & Transitions

### Count-Up Animation
```tsx
const animatedValue = useCountUp(targetValue, 750, decimals);
```

Animate headline numbers from 0 → target on mount. Must respect `prefers-reduced-motion`.

### Flip Transitions
```css
.flipFace {
  transition: opacity 0.3s ease, transform 0.6s cubic-bezier(0.4, 0, 0.2, 1);
  backface-visibility: hidden;
}

@media (prefers-reduced-motion: reduce) {
  .flipFace { transition: none; }
}
```

### Smooth Transitions
- **Fast**: 0.2s (hover states, dimming)
- **Medium**: 0.3s (opacity changes)
- **Slow**: 0.6s (transforms, flips)
- **Easing**: `ease` for simple, `cubic-bezier(0.4, 0, 0.2, 1)` for complex

**Always provide reduced-motion fallback:**
```css
@media (prefers-reduced-motion: reduce) {
  .animated { transition: none; animation: none; }
}
```

## Product Tour Component

Reusable tour for onboarding users. Lives in `/components/ProductTour/`.

```tsx
import { ProductTour, type TourStep } from '@/components/ProductTour';

const TOUR_STEPS: TourStep[] = [
  {
    target: '[data-tour="selector"]',  // CSS selector
    title: 'Step title',
    description: 'Explanation of this element.',
    position: 'bottom' | 'top' | 'left' | 'right'
  },
  // ...
];

// In component:
const [showTour, setShowTour] = useState(false);
const [tourSeen, setTourSeen] = useState(false);

useEffect(() => {
  const seen = localStorage.getItem('page-tour-seen');
  if (seen) {
    setTourSeen(true);
  } else {
    setTimeout(() => setShowTour(true), 1000);
  }
}, []);

const handleTourComplete = () => {
  setShowTour(false);
  setTourSeen(true);
  localStorage.setItem('page-tour-seen', 'true');
};

return (
  <>
    {showTour && (
      <ProductTour
        steps={TOUR_STEPS}
        tourId="page-id"
        onComplete={handleTourComplete}
      />
    )}
    
    {/* Tour trigger button */}
    <div style={{ position: 'relative' }}>
      <Button variant="link" onClick={() => setShowTour(true)}>
        Take a tour
      </Button>
      {!tourSeen && <div className={styles.tourBeacon} />}
    </div>
  </>
);
```

**Tour beacon CSS** (periodic blinking attention-getter):
```css
.tourBeacon {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  width: 100%;
  height: 100%;
  pointer-events: none;
}

.tourBeacon::before {
  content: '';
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: #0066cc;
  animation: beaconBlinkPeriodic 3s ease-in-out infinite;
}

.tourBeacon::after {
  content: '';
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  width: 28px;
  height: 28px;
  border-radius: 50%;
  border: 2px solid #0066cc;
  animation: beaconBlinkPeriodic 3s ease-in-out infinite;
}

@keyframes beaconBlinkPeriodic {
  0% { opacity: 1; transform: translate(-50%, -50%) scale(1); }
  5% { opacity: 0.15; transform: translate(-50%, -50%) scale(0.8); }
  10% { opacity: 1; transform: translate(-50%, -50%) scale(1); }
  15% { opacity: 0.15; transform: translate(-50%, -50%) scale(0.8); }
  20% { opacity: 1; transform: translate(-50%, -50%) scale(1); }
  25% { opacity: 0.15; transform: translate(-50%, -50%) scale(0.8); }
  30%, 100% { opacity: 1; transform: translate(-50%, -50%) scale(1); }
}

@media (prefers-reduced-motion: reduce) {
  .tourBeacon::before,
  .tourBeacon::after {
    animation: none;
  }
}
```

## Accessibility Requirements

### Keyboard Navigation
- All interactive elements must be keyboard accessible
- Flip tiles: flip on click and through their keyboard-operable chart/table button
- Tour: closes on Esc key
- Proper `tabIndex` on custom interactive elements

### Screen Readers
- All icons need `aria-label` when standalone
- Form inputs must have associated labels
- Status messages should use `role="status"` or `aria-live`

### Reduced Motion
- **Every animation and transition** must have a `prefers-reduced-motion` fallback
- Fallback = `transition: none; animation: none;`
- Applies to: flips, count-ups, pulses, fades, transforms

### Focus Styles
```css
.interactive:focus-visible {
  outline: 2px solid var(--text-link);
  outline-offset: 2px;
  border-radius: 4px;
}
```

## Form Patterns

### Input Styling (Consistency)
All text inputs should match:
```css
.input {
  width: 100%;
  height: 42px;
  border: 1px solid var(--border-strong);
  border-radius: 4px;
  padding: 7px 11px;
  font-size: 14px;
  font-family: "Red Hat Text";
  background: var(--bg-page);
  box-sizing: border-box;
}

.input:focus {
  border: 2px solid var(--text-link);
  outline: none;
  padding: 6px 10px; /* Adjust for 2px border */
}
```

### Field Labels
```tsx
<label className={styles.fieldLabel} htmlFor="input-id">
  Label Text
  <InfoCircleIcon style={{ width: 12, height: 12, opacity: 0.7 }} />
</label>
```

```css
.fieldLabel {
  font-family: "Red Hat Mono";
  font-size: 11.5px;
  font-weight: 600;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--text-secondary);
  display: flex;
  align-items: center;
  gap: 6px;
}
```

## Usage Guidelines

### When to Use Each Font
- **Red Hat Display**: Headings and prominent metric values
- **Red Hat Text**: UI text, labels, descriptions, and body copy
- **Red Hat Mono**: Numbers in tables/metrics, code snippets, and technical uppercase labels ONLY

### When to Use Each Component
- **Flip Tile**: Any metric where users might want to see "how it's calculated"
- **Search**: Pages with 4+ sections or information-dense content
- **Accordion**: Grouped settings, optional details, progressive disclosure
- **Tour**: First-time onboarding for pages with 3+ distinct UI areas
- **Warning Strip**: Default assumptions, important context, actionable notices

### Color Usage
- **Blue (#0066cc)**: Primary actions, links, selected states, tour highlights
- **Never red**: No red buttons or red primary actions (red is logo-only)
- **Status colors**: Only for status pills, constraint indicators, success/warning messages

## File Organization

```
app/
  [page]/
    page.tsx              # Main component
    [Page].module.css     # Page-specific styles
    [page]Helpers.tsx     # Helper components (FlipTile, Term, etc.)

components/
  ProductTour/            # Reusable tour component
    ProductTour.tsx
    ProductTour.module.css
    index.ts
```

## CSS Module Patterns

### Variable Definitions
```css
.page {
  /* Token fallbacks */
  --t: var(--gc-text, #151515);
  --t2: var(--gc-text-2, #3c3f42);
  --t3: var(--gc-text-3, #54585c);
  --bg: var(--gc-bg, #ffffff);
  --blue: var(--gc-link, #0066cc);
  
  /* Fonts */
  --display: "Red Hat Display", system-ui, sans-serif;
  --sans: "Red Hat Text", system-ui, sans-serif;
  --mono: "Red Hat Mono", ui-monospace, monospace;
  
  /* Shared values */
  --radius: 6px;
}
```

### Section Comments
```css
/* ---------- section name ---------- */
.selector { }
```

Use ASCII comment dividers to separate major sections in CSS files.

## Testing Checklist

Before shipping a new page/component:

- [ ] All text is readable at stated sizes (nothing below 11.5px)
- [ ] All interactive elements work with keyboard
- [ ] Tour (if present) highlights correct elements
- [ ] All animations have reduced-motion fallbacks
- [ ] Colors meet contrast requirements (WCAG AA minimum)
- [ ] Layout is responsive (test 1920px, 1280px, 768px)
- [ ] Flip tiles flip correctly on click and through their chart/table button
- [ ] Search (if present) dims non-matching sections
- [ ] Form inputs are 42px tall and styled consistently
- [ ] Numbers use tabular-nums
- [ ] Icons have appropriate aria-labels

---

**Summary**: This design system prioritizes readability, consistency, and accessibility. Every pattern here has been battle-tested in the Performance page. Use these guidelines as the foundation for all new pages — don't invent new patterns unless there's a clear gap.
