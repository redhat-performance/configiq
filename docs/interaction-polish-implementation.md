# Interaction & Polish - Implementation Guide

## ✅ Components Created

### 1. **FlipCard Component** (`components/ui/FlipCard.tsx`)
- Robust CSS transform flip animation
- Uses `opacity` + `rotateY` (not bare backface-visibility)
- 600ms cubic-bezier transition
- Click to flip front/back
- Prevents flickering with opacity transitions

### 2. **useCountUp Hook** (`hooks/useCountUp.ts`)
- Animates numbers from 0 to target value
- Ease-out cubic timing function
- 1500ms duration (customizable)
- requestAnimationFrame for smooth 60fps

### 3. **JargonPopover Component** (`components/ui/JargonPopover.tsx`)
- PatternFly Popover with "?" HelpIcon
- Pre-defined explanations for all jargon terms
- 360px max-width for readability
- Auto-positioning

**Jargon terms covered:**
- KV cache
- max_num_seqs
- prefills/step
- tensor parallel (TP)
- active-request ratio
- range drivers
- GQA/MHA/MQA
- worst-case context
- ISL/OSL
- BF16/FP8
- vLLM
- chunked prefill
- prefix caching

---

## 📋 Implementation Checklist

### **Theme & Typography** ✅
- [x] Import `app/theme.css` in `app/layout.tsx`
- [x] Replace all inline colors with CSS custom properties:
  - `var(--gc-text)` for primary
  - `var(--gc-text-2)` for body/details
  - `var(--gc-text-3)` for captions only
  - `var(--gc-link)` for links
  - `var(--gc-brand-red)` for logo ONLY
- [x] Use utility classes: `.gc-page-title`, `.gc-card-title`, `.gc-metric`, `.gc-body`, `.gc-detail`, `.gc-label`
- [x] Enforce minimum 11.5px font-size (`.gc-caption`)
- [x] Add `font-variant-numeric: tabular-nums` to all numbers

### **Flip Card Animation** 🔲
- [ ] Wrap result cards in `<FlipCard>` component
- [ ] Front: Display metric value
- [ ] Back: Show formula/calculation
- [ ] Example:
  ```tsx
  <FlipCard
    front={<MetricDisplay value={16} unit="GB" label="Weight Memory" />}
    back={<Formula>8B params × 2 bytes (BF16)</Formula>}
  />
  ```

### **Count-Up Animation** 🔲
- [ ] Use `useCountUp()` hook on all big numbers
- [ ] Trigger on page load / calculate button click
- [ ] Example:
  ```tsx
  const gpuCount = useCountUp(result.gpuCount, 1500);
  return <span className="gc-metric">{gpuCount}</span>;
  ```

### **Accordion Animation** 🔲
- [ ] Use PatternFly `<Accordion>` component
- [ ] Sections to make collapsible:
  - "Why this GPU count?" constraints table
  - KV cache scenarios
  - Range drivers
- [ ] Default: "Why this GPU count?" expanded, others collapsed
- [ ] Add smooth expand/collapse transition

### **Jargon Popovers** 🔲
Add `<JargonPopover>` next to every jargon term:

**Result cards:**
- [ ] "KV cache / request" 
- [ ] "Tensor parallel"

**Constraints section:**
- [ ] "max_num_seqs"
- [ ] "prefills/step"
- [ ] "tensor parallel minimum"

**KV scenarios:**
- [ ] "worst-case context"

**Other:**
- [ ] "GQA" badge
- [ ] "range drivers" heading
- [ ] "ISL" / "OSL" in workload banner
- [ ] "BF16" / "FP8" in precision descriptions
- [ ] "chunked prefill" in vLLM config
- [ ] "prefix caching" in optimization suggestions

Example usage:
```tsx
<span>
  KV cache <JargonPopover term="KV cache" explanation={JARGON["KV cache"]} />
</span>
```

### **Spacing & Hierarchy** 🔲
Per theme.css guidelines:

**Generous spacing:**
- [ ] Page margins: `3rem` (48px) on desktop
- [ ] Section gaps: `2.5rem` (40px) between major sections
- [ ] Card padding: `2rem` (32px) inside cards
- [ ] Grid gaps: `1.5rem` (24px) between cards
- [ ] Line height: `1.5` for body text, `1.2` for numbers

**Clear hierarchy:**
- [ ] Page title: 26px Display 500
- [ ] Section titles: 16px Display 600
- [ ] Hero number: 80px Display 700 (dark card)
- [ ] Supporting numbers: 32px Display 700 (light cards)
- [ ] Body text: 14px Sans (var(--gc-text-2))
- [ ] Details: 13px Mono (var(--gc-text-2))

**High contrast:**
- [ ] No text lighter than `var(--gc-text-3)` (#54585c)
- [ ] Body text uses `var(--gc-text-2)` (#3c3f42)
- [ ] Links use `var(--gc-link)` (#0066cc)
- [ ] Success badges: `var(--gc-success)` (#3e8635)

---

## 🎨 CSS Additions Needed

Add to `app/globals.css` or `app/theme.css`:

```css
/* ═══════════════════════════════════════════
   ACCORDION ANIMATION
   ═══════════════════════════════════════════ */

.pf-v5-c-accordion__expanded-content {
  transition: max-height 0.3s cubic-bezier(0.4, 0, 0.2, 1),
              opacity 0.3s ease;
  overflow: hidden;
}

.pf-v5-c-accordion__expanded-content[hidden] {
  max-height: 0;
  opacity: 0;
}

.pf-v5-c-accordion__expanded-content:not([hidden]) {
  max-height: 2000px;
  opacity: 1;
}

/* ═══════════════════════════════════════════
   COUNT-UP NUMBER ANIMATION
   ═══════════════════════════════════════════ */

@keyframes countUpFade {
  from {
    opacity: 0;
    transform: translateY(10px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

.gc-metric,
.gc-scenario {
  animation: countUpFade 0.4s ease-out;
}

/* ═══════════════════════════════════════════
   FLIP CARD HINT
   ═══════════════════════════════════════════ */

.flip-card-hint {
  position: absolute;
  bottom: 8px;
  right: 12px;
  font-size: 11px;
  color: var(--gc-text-3);
  opacity: 0.7;
  pointer-events: none;
}

.flip-card:hover .flip-card-hint {
  opacity: 1;
}
```

---

## 🧪 Testing Checklist

### **Visual Polish**
- [ ] All text meets minimum 11.5px size
- [ ] No faint gray (#6a6e73 or lighter) on body text
- [ ] Generous white space between sections
- [ ] Clear visual hierarchy (big → medium → small)
- [ ] High contrast on all interactive elements

### **Flip Cards**
- [ ] Click to flip works smoothly
- [ ] No backface flicker
- [ ] 600ms animation feels natural
- [ ] "Tap to flip" hint visible
- [ ] Front/back content both readable

### **Count-Up Animation**
- [ ] Numbers animate from 0 to value on load
- [ ] Smooth 60fps animation
- [ ] Ease-out timing feels natural
- [ ] Works on Calculate button click
- [ ] Multiple numbers don't stutter

### **Popovers**
- [ ] "?" icon visible next to jargon
- [ ] Click opens popover
- [ ] Explanation is clear and concise
- [ ] Popover positions correctly
- [ ] Can dismiss by clicking away

### **Accordions**
- [ ] Expand/collapse smooth animation
- [ ] Summary visible when collapsed
- [ ] Content fully hidden when collapsed
- [ ] No layout shift on expand
- [ ] Icons rotate with animation

---

## 📦 Quick Implementation

### **Step 1: Import theme.css**

```tsx
// app/layout.tsx
import "./theme.css";
```

### **Step 2: Update Predict performance page**

```tsx
// app/predict/page.tsx
"use client";

import { useCountUp } from "@/hooks/useCountUp";
import { FlipCard } from "@/components/ui/FlipCard";
import { JargonPopover, JARGON } from "@/components/ui/JargonPopover";
import { Accordion, AccordionItem, AccordionContent, AccordionToggle } from "@patternfly/react-core";

// In your component:
const gpuCount = useCountUp(result.gpuCount, 1500);
const weightGB = useCountUp(result.weightGB, 1500);

// Wrap cards:
<FlipCard
  front={
    <Card>
      <CardBody>
        <span className="gc-label">GPUs REQUIRED</span>
        <div className="gc-metric gc-num">{gpuCount}</div>
        <span className="gc-unit">GPUs</span>
      </CardBody>
    </Card>
  }
  back={
    <Card style={{ background: "var(--gc-text)" }}>
      <CardBody>
        <span className="gc-label" style={{ color: "#fff" }}>CALCULATION</span>
        <div className="gc-detail" style={{ color: "#fff" }}>
          TP size × Replicas = {result.tpSize} × {result.replicas}
        </div>
      </CardBody>
    </Card>
  }
/>

// Add popovers:
<span>
  KV cache <JargonPopover term="KV cache" explanation={JARGON["KV cache"]} />
</span>

// Accordion:
<Accordion>
  <AccordionItem>
    <AccordionToggle>
      Why this GPU count? <JargonPopover term="tensor parallel" />
    </AccordionToggle>
    <AccordionContent>
      {/* Constraint table */}
    </AccordionContent>
  </AccordionItem>
</Accordion>
```

---

## 🎯 Final Polish Points

1. **No faint gray micro-text** - All body content uses `var(--gc-text-2)` (#3c3f42) minimum
2. **Generous spacing** - 2.5rem between sections, 2rem card padding
3. **Clear hierarchy** - 80px hero number → 32px supporting → 14px body → 13px detail
4. **Every jargon explained** - 15+ popover help icons throughout
5. **Smooth animations** - Flip (600ms), count-up (1500ms), accordion (300ms)

**Status:** Components created, implementation guide ready.
**Next:** Apply to Predict performance page systematically.
