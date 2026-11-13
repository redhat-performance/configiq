'use client';

import * as React from 'react';
import { createPortal } from 'react-dom';
import { Button } from '@patternfly/react-core';
import type { TourPosition, TourStep } from '@/lib/help/tours';
import styles from './ProductTour.module.css';

export type { TourPosition, TourStep };

interface ProductTourProps {
  steps: TourStep[];
  onComplete: () => void;
}

interface Box {
  top: number;
  left: number;
  width: number;
  height: number;
}

interface TooltipBox {
  top: number;
  left: number;
}

interface TourLayout {
  spotlight: Box;
  tooltip: TooltipBox;
  placement: TourPosition;
}

const SPOTLIGHT_PADDING = 8;
const TOOLTIP_GAP = 16;
const VIEWPORT_MARGIN = 12;
const FALLBACK_WIDTH = 360;
const FALLBACK_HEIGHT = 200;
/** Steps whose target never appears are skipped instead of stalling the tour. */
const FALLBACK_ORDER: TourPosition[] = ['bottom', 'top', 'right', 'left'];

function prefersReducedMotion(): boolean {
  return (
    typeof window !== 'undefined' &&
    typeof window.matchMedia === 'function' &&
    window.matchMedia('(prefers-reduced-motion: reduce)').matches
  );
}

function queryTarget(selector: string): Element | null {
  if (!selector) return null;
  try {
    return document.querySelector(selector);
  } catch {
    return null;
  }
}

/** Index of the next step at or after `from` whose target exists, or -1. */
function nextAvailableStep(steps: TourStep[], from: number): number {
  for (let i = Math.max(from, 0); i < steps.length; i += 1) {
    if (queryTarget(steps[i].target)) return i;
  }
  return -1;
}

function sideHasRoom(
  side: TourPosition,
  spotlight: Box,
  tooltipWidth: number,
  tooltipHeight: number,
  viewportWidth: number,
  viewportHeight: number,
): boolean {
  switch (side) {
    case 'top':
      return spotlight.top >= tooltipHeight + TOOLTIP_GAP;
    case 'bottom':
      return viewportHeight - (spotlight.top + spotlight.height) >= tooltipHeight + TOOLTIP_GAP;
    case 'left':
      return spotlight.left >= tooltipWidth + TOOLTIP_GAP;
    case 'right':
      return viewportWidth - (spotlight.left + spotlight.width) >= tooltipWidth + TOOLTIP_GAP;
    default:
      return false;
  }
}

function placeTooltip(
  spotlight: Box,
  preferred: TourPosition,
  tooltipWidth: number,
  tooltipHeight: number,
): { position: TooltipBox; placement: TourPosition } {
  const viewportWidth = window.innerWidth;
  const viewportHeight = window.innerHeight;

  const candidates = [preferred, ...FALLBACK_ORDER.filter((side) => side !== preferred)];
  const placement =
    candidates.find((side) =>
      sideHasRoom(side, spotlight, tooltipWidth, tooltipHeight, viewportWidth, viewportHeight),
    ) ?? preferred;

  const centerX = spotlight.left + spotlight.width / 2;
  const centerY = spotlight.top + spotlight.height / 2;

  let top = 0;
  let left = 0;
  switch (placement) {
    case 'top':
      top = spotlight.top - TOOLTIP_GAP - tooltipHeight;
      left = centerX - tooltipWidth / 2;
      break;
    case 'right':
      top = centerY - tooltipHeight / 2;
      left = spotlight.left + spotlight.width + TOOLTIP_GAP;
      break;
    case 'left':
      top = centerY - tooltipHeight / 2;
      left = spotlight.left - TOOLTIP_GAP - tooltipWidth;
      break;
    case 'bottom':
    default:
      top = spotlight.top + spotlight.height + TOOLTIP_GAP;
      left = centerX - tooltipWidth / 2;
      break;
  }

  // Never let the tooltip leave the viewport, even when no side has room.
  const maxLeft = Math.max(VIEWPORT_MARGIN, viewportWidth - tooltipWidth - VIEWPORT_MARGIN);
  const maxTop = Math.max(VIEWPORT_MARGIN, viewportHeight - tooltipHeight - VIEWPORT_MARGIN);

  return {
    placement,
    position: {
      top: Math.min(Math.max(top, VIEWPORT_MARGIN), maxTop),
      left: Math.min(Math.max(left, VIEWPORT_MARGIN), maxLeft),
    },
  };
}

function spotlightFor(element: Element): Box {
  const rect = element.getBoundingClientRect();
  return {
    top: rect.top - SPOTLIGHT_PADDING,
    left: rect.left - SPOTLIGHT_PADDING,
    width: rect.width + SPOTLIGHT_PADDING * 2,
    height: rect.height + SPOTLIGHT_PADDING * 2,
  };
}

export function ProductTour({ steps, onComplete }: ProductTourProps) {
  const [currentStep, setCurrentStep] = React.useState(0);
  const [layout, setLayout] = React.useState<TourLayout | null>(null);
  const [mounted, setMounted] = React.useState(false);

  const dialogRef = React.useRef<HTMLDivElement>(null);
  const previousFocus = React.useRef<HTMLElement | null>(null);
  const completedRef = React.useRef(false);
  const onCompleteRef = React.useRef(onComplete);

  React.useEffect(() => {
    onCompleteRef.current = onComplete;
  }, [onComplete]);

  const finish = React.useCallback(() => {
    if (completedRef.current) return;
    completedRef.current = true;
    onCompleteRef.current();
  }, []);

  const step = steps[currentStep];

  // Only render the portal on the client.
  React.useEffect(() => {
    setMounted(true);
  }, []);

  // Move past steps whose target is not in the DOM. This is what keeps a
  // first visit working: result tiles only exist after a calculation.
  React.useEffect(() => {
    if (!mounted) return;
    if (steps.length === 0) {
      finish();
      return;
    }
    if (queryTarget(steps[currentStep]?.target ?? '')) return;
    const next = nextAvailableStep(steps, currentStep + 1);
    if (next === -1) {
      finish();
      return;
    }
    setCurrentStep(next);
  }, [mounted, steps, currentStep, finish]);

  // Measure the step target and keep the spotlight and tooltip in place.
  React.useEffect(() => {
    if (!mounted || !step) return;

    let frame = 0;
    let resizeObserver: ResizeObserver | null = null;

    const update = () => {
      const target = queryTarget(step.target);
      if (!target) {
        setLayout(null);
        return;
      }
      const spotlight = spotlightFor(target);
      const tooltipEl = dialogRef.current;
      const tooltipWidth = tooltipEl?.offsetWidth || FALLBACK_WIDTH;
      const tooltipHeight = tooltipEl?.offsetHeight || FALLBACK_HEIGHT;
      const { position, placement } = placeTooltip(
        spotlight,
        step.position ?? 'bottom',
        tooltipWidth,
        tooltipHeight,
      );
      setLayout({ spotlight, tooltip: position, placement });
    };

    const scheduleUpdate = () => {
      cancelAnimationFrame(frame);
      frame = requestAnimationFrame(update);
    };

    const target = queryTarget(step.target);

    // Bring the target into view before measuring. Smooth scrolling is
    // skipped when the user asks for reduced motion.
    target?.scrollIntoView({
      behavior: prefersReducedMotion() ? 'auto' : 'smooth',
      block: 'center',
      inline: 'nearest',
    });

    // The first measurement runs after layout settles.
    frame = requestAnimationFrame(update);
    const settleTimer = window.setTimeout(update, 320);

    window.addEventListener('scroll', scheduleUpdate, { passive: true });
    window.addEventListener('resize', scheduleUpdate);

    if (target && typeof ResizeObserver !== 'undefined') {
      resizeObserver = new ResizeObserver(scheduleUpdate);
      resizeObserver.observe(target);
    }

    return () => {
      cancelAnimationFrame(frame);
      window.clearTimeout(settleTimer);
      window.removeEventListener('scroll', scheduleUpdate);
      window.removeEventListener('resize', scheduleUpdate);
      resizeObserver?.disconnect();
    };
  }, [mounted, step, currentStep]);

  // Move focus into the dialog and restore it when the tour closes.
  React.useEffect(() => {
    if (!mounted) return;
    previousFocus.current = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    const frame = requestAnimationFrame(() => dialogRef.current?.focus());

    return () => {
      cancelAnimationFrame(frame);
      const element = previousFocus.current;
      if (element && document.contains(element)) {
        element.focus();
      }
    };
  }, [mounted]);

  // Esc closes the tour from anywhere on the page.
  React.useEffect(() => {
    if (!mounted) return;
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        event.preventDefault();
        finish();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [mounted, finish]);

  const handleNext = React.useCallback(() => {
    const next = nextAvailableStep(steps, currentStep + 1);
    if (next === -1) {
      finish();
      return;
    }
    setCurrentStep(next);
  }, [steps, currentStep, finish]);

  const handleBack = React.useCallback(() => {
    // Walk backwards to the nearest previously shown step.
    for (let i = currentStep - 1; i >= 0; i -= 1) {
      if (queryTarget(steps[i].target)) {
        setCurrentStep(i);
        return;
      }
    }
  }, [steps, currentStep]);

  const handleKeyDown = (event: React.KeyboardEvent<HTMLDivElement>) => {
    if (event.key !== 'Tab') return;
    const focusable = dialogRef.current?.querySelectorAll<HTMLElement>(
      'button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])',
    );
    if (!focusable || focusable.length === 0) {
      event.preventDefault();
      return;
    }
    const first = focusable[0];
    const last = focusable[focusable.length - 1];
    const active = document.activeElement;

    if (event.shiftKey && (active === first || active === dialogRef.current)) {
      event.preventDefault();
      last.focus();
    } else if (!event.shiftKey && active === last) {
      event.preventDefault();
      first.focus();
    }
  };

  if (!mounted || !step) return null;

  const titleId = 'product-tour-title';
  const descriptionId = 'product-tour-description';
  // "Back" is only offered when an earlier step is actually on screen.
  const hasPreviousStep = steps
    .slice(0, currentStep)
    .some((earlier) => queryTarget(earlier.target) !== null);
  const isLastStep = nextAvailableStep(steps, currentStep + 1) === -1;

  const tourContent = (
    <>
      {/* Blocks page interaction while the tour is open. Clicking it is
          deliberately inert — only Esc or "Skip tour" ends the tour. */}
      <div className={styles.overlay} aria-hidden="true" />

      {layout && (
        <div
          className={styles.spotlight}
          style={{
            top: `${layout.spotlight.top}px`,
            left: `${layout.spotlight.left}px`,
            width: `${layout.spotlight.width}px`,
            height: `${layout.spotlight.height}px`,
          }}
          aria-hidden="true"
        />
      )}

      <div
        ref={dialogRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        aria-describedby={descriptionId}
        tabIndex={-1}
        className={`${styles.tooltip} ${styles[`tooltip-${layout?.placement ?? step.position ?? 'bottom'}`]}`}
        style={{
          top: `${layout?.tooltip.top ?? 0}px`,
          left: `${layout?.tooltip.left ?? 0}px`,
          visibility: layout ? 'visible' : 'hidden',
        }}
        onKeyDown={handleKeyDown}
      >
        <div className={styles.tooltipHeader}>
          <h3 id={titleId} className={styles.tooltipTitle}>
            {step.title}
          </h3>
          <span className={styles.tooltipProgress}>
            {currentStep + 1} of {steps.length}
          </span>
        </div>
        <p id={descriptionId} className={styles.tooltipDescription}>
          {step.description}
        </p>

        <div className={styles.progressDots} aria-hidden="true">
          {steps.map((_, i) => (
            <div key={i} className={`${styles.dot} ${i === currentStep ? styles.dotActive : ''}`} />
          ))}
        </div>

        <div className={styles.tooltipActions}>
          <div className={styles.secondaryActions}>
            {hasPreviousStep && (
              <button type="button" className={styles.backButton} onClick={handleBack}>
                Back
              </button>
            )}
            <button type="button" className={styles.skipButton} onClick={finish}>
              Skip tour
            </button>
          </div>
          <Button variant="primary" onClick={handleNext}>
            {isLastStep ? 'Got it' : 'Next'}
          </Button>
        </div>
      </div>
    </>
  );

  return createPortal(tourContent, document.body);
}
