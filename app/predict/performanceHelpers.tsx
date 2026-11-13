'use client';

import * as React from 'react';
import { Popover, Button } from '@patternfly/react-core';
import OutlinedQuestionCircleIcon from '@patternfly/react-icons/dist/esm/icons/outlined-question-circle-icon';
import styles from './Performance.module.css';
import { GLOSSARY, type GlossaryId } from '@/lib/help/glossary';
import { PRECISION_METADATA, KV_PRECISION_METADATA } from '@/lib/help/precision';

export { GLOSSARY, PRECISION_METADATA, KV_PRECISION_METADATA };

/* ----------------------------------------------------------------------------
   Term — a "?" popover for jargon. Plain-language explanations so business
   users aren't lost. Drop <Term k="kvCache" /> next to any label.
   ---------------------------------------------------------------------------- */

export function Term({ k }: { k: GlossaryId }) {
  const g = GLOSSARY[k];
  if (!g) return null;
  return (
    <Popover headerContent={g.title} bodyContent={g.body} maxWidth="320px">
      <button type="button" className={styles.termBtn} aria-label={`What is ${g.title}?`}>
        <OutlinedQuestionCircleIcon style={{ width: 13, height: 13 }} />
      </button>
    </Popover>
  );
}

/* ----------------------------------------------------------------------------
   useCountUp — animate a number from 0 → target on mount (and on change).
   Respects prefers-reduced-motion.
   ---------------------------------------------------------------------------- */

export function useCountUp(target: number, duration = 750, decimals = 0) {
  const [val, setVal] = React.useState(target);
  const raf = React.useRef<number>(0);
  React.useEffect(() => {
    if (typeof window !== 'undefined' &&
        window.matchMedia?.('(prefers-reduced-motion: reduce)').matches) {
      setVal(target);
      return;
    }
    const from = 0;
    const start = performance.now();
    const tick = (now: number) => {
      const p = Math.min((now - start) / duration, 1);
      const e = 1 - Math.pow(1 - p, 4); // easeOutQuart
      setVal(from + (target - from) * e);
      if (p < 1) raf.current = requestAnimationFrame(tick);
    };
    raf.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf.current);
  }, [target, duration]);
  const factor = Math.pow(10, decimals);
  return Math.round(val * factor) / factor;
}

/* ----------------------------------------------------------------------------
   FlipTile — front face + math back face. Robust opacity+rotate flip
   (no bare backface-visibility dependency). Click or Enter/Space to flip.
   ---------------------------------------------------------------------------- */

export function FlipTile({
  dark = false,
  sparkline,
  front,
  back,
}: {
  dark?: boolean;
  sparkline?: React.ReactNode;
  front: React.ReactNode;
  back: React.ReactNode;
}) {
  const [flipped, setFlipped] = React.useState(false);
  return (
    <div
      className={`${styles.flip} ${dark ? styles.tileDark : ''} ${flipped ? styles.flipped : ''}`}
      role="button"
      tabIndex={0}
      aria-pressed={flipped}
      onClick={() => setFlipped((f) => !f)}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); setFlipped((f) => !f); }
      }}
    >
      <div className={`${styles.flipFace} ${styles.flipFront}`}>
        {sparkline ? <div className={styles.sparkline}>{sparkline}</div> : null}
        {front}
        <span className={styles.seeMath}>↻ see math</span>
      </div>
      <div className={`${styles.flipFace} ${styles.flipBack}`}>
        {back}
        <span className={styles.seeMath}>↻ flip back</span>
      </div>
    </div>
  );
}

/* ----------------------------------------------------------------------------
   Sparkline — tiny GPU-count-vs-concurrency line for the dark hero tile.
   ---------------------------------------------------------------------------- */

export function Sparkline({
  points,
  currentX,
  width = 96,
  height = 36,
  stroke = 'rgba(255,255,255,0.85)',
}: {
  points: [number, number][];
  currentX?: number;
  width?: number;
  height?: number;
  stroke?: string;
}) {
  const xs = points.map((p) => p[0]);
  const ys = points.map((p) => p[1]);
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys), maxY = Math.max(...ys);
  const nx = (x: number) => ((x - minX) / (maxX - minX || 1)) * (width - 2) + 1;
  const ny = (y: number) => height - 2 - ((y - minY) / (maxY - minY || 1)) * (height - 4);
  const d = points.map((p, i) => `${i === 0 ? 'M' : 'L'}${nx(p[0]).toFixed(1)},${ny(p[1]).toFixed(1)}`).join(' ');

  // Find current point
  const currentPoint = currentX !== undefined ? points.find(p => p[0] >= currentX) || points[points.length - 1] : null;

  return (
    <div style={{ position: 'relative', display: 'inline-block' }}>
      <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`}>
        <path d={d} fill="none" stroke={stroke} strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
        {points.map((p, i) => (
          <circle
            key={i}
            cx={nx(p[0])}
            cy={ny(p[1])}
            r={currentPoint && p[0] === currentPoint[0] ? 3 : 1.4}
            fill={currentPoint && p[0] === currentPoint[0] ? stroke : stroke}
            opacity={currentPoint && p[0] === currentPoint[0] ? 1 : 0.7}
          />
        ))}
      </svg>
      <div style={{
        position: 'absolute',
        bottom: '-18px',
        left: 0,
        fontSize: '9px',
        color: 'rgba(255,255,255,0.5)',
        fontFamily: 'var(--mono)',
        letterSpacing: '0.03em',
        textTransform: 'uppercase'
      }}>
        GPUs vs. concurrency
      </div>
      {currentPoint && (
        <div style={{
          position: 'absolute',
          top: '-14px',
          right: 0,
          fontSize: '10px',
          color: 'rgba(255,255,255,0.65)',
          fontFamily: 'var(--mono)',
          whiteSpace: 'nowrap'
        }}>
          now · {currentPoint[0]} → {currentPoint[1]} GPU
        </div>
      )}
    </div>
  );
}
