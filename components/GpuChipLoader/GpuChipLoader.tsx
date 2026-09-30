'use client';

import * as React from 'react';
import CheckCircleIcon from '@patternfly/react-icons/dist/esm/icons/check-circle-icon';
import styles from './GpuChipLoader.module.css';

const PHASES = [
  'Analyzing model architecture',
  'Evaluating memory requirements',
  'Calculating tensor parallelism',
  'Optimizing GPU topology',
  'Estimating throughput',
  'Finalizing configuration',
] as const;

const PHASE_DURATION = 3;

interface GpuChipLoaderProps {
  elapsed: number;
  /** Upper-bound wait shown in the hint, in seconds. Pass
   * DEFAULT_GATEWAY_TIMEOUT_SECONDS from lib/api/timeout. */
  timeoutSeconds: number;
  progressMessages?: string[];
}

export function GpuChipLoader({ elapsed, timeoutSeconds, progressMessages }: GpuChipLoaderProps) {
  const activeIndex = Math.min(Math.floor(elapsed / PHASE_DURATION), PHASES.length - 1);

  return (
    <div className={styles.wrap}>
      <div className={styles.chipAndPhases}>
        <GpuChipSvg />
        <div className={styles.phases} aria-live="polite">
          {progressMessages?.length ? progressMessages.map((message, i) => {
            const active = i === progressMessages.length - 1;
            return (
              <div key={`${message}-${i}`} className={`${styles.phase} ${active ? styles.phaseActive : styles.phaseDone}`}>
                {active ? <span className={styles.activeDot} /> : <CheckCircleIcon className={styles.checkIcon} />}
                {message}
              </div>
            );
          }) : PHASES.map((label, i) => {
            const done = i < activeIndex;
            const active = i === activeIndex;
            return (
              <div
                key={label}
                className={`${styles.phase} ${done ? styles.phaseDone : ''} ${active ? styles.phaseActive : ''}`}
              >
                {done ? (
                  <CheckCircleIcon className={styles.checkIcon} />
                ) : active ? (
                  <span className={styles.activeDot} />
                ) : (
                  <span className={styles.pendingDot} />
                )}
                {label}...
              </div>
            );
          })}
        </div>
      </div>
      <div style={{ textAlign: 'center' }}>
        <div className={styles.timer}>{elapsed}s elapsed</div>
        <div className={styles.timerNote}>This typically takes 10–20 seconds, but can take up to {timeoutSeconds} seconds</div>
      </div>
    </div>
  );
}

function GpuChipSvg() {
  const pinPositions = {
    top: Array.from({ length: 8 }, (_, i) => ({ x: 52 + i * 13, y: 28 })),
    bottom: Array.from({ length: 8 }, (_, i) => ({ x: 52 + i * 13, y: 162 })),
    left: Array.from({ length: 8 }, (_, i) => ({ x: 28, y: 52 + i * 13 })),
    right: Array.from({ length: 8 }, (_, i) => ({ x: 162, y: 52 + i * 13 })),
  };

  const cores = Array.from({ length: 16 }, (_, i) => ({
    x: 62 + (i % 4) * 22,
    y: 62 + Math.floor(i / 4) * 22,
    idx: i,
  }));

  return (
    <svg className={styles.chip} viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">
      {/* Outer chip package */}
      <rect x="40" y="40" width="120" height="120" rx="4" fill="#151515" stroke="#3c3f42" strokeWidth="1.5" />

      {/* Substrate border */}
      <rect x="46" y="46" width="108" height="108" rx="2" fill="none" stroke="#2a2a2a" strokeWidth="1" />

      {/* Pins — top */}
      {pinPositions.top.map((p, i) => (
        <rect key={`t${i}`} className={styles.pin} style={{ '--i': i } as React.CSSProperties}
          x={p.x} y={p.y} width="6" height="12" rx="1" />
      ))}
      {/* Pins — bottom */}
      {pinPositions.bottom.map((p, i) => (
        <rect key={`b${i}`} className={styles.pin} style={{ '--i': i + 8 } as React.CSSProperties}
          x={p.x} y={p.y} width="6" height="12" rx="1" />
      ))}
      {/* Pins — left */}
      {pinPositions.left.map((p, i) => (
        <rect key={`l${i}`} className={styles.pin} style={{ '--i': i + 16 } as React.CSSProperties}
          x={p.x} y={p.y} width="12" height="6" rx="1" />
      ))}
      {/* Pins — right */}
      {pinPositions.right.map((p, i) => (
        <rect key={`r${i}`} className={styles.pin} style={{ '--i': i + 24 } as React.CSSProperties}
          x={p.x} y={p.y} width="12" height="6" rx="1" />
      ))}

      {/* Circuit traces — vertical from top pins to core grid */}
      {pinPositions.top.map((p, i) => (
        <line key={`tt${i}`} className={styles.trace} style={{ '--i': i } as React.CSSProperties}
          x1={p.x + 3} y1={p.y + 12} x2={p.x + 3} y2={58} />
      ))}
      {/* Circuit traces — vertical from bottom pins */}
      {pinPositions.bottom.map((p, i) => (
        <line key={`tb${i}`} className={styles.trace} style={{ '--i': i + 8 } as React.CSSProperties}
          x1={p.x + 3} y1={p.y} x2={p.x + 3} y2={142} />
      ))}
      {/* Circuit traces — horizontal from left pins */}
      {pinPositions.left.map((p, i) => (
        <line key={`tl${i}`} className={styles.trace} style={{ '--i': i + 16 } as React.CSSProperties}
          x1={p.x + 12} y1={p.y + 3} x2={58} y2={p.y + 3} />
      ))}
      {/* Circuit traces — horizontal from right pins */}
      {pinPositions.right.map((p, i) => (
        <line key={`tr${i}`} className={styles.trace} style={{ '--i': i + 24 } as React.CSSProperties}
          x1={p.x} y1={p.y + 3} x2={142} y2={p.y + 3} />
      ))}

      {/* Core grid — 4×4 */}
      {cores.map(c => (
        <rect key={`c${c.idx}`} className={styles.core} style={{ '--i': c.idx } as React.CSSProperties}
          x={c.x} y={c.y} width="14" height="14" rx="2" />
      ))}

      {/* Center label */}
      <text x="100" y="105" textAnchor="middle" fontFamily="'JetBrains Mono', monospace"
        fontSize="11" fontWeight="600" fill="rgba(255,255,255,0.5)" letterSpacing="0.08em">
        GPU
      </text>
    </svg>
  );
}
