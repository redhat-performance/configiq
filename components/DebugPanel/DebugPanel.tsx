'use client';

import * as React from 'react';
import styles from './DebugPanel.module.css';
import type { DebugPane, DebugRun } from '@/lib/debug/types';

interface DebugPanelProps {
  /** Runs to show, in display order. An empty list hides the panel. */
  runs: DebugRun[];
  status?: number | null;
  duration?: number | null;
  endpoint?: string;
  open: boolean;
  onToggle: (open: boolean) => void;
}

const COPY_FEEDBACK_MS = 1500;

interface TabStripProps {
  idPrefix: string;
  label: string;
  items: { id: string; label: string }[];
  activeId: string | null;
  onSelect: (id: string) => void;
  variant: 'run' | 'group' | 'artifact';
  /**
   * The one region these tabs switch. Every strip picks a pane for the same
   * panel, so they all name it rather than each claiming content that is not
   * rendered. Undefined when there is no pane to control.
   */
  panelId?: string;
}

/** A tablist with arrow/Home/End movement, rendered only when it can switch. */
function TabStrip({
  idPrefix,
  label,
  items,
  activeId,
  onSelect,
  variant,
  panelId,
}: TabStripProps) {
  if (items.length < 2) return null;

  const handleKeyDown = (event: React.KeyboardEvent<HTMLDivElement>) => {
    const index = items.findIndex(item => item.id === activeId);
    if (index < 0) return;

    const next =
      event.key === 'ArrowRight' ? items[(index + 1) % items.length]
        : event.key === 'ArrowLeft' ? items[(index - 1 + items.length) % items.length]
          : event.key === 'Home' ? items[0]
            : event.key === 'End' ? items[items.length - 1]
              : null;
    if (!next) return;

    event.preventDefault();
    onSelect(next.id);
    const buttons = Array.from(
      event.currentTarget.querySelectorAll<HTMLButtonElement>('button[role="tab"]'),
    );
    buttons.find(button => button.dataset.tabId === next.id)?.focus();
  };

  return (
    <div
      role="tablist"
      aria-label={label}
      className={`${styles.tabStrip} ${styles[`tabStrip_${variant}`] ?? ''}`}
      onKeyDown={handleKeyDown}
    >
      {items.map(item => {
        const selected = item.id === activeId;
        return (
          <button
            key={item.id}
            type="button"
            role="tab"
            id={`${idPrefix}-tab-${item.id}`}
            aria-selected={selected}
            aria-controls={panelId}
            tabIndex={selected ? 0 : -1}
            data-tab-id={item.id}
            className={`${styles.tab} ${selected ? styles.tabSelected : ''}`}
            onClick={() => onSelect(item.id)}
          >
            {item.label}
          </button>
        );
      })}
    </div>
  );
}

export function DebugPanel({
  runs,
  status = null,
  duration = null,
  endpoint,
  open,
  onToggle,
}: DebugPanelProps) {
  const baseId = React.useId();
  const [runId, setRunId] = React.useState<string | null>(null);
  const [groupId, setGroupId] = React.useState<string | null>(null);
  const [paneId, setPaneId] = React.useState<string | null>(null);
  const [copiedId, setCopiedId] = React.useState<string | null>(null);
  const copyTimer = React.useRef<ReturnType<typeof setTimeout> | null>(null);

  React.useEffect(() => () => {
    if (copyTimer.current) clearTimeout(copyTimer.current);
  }, []);

  if (runs.length === 0) return null;

  // A new result replaces the runs, so an id that no longer exists falls back
  // to the first run rather than rendering an empty panel.
  const run = runs.find(candidate => candidate.id === runId) ?? runs[0];
  const groups = Array.from(new Set(
    run.panes
      .map(pane => pane.group)
      .filter((group): group is string => typeof group === 'string' && group !== ''),
  ));
  const group = groups.find(candidate => candidate === groupId) ?? groups[0] ?? null;
  // A group only partitions the artifacts that belong to one. Ungrouped panes,
  // such as the run script and the version pins, describe the whole run and stay
  // visible behind every group tab. An empty group names no tab, so it counts
  // as ungrouped here, matching how the tabs are collected above.
  const panes = group === null
    ? run.panes
    : [
        ...run.panes.filter(pane => pane.group === group),
        ...run.panes.filter(pane => !pane.group),
      ];
  const pane = panes.find(candidate => candidate.id === paneId) ?? panes[0];

  const copy = async (target: DebugPane) => {
    const clipboard = navigator.clipboard;
    // Without the Clipboard API there is nothing to write, and reporting
    // success would claim a copy that did not happen.
    if (!clipboard) return;
    try {
      await clipboard.writeText(target.contents);
      setCopiedId(target.id);
      if (copyTimer.current) clearTimeout(copyTimer.current);
      copyTimer.current = setTimeout(() => setCopiedId(null), COPY_FEEDBACK_MS);
    } catch {
      // A denied clipboard leaves the pane readable, which is the fallback.
    }
  };

  const hasTabStrip = panes.length > 1;
  const activeTabId = hasTabStrip && pane ? `${baseId}-artifact-tab-${pane.id}` : undefined;
  const panePanelId = pane ? `${baseId}-pane` : undefined;

  return (
    <div className={styles.debugSection}>
      <button
        type="button"
        className={styles.debugToggle}
        onClick={() => onToggle(!open)}
        aria-expanded={open}
      >
        <span className={styles.debugToggleIcon} aria-hidden="true">{open ? '▾' : '▸'}</span>
        Debug panel
        {endpoint && <span className={styles.debugEndpoint}>{endpoint}</span>}
        {status !== null && (
          <span className={`${styles.debugStatusBadge} ${status >= 200 && status < 300 ? styles.debugStatusOk : styles.debugStatusErr}`}>
            {status}
          </span>
        )}
        {duration !== null && (
          <span className={styles.debugDuration}>{duration} ms</span>
        )}
      </button>

      {open && (
        <div className={styles.debugBody}>
          {run.notes && run.notes.length > 0 && (
            <ul className={styles.debugNotes}>
              {run.notes.map(note => <li key={note}>{note}</li>)}
            </ul>
          )}

          <TabStrip
            idPrefix={`${baseId}-run`}
            label="Debug runs"
            variant="run"
            items={runs.map(item => ({ id: item.id, label: item.badge ? `${item.label} · ${item.badge}` : item.label }))}
            activeId={run.id}
            panelId={panePanelId}
            onSelect={(id) => {
              setRunId(id);
              setGroupId(null);
              setPaneId(null);
            }}
          />

          <TabStrip
            idPrefix={`${baseId}-group`}
            label="Artifact groups"
            variant="group"
            items={groups.map(item => ({ id: item, label: item }))}
            activeId={group}
            panelId={panePanelId}
            onSelect={(id) => {
              setGroupId(id);
              setPaneId(null);
            }}
          />

          <div className={styles.paneHeader}>
            <TabStrip
              idPrefix={`${baseId}-artifact`}
              label="Artifacts"
              variant="artifact"
              items={panes.map(item => ({ id: item.id, label: item.label }))}
              activeId={pane?.id ?? null}
              onSelect={setPaneId}
              panelId={panePanelId}
            />
            {pane && (
              <div className={styles.paneActions}>
                <span className={styles.paneLanguage}>{pane.language}</span>
                <button
                  type="button"
                  className={styles.copyButton}
                  onClick={() => copy(pane)}
                  aria-label={`Copy ${pane.label}`}
                >
                  {copiedId === pane.id ? 'Copied' : 'Copy'}
                </button>
              </div>
            )}
          </div>

          {pane && (
            <div
              role="tabpanel"
              id={panePanelId}
              aria-labelledby={activeTabId}
              // A single pane has no artifact tab to name the panel, so the
              // pane label carries it instead of leaving the panel unnamed.
              aria-label={activeTabId ? undefined : pane.label}
              className={styles.paneBody}
            >
              <pre className={styles.debugPre}>{pane.contents}</pre>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
