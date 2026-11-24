'use client';

import * as React from 'react';
import {
  Alert,
  Button,
  EmptyState,
  EmptyStateActions,
  EmptyStateBody,
  EmptyStateFooter,
  Modal,
  ModalBody,
  ModalFooter,
  ModalHeader,
  Radio,
  Spinner,
  Toolbar,
  ToolbarContent,
  ToolbarItem,
} from '@patternfly/react-core';
import DownloadIcon from '@patternfly/react-icons/dist/esm/icons/download-icon';
import FileExportIcon from '@patternfly/react-icons/dist/esm/icons/file-export-icon';
import FileImportIcon from '@patternfly/react-icons/dist/esm/icons/file-import-icon';
import TimesIcon from '@patternfly/react-icons/dist/esm/icons/times-icon';
import TrashIcon from '@patternfly/react-icons/dist/esm/icons/trash-icon';
import Link from 'next/link';

import { comparisonCsv } from '@/lib/comparison/csv';
import { backfillComparisonCosts, type ComparisonCostSources } from '@/lib/comparison/backfill';
import {
  COMPARISON_SECTIONS,
  formatMetric,
  formatNumber,
  metricsFor,
  simulatorLabel,
  type ComparisonMetric,
} from '@/lib/comparison/metrics';
import {
  clearComparison,
  exportComparisonJson,
  importComparison,
  loadComparison,
  removeComparisonEntry,
  writeComparison,
} from '@/lib/comparison/storage';
import {
  MAX_COMPARISON_ENTRIES,
  emptyComparisonDocument,
  type ComparisonDocument,
  type ComparisonEntry,
} from '@/lib/comparison/types';

import styles from './CompareResults.module.css';

type Notice =
  | { kind: 'success' | 'danger' | 'info'; message: string }
  | null;

interface CompareResultsProps {
  /** Overrides for tests; the page reads the browser's own set. */
  load?: () => ReturnType<typeof loadComparison>;
  /**
   * Prices that can complete a saved result whose cost figures were missing,
   * or null when no catalogue has loaded yet.
   */
  costSources?: ComparisonCostSources | null;
}

function timestamp(date: string): string {
  const parsed = new Date(date);
  if (Number.isNaN(parsed.getTime())) return date;
  return parsed.toLocaleString(undefined, {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
}

function fileName(extension: 'csv' | 'json'): string {
  const day = new Date().toISOString().slice(0, 10);
  return `configiq-comparison-${day}.${extension}`;
}

function download(contents: string, name: string, type: string): void {
  const blob = new Blob([contents], { type });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = name;
  document.body.append(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
}

function formatDelta(metric: ComparisonMetric, difference: number): string {
  const magnitude = Math.abs(difference);
  const formatted = metric.formatter ? metric.formatter(magnitude) : formatNumber(magnitude, metric.digits ?? 1);
  return difference > 0 ? `▲ ${formatted}` : `▼ ${formatted}`;
}

/**
 * The direction of a difference, not a judgement.
 *
 * Higher is better for throughput and lower is better for latency, so the
 * comparison reports which way a value moved and leaves the reading to the
 * user.
 */
function deltaFor(
  metric: ComparisonMetric,
  value: number | string | null,
  baseline: ComparisonEntry | null,
): string | null {
  if (metric.kind !== 'number' || baseline == null) return null;
  if (typeof value !== 'number') return null;
  const base = metric.value(baseline);
  if (typeof base !== 'number' || !Number.isFinite(base)) return null;
  const difference = value - base;
  return difference === 0 ? 'no change' : formatDelta(metric, difference);
}

interface EntryColumnProps {
  entry: ComparisonEntry;
  isBaseline: boolean;
  onBaseline: (id: string) => void;
  onDetails: (entry: ComparisonEntry) => void;
  onRemove: (entry: ComparisonEntry) => void;
}

function EntryColumn({ entry, isBaseline, onBaseline, onDetails, onRemove }: EntryColumnProps) {
  return (
    <th scope="col" className={styles.entryCell}>
      <div className={styles.entryHead}>
        <span className={styles.entryName}>{entry.name}</span>
        <span className={styles.entryMeta}>{entry.model}</span>
        <span className={styles.entryMeta}>
          {entry.systemLabel} · {simulatorLabel(entry)}
        </span>
        <span className={styles.entryMeta}>
          {entry.engineVersion == null ? 'Engine not reported' : `Engine ${entry.engineVersion}`}
        </span>
        <span className={styles.entrySaved}>Saved {timestamp(entry.savedAt)}</span>
        <div className={styles.entryActions}>
          <Radio
            id={`baseline-${entry.id}`}
            name="comparison-baseline"
            label="Baseline"
            isChecked={isBaseline}
            onChange={() => onBaseline(entry.id)}
          />
          <Button variant="link" size="sm" onClick={() => onDetails(entry)}>
            Details
          </Button>
          <Button
            variant="plain"
            size="sm"
            aria-label={`Remove ${entry.name} from the comparison`}
            onClick={() => onRemove(entry)}
            icon={<TrashIcon />}
          />
        </div>
      </div>
    </th>
  );
}

export function CompareResults({ load = loadComparison, costSources = null }: CompareResultsProps) {
  const [document, setDocument] = React.useState<ComparisonDocument>(emptyComparisonDocument);
  const [loaded, setLoaded] = React.useState(false);
  const [baselineId, setBaselineId] = React.useState<string | null>(null);
  const [notice, setNotice] = React.useState<Notice>(null);
  const [details, setDetails] = React.useState<ComparisonEntry | null>(null);
  const [confirmClear, setConfirmClear] = React.useState(false);
  const fileInput = React.useRef<HTMLInputElement>(null);

  React.useEffect(() => {
    const loadedSet = load();
    setDocument(loadedSet.document);
    setLoaded(true);
    if (loadedSet.problem != null) {
      setNotice({ kind: 'danger', message: loadedSet.problem });
    }
  }, [load]);

  // A saved result keeps the cost figures its page could see. Completing a
  // missing one here means an old or costings-off snapshot still shows a price,
  // and the table and an export of it agree because both read this document.
  React.useEffect(() => {
    if (!loaded || costSources == null) return;
    const backfilled = backfillComparisonCosts(document, costSources);
    if (backfilled.filled === 0) return;
    setDocument(backfilled.document);
    const written = writeComparison(backfilled.document);
    if (!written.ok) {
      setNotice({
        kind: 'danger',
        message: `Cost figures are shown but could not be saved: ${written.problem}`,
      });
    }
  }, [loaded, costSources, document]);

  const entries = document.entries;
  const baseline = entries.find(entry => entry.id === baselineId) ?? null;

  const replaceDocument = (next: ComparisonDocument) => {
    setDocument(next);
    if (!next.entries.some(entry => entry.id === baselineId)) setBaselineId(null);
  };

  const handleExportCsv = () => {
    download(comparisonCsv(document), fileName('csv'), 'text/csv;charset=utf-8');
  };

  const handleExportJson = () => {
    download(exportComparisonJson(document), fileName('json'), 'application/json');
  };

  const handleImport = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    event.target.value = '';
    if (!file) return;
    let raw: string;
    try {
      raw = await file.text();
    } catch {
      setNotice({ kind: 'danger', message: 'The selected file could not be read.' });
      return;
    }
    const result = importComparison(raw);
    if (!result.ok) {
      setNotice({ kind: 'danger', message: result.problem });
      return;
    }
    replaceDocument(result.document);
    setNotice({
      kind: 'success',
      message: `Imported ${result.document.entries.length} result${result.document.entries.length === 1 ? '' : 's'} from ${file.name}.`,
    });
  };

  const handleRemove = (entry: ComparisonEntry) => {
    const result = removeComparisonEntry(entry.id);
    if (!result.ok) {
      setNotice({ kind: 'danger', message: result.problem });
      return;
    }
    replaceDocument(result.document);
    setNotice({ kind: 'info', message: `Removed ${entry.name} from the comparison.` });
  };

  const handleClear = () => {
    const result = clearComparison();
    setConfirmClear(false);
    if (!result.ok) {
      setNotice({ kind: 'danger', message: result.problem });
      return;
    }
    replaceDocument(result.document);
    setNotice({ kind: 'info', message: 'Cleared every saved result.' });
  };

  // The same control serves the toolbar and the empty state: a fresh browser
  // restores an exported set before it has anything saved.
  const importControl = (
    <>
      <Button variant="secondary" icon={<FileImportIcon />} onClick={() => fileInput.current?.click()}>
        Import JSON
      </Button>
      <input
        ref={fileInput}
        type="file"
        accept="application/json,.json"
        className={styles.fileInput}
        aria-label="Import a comparison set"
        onChange={handleImport}
      />
    </>
  );

  const toolbar = (
    <Toolbar id="comparison-toolbar" ouiaId="ComparisonToolbar">
      <ToolbarContent>
        <ToolbarItem>
          <span className={styles.capacity}>
            {entries.length} of {MAX_COMPARISON_ENTRIES} results
          </span>
        </ToolbarItem>
        {baseline != null && (
          <ToolbarItem>
            <Button variant="link" onClick={() => setBaselineId(null)}>
              Clear baseline
            </Button>
          </ToolbarItem>
        )}
        <ToolbarItem align={{ default: 'alignEnd' }} variant="separator" />
        <ToolbarItem>
          <Button
            variant="secondary"
            icon={<FileExportIcon />}
            onClick={handleExportCsv}
          >
            Export CSV
          </Button>
        </ToolbarItem>
        <ToolbarItem>
          <Button
            variant="secondary"
            icon={<DownloadIcon />}
            onClick={handleExportJson}
          >
            Export JSON
          </Button>
        </ToolbarItem>
        <ToolbarItem>
          {importControl}
        </ToolbarItem>
        <ToolbarItem>
          <Button
            variant="danger"
            icon={<TrashIcon />}
            onClick={() => setConfirmClear(true)}
          >
            Clear all
          </Button>
        </ToolbarItem>
      </ToolbarContent>
    </Toolbar>
  );

  const detailsModal = details != null && (
    <Modal
      isOpen
      onClose={() => setDetails(null)}
      variant="medium"
      aria-labelledby="comparison-details-title"
    >
      <ModalHeader title={details.name} labelId="comparison-details-title" />
      <ModalBody>
        <dl className={styles.details}>
          {COMPARISON_SECTIONS.flatMap(section =>
            metricsFor(section.id).map(metric => (
              <div key={metric.id}>
                <dt>{metric.label}</dt>
                <dd>{formatMetric(metric, metric.value(details))}</dd>
              </div>
            )),
          )}
          <div>
            <dt>Saved</dt>
            <dd>{timestamp(details.savedAt)}</dd>
          </div>
        </dl>
      </ModalBody>
      <ModalFooter>
        <Link href={details.link}>Open the view behind this result</Link>
        <Button variant="link" onClick={() => setDetails(null)}>
          Close
        </Button>
      </ModalFooter>
    </Modal>
  );

  const clearModal = (
    <Modal
      isOpen={confirmClear}
      onClose={() => setConfirmClear(false)}
      variant="small"
      aria-labelledby="comparison-clear-title"
    >
      <ModalHeader title="Clear every saved result?" labelId="comparison-clear-title" />
      <ModalBody>
        This removes all {entries.length} saved results from this browser. Exported files
        are not affected.
      </ModalBody>
      <ModalFooter>
        <Button variant="danger" onClick={handleClear}>
          Clear all
        </Button>
        <Button variant="link" onClick={() => setConfirmClear(false)}>
          Cancel
        </Button>
      </ModalFooter>
    </Modal>
  );

  if (!loaded) {
    return (
      <div className={styles.page} aria-busy="true">
        <Spinner aria-label="Loading the comparison" />
      </div>
    );
  }

  return (
    <div className={styles.page}>
      <div className={styles.header}>
        <div>
          <h1 className={styles.pageTitle}>Saved results</h1>
          <p className={styles.subtitle}>
            Results you saved from a result page, side by side. The set lives in this
            browser only and is never sent to a ConfigIQ service.
          </p>
        </div>
      </div>

      {notice != null && (
        <Alert
          className={styles.notice}
          variant={notice.kind}
          title={notice.message}
          isInline
          actionClose={<Button variant="plain" aria-label="Dismiss" onClick={() => setNotice(null)} icon={<TimesIcon />} />}
        />
      )}

      {entries.length === 0 ? (
        <EmptyState className={styles.empty} titleText="No results saved yet" headingLevel="h2">
          <EmptyStateBody>
            Open a result page and choose <strong>Save for comparison</strong> to keep its
            figures here. Saved results can be shown side by side, exported, and imported
            again later.
          </EmptyStateBody>
          <EmptyStateFooter>
            <EmptyStateActions>
              <Link href="/predict" className={styles.emptyAction}>
                Predict performance
              </Link>
              <Link href="/recommend" className={styles.emptyActionSecondary}>
                Recommend sizing
              </Link>
              {importControl}
            </EmptyStateActions>
          </EmptyStateFooter>
        </EmptyState>
      ) : (
        <>
          {toolbar}
          <div className={styles.tableWrap}>
            <table className={styles.table}>
              <caption className={styles.caption}>
                Saved results, one per column. Choose a baseline to see each value&apos;s
                difference from it.
              </caption>
              <thead>
                <tr>
                  <th scope="col" className={styles.metricHead} />
                  {entries.map(entry => (
                    <EntryColumn
                      key={entry.id}
                      entry={entry}
                      isBaseline={entry.id === baselineId}
                      onBaseline={setBaselineId}
                      onDetails={setDetails}
                      onRemove={handleRemove}
                    />
                  ))}
                </tr>
              </thead>
              {COMPARISON_SECTIONS.map(section => (
                <tbody key={section.id} className={styles.section}>
                  <tr className={styles.sectionRow}>
                    <th scope="rowgroup">
                      {section.label}
                    </th>
                    {/* The baseline sits above the column it belongs to, once
                        per section, so it does not add a line to every metric
                        row. Empty cells keep the section band full width. */}
                    {entries.map(entry => (
                      <td key={entry.id}>
                        {entry.id === baselineId ? (
                          <span className={styles.baselineTag}>baseline</span>
                        ) : null}
                      </td>
                    ))}
                  </tr>
                  {metricsFor(section.id).map(metric => (
                    <tr key={metric.id}>
                      <th scope="row" className={styles.metricLabel}>
                        {metric.label}
                      </th>
                      {entries.map(entry => {
                        const value = metric.value(entry);
                        const isBaseline = entry.id === baselineId;
                        const delta = isBaseline ? null : deltaFor(metric, value, baseline);
                        return (
                          <td
                            key={entry.id}
                            className={metric.kind === 'number' ? styles.numeric : styles.text}
                          >
                            <span className={styles.value}>{formatMetric(metric, value)}</span>
                            {delta != null && <span className={styles.delta}>{delta}</span>}
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              ))}
            </table>
          </div>
        </>
      )}

      {detailsModal}
      {clearModal}
    </div>
  );
}
