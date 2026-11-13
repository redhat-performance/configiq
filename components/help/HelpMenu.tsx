'use client';

import * as React from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { Button, Popover, TextInput } from '@patternfly/react-core';
import HelpIcon from '@patternfly/react-icons/dist/esm/icons/help-icon';
import CompassIcon from '@patternfly/react-icons/dist/esm/icons/compass-icon';
import SearchIcon from '@patternfly/react-icons/dist/esm/icons/search-icon';
import { usePageTour } from './usePageTour';
import { AskPanel } from './AskPanel';
import {
  buildHelpIndex,
  parseHelpIndex,
  routeLabel,
  searchHelp,
  suggestionsForRoute,
  type DocsSearchEntry,
  type HelpIndex,
  type HelpResult,
} from '@/lib/help/search';
import { recordSearchEvent } from '@/lib/help/telemetry';
import styles from './HelpMenu.module.css';

type IndexStatus = 'idle' | 'loading' | 'ready' | 'error';

interface HelpPanelProps {
  status: IndexStatus;
  query: string;
  onQueryChange: (value: string) => void;
  results: HelpResult[];
  suggestions: HelpResult[];
  currentRoute: string;
  /** Close the popover. Passed by PatternFly's `bodyContent` callback. */
  hide: () => void;
  onStartTour?: () => void;
  tourTitle?: string;
  /** Assistant model name when enabled; null hides the question box. */
  assistantModel: string | null;
  assistantReason: string | null;
}

function ResultLink({ result, hide }: { result: HelpResult; hide: () => void }) {
  if (!result.url) return null;
  return (
    <Link
      href={result.url}
      className={styles.result}
      data-help-item
      onClick={hide}
      // The docs routes are static; a client-side transition is fine and keeps
      // the popover from being torn down mid-navigation.
      prefetch={false}
    >
      <span className={styles.resultTitle}>{result.title}</span>
      <span className={styles.resultContext}>{result.context}</span>
    </Link>
  );
}

function GlossaryResult({ result }: { result: HelpResult }) {
  return (
    <div className={styles.glossaryResult}>
      <span className={styles.resultTitle}>{result.title}</span>
      <span className={styles.resultBody}>{result.body}</span>
    </div>
  );
}

function HelpPanel({
  status,
  query,
  onQueryChange,
  results,
  suggestions,
  currentRoute,
  hide,
  onStartTour,
  tourTitle,
  assistantModel,
  assistantReason,
}: HelpPanelProps) {
  const panelRef = React.useRef<HTMLDivElement>(null);
  const trimmed = query.trim();

  const groups = React.useMemo(() => {
    const page = results.filter((result) => result.kind === 'page').slice(0, 4);
    const glossary = results.filter((result) => result.kind === 'glossary').slice(0, 3);
    const docs = results.filter((result) => result.kind === 'docs').slice(0, 6);
    return { page, glossary, docs };
  }, [results]);

  const hasResults =
    groups.page.length > 0 || groups.glossary.length > 0 || groups.docs.length > 0;

  // Arrow keys move through the actionable rows. Upstream/downstream order is
  // the same as the visual order, so this needs no per-group bookkeeping.
  const handleKeyDown = (event: React.KeyboardEvent<HTMLDivElement>) => {
    if (event.key !== 'ArrowDown' && event.key !== 'ArrowUp') return;
    const items = panelRef.current?.querySelectorAll<HTMLElement>('[data-help-item]');
    if (!items || items.length === 0) return;
    event.preventDefault();
    const list = Array.from(items);
    const activeIndex = list.indexOf(document.activeElement as HTMLElement);
    const nextIndex =
      event.key === 'ArrowDown'
        ? Math.min(activeIndex + 1, list.length - 1)
        : Math.max(activeIndex - 1, 0);
    list[activeIndex === -1 ? 0 : nextIndex].focus();
  };

  return (
    <div className={styles.panel} ref={panelRef} onKeyDown={handleKeyDown}>
      <TextInput
        id="help-search-input"
        type="search"
        aria-label="Search help"
        placeholder="Search help, glossary, and docs"
        value={query}
        autoComplete="off"
        spellCheck={false}
        customIcon={<SearchIcon />}
        onChange={(_event, value) => onQueryChange(value)}
      />

      {onStartTour && (
        <button
          type="button"
          className={styles.tourRow}
          data-help-item
          onClick={() => {
            hide();
            onStartTour();
          }}
        >
          <CompassIcon className={styles.tourIcon} />
          <span>
            <span className={styles.resultTitle}>Take a tour on this page</span>
            <span className={styles.resultContext}>{tourTitle}</span>
          </span>
        </button>
      )}

      {!trimmed && (
        <>
          {suggestions.length > 0 && (
            <section className={styles.group} aria-label={`Controls on ${routeLabel(currentRoute)}`}>
              <h3 className={styles.groupTitle}>On this page</h3>
              {suggestions.map((result) => (
                <ResultLink key={result.id} result={result} hide={hide} />
              ))}
            </section>
          )}

          <section className={styles.group} aria-label="Documentation">
            <h3 className={styles.groupTitle}>Documentation</h3>
            <Link href="/docs" className={styles.result} data-help-item onClick={hide} prefetch={false}>
              <span className={styles.resultTitle}>All documentation</span>
              <span className={styles.resultContext}>Guides, API reference, and caveats</span>
            </Link>
          </section>
        </>
      )}

      {trimmed && status === 'loading' && !hasResults && (
        <p className={styles.status} role="status">Searching documentation…</p>
      )}

      {trimmed && status === 'error' && (
        <p className={styles.status} role="status">
          Documentation search is unavailable right now. Glossary and page results still work.
        </p>
      )}

      {trimmed && !hasResults && status !== 'loading' && (
        <p className={styles.status} role="status">
          No matches. Try a page name, a control such as concurrency, or a term such as TTFT.
        </p>
      )}

      {groups.page.length > 0 && (
        <section className={styles.group} aria-label="On this page">
          <h3 className={styles.groupTitle}>On this page</h3>
          {groups.page.map((result) => (
            <ResultLink key={result.id} result={result} hide={hide} />
          ))}
        </section>
      )}

      {groups.glossary.length > 0 && (
        <section className={styles.group} aria-label="Glossary">
          <h3 className={styles.groupTitle}>Glossary</h3>
          {groups.glossary.map((result) => (
            <GlossaryResult key={result.id} result={result} />
          ))}
        </section>
      )}

      {groups.docs.length > 0 && (
        <section className={styles.group} aria-label="Pages and docs">
          <h3 className={styles.groupTitle}>Pages and docs</h3>
          {groups.docs.map((result) => (
            <ResultLink key={result.id} result={result} hide={hide} />
          ))}
        </section>
      )}

      <p className={styles.footer}>
        <kbd className={styles.kbd}>/</kbd> to search · <kbd className={styles.kbd}>Esc</kbd> to close
      </p>

      {assistantModel ? (
        <AskPanel currentRoute={currentRoute} model={assistantModel} hide={hide} />
      ) : assistantReason ? (
        <p className={styles.assistantNote}>Ask is unavailable. {assistantReason}</p>
      ) : null}
    </div>
  );
}

/**
 * Masthead help entry point: one popover with search, page controls, glossary,
 * documentation, and the current page's tour.
 *
 * The docs index is a static build artifact fetched on first open, so the
 * corpus never ships in the initial bundle and nothing parses Markdown in the
 * browser.
 */
export function HelpMenu() {
  const pathname = usePathname();
  const currentRoute = pathname ?? '/';
  const { tour, startTour } = usePageTour();

  const triggerRef = React.useRef<HTMLButtonElement>(null);
  const requested = React.useRef(false);
  const abortRef = React.useRef<AbortController | null>(null);

  const [status, setStatus] = React.useState<IndexStatus>('idle');
  const [docs, setDocs] = React.useState<DocsSearchEntry[]>([]);
  const [query, setQuery] = React.useState('');
  const [open, setOpen] = React.useState(false);
  // Assistant availability is a server fact, probed once on first open. Absent
  // or disabled means the panel explains why instead of offering a dead input.
  const [assistantModel, setAssistantModel] = React.useState<string | null>(null);
  const [assistantReason, setAssistantReason] = React.useState<string | null>(null);
  const assistantRequested = React.useRef(false);

  const index = React.useMemo<HelpIndex>(() => buildHelpIndex(docs), [docs]);

  const loadAssistant = React.useCallback(() => {
    if (assistantRequested.current) return;
    assistantRequested.current = true;

    fetch('/api/assistant', { headers: { Accept: 'application/json' } })
      .then((response) => (response.ok ? (response.json() as Promise<unknown>) : null))
      .then((raw) => {
        if (!raw || typeof raw !== 'object') return;
        const payload = raw as { enabled?: unknown; model?: unknown; reason?: unknown };
        const model = typeof payload.model === 'string' ? payload.model : null;
        if (payload.enabled === true && model) {
          setAssistantModel(model);
          setAssistantReason(null);
          return;
        }
        setAssistantModel(null);
        setAssistantReason(
          typeof payload.reason === 'string' && payload.reason.trim().length > 0
            ? payload.reason
            : 'The assistant is not available on this deployment.',
        );
      })
      .catch(() => {
        // A failed probe is not an error the user can act on; leave the panel
        // out rather than promising a feature that may not answer.
        setAssistantModel(null);
        setAssistantReason(null);
      });
  }, []);

  const loadIndex = React.useCallback(() => {
    if (requested.current) return;
    requested.current = true;
    setStatus('loading');
    loadAssistant();

    const controller = new AbortController();
    abortRef.current = controller;

    fetch('/help-index.json', { signal: controller.signal })
      .then((response) => {
        if (!response.ok) throw new Error(`help index: ${response.status}`);
        return response.json() as Promise<unknown>;
      })
      .then((raw) => {
        setDocs(parseHelpIndex(raw));
        setStatus('ready');
      })
      .catch((error: unknown) => {
        if (error instanceof DOMException && error.name === 'AbortError') return;
        // Allow a retry on the next open rather than caching the failure.
        requested.current = false;
        setStatus('error');
      });
  }, [loadAssistant]);

  React.useEffect(
    () => () => {
      abortRef.current?.abort();
    },
    [],
  );

  // `/` opens help from anywhere that is not already a text field.
  React.useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key !== '/' || event.metaKey || event.ctrlKey || event.altKey) return;
      const target = event.target;
      if (target instanceof HTMLElement && target.closest('input, textarea, select, [contenteditable="true"]')) {
        return;
      }
      event.preventDefault();
      loadIndex();
      triggerRef.current?.click();
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [loadIndex]);

  const results = React.useMemo(
    () => (query.trim() ? searchHelp(query, index, currentRoute) : []),
    [query, index, currentRoute],
  );

  const suggestions = React.useMemo(
    () => suggestionsForRoute(index, currentRoute),
    [index, currentRoute],
  );

  // Count a settled query once. Waiting for a pause keeps this to one event per
  // search rather than one per keystroke.
  const countedQuery = React.useRef<string | null>(null);
  React.useEffect(() => {
    const trimmed = query.trim();
    if (trimmed.length < 2 || status !== 'ready' || countedQuery.current === trimmed) return;
    const timer = window.setTimeout(() => {
      countedQuery.current = trimmed;
      recordSearchEvent(results.length);
    }, 600);
    return () => window.clearTimeout(timer);
  }, [query, results.length, status]);

  return (
    <Popover
      aria-label="Help and search"
      position="bottom-end"
      hasAutoWidth
      minWidth="420px"
      maxWidth="540px"
      distance={12}
      elementToFocus="#help-search-input"
      onShown={() => setOpen(true)}
      onHidden={() => {
        setOpen(false);
        setQuery('');
      }}
      bodyContent={(hide) => (
        <HelpPanel
          status={status}
          query={query}
          onQueryChange={setQuery}
          results={results}
          suggestions={suggestions}
          currentRoute={currentRoute}
          hide={hide}
          onStartTour={tour ? startTour : undefined}
          tourTitle={tour?.title}
          assistantModel={assistantModel}
          assistantReason={assistantReason}
        />
      )}
    >
      <Button
        ref={triggerRef}
        variant="plain"
        className={styles.trigger}
        style={{ color: 'rgba(255,255,255,0.85)' }}
        aria-label="Help, search, and tours"
        aria-expanded={open}
        icon={<HelpIcon />}
        onClick={loadIndex}
      >
        Help
      </Button>
    </Popover>
  );
}
