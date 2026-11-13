'use client';

import * as React from 'react';
import Link from 'next/link';
import { Button, Spinner, TextArea } from '@patternfly/react-core';
import { useSettings } from '@/contexts/SettingsContext';
import type { AssistantEvent, AssistantLink, AssistantCitation } from '@/lib/assistant/types';
import styles from './AskPanel.module.css';

type AskStatus = 'idle' | 'asking' | 'done' | 'error';

interface AskPanelProps {
  /** Route the question is about, sent so the answer can be scoped. */
  currentRoute: string;
  /** Availability comes from the server; absent means the assistant is off. */
  model: string | null;
  /** Close the popover when the user opens a linked page. */
  hide: () => void;
}

interface ProgressLine {
  id: string;
  text: string;
}

function readEvent(line: string): AssistantEvent | null {
  try {
    const parsed = JSON.parse(line) as AssistantEvent;
    return parsed && typeof parsed === 'object' && typeof parsed.type === 'string' ? parsed : null;
  } catch {
    return null;
  }
}

/**
 * Opt-in question box inside the help popover.
 *
 * The route streams progress events and one final verified result. The answer
 * text is rendered only from that result: if the deterministic verifier could
 * not trace a number, the panel shows the sources instead of the prose.
 */
export function AskPanel({ currentRoute, model, hide }: AskPanelProps) {
  const { simulator, inferenceBackend, backendVersion } = useSettings();

  const [question, setQuestion] = React.useState('');
  const [status, setStatus] = React.useState<AskStatus>('idle');
  const [progress, setProgress] = React.useState<ProgressLine[]>([]);
  const [answer, setAnswer] = React.useState('');
  const [citations, setCitations] = React.useState<AssistantCitation[]>([]);
  const [links, setLinks] = React.useState<AssistantLink[]>([]);
  const [unsupported, setUnsupported] = React.useState<string[]>([]);
  const [answeredBy, setAnsweredBy] = React.useState<string | null>(null);
  const [error, setError] = React.useState<string | null>(null);
  const abortRef = React.useRef<AbortController | null>(null);

  React.useEffect(
    () => () => {
      abortRef.current?.abort();
    },
    [],
  );

  const reset = () => {
    setProgress([]);
    setAnswer('');
    setCitations([]);
    setLinks([]);
    setUnsupported([]);
    setAnsweredBy(null);
    setError(null);
  };

  const handleEvent = (event: AssistantEvent) => {
    if (event.type === 'retrieval') {
      if (event.sections.length > 0) {
        setProgress((current) => [
          ...current,
          {
            id: 'retrieval',
            text: `Read ${event.sections.length} documentation section${event.sections.length === 1 ? '' : 's'}`,
          },
        ]);
      }
      return;
    }
    if (event.type === 'tool') {
      setProgress((current) => [
        ...current,
        { id: `tool-${current.length}`, text: event.summary },
      ]);
      return;
    }
    if (event.type === 'error') {
      setStatus('error');
      setError(event.message);
      return;
    }
    if (event.type === 'result') {
      setStatus('done');
      setAnsweredBy(event.model);
      setCitations(event.citations);
      setLinks(event.links);
      setUnsupported(event.unsupportedNumbers);
      // Suppressed prose is never shown; the sources below carry the answer.
      setAnswer(event.status === 'answered' ? event.answer : '');
    }
  };

  const ask = async () => {
    const trimmed = question.trim();
    if (trimmed.length < 3) return;

    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    reset();
    setStatus('asking');

    try {
      const response = await fetch('/api/assistant', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: trimmed,
          route: currentRoute,
          simulator: simulator || undefined,
          backend: inferenceBackend || undefined,
          backendVersion: backendVersion || undefined,
        }),
        signal: controller.signal,
      });

      if (!response.ok) {
        const payload = (await response.json().catch(() => null)) as { message?: string } | null;
        setStatus('error');
        setError(payload?.message ?? 'The assistant is unavailable right now. Search still works.');
        return;
      }

      if (!response.body) {
        setStatus('error');
        setError('The assistant returned no answer. Search still works.');
        return;
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() ?? '';
        for (const line of lines) {
          if (!line.trim()) continue;
          const event = readEvent(line);
          if (event) handleEvent(event);
        }
      }

      if (buffer.trim()) {
        const event = readEvent(buffer);
        if (event) handleEvent(event);
      }
    } catch (caught) {
      if (caught instanceof DOMException && caught.name === 'AbortError') return;
      setStatus('error');
      setError('The assistant could not be reached. Search still works.');
    }
  };

  if (!model) return null;

  return (
    <section className={styles.panel} aria-label="Ask about ConfigIQ">
      <label className={styles.label} htmlFor="assistant-question">
        Ask about ConfigIQ
      </label>
      <TextArea
        id="assistant-question"
        className={styles.input}
        value={question}
        rows={2}
        resizeOrientation="vertical"
        placeholder="e.g. Where do I set concurrency?"
        isDisabled={status === 'asking'}
        onChange={(_event, value) => setQuestion(value)}
        onKeyDown={(event) => {
          if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault();
            void ask();
          }
        }}
      />
      <div className={styles.actions}>
        <Button
          variant="secondary"
          size="sm"
          isDisabled={status === 'asking' || question.trim().length < 3}
          onClick={() => void ask()}
        >
          {status === 'asking' ? 'Asking…' : 'Ask'}
        </Button>
        <span className={styles.hint}>
          Explains the docs and looks up read-only facts. It never runs sizing;
          links open pre-filled so you can calculate.
        </span>
      </div>

      {status === 'asking' && (
        <ul className={styles.progress} aria-live="polite">
          <li className={styles.progressRow}>
            <Spinner size="sm" aria-label="Reading documentation" />
            <span>Reading documentation and catalog…</span>
          </li>
          {progress.map((line) => (
            <li key={line.id} className={styles.progressRow}>
              <span className={styles.check} aria-hidden="true">
                ✓
              </span>
              <span>{line.text}</span>
            </li>
          ))}
        </ul>
      )}

      {status === 'error' && error && (
        <p className={styles.status} role="status">
          {error}
        </p>
      )}

      {status === 'done' && (
        <div className={styles.answerBlock} aria-live="polite">
          {answer ? (
            <p className={styles.answer}>{answer}</p>
          ) : (
            <p className={styles.status}>
              I could not verify the numbers in that answer, so it is not shown. The sources below
              are what I read; open one to check the figure there.
              {unsupported.length > 0 && (
                <span className={styles.unverified}> Unverified: {unsupported.join(', ')}.</span>
              )}
            </p>
          )}

          {citations.length > 0 && (
            <ul className={styles.citations}>
              {citations.map((citation) => (
                <li key={citation.url}>
                  <Link className={styles.citation} href={citation.url} prefetch={false} onClick={hide}>
                    <span>{citation.title}</span>
                    <span className={styles.citationSection}>{citation.section}</span>
                  </Link>
                </li>
              ))}
            </ul>
          )}

          {links.length > 0 && (
            <div className={styles.links}>
              {links.map((link, index) => (
                <Link
                  key={link.href}
                  className={index === 0 ? styles.primaryLink : styles.secondaryLink}
                  href={link.href}
                  prefetch={false}
                  onClick={hide}
                >
                  {link.label}
                </Link>
              ))}
            </div>
          )}

          {answeredBy && (
            <p className={styles.provenance}>
              Answer from the documentation, checked against the sources above.
              <span className={styles.modelName}> {answeredBy}</span>
            </p>
          )}
        </div>
      )}
    </section>
  );
}
