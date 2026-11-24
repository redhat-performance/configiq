'use client';

import * as React from 'react';
import { Button } from '@patternfly/react-core';
import CopyIcon from '@patternfly/react-icons/dist/esm/icons/copy-icon';

import styles from './CopyLinkButton.module.css';

const COPIED_LABEL = 'Copied';
const FAILED_LABEL = 'Copy failed';

interface CopyLinkButtonProps {
  /** Build the absolute or root-relative URL to copy. */
  getUrl: () => string | null;
  label?: string;
}

/**
 * Copies the current view as a shareable link.
 *
 * The control reports itself by swapping its label, so the toolbar never grows
 * a status sentence and the actions beside it stay put. The full confirmation
 * or failure is announced from a hidden live region, and a failure carries its
 * reason on the button for sighted users.
 */
export function CopyLinkButton({ getUrl, label = 'Copy link' }: CopyLinkButtonProps) {
  const [state, setState] = React.useState<'idle' | 'copied' | 'error'>('idle');
  // Each attempt gets a new value so a second click re-arms the confirmation
  // instead of letting the first click's timer hide it a moment later.
  const [attempt, setAttempt] = React.useState(0);

  React.useEffect(() => {
    if (state === 'idle') return;
    const timer = window.setTimeout(() => setState('idle'), 2500);
    return () => window.clearTimeout(timer);
  }, [state, attempt]);

  const handleClick = async () => {
    setAttempt(current => current + 1);
    const url = getUrl();
    if (!url) {
      setState('error');
      return;
    }

    const absolute = url.startsWith('http') ? url : `${window.location.origin}${url}`;
    try {
      await navigator.clipboard.writeText(absolute);
      setState('copied');
    } catch {
      setState('error');
    }
  };

  const visibleLabel = state === 'copied'
    ? COPIED_LABEL
    : state === 'error' ? FAILED_LABEL : label;

  const announcement = state === 'copied'
    ? 'Link copied to the clipboard.'
    : state === 'error' ? 'Could not copy the link. Copy the address from the browser instead.' : '';

  return (
    <span className={styles.root}>
      <span className={styles.stack}>
        {[label, COPIED_LABEL, FAILED_LABEL].map(text => (
          <span key={text} className={styles.ghost} aria-hidden="true">
            <Button component="span" variant="link" icon={<CopyIcon />}>{text}</Button>
          </span>
        ))}
        <Button
          variant="link"
          icon={<CopyIcon />}
          onClick={handleClick}
          title={state === 'error' ? 'Could not copy the link' : undefined}
        >
          {visibleLabel}
        </Button>
      </span>
      <span role="status" aria-live="polite" className={styles.status}>
        {announcement}
      </span>
    </span>
  );
}
