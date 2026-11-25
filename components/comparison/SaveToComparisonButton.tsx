'use client';

import * as React from 'react';
import { Button, Tooltip } from '@patternfly/react-core';
import BookmarkIcon from '@patternfly/react-icons/dist/esm/icons/bookmark-icon';
import CheckIcon from '@patternfly/react-icons/dist/esm/icons/check-icon';
import ExclamationCircleIcon from '@patternfly/react-icons/dist/esm/icons/exclamation-circle-icon';

import { upsertComparisonEntry } from '@/lib/comparison/storage';
import type { ComparisonEntry } from '@/lib/comparison/types';

import styles from './SaveToComparisonButton.module.css';

const FEEDBACK_MS = 4000;

const IDLE_LABEL = 'Save result';
const SAVED_LABEL = 'Saved';
const ERROR_LABEL = 'Not saved';

type SaveState =
  | { kind: 'idle' }
  | { kind: 'saved' }
  | { kind: 'error'; message: string };

interface SaveToComparisonButtonProps {
  /**
   * Build the snapshot at the moment of saving, or null when there is no
   * result to save. Building on click keeps the saved figures the ones the
   * button was pressed against.
   */
  buildEntry: () => ComparisonEntry | null;
  /** Why saving is unavailable right now, or null when it is available. */
  disabledReason: string | null;
}

/**
 * Save the current result to the browser-local comparison set.
 *
 * The snapshot is written on this device only; nothing is sent to a ConfigIQ
 * service. Feedback is the control's own label, so the header never grows a
 * status sentence and the neighbouring actions do not move. A refusal is
 * announced in a hidden live region and its reason is available on the button
 * itself, rather than printed inline where it would reflow the toolbar.
 */
export function SaveToComparisonButton({ buildEntry, disabledReason }: SaveToComparisonButtonProps) {
  const [state, setState] = React.useState<SaveState>({ kind: 'idle' });

  React.useEffect(() => {
    if (state.kind === 'idle') return;
    const timer = window.setTimeout(() => setState({ kind: 'idle' }), FEEDBACK_MS);
    return () => window.clearTimeout(timer);
  }, [state]);

  const handleClick = () => {
    const entry = buildEntry();
    if (entry == null) {
      setState({ kind: 'error', message: 'There is no result to save yet.' });
      return;
    }
    const written = upsertComparisonEntry(entry);
    setState(written.ok
      ? { kind: 'saved' }
      : { kind: 'error', message: written.problem });
  };

  const label = state.kind === 'saved'
    ? SAVED_LABEL
    : state.kind === 'error' ? ERROR_LABEL : IDLE_LABEL;

  const announcement = state.kind === 'saved'
    ? 'Saved to comparison.'
    : state.kind === 'error' ? `Not saved. ${state.message}` : '';

  const button = (
    <Button
      variant="link"
      className={styles.button}
      icon={state.kind === 'saved'
        ? <CheckIcon />
        : state.kind === 'error' ? <ExclamationCircleIcon /> : <BookmarkIcon />}
      onClick={handleClick}
      isDisabled={disabledReason != null}
      title={state.kind === 'error' ? state.message : undefined}
      aria-label={disabledReason == null ? undefined : `Save result unavailable. ${disabledReason}`}
    >
      {label}
    </Button>
  );

  // A hidden copy of each label reserves the widest state inside one grid cell,
  // so swapping the label never changes the control's width or nudges the
  // actions beside it. The live region is off-screen for the same reason.
  return (
    <div className={styles.root}>
      <span className={styles.stack}>
        {[IDLE_LABEL, SAVED_LABEL, ERROR_LABEL].map(text => (
          <span key={text} className={styles.ghost} aria-hidden="true">
            <Button component="span" variant="link" icon={<BookmarkIcon />}>{text}</Button>
          </span>
        ))}
        {disabledReason == null ? button : (
          <Tooltip content={disabledReason}>
            <span tabIndex={0} className={styles.disabled}>
              {button}
            </span>
          </Tooltip>
        )}
      </span>
      <span role="status" aria-live="polite" className={styles.status}>
        {announcement}
      </span>
    </div>
  );
}
