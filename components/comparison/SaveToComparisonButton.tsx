'use client';

import * as React from 'react';
import { Button, Tooltip } from '@patternfly/react-core';
import BookmarkIcon from '@patternfly/react-icons/dist/esm/icons/bookmark-icon';
import CheckIcon from '@patternfly/react-icons/dist/esm/icons/check-icon';
import ExclamationCircleIcon from '@patternfly/react-icons/dist/esm/icons/exclamation-circle-icon';
import Link from 'next/link';

import { upsertComparisonEntry } from '@/lib/comparison/storage';
import type { ComparisonEntry } from '@/lib/comparison/types';

import styles from './SaveToComparisonButton.module.css';

const FEEDBACK_MS = 4000;

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
 * service. A refusal (full set, storage disabled) is shown inline beside the
 * button rather than as a toast, so the reason stays readable and announced.
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

  const button = (
    <Button
      variant="secondary"
      icon={state.kind === 'saved' ? <CheckIcon /> : <BookmarkIcon />}
      onClick={handleClick}
      isDisabled={disabledReason != null}
      aria-label={disabledReason == null ? 'Save for comparison' : `Save for comparison unavailable. ${disabledReason}`}
    >
      {state.kind === 'saved' ? 'Saved' : 'Save for comparison'}
    </Button>
  );

  // The status sits beside the control in a nowrap box: the box may grow when a
  // message appears, but it cannot wrap the empty live region onto a second
  // line, which would push the button above its neighbours in the header row.
  return (
    <div className={styles.root}>
      {disabledReason == null ? button : (
        <Tooltip content={disabledReason}>
          <span tabIndex={0} className={styles.disabled}>
            {button}
          </span>
        </Tooltip>
      )}
      <span role="status" aria-live="polite" className={styles.status}>
        {state.kind === 'saved' ? (
          <>
            Saved to comparison. <Link href="/compare">View comparison</Link>
          </>
        ) : state.kind === 'error' ? (
          <>
            <ExclamationCircleIcon /> {state.message}
          </>
        ) : ''}
      </span>
    </div>
  );
}
