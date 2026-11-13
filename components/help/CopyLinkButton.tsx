'use client';

import * as React from 'react';
import { Button } from '@patternfly/react-core';
import CopyIcon from '@patternfly/react-icons/dist/esm/icons/copy-icon';

interface CopyLinkButtonProps {
  /** Build the absolute or root-relative URL to copy. */
  getUrl: () => string | null;
  label?: string;
}

/**
 * Copies the current view as a shareable link. Renders an inline confirmation
 * rather than a toast so the result is announced to screen readers.
 */
export function CopyLinkButton({ getUrl, label = 'Copy link' }: CopyLinkButtonProps) {
  const [state, setState] = React.useState<'idle' | 'copied' | 'error'>('idle');

  React.useEffect(() => {
    if (state === 'idle') return;
    const timer = window.setTimeout(() => setState('idle'), 2500);
    return () => window.clearTimeout(timer);
  }, [state]);

  const handleClick = async () => {
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

  return (
    <span style={{ display: 'inline-flex', alignItems: 'center', gap: 8 }}>
      <Button variant="link" icon={<CopyIcon />} onClick={handleClick}>
        {label}
      </Button>
      <span role="status" aria-live="polite" style={{ fontSize: 13, color: 'var(--gc-text-2)' }}>
        {state === 'copied' ? 'Link copied' : state === 'error' ? 'Could not copy the link' : ''}
      </span>
    </span>
  );
}
