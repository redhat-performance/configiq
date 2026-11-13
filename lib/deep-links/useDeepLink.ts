'use client';

import * as React from 'react';
import type { LinkFieldMap, LinkValues, PageLink, ParsedLink } from './common';
export interface DeepLinkState<F extends LinkFieldMap> {
  parsed: ParsedLink<F> | null;
  /** Parameters dropped because they were unknown or invalid. */
  warnings: string[];
  /** True once the link has been applied, or once the page was ready and had none. */
  applied: boolean;
}

/**
 * Read a page's shared-view parameters once and apply them after the page is
 * ready.
 *
 * `ready` is the page's own hydration gate (settings loaded, catalog settled).
 * Applying earlier would flash default state and then overwrite it, and would
 * let a catalog response replace a value the link asked for.
 *
 * Nothing here writes to persisted settings: opening a link changes this view
 * only, so a recipient's defaults survive.
 */
export function useDeepLink<F extends LinkFieldMap>(
  link: PageLink<F>,
  ready: boolean,
  apply: (values: LinkValues<F>) => void,
): DeepLinkState<F> {
  const parsedRef = React.useRef<ParsedLink<F> | null>(null);
  if (parsedRef.current === null && typeof window !== 'undefined') {
    parsedRef.current = link.parse(window.location.search);
  }

  const [applied, setApplied] = React.useState(false);
  const applyRef = React.useRef(apply);
  applyRef.current = apply;

  React.useEffect(() => {
    if (!ready || applied) return;
    setApplied(true);
    const values = parsedRef.current?.values;
    if (values && Object.keys(values).length > 0) applyRef.current(values);
  }, [ready, applied]);

  return {
    parsed: parsedRef.current,
    warnings: parsedRef.current?.warnings ?? [],
    applied,
  };
}
