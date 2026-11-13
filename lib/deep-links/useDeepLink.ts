'use client';

import * as React from 'react';
import { useSearchParams } from 'next/navigation';
import type { LinkFieldMap, LinkValues, PageLink, ParsedLink } from './common';

export interface DeepLinkState<F extends LinkFieldMap> {
  parsed: ParsedLink<F> | null;
  /** Parameters dropped because they were unknown or invalid. */
  warnings: string[];
  /** True once the link has been applied, or once the page was ready and had none. */
  applied: boolean;
}

/**
 * Read a page's shared-view parameters and apply them after the page is ready.
 *
 * `ready` is the page's own hydration gate (settings loaded, catalog settled).
 * Applying earlier would flash default state and then overwrite it, and would
 * let a catalog response replace a value the link asked for.
 *
 * The search string is the source of truth, so a link opened while the page is
 * already mounted (a pasted URL, a back/forward entry) applies too. After each
 * application the address bar is rewritten to the canonical link: unknown and
 * invalid parameters are gone, and `v=1` marks what the reader understood.
 *
 * Nothing here writes to persisted settings: opening a link changes this view
 * only, so a recipient's defaults survive.
 */
export function useDeepLink<F extends LinkFieldMap>(
  link: PageLink<F>,
  ready: boolean,
  apply: (values: LinkValues<F>) => void,
): DeepLinkState<F> {
  const searchParams = useSearchParams();
  const search = searchParams.toString();
  const parsed = React.useMemo(
    () => link.parse(search ? `?${search}` : ''),
    [link, search],
  );

  const [applied, setApplied] = React.useState(false);
  const appliedSearch = React.useRef<string | null>(null);
  const applyRef = React.useRef(apply);
  applyRef.current = apply;

  React.useEffect(() => {
    if (!ready || appliedSearch.current === search) return;
    appliedSearch.current = search;
    setApplied(true);

    const values = parsed.values;
    if (Object.keys(values).length > 0) applyRef.current(values);

    // Keep `history.state` intact: Next keeps its own navigation state there,
    // and replacing the object would discard it. A link from an unknown format
    // version is left as written rather than relabelled as this build's link.
    if (!parsed.unsupportedVersion) {
      const canonical = link.build(values, window.location.pathname);
      if (window.location.search !== canonical.slice(canonical.indexOf('?'))) {
        window.history.replaceState(window.history.state, '', canonical);
      }
    }
  }, [ready, search, parsed, link]);

  return {
    parsed,
    warnings: parsed.warnings,
    applied,
  };
}
