'use client';

import * as React from 'react';
import type { LinkFieldMap, LinkValues, PageLink } from './common';
import { applyLinkValues, collectLinkValues, type FieldAccessors } from './shared-view';
import { useDeepLink, type DeepLinkState } from './useDeepLink';

export interface SharedViewOptions<F extends LinkFieldMap, B extends keyof F> {
  /** One accessor per field this page owns; the compiler requires all of them. */
  fields: FieldAccessors<F, B>;
  /** The page's hydration gate, as for `useDeepLink`. */
  ready: boolean;
  /** The page's own starting values, so a link describes the view. */
  defaults?: LinkValues<F>;
  /** Values this page always sends but does not own as state (provenance). */
  provenance?: Partial<LinkValues<F>>;
  /** Runs after a link's values are applied: mark a retained result stale, etc. */
  onApplied?: () => void;
}

export interface SharedView<F extends LinkFieldMap> extends DeepLinkState<F> {
  /** The canonical link for the view as it stands. */
  build: () => string;
}

/**
 * Bind a page's link contract to its state.
 *
 * Builds the link from the accessor table's read side and applies it through
 * the write side, so the two directions can never disagree about which fields
 * exist. A field added to the contract without a page binding is a compile
 * error, which is the point: the contract is enforced by the compiler rather
 * than by review.
 */
export function useSharedView<F extends LinkFieldMap, B extends keyof F>(
  link: PageLink<F, B>,
  options: SharedViewOptions<F, B>,
): SharedView<F> {
  const { ready, defaults, provenance, onApplied } = options;
  const fields = React.useRef(options.fields);
  fields.current = options.fields;
  const applied = React.useRef(onApplied);
  applied.current = onApplied;

  const state = useDeepLink(link, ready, (values) => {
    applyLinkValues(values, fields.current);
    applied.current?.();
  });

  const build = React.useCallback(
    () => {
      const values = collectLinkValues<F>(fields.current, provenance);
      return link.build(defaults ? link.omitDefaults(values, defaults) : values);
    },
    [link, defaults, provenance],
  );

  return { ...state, build };
}
