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
  /**
   * Values a field takes when an opened link does not name it.
   *
   * A field whose page default has moved must keep the meaning absence had in
   * links that were already shared: applying the recipient's new default would
   * state a value the link never described. Pages that rely on this encode the
   * field in every link they build, so only older links leave it out.
   */
  absent?: LinkValues<F>;
  /** Values this page always sends but does not own as state (provenance). */
  provenance?: Partial<LinkValues<F>>;
  /** Runs after a link's values are applied: mark a retained result stale, etc. */
  onApplied?: (values: LinkValues<F>) => void;
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
  const absent = React.useRef(options.absent);
  absent.current = options.absent;

  const state = useDeepLink(link, ready, (values) => {
    const fill = absent.current;
    if (fill) {
      // An opened link states every field it means to change; a field it leaves
      // out keeps the meaning absence had when the link was written.
      const merged = { ...fill } as Record<string, unknown>;
      for (const [key, value] of Object.entries(values)) merged[key] = value;
      applyLinkValues(merged as LinkValues<F>, fields.current);
    } else {
      applyLinkValues(values, fields.current);
    }
    // The hook's caller sees what the link itself stated, not the absence fill,
    // so a field the link omits can be reported as unaddressed.
    applied.current?.(values);
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
