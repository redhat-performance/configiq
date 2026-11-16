import type { z } from 'zod';
import type { LinkFieldMap, LinkValues, PageLink } from './common';

/**
 * Binding a link contract to page state.
 *
 * A link field lives in three places: the contract (what a link may carry), the
 * builder (what this view exports), and the applier (what an incoming value
 * changes). Writing those three by hand is how a field silently stops
 * round-tripping, so the export and import sides are declared once, per field,
 * as an accessor table. `useSharedView` derives the builder and the applier
 * from that table.
 *
 * The table is total over the fields the page owns (`PageFieldKeys`): adding a
 * field to `defineLink` is a compile error until the page binds it. A page that
 * spreads `PROVENANCE_FIELDS` binds one fewer set, because `useLinkProvenance`
 * owns the trio. A page marked `withBackendProvenance` binds fewer by the
 * backend pair, which it reads from settings instead of editing.
 */
export interface LinkAccessor<T, V = unknown> {
  /** The value this view would put in a link, or undefined when it does not apply. */
  read: () => T | undefined;
  /**
   * Apply a value from a link to this view's state.
   *
   * `values` is everything the link carried, for the rare field whose value
   * implies another control (a preset implying its axes). A field that needs
   * the whole set staged before publishing uses `onApplied` on the hook.
   */
  write: (value: T, values: V) => void;
}

export type FieldAccessors<F extends LinkFieldMap, B extends keyof F = keyof F> = {
  [K in B]-?: LinkAccessor<z.infer<F[K]>, LinkValues<F>>;
};

/**
 * Declare the accessor table for a page's link.
 *
 * Identity at runtime; the value is the type check. A page that drops a field,
 * or renames one, fails to compile here rather than shipping a link that
 * carries a control the page ignores.
 */
export function defineAccessors<F extends LinkFieldMap, B extends keyof F>(
  _link: PageLink<F, B>,
  accessors: FieldAccessors<F, B>,
): FieldAccessors<F, B> {
  return accessors;
}

/** A table seen through its write side only, so accessor types stay assignable. */
export type WritableAccessors =
  Partial<Record<string, { write: (value: never, values: never) => void }>>;

/** A table seen through its read side only, so accessor types stay assignable. */
export type ReadableAccessors = Record<string, { read: () => unknown }>;

/**
 * Apply the values an incoming link carries.
 *
 * Present values only: an omitted field keeps the recipient's own value, and a
 * field the page does not bind (provenance) is handled by its own hook.
 */
export function applyLinkValues<F extends LinkFieldMap>(
  values: LinkValues<F>,
  accessors: WritableAccessors,
): void {
  for (const [key, value] of Object.entries(values)) {
    if (value === undefined) continue;
    accessors[key]?.write(value as never, values as never);
  }
}

/**
 * Collect the values this view would export.
 *
 * `extra` carries values a page does not own as state but still always sends —
 * the provenance trio, for instance.
 */
export function collectLinkValues<F extends LinkFieldMap>(
  accessors: ReadableAccessors,
  extra?: Partial<LinkValues<F>>,
): LinkValues<F> {
  const values: Record<string, unknown> = {};
  for (const [key, accessor] of Object.entries(accessors)) {
    const value = accessor.read();
    if (value !== undefined) values[key] = value;
  }
  for (const [key, value] of Object.entries(extra ?? {})) {
    if (value !== undefined) values[key] = value;
  }
  return values as LinkValues<F>;
}
