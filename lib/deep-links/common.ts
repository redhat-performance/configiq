import { z } from 'zod';
import { normalizeSimulatorId } from '@/lib/simulator-ids';

/**
 * Shared deep-link primitives.
 *
 * A shared link is a partial description of a view. Every field is optional,
 * malformed values are dropped with a notice rather than crashing the page, and
 * unknown parameters are reported instead of being forwarded. Pages differ only
 * in which fields they accept, so the parsing, building, and default-elision
 * rules live here once.
 */

export const LINK_VERSION = '1';

export const MAX_ID_LENGTH = 120;
export const MAX_MODEL_LENGTH = 200;
export const MAX_SEQUENCE_LENGTH = 10_000_000;

/**
 * Ids reach the API, so reject anything that could carry a scheme or path
 * escape. Model names contain `/` (org/name) and ids contain `.`, `_`, `-`.
 */
export const SAFE_ID = /^[A-Za-z0-9][A-Za-z0-9._\-/]*$/;

export const text = (max: number) => z.string().trim().min(1).max(max);
export const modelPath = text(MAX_MODEL_LENGTH).regex(SAFE_ID, 'invalid model id');
export const systemId = text(MAX_ID_LENGTH).regex(SAFE_ID, 'invalid system id');
export const positiveInt = (max: number) => z.coerce.number().int().min(1).max(max);
export const positiveNumber = (max: number) => z.coerce.number().positive().max(max);
export const nonNegativeInt = (max: number) => z.coerce.number().int().min(0).max(max);

export const SERVING_MODES = ['agg', 'disagg'] as const;
export const INFERENCE_BACKENDS = ['vllm', 'tensorrt-llm', 'sglang'] as const;
export const WEIGHT_PRECISIONS = ['FP16', 'FP8', 'INT8', 'INT4', 'MXFP4', 'NVFP4'] as const;
export const KV_PRECISIONS = ['FP16', 'FP8', 'NVFP4'] as const;

/**
 * `fabric=none` is the explicit "aggregate candidates only" choice. It is a
 * sentinel because the page's own None option is an empty string, and `build`
 * omits empty values; omitting the field instead means "use the recipient's
 * default", which is a different view.
 */
export const FABRIC_NONE = 'none';

/**
 * Provenance trio. Generated links always carry all three so a recipient can
 * tell which implementation produced the numbers, even when their own settings
 * differ.
 */
export const PROVENANCE_FIELDS = {
  backend: z.enum(INFERENCE_BACKENDS),
  backendVersion: text(MAX_ID_LENGTH),
  // A link shared before the simulator rename still names the old id; read it
  // as the current one so the recipient sees the same implementation.
  simulator: text(MAX_ID_LENGTH).transform(normalizeSimulatorId),
} as const;

export type LinkFieldMap = Record<string, z.ZodTypeAny>;

export type LinkValues<F extends LinkFieldMap> = {
  [K in keyof F]?: z.infer<F[K]>;
};

export interface ParsedLink<F extends LinkFieldMap> {
  values: LinkValues<F>;
  /** Parameters dropped because they were malformed, plus unknown keys. */
  warnings: string[];
  /** The link declared a format version this build does not understand. */
  unsupportedVersion?: boolean;
}

/** The keys of `PROVENANCE_FIELDS`, as a type. */
export type ProvenanceKey = keyof typeof PROVENANCE_FIELDS;

/**
 * The fields a page binds to its own state.
 *
 * A page that spreads `PROVENANCE_FIELDS` reads those three from settings
 * instead: they are applied for the session by `useLinkProvenance`, never by
 * the page's own bindings.
 */
export type PageFieldKeys<F extends LinkFieldMap> = Exclude<keyof F, ProvenanceKey>;

export interface ValidatedLink<F extends LinkFieldMap> {
  /** The built URL, carrying only the values the contract accepted. */
  href: string;
  values: LinkValues<F>;
  /** Candidates the contract rejected, with the raw input so a caller can report them. */
  ignored: Array<{ key: string; value: unknown }>;
}

export interface PageLink<F extends LinkFieldMap, B extends keyof F = keyof F> {
  path: string;
  fields: F;
  parse: (search: string) => ParsedLink<F>;
  build: (values: LinkValues<F>, path?: string) => string;
  omitDefaults: (values: LinkValues<F>, defaults: LinkValues<F>) => LinkValues<F>;
  /**
   * Phantom marker: the fields this page is responsible for binding. It is
   * never present at runtime; it exists so a binding table that misses a
   * contract field fails to compile.
   */
  readonly boundFields?: B;
}

/**
 * Declare the link contract for one page.
 *
 * `path` is used for generated links and in messages; `fields` is the complete
 * set of parameters the page accepts.
 */
export function defineLink<F extends LinkFieldMap>(path: string, fields: F): PageLink<F, keyof F> {
  type Values = LinkValues<F>;
  const keys = Object.keys(fields) as (keyof F & string)[];

  function parse(search: string): ParsedLink<F> {
    const params = new URLSearchParams(search);
    const values: Values = {};
    const warnings: string[] = [];

    for (const key of params.keys()) {
      if (key === 'v' || key in fields) continue;
      warnings.push(`Ignored unknown parameter "${key}".`);
    }

    // A link without `v` predates versioning, so it is read as version 1.
    // A link from a version this build does not know may use the same names
    // for different values, so it is refused whole rather than half-applied.
    const version = params.get('v');
    if (version !== null && version !== LINK_VERSION) {
      return {
        values: {},
        unsupportedVersion: true,
        warnings: [
          ...warnings,
          `This link uses an unsupported format version "${version}"; its settings were not applied.`,
        ],
      };
    }

    for (const key of keys) {
      const raw = params.get(key);
      if (raw === null) continue;
      const parsed = fields[key].safeParse(raw);
      if (parsed.success) {
        (values as Record<string, unknown>)[key] = parsed.data;
      } else {
        warnings.push(`Ignored "${key}": the value was not valid.`);
      }
    }

    return { values, warnings };
  }

  function build(values: Values, targetPath = path): string {
    const params = new URLSearchParams();
    params.set('v', LINK_VERSION);
    for (const key of keys) {
      const value = values[key];
      if (value === undefined || value === null || value === '') continue;
      params.set(key, String(value));
    }
    return `${targetPath}?${params.toString()}`;
  }

  function omitDefaults(values: Values, defaults: Values): Values {
    const result: Values = {};
    for (const key of keys) {
      const value = values[key];
      if (value === undefined || value === null || value === '') continue;
      if (defaults[key] !== undefined && String(defaults[key]) === String(value)) continue;
      (result as Record<string, unknown>)[key] = value;
    }
    return result;
  }

  return { path, fields, parse, build, omitDefaults };
}

/**
 * Validate candidate values through a contract and build the URL.
 *
 * `build` serializes whatever it is handed, so a caller assembling values from
 * free text must not call it directly: an out-of-range number would become a
 * link the page then applies. Every candidate goes through the same schema the
 * page parses with, and the rejected ones are returned instead of written.
 */
export function buildValidated<F extends LinkFieldMap>(
  link: PageLink<F>,
  candidates: Record<string, unknown>,
): ValidatedLink<F> {
  const values = {} as LinkValues<F>;
  const ignored: Array<{ key: string; value: unknown }> = [];

  for (const key of Object.keys(link.fields)) {
    const candidate = candidates[key];
    if (candidate === undefined || candidate === null || candidate === '') continue;
    const parsed = link.fields[key].safeParse(candidate);
    if (parsed.success) {
      (values as Record<string, unknown>)[key] = parsed.data;
    } else {
      ignored.push({ key, value: candidate });
    }
  }

  return { href: link.build(values), values, ignored };
}

/**
 * Mark a link whose provenance trio the page reads from settings.
 *
 * The marker narrows the page's binding obligation to the fields it owns, so
 * `useSharedView` asks for a bound accessor per remaining field and the trio is
 * handled once, by `useLinkProvenance`. It cannot be applied to a contract that
 * lacks a provenance field: the compiler rejects the argument.
 */
export function withProvenance<F extends LinkFieldMap & Record<ProvenanceKey, z.ZodTypeAny>>(
  link: PageLink<F, keyof F>,
): PageLink<F, PageFieldKeys<F>> {
  // The marker is type-only: `boundFields` is never set at runtime. Narrowing
  // it is the whole point of this helper, so it cannot be expressed by copying
  // the object.
  return link as PageLink<F, PageFieldKeys<F>>;
}

/** The provenance fields a capability-only view carries without a simulator. */
export type BackendProvenanceKey = 'backend' | 'backendVersion';

/** The keys a page with backend provenance still binds itself. */
export type BackendPageFieldKeys<F extends LinkFieldMap> = Exclude<keyof F, BackendProvenanceKey>;

/**
 * Mark a link whose backend and version the page reads from settings.
 *
 * A view with no simulator choice — the KV cache calculator, for example —
 * still records the backend and version that produced it, but it has no
 * simulator to offer the reader. Its backend pair is applied for the
 * session by the same handling as the provenance trio, so it binds one fewer
 * set of accessors without claiming a simulator the page cannot switch to.
 */
export function withBackendProvenance<
  F extends LinkFieldMap & Record<BackendProvenanceKey, z.ZodTypeAny>,
>(
  link: PageLink<F, keyof F>,
): PageLink<F, BackendPageFieldKeys<F>> {
  return link as PageLink<F, BackendPageFieldKeys<F>>;
}
