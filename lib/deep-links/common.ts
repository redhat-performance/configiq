import { z } from 'zod';

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
export const nonNegativeInt = (max: number) => z.coerce.number().int().min(0).max(max);

export const SERVING_MODES = ['agg', 'disagg'] as const;
export const INFERENCE_BACKENDS = ['vllm', 'tensorrt-llm', 'sglang'] as const;
export const WEIGHT_PRECISIONS = ['FP16', 'FP8', 'INT8', 'INT4', 'MXFP4', 'NVFP4'] as const;
export const KV_PRECISIONS = ['FP16', 'FP8', 'NVFP4'] as const;

/**
 * Provenance trio. Generated links always carry all three so a recipient can
 * tell which implementation produced the numbers, even when their own settings
 * differ.
 */
export const PROVENANCE_FIELDS = {
  backend: z.enum(INFERENCE_BACKENDS),
  backendVersion: text(MAX_ID_LENGTH),
  simulator: text(MAX_ID_LENGTH),
} as const;

export type LinkFieldMap = Record<string, z.ZodTypeAny>;

export type LinkValues<F extends LinkFieldMap> = {
  [K in keyof F]?: z.infer<F[K]>;
};

export interface ParsedLink<F extends LinkFieldMap> {
  values: LinkValues<F>;
  /** Parameters dropped because they were malformed, plus unknown keys. */
  warnings: string[];
}

export interface PageLink<F extends LinkFieldMap> {
  path: string;
  fields: F;
  parse: (search: string) => ParsedLink<F>;
  build: (values: LinkValues<F>, path?: string) => string;
  omitDefaults: (values: LinkValues<F>, defaults: LinkValues<F>) => LinkValues<F>;
}

/**
 * Declare the link contract for one page.
 *
 * `path` is used for generated links and in messages; `fields` is the complete
 * set of parameters the page accepts.
 */
export function defineLink<F extends LinkFieldMap>(path: string, fields: F): PageLink<F> {
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
