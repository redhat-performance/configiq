/**
 * The query strings for the validation benchmark requests.
 *
 * Kept beside the page rather than inline in its effects, so the requests the
 * page sends and the deep-link contract can be audited against the same
 * builders instead of two hand-kept copies.
 */

import {
  BENCHMARK_FILTER_KEYS,
  type BenchmarkFilters,
} from '@/lib/validation/benchmark-types';

export type SelectionKey = 'accelerator' | 'model' | 'isl' | 'osl' | 'tp';

export type Selection = Pick<BenchmarkFilters, SelectionKey>;

export const SELECTION_KEYS: readonly SelectionKey[] = [
  'accelerator',
  'model',
  'isl',
  'osl',
  'tp',
];

/**
 * Merge the fields a link staged over the page's current selection.
 *
 * A link carries a selection one field at a time, but comparing is one request:
 * the page stages each field and publishes the merged result once. A blank or
 * omitted field keeps the current value, so a partial link narrows the
 * comparison instead of clearing the filters it says nothing about.
 */
export function mergeSelection(current: Selection, staged: Partial<Selection>): Selection {
  return {
    model: staged.model ?? current.model,
    accelerator: staged.accelerator ?? current.accelerator,
    isl: staged.isl ?? current.isl,
    osl: staged.osl ?? current.osl,
    tp: staged.tp ?? current.tp,
  };
}

/**
 * Build the resolve query for one selection.
 *
 * A blank selection field is omitted rather than sent empty: the dataset
 * resolves the remaining filters, and an empty string would narrow to nothing.
 * The backend and version describe the simulator profile the comparison is
 * resolved against, not the measured row itself.
 */
export function buildResolveParams(
  selection: Selection,
  backend: string,
  backendVersion: string,
): URLSearchParams {
  const params = new URLSearchParams({ mode: 'resolve' });
  for (const key of SELECTION_KEYS) {
    if (selection[key]) params.set(key, selection[key]);
  }
  if (backend) params.set('backend', backend);
  if (backendVersion) params.set('backendVersion', backendVersion);
  return params;
}

/**
 * Build the data query for one measured configuration.
 *
 * The filter values are already resolved by the service, so every one of them
 * is sent: an omitted filter here would widen the dataset rather than narrow
 * the comparison.
 */
export function buildDataParams(
  filters: BenchmarkFilters,
  configurationKey: string,
): URLSearchParams {
  const params = new URLSearchParams({ mode: 'data', configuration: configurationKey });
  for (const key of BENCHMARK_FILTER_KEYS) params.set(key, filters[key]);
  return params;
}
