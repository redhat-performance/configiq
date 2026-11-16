import type { ConfigurationOption, DatasetConfiguration } from '@/lib/validation/benchmark-types';

/**
 * Keeping the visible measured row in step with an incoming list.
 *
 * The comparison page fetches a fresh set of runtime configurations for every
 * selection, so the row it shows has to be chosen again each time. The helper
 * lives here rather than in the page so the precedence can be tested without
 * mounting it.
 */

/**
 * How different two measured runtime configurations are. Keeping the same
 * source matters most, then the recorded precision and runtime settings, so a
 * changed workload dimension repairs to the run that still describes the same
 * deployment.
 */
function configurationDistance(left: DatasetConfiguration, right: DatasetConfiguration): number {
  let score = left.measurement_source === right.measurement_source ? 0 : 100;
  for (const field of Object.keys(left)) {
    if (field === 'measurement_source') continue;
    const leftValue = left[field] ?? '';
    const rightValue = right[field] ?? '';
    if (leftValue !== rightValue) score += leftValue && rightValue ? 4 : 1;
  }
  return score;
}

function nearestConfiguration(
  options: ConfigurationOption[],
  previous: DatasetConfiguration | null,
): ConfigurationOption | null {
  if (options.length === 0) return null;
  if (!previous) return options[0];
  return [...options].sort((left, right) =>
    configurationDistance(left.values, previous) - configurationDistance(right.values, previous),
  )[0];
}

/**
 * The measured row a view should show.
 *
 * A shared link names one explicitly, and that choice wins even when the
 * recipient's previous selection points at a different row. Otherwise the
 * previous choice is kept, and a first visit takes the nearest row to it.
 */
export function selectConfiguration(
  options: ConfigurationOption[],
  requestedKey: string | null,
  previous: ConfigurationOption | null,
): ConfigurationOption | null {
  return options.find(option => option.key === requestedKey)
    ?? options.find(option => option.key === previous?.key)
    ?? nearestConfiguration(options, previous?.values ?? null);
}
