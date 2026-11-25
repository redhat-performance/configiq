import { z } from 'zod';
import {
  INFERENCE_BACKENDS,
  MAX_ID_LENGTH,
  MAX_SEQUENCE_LENGTH,
  SERVING_MODES,
  defineLink,
  nonNegativeInt,
  positiveInt,
  positiveNumber,
  text,
  type LinkValues,
} from './common';

/**
 * The tested-model detail view.
 *
 * The page addresses one measured pair by its path segment; the link carries
 * the curated view of that pair: which measured workload is on the chart, and
 * optionally a prediction snapshot to plot against it.
 *
 * The prediction is a *snapshot*, not a request. Opening the link must not
 * re-run a simulator or re-classify anything, so every value the overlay draws
 * is encoded here: the workload, concurrency, parallelism, predicted throughput,
 * the verdict the prediction page showed, and the provenance that produced it.
 * The verdict is therefore reader-visible and user-editable, and the page labels
 * it as coming from the shared prediction rather than asserting it as its own.
 */

const CLASSIFIER_STATUSES = [
  'within_validated_range',
  'near_validated_boundary',
  'outside_validated_range',
  'measured_no_envelope',
  'no_measurements',
  'validation_unavailable',
] as const;

export const testedModelLink = defineLink('/tested-models/[pairId]', {
  // Chart selection. `turns` is a workload dimension, not a numeric axis: a
  // multi-turn aggregate is a different measurement, kept selectable and
  // labelled rather than mixed into a single-turn curve.
  isl: nonNegativeInt(MAX_SEQUENCE_LENGTH),
  osl: nonNegativeInt(MAX_SEQUENCE_LENGTH),
  turns: positiveInt(1_000),
  // The line the reader is looking at, when they have focused one. The chart
  // still draws every measured TP so the comparison has context.
  tp: positiveInt(1_024),

  // Prediction snapshot. Required together to plot a point; a link that names
  // only some of them shows no overlay rather than inventing the rest.
  predIsl: nonNegativeInt(MAX_SEQUENCE_LENGTH),
  predOsl: nonNegativeInt(MAX_SEQUENCE_LENGTH),
  predConcurrency: positiveInt(1_000_000),
  predTp: positiveInt(1_024),
  predThroughput: positiveNumber(1_000_000_000),
  predStatus: z.enum(CLASSIFIER_STATUSES),
  predConfidence: z.coerce.number().min(0).max(1),
  predBackend: z.enum(INFERENCE_BACKENDS),
  predBackendVersion: text(MAX_ID_LENGTH),
  predServingMode: z.enum(SERVING_MODES),
});

export type TestedModelLinkValues = LinkValues<typeof testedModelLink.fields>;
