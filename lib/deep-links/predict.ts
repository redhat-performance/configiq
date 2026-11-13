import { z } from 'zod';
import {
  INFERENCE_BACKENDS,
  KV_PRECISIONS,
  LINK_VERSION,
  MAX_ID_LENGTH,
  MAX_SEQUENCE_LENGTH,
  SERVING_MODES,
  WEIGHT_PRECISIONS,
  defineLink,
  modelPath,
  nonNegativeInt,
  positiveInt,
  systemId,
  text,
  type LinkValues,
  type ParsedLink,
} from './common';

/**
 * Deep-link contract for `/predict`.
 *
 * Parameter names are the API field names, not UI labels, so a link stays
 * readable to anyone who knows the API. Every field is optional: a partial
 * link is valid, and anything absent keeps the recipient's own value.
 */

export {
  INFERENCE_BACKENDS,
  KV_PRECISIONS,
  LINK_VERSION,
  SERVING_MODES,
  WEIGHT_PRECISIONS,
};

const FIELDS = {
  model: modelPath,
  system: systemId,
  isl: positiveInt(MAX_SEQUENCE_LENGTH),
  osl: positiveInt(MAX_SEQUENCE_LENGTH),
  concurrency: positiveInt(1_000_000),
  backend: z.enum(INFERENCE_BACKENDS),
  backendVersion: text(MAX_ID_LENGTH),
  simulator: text(MAX_ID_LENGTH),
  servingMode: z.enum(SERVING_MODES),
  fabric: text(MAX_ID_LENGTH),
  tp: positiveInt(1024),
  pp: positiveInt(1024),
  weightPrecision: z.enum(WEIGHT_PRECISIONS),
  kvPrecision: z.enum(KV_PRECISIONS),
  maxModelLen: positiveInt(MAX_SEQUENCE_LENGTH),
  prefillMaxSeqLen: positiveInt(MAX_SEQUENCE_LENGTH),
  decodeMaxSeqLen: positiveInt(MAX_SEQUENCE_LENGTH),
  prefix: nonNegativeInt(MAX_SEQUENCE_LENGTH),
  maxNumSeqs: positiveInt(1_000_000),
  gpuMemUtil: z.coerce.number().min(0.01).max(1),
  moeTp: positiveInt(1024),
  moeEp: positiveInt(1024),
} as const;

export const predictLink = defineLink('/predict', FIELDS);

export type PredictLinkField = keyof typeof FIELDS;

export type PredictLinkValues = LinkValues<typeof FIELDS>;

export type ParsedPredictLink = ParsedLink<typeof FIELDS>;

/**
 * Parse a query string into link values.
 *
 * Each parameter is validated independently so one bad value (say `isl=abc`)
 * cannot discard the rest of an otherwise usable link. Unrecognised keys are
 * reported and ignored rather than being forwarded to the page.
 */
export const parsePredictLink = predictLink.parse;

/** Every value currently set, encoded as a shareable query string. */
export const buildPredictLink = predictLink.build;

/**
 * Remove values that equal the page's resolved defaults, so a link describes
 * the view rather than restating every default.
 */
export const omitDefaults = predictLink.omitDefaults;
