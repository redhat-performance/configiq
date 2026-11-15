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
  withProvenance,
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

/**
 * Weights quantisation schemes a MoE model can be run with, as the vLLM
 * override panel offers them.
 */
const MOE_QUANT_MODES = [
  'w4a16_mxfp4',
  'w4a8_mxfp4_mxfp8',
  'w4a16_mxfp4_cutlass',
  'w4a8_mxfp4_mxfp8_trtllm',
] as const;

/** A MoE weights quantisation scheme the override panel can select. */
export type MoeQuantMode = (typeof MOE_QUANT_MODES)[number];

/** The scheme a model starts on; a link omits it rather than restating it. */
export const DEFAULT_MOE_QUANT_MODE = MOE_QUANT_MODES[0];

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
  // The manual vLLM override panel. A view that pins chunked prefill is a
  // different view from one that leaves the backend to choose, so the choice
  // travels with the link; absent means the backend decides.
  chunkedPrefill: z.enum(['on', 'off']),
  gpuMemUtil: z.coerce.number().min(0.01).max(1),
  moeTp: positiveInt(1024),
  moeEp: positiveInt(1024),
  // MoE weights quantisation: sent only for a MoE model, and only when it is
  // not the starting scheme.
  moePrecision: z.enum(MOE_QUANT_MODES),
  // Disaggregated runs configure each pool independently: a prefill pool and a
  // decode pool can differ in tensor/pipeline parallelism, workers, and batch.
  prefillTp: positiveInt(1024),
  prefillPp: positiveInt(1024),
  prefillWorkers: positiveInt(1024),
  prefillBatch: positiveInt(1_000_000),
  decodeTp: positiveInt(1024),
  decodePp: positiveInt(1024),
  decodeWorkers: positiveInt(1024),
  decodeBatch: positiveInt(1_000_000),
} as const;

export const predictLink = withProvenance(defineLink('/predict', FIELDS));

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
