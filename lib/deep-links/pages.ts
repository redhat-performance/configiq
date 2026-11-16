import { z } from 'zod';
import {
  INFERENCE_BACKENDS,
  MAX_ID_LENGTH,
  MAX_SEQUENCE_LENGTH,
  PROVENANCE_FIELDS,
  SERVING_MODES,
  defineLink,
  withBackendProvenance,
  withProvenance,
  modelPath,
  nonNegativeInt,
  positiveInt,
  positiveNumber,
  systemId,
  text,
  type LinkValues,
} from './common';

/**
 * Deep-link contracts for the remaining pages.
 *
 * Each page declares only the fields its own state owns. `/predict` keeps its
 * module because its contract is the largest and predates this registry.
 *
 * Rules shared by every page:
 * - field names are API field names, not UI labels;
 * - every field is optional, so a partial link is valid;
 * - results are never encoded, and no link can trigger a calculation.
 */

/**
 * A measured row's configuration key is the dataset's JSON identifier over
 * every runtime field, not a short catalog id: real keys are several hundred
 * characters. Keep a bound comfortably above the producer's output so a link
 * names the measured run instead of being dropped as malformed.
 */
const MAX_CONFIGURATION_KEY_LENGTH = 4096;

export const recommendLink = withProvenance(defineLink('/recommend', {
  model: modelPath,
  system: systemId,
  goal: z.enum(['latency', 'demand', 'capacity']),
  isl: positiveInt(MAX_SEQUENCE_LENGTH),
  osl: positiveInt(MAX_SEQUENCE_LENGTH),
  concurrency: positiveInt(1_000_000),
  // The form and the request contract both accept a fractional latency target,
  // so the link must too: `ttft=1.4` is a real view, not a malformed one.
  ttft: positiveNumber(600_000),
  tpot: z.coerce.number().min(0.001).max(600_000),
  e2eLatency: positiveNumber(3_600_000),
  demandRate: z.coerce.number().min(0.0001).max(1_000_000),
  prefix: nonNegativeInt(MAX_SEQUENCE_LENGTH),
  maxSeqLen: positiveInt(MAX_SEQUENCE_LENGTH),
  // Declared before the pool lengths and fabric so a link applies the mode
  // first; those fields are only submitted for a disaggregated view.
  servingMode: z.enum(SERVING_MODES),
  prefillMaxSeqLen: positiveInt(MAX_SEQUENCE_LENGTH),
  decodeMaxSeqLen: positiveInt(MAX_SEQUENCE_LENGTH),
  fabric: text(MAX_ID_LENGTH),
  ...PROVENANCE_FIELDS,
}));

/**
 * The fraction kinds the memory request accepts. The link and the page state
 * share this list so a link cannot carry a value the form cannot display and
 * the request would send outside its documented enum.
 */
export const MEMORY_FRACTION_KINDS = ['of_total', 'of_free'] as const;

export type MemoryFractionKind = (typeof MEMORY_FRACTION_KINDS)[number];

/**
 * The KV cache calculator is AISimulate-only by contract, so its link carries
 * the backend but never a simulator: adding one would imply a choice the page
 * does not offer. The backend pair is provenance read from settings and applied
 * for the session, not a pair of controls the page edits.
 */
export const kvCacheLink = withBackendProvenance(defineLink('/kv-cache', {
  model: modelPath,
  system: systemId,
  // Provenance, applied for the session rather than edited here. The backend
  // is the same closed set the other tools accept, so a link always names a
  // value this build can hand to settings.
  backend: z.enum(INFERENCE_BACKENDS),
  backendVersion: text(MAX_ID_LENGTH),
  maxNumTokens: positiveInt(MAX_SEQUENCE_LENGTH),
  maxBatchSize: positiveInt(1_000_000),
  tp: positiveInt(1024),
  pp: positiveInt(1024),
  moeTp: positiveInt(1024),
  moeEp: positiveInt(1024),
  servingMode: z.enum(SERVING_MODES),
  // Both pools carry their own parallelism when the view is disaggregated.
  prefillTp: positiveInt(1024),
  prefillPp: positiveInt(1024),
  prefillMoeTp: positiveInt(1024),
  prefillMoeEp: positiveInt(1024),
  decodeTp: positiveInt(1024),
  decodePp: positiveInt(1024),
  decodeMoeTp: positiveInt(1024),
  decodeMoeEp: positiveInt(1024),
  memFractionKind: z.enum(MEMORY_FRACTION_KINDS),
  memFractionValue: z.coerce.number().min(0).max(1),
}));

export const gpuExplorerLink = defineLink('/gpu-explorer', {
  preset: z.enum(['balanced', 'cost-efficiency', 'performance']),
  vendor: z.enum(['all', 'nvidia', 'amd']),
  x: z.enum(['vram', 'price', 'throughput-index', 'mem-bw']),
  y: z.enum(['vram', 'price', 'throughput-index', 'mem-bw']),
});

export const hybridSavingsLink = withProvenance(defineLink('/hybrid-savings', {
  model: modelPath,
  apiModel: text(200),
  monthlyInputTokens: nonNegativeInt(1e15),
  monthlyOutputTokens: nonNegativeInt(1e15),
  averageInputTokens: positiveInt(MAX_SEQUENCE_LENGTH),
  averageOutputTokens: positiveInt(MAX_SEQUENCE_LENGTH),
  ttft: positiveNumber(600_000),
  tpot: z.coerce.number().min(0.001).max(600_000),
  concurrency: positiveInt(1_000_000),
  prefix: nonNegativeInt(MAX_SEQUENCE_LENGTH),
  hours: z.coerce.number().min(0.01).max(744),
  peak: z.coerce.number().min(1).max(100),
  billingMode: z.enum(['scale-to-zero', 'active-window', 'always-on']),
  costLens: z.enum(['fully-loaded', 'marginal']),
  ...PROVENANCE_FIELDS,
}));

/**
 * Benchmark comparison selects an accelerator by the label the benchmark
 * dataset publishes, not by a catalog system id: the dataset, not the catalog,
 * is the source of that value, so a system id would not resolve for a measured
 * row whose GPU is missing from the catalog.
 */
export const validationCompareLink = withProvenance(defineLink('/validation/compare', {
  model: text(200),
  accelerator: text(MAX_ID_LENGTH),
  isl: positiveInt(MAX_SEQUENCE_LENGTH),
  osl: positiveInt(MAX_SEQUENCE_LENGTH),
  tp: positiveInt(1024),
  // A measured row is identified by its configuration key, not just by the
  // model/accelerator pair: precision and quantisation rows share both.
  configurationKey: text(MAX_CONFIGURATION_KEY_LENGTH),
  ...PROVENANCE_FIELDS,
}));

export type RecommendLinkValues = LinkValues<typeof recommendLink.fields>;
export type KvCacheLinkValues = LinkValues<typeof kvCacheLink.fields>;
export type GpuExplorerLinkValues = LinkValues<typeof gpuExplorerLink.fields>;
export type HybridSavingsLinkValues = LinkValues<typeof hybridSavingsLink.fields>;
export type ValidationCompareLinkValues = LinkValues<typeof validationCompareLink.fields>;
