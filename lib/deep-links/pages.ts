import { z } from 'zod';
import {
  MAX_ID_LENGTH,
  MAX_SEQUENCE_LENGTH,
  PROVENANCE_FIELDS,
  SERVING_MODES,
  defineLink,
  modelPath,
  nonNegativeInt,
  positiveInt,
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

export const recommendLink = defineLink('/recommend', {
  model: modelPath,
  system: systemId,
  goal: z.enum(['latency', 'demand', 'capacity']),
  isl: positiveInt(MAX_SEQUENCE_LENGTH),
  osl: positiveInt(MAX_SEQUENCE_LENGTH),
  concurrency: positiveInt(1_000_000),
  ttft: positiveInt(600_000),
  tpot: z.coerce.number().min(0.001).max(600_000),
  e2eLatency: positiveInt(3_600_000),
  demandRate: z.coerce.number().min(0.0001).max(1_000_000),
  prefix: nonNegativeInt(MAX_SEQUENCE_LENGTH),
  maxSeqLen: positiveInt(MAX_SEQUENCE_LENGTH),
  fabric: text(MAX_ID_LENGTH),
  ...PROVENANCE_FIELDS,
});

/**
 * The KV cache calculator is AISimulate-only by contract, so its link carries
 * the backend but never a simulator: adding one would imply a choice the page
 * does not offer.
 */
export const kvCacheLink = defineLink('/kv-cache', {
  model: modelPath,
  system: systemId,
  // The backend list comes from the catalog, so a link accepts any backend id
  // rather than a fixed enum that would go stale as versions are added.
  backend: text(MAX_ID_LENGTH),
  backendVersion: text(MAX_ID_LENGTH),
  maxNumTokens: positiveInt(MAX_SEQUENCE_LENGTH),
  maxBatchSize: positiveInt(1_000_000),
  tp: positiveInt(1024),
  pp: positiveInt(1024),
  moeTp: positiveInt(1024),
  moeEp: positiveInt(1024),
  servingMode: z.enum(SERVING_MODES),
  memFractionKind: text(64),
  memFractionValue: z.coerce.number().min(0).max(1),
});

export const gpuExplorerLink = defineLink('/gpu-explorer', {
  preset: z.enum(['balanced', 'cost-efficiency', 'performance']),
  vendor: z.enum(['all', 'nvidia', 'amd']),
  x: z.enum(['vram', 'price', 'throughput-index', 'mem-bw']),
  y: z.enum(['vram', 'price', 'throughput-index', 'mem-bw']),
});

export const hybridSavingsLink = defineLink('/hybrid-savings', {
  model: modelPath,
  apiModel: text(200),
  monthlyInputTokens: nonNegativeInt(1e15),
  monthlyOutputTokens: nonNegativeInt(1e15),
  averageInputTokens: positiveInt(MAX_SEQUENCE_LENGTH),
  averageOutputTokens: positiveInt(MAX_SEQUENCE_LENGTH),
  ttft: positiveInt(600_000),
  tpot: z.coerce.number().min(0.001).max(600_000),
  concurrency: positiveInt(1_000_000),
  prefix: nonNegativeInt(MAX_SEQUENCE_LENGTH),
  hours: z.coerce.number().min(0.01).max(744),
  peak: z.coerce.number().min(1).max(100),
  billingMode: z.enum(['scale-to-zero', 'active-window', 'always-on']),
  costLens: z.enum(['fully-loaded', 'marginal']),
  ...PROVENANCE_FIELDS,
});

/**
 * Benchmark comparison selects an accelerator by the label the benchmark
 * dataset publishes, not by a catalog system id: the dataset, not the catalog,
 * is the source of that value, so a system id would not resolve for a measured
 * row whose GPU is missing from the catalog.
 */
export const validationCompareLink = defineLink('/validation/compare', {
  model: text(200),
  accelerator: text(MAX_ID_LENGTH),
  isl: positiveInt(MAX_SEQUENCE_LENGTH),
  osl: positiveInt(MAX_SEQUENCE_LENGTH),
  tp: positiveInt(1024),
  ...PROVENANCE_FIELDS,
});

export type RecommendLinkValues = LinkValues<typeof recommendLink.fields>;
export type KvCacheLinkValues = LinkValues<typeof kvCacheLink.fields>;
export type GpuExplorerLinkValues = LinkValues<typeof gpuExplorerLink.fields>;
export type HybridSavingsLinkValues = LinkValues<typeof hybridSavingsLink.fields>;
export type ValidationCompareLinkValues = LinkValues<typeof validationCompareLink.fields>;
