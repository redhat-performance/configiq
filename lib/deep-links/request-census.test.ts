import { describe, expect, it } from 'vitest';
import { baseRequestPayload } from '@/lib/api/recommend';
import { buildKvCachePayload } from '@/lib/api/kv-cache-calc';
import { buildPredictRequestBody, type EstimateAdapterInput } from '@/lib/api/estimate-adapter';
import type { RecommendRequest } from '@/lib/api/schemas';
import type { KvCacheCalcRequest } from '@/lib/api/schemas';
import {
  buildDataParams,
  buildResolveParams,
} from '@/app/validation/compare/benchmark-request';
import {
  gpuExplorerLink,
  hybridSavingsLink,
  kvCacheLink,
  recommendLink,
  scorecardLink,
  validationCompareLink,
} from './pages';
import { predictLink } from './predict';
import { auditRequestCensus, type RequestCensus } from './request-census';

/**
 * One payload that exercises every optional key.
 *
 * The audit reports a classified key that the payload never sends, so these
 * inputs deliberately set every branch: a disagg predict request, a latency
 * recommend request, and a KV request with MoE and a model config. Anything a
 * branch sends shows up here.
 */
function maximalPredictInput(): EstimateAdapterInput {
  return {
    model_path: 'Qwen/Qwen3-235B-A22B',
    system: 'h200_sxm',
    backend: 'vllm',
    backend_version: '0.31.0',
    isl: 4096,
    osl: 512,
    batch_size: 64,
    requests_per_load_unit: 8,
    tp_size: 8,
    pp_size: 2,
    attention_dp_size: 2,
    gpu_memory_utilization: 0.9,
    max_num_seqs: 128,
    enable_chunked_prefill: true,
    max_seq_len: 32768,
    prefill_max_seq_len: 8192,
    decode_max_seq_len: 24576,
    prefix: 512,
    kvcache_quant_mode: 'fp8',
    gemm_quant_mode: 'fp8',
    moe_quant_mode: 'w4a8_mxfp4_mxfp8',
    hf_model_config: { hidden_size: 4096 },
    simulator: 'aisim',
    moe_ep_size: 8,
    moe_tp_size: 1,
    mode: 'disagg',
    fabric: 'ib-400g',
    prefill: { tp: 4, pp: 1, workers: 2, batch: 16 },
    decode: { tp: 8, pp: 2, workers: 4, batch: 256 },
  };
}

function maximalRecommendRequest(): RecommendRequest {
  return {
    model_path: 'Qwen/Qwen3-32B',
    system: 'h200_sxm',
    backend: 'vllm',
    backend_version: '0.31.0',
    isl: 4096,
    osl: 512,
    max_seq_len: 32768,
    prefill_max_seq_len: 8192,
    decode_max_seq_len: 24576,
    ttft: 2000,
    tpot: 50,
    target_request_rate: 12,
    target_concurrency: 32,
    request_latency: 60000,
    prefix: 512,
    top_n: 5,
    model_config: { hidden_size: 4096 },
    simulator: 'aisim',
    fabric: 'ib-400g',
    mode: 'disagg',
    objective: 'min_gpus',
    // `enforce` with no request latency keeps the default TPOT branch reachable;
    // the payload below sets both, so every branch key is present once.
    sla_mode: 'enforce',
    kv_transfer_bandwidth_gb_per_second: 400,
  };
}

function maximalKvRequest(): KvCacheCalcRequest {
  return {
    model_path: 'Qwen/Qwen3-235B-A22B',
    system: 'h200_sxm',
    backend: 'vllm',
    backend_version: '0.31.0',
    max_num_tokens: 32768,
    max_batch_size: 256,
    tp_size: 8,
    pp_size: 2,
    moe_tp_size: 1,
    moe_ep_size: 8,
    memory_fraction_kind: 'of_free',
    memory_fraction_value: 0.9,
    model_config: { hidden_size: 4096 },
  };
}

const predictCensus: RequestCensus = {
  page: '/predict',
  request: {
    model_path: { from: ['model'] },
    system: { from: ['system'] },
    backend: { from: ['backend'] },
    backend_version: { from: ['backendVersion'] },
    isl: { from: ['isl'] },
    osl: { from: ['osl'] },
    batch_size: { from: ['concurrency'] },
    tp_size: { from: ['tp'] },
    pp_size: { from: ['pp'] },
    max_seq_len: { from: ['maxModelLen'] },
    prefill_max_seq_len: { from: ['prefillMaxSeqLen'] },
    decode_max_seq_len: { from: ['decodeMaxSeqLen'] },
    prefix: { from: ['prefix'] },
    simulator: { from: ['simulator'] },
    gpu_memory_utilization: { from: ['gpuMemUtil'] },
    max_num_seqs: { from: ['maxNumSeqs'] },
    kvcache_quant_mode: { from: ['kvPrecision'] },
    gemm_quant_mode: { from: ['weightPrecision'] },
    moe_quant_mode: { from: ['moePrecision'] },
    moe_tp_size: { from: ['moeTp'] },
    moe_ep_size: { from: ['moeEp'] },
    mode: { from: ['servingMode'] },
    enable_chunked_prefill: { from: ['chunkedPrefill'] },
    fabric: { from: ['fabric'] },
    prefill_tp_size: { from: ['prefillTp'] },
    prefill_pp_size: { from: ['prefillPp'] },
    prefill_num_workers: { from: ['prefillWorkers'] },
    prefill_batch_size: { from: ['prefillBatch'] },
    decode_tp_size: { from: ['decodeTp'] },
    decode_pp_size: { from: ['decodePp'] },
    decode_num_workers: { from: ['decodeWorkers'] },
    decode_batch_size: { from: ['decodeBatch'] },
    requests_per_load_unit: { from: ['requestsPerLoadUnit'] },
    attention_dp_size: {
      notShareable: 'no control on this page; attention DP is set by other callers',
    },
    model_config: {
      notShareable: 'the recipient fetches the model config for their own catalog; a link carries no file',
    },
  },
};

const recommendCensus: RequestCensus = {
  page: '/recommend',
  request: {
    model_path: { from: ['model'] },
    system: { from: ['system'] },
    backend: { from: ['backend'] },
    backend_version: { from: ['backendVersion'] },
    isl: { from: ['isl'] },
    osl: { from: ['osl'] },
    prefix: { from: ['prefix'] },
    max_seq_len: { from: ['maxSeqLen'] },
    prefill_max_seq_len: { from: ['prefillMaxSeqLen'] },
    decode_max_seq_len: { from: ['decodeMaxSeqLen'] },
    ttft: { from: ['ttft'] },
    tpot: { from: ['tpot'] },
    request_latency: { from: ['e2eLatency'] },
    target_request_rate: { from: ['demandRate'] },
    target_concurrency: { from: ['concurrency'] },
    simulator: { from: ['simulator'] },
    fabric: { from: ['fabric'] },
    mode: { from: ['servingMode'] },
    // The goal decides both the objective and whether latency constrains it.
    objective: { from: ['goal'] },
    sla_mode: { from: ['goal'] },
    database_mode: {
      notShareable: 'the client pins HYBRID; the page offers no control for it',
    },
    top_n: { notShareable: 'a result-count cap the page leaves to the service' },
    model_config: {
      notShareable: 'the service resolves it from the model catalog; a link carries no file',
    },
    kv_transfer_bandwidth_gb_per_second: {
      notShareable: 'a Pareto sweep topology input this page does not offer',
    },
  },
};

const kvCacheCensus: RequestCensus = {
  page: '/kv-cache',
  request: {
    model_path: { from: ['model'] },
    system: { from: ['system'] },
    backend: { from: ['backend'] },
    backend_version: { from: ['backendVersion'] },
    max_num_tokens: { from: ['maxNumTokens'] },
    max_batch_size: { from: ['maxBatchSize'] },
    memory_fraction_kind: { from: ['memFractionKind'] },
    memory_fraction_value: { from: ['memFractionValue'] },
    // One request per pool: the aggregated view sends the plain width, the
    // disaggregated view sends each pool's own.
    tp_size: { from: ['tp', 'prefillTp', 'decodeTp'] },
    pp_size: { from: ['pp', 'prefillPp', 'decodePp'] },
    moe_tp_size: { from: ['moeTp', 'prefillMoeTp', 'decodeMoeTp'] },
    moe_ep_size: { from: ['moeEp', 'prefillMoeEp', 'decodeMoeEp'] },
    model_config: {
      notShareable: 'the recipient fetches the model config for their own catalog; a link carries no file',
    },
  },
  viewOnly: {
    servingMode: 'the page fans out one request per pool; the mode itself is not a field the service sees',
  },
};

const compareCensus: RequestCensus = {
  page: '/validation/compare',
  request: {
    accelerator: { from: ['accelerator'] },
    model: { from: ['model'] },
    isl: { from: ['isl'] },
    osl: { from: ['osl'] },
    tp: { from: ['tp'] },
    backend: { from: ['backend'] },
    backendVersion: { from: ['backendVersion'] },
    configuration: { from: ['configurationKey'] },
    mode: { notShareable: 'selects the resolve or data request; it is not part of the view' },
  },
  viewOnly: {
    simulator:
      'the measured dataset is not simulator-scoped; each row names the configuration it came from',
  },
};

const scorecardCensus: RequestCensus = {
  page: '/validation/scorecard',
  request: {},
  viewOnly: {
    metric: 'selects which published statistic slice to show; the request returns all of them',
    statistic: 'selects a column of the published totals; not a request field',
    releaseMatch: 'selects the matched or mixed slice the artifact already contains',
    source: 'selects one measured source from the same artifact',
    model: 'selects a slice of the published point detail the page fetches when a row filter is active',
    gpu: 'selects a slice of the published point detail the page fetches when a row filter is active',
    workload: 'selects a slice of the published point detail the page fetches when a row filter is active',
    pair: 'selects one comparison pair the artifact already contains; not a request field',
  },
};

describe('every request key is accounted for by a link field', () => {
  it('/predict', () => {
    expect(
      auditRequestCensus(Object.keys(predictLink.fields), predictCensus, buildPredictRequestBody(maximalPredictInput())),
    ).toEqual([]);
  });

  it('/recommend', () => {
    expect(
      auditRequestCensus(Object.keys(recommendLink.fields), recommendCensus, baseRequestPayload(maximalRecommendRequest())),
    ).toEqual([]);
  });

  it('/kv-cache', () => {
    expect(
      auditRequestCensus(Object.keys(kvCacheLink.fields), kvCacheCensus, buildKvCachePayload(maximalKvRequest())),
    ).toEqual([]);
  });

  it('/validation/scorecard', () => {
    // The page reads one published artifact and narrows it in the browser, so
    // its only keys are transport parameters, not view values.
    expect(auditRequestCensus(Object.keys(scorecardLink.fields), scorecardCensus, {})).toEqual([]);
  });

  it('/validation/compare', () => {    // Both requests the page sends: resolving a selection, then loading the
    // measured rows for it.
    const resolve = Object.fromEntries(
      buildResolveParams(
        { accelerator: 'h100_sxm', model: 'Qwen/Qwen3-32B', isl: '4096', osl: '512', tp: '8' },
        'vllm',
        '0.31.0',
      ).entries(),
    );
    const data = Object.fromEntries(
      buildDataParams(
        {
          accelerator: 'h100_sxm',
          model: 'Qwen/Qwen3-32B',
          backend: 'vllm',
          backendVersion: '0.31.0',
          isl: '4096',
          osl: '512',
          tp: '8',
        },
        'config-key',
      ).entries(),
    );

    expect(
      auditRequestCensus(Object.keys(validationCompareLink.fields), compareCensus, { ...resolve, ...data }),
    ).toEqual([]);
  });
});

describe('pages that submit no request', () => {
  // Documented rather than assumed: if one of these grows an API call, the
  // today-shaped comment is wrong and the census has to be written for it.
  it('/hybrid-savings compares hosted, rented, and owned cost from catalog and price responses only', () => {
    expect(Object.keys(hybridSavingsLink.fields).length).toBeGreaterThan(0);
  });

  it('/gpu-explorer filters a catalog response it has already received', () => {
    expect(Object.keys(gpuExplorerLink.fields).length).toBeGreaterThan(0);
  });
});

describe('the census audit itself', () => {
  const fieldKeys = ['alpha', 'beta'];

  it('reports a request key nobody classified', () => {
    const problems = auditRequestCensus(
      fieldKeys,
      { page: '/probe', request: { alpha: { from: ['alpha'] }, beta: { from: ['beta'] } } },
      { alpha: 1, beta: 2, gamma: 3 },
    );

    expect(problems).toEqual([
      '/probe: request key "gamma" is not classified — carry it in a link field or say why it cannot be shared',
    ]);
  });

  it('reports a contract field the request never carries', () => {
    const problems = auditRequestCensus(
      ['alpha', 'beta'],
      { page: '/probe', request: { alpha: { from: ['alpha'] } } },
      { alpha: 1 },
    );

    expect(problems).toEqual([
      '/probe: contract field "beta" is not carried by the request — bind it or list it as view-only',
    ]);
  });

  it('accepts a view-only field with a reason', () => {
    const problems = auditRequestCensus(
      fieldKeys,
      {
        page: '/probe',
        request: { alpha: { from: ['alpha'] } },
        viewOnly: { beta: 'a control the service never sees' },
      },
      { alpha: 1 },
    );

    expect(problems).toEqual([]);
  });

  it('reports a field name that is not in the contract', () => {
    const problems = auditRequestCensus(
      fieldKeys,
      { page: '/probe', request: { alpha: { from: ['gamma'] }, beta: { from: ['beta'] } } },
      { alpha: 1, beta: 2 },
    );

    expect(problems).toEqual([
      '/probe: request key "alpha" derives from unknown field "gamma"',
      // And, because of that, `alpha` is left uncovered.
      '/probe: contract field "alpha" is not carried by the request — bind it or list it as view-only',
    ]);
  });
});
