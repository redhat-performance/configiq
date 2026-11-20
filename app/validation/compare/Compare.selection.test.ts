import { describe, expect, it } from 'vitest';
import type { ConfigurationOption, DatasetConfiguration } from '@/lib/validation/benchmark-types';
import { selectConfiguration } from './configuration-selection';

function configuration(values: Partial<DatasetConfiguration>): DatasetConfiguration {
  return {
    measurement_source: 'redhat',
    benchmark_version: '1',
    max_num_batched_tokens: null,
    pp: null,
    dp: null,
    ep: null,
    cp: null,
    moe_tp_size: null,
    moe_ep_size: null,
    enable_expert_parallel: null,
    vllm_rocm_use_aiter: null,
    vllm_rocm_use_aiter_moe: null,
    vllm_rocm_use_aiter_moe_situv2: null,
    backend: 'vllm',
    backend_version: '0.9.0',
    precision: 'fp16',
    weight_quantization: 'none',
    kv_quantization: 'none',
    prefix_caching: 'false',
    prefix_tokens: '0',
    prefix_count: '0',
    max_seq_len: '4096',
    max_num_seqs: '128',
    gpu_memory_utilization: '0.9',
    chunked_prefill_enabled: 'true',
    spec_decoding: 'none',
    request_type: 'chat',
    turns: '1',
    tp_prefill: '1',
    tp_decode: '1',
    inferencex_topology: null,
    inferencex_curve: null,
    inferencex_offload: null,
    ...values,
  };
}

const FP16: ConfigurationOption = {
  key: 'fp16-tp1',
  label: 'FP16 · TP1',
  values: configuration({ tp_prefill: '1', tp_decode: '1' }),
  rows: 12,
};
const FP8: ConfigurationOption = {
  key: 'fp8-tp4',
  label: 'FP8 · TP4',
  values: configuration({ precision: 'fp8', tp_prefill: '4', tp_decode: '4' }),
  rows: 9,
};
const INT4: ConfigurationOption = {
  key: 'int4-tp8',
  label: 'INT4 · TP8',
  values: configuration({ precision: 'int4', tp_prefill: '8', tp_decode: '8' }),
  rows: 4,
};
const OPTIONS = [FP16, FP8, INT4];

describe('measured configuration selection', () => {
  it('takes the row a shared link names', () => {
    expect(selectConfiguration(OPTIONS, 'int4-tp8', FP16)).toBe(INT4);
  });

  it('prefers the linked row over the closest match', () => {
    // The nearest row to FP16 would be FP16 itself; the link has to win.
    expect(selectConfiguration(OPTIONS, 'fp8-tp4', FP16)).toBe(FP8);
  });

  it('keeps the previous row when the link names none', () => {
    expect(selectConfiguration(OPTIONS, null, FP8)).toBe(FP8);
  });

  it('falls back to the nearest row when the linked row is gone', () => {
    // A recipient whose dataset lacks the linked row still gets a usable view.
    expect(selectConfiguration(OPTIONS, 'removed-row', FP8)).toBe(FP8);
  });

  it('has nothing to select from an empty list', () => {
    expect(selectConfiguration([], 'fp8-tp4', FP16)).toBeNull();
  });
});
