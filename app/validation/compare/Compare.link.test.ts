import { describe, expect, it } from 'vitest';
import { validationCompareLink } from '@/lib/deep-links/pages';
import { getConfigurationOptions } from '@/lib/validation/benchmark-dataset';

/**
 * A shared benchmark link must name the measured run it was copied from.
 *
 * The configuration key is the dataset's own identifier: a JSON object over the
 * full runtime configuration, not a short catalog id. These tests use a key the
 * producer actually generates so the contract cannot pass against a short
 * fixture that no real link ever carries.
 */
const ROW = {
  benchmark_source: 'redhat',
  version: '0.24.0',
  backend: 'vLLM',
  backend_version: '0.9.0',
  precision: 'FP16',
  weight_quantization: 'FP8',
  kv_quantization: 'FP8',
  prefix_caching: true,
  prefix_tokens: 0,
  prefix_count: 0,
  max_seq_len: 2048,
  max_num_seqs: 128,
  gpu_memory_utilization: 0.9,
  chunked_prefill_enabled: true,
  spec_decoding: null,
  request_type: 'concurrent',
  turns: 1,
  tp_prefill: 1,
  tp_decode: 1,
  inferencex_topology: null,
  inferencex_curve: null,
  inferencex_offload: null,
};

function measuredConfigurationKey(): string {
  const [option] = getConfigurationOptions([ROW]);
  expect(option).toBeDefined();
  return option.key;
}

describe('benchmark comparison shared views', () => {
  it('parses the configuration key the page actually writes', () => {
    const key = measuredConfigurationKey();

    // Guard the fixture: a short key would hide the truncated-key bug.
    expect(key.length).toBeGreaterThan(120);

    const parsed = validationCompareLink.parse(
      `?configurationKey=${encodeURIComponent(key)}`,
    );

    expect(parsed.warnings).toEqual([]);
    expect(parsed.values.configurationKey).toBe(key);
  });

  it('round-trips the measured configuration through build and parse', () => {
    const key = measuredConfigurationKey();

    const link = validationCompareLink.build({ configurationKey: key });
    const parsed = validationCompareLink.parse(link.slice(link.indexOf('?')));

    expect(parsed.values.configurationKey).toBe(key);
    expect(link).toContain('v=1');
  });
});
