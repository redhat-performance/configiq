import { describe, expect, it } from 'vitest';
import { buildDataParams, buildResolveParams, mergeSelection, type Selection } from './benchmark-request';

const selection: Selection = {
  model: 'Qwen/Qwen3-32B',
  accelerator: 'h100_sxm',
  isl: '4096',
  osl: '512',
  tp: '8',
};

describe('mergeSelection', () => {
  it('keeps the current value for a field the link did not carry', () => {
    expect(mergeSelection(selection, { isl: '8192' })).toEqual({
      ...selection,
      isl: '8192',
    });
  });

  it('does not clear a filter the link says nothing about', () => {
    // A link naming only the model must not empty the accelerator and widths
    // the reader already had open.
    expect(mergeSelection(selection, { model: 'meta-llama/Llama-3.1-70B-Instruct' })).toEqual({
      ...selection,
      model: 'meta-llama/Llama-3.1-70B-Instruct',
    });
  });
});

describe('buildResolveParams', () => {
  it('omits a blank filter rather than sending it empty', () => {
    const params = buildResolveParams({ ...selection, tp: '' }, 'vllm', '');

    expect([...params.keys()]).toEqual(['mode', 'accelerator', 'model', 'isl', 'osl', 'backend']);
    expect(params.get('backend')).toBe('vllm');
  });
});

describe('buildDataParams', () => {
  it('sends every resolved filter, including the configuration', () => {
    const params = buildDataParams(
      {
        accelerator: 'h100_sxm',
        model: 'Qwen/Qwen3-32B',
        backend: 'vllm',
        backendVersion: '0.31.0',
        isl: '4096',
        osl: '512',
        tp: '8',
      },
      'fp16-tp8',
    );

    expect(Object.fromEntries(params.entries())).toEqual({
      mode: 'data',
      configuration: 'fp16-tp8',
      accelerator: 'h100_sxm',
      model: 'Qwen/Qwen3-32B',
      backend: 'vllm',
      backendVersion: '0.31.0',
      isl: '4096',
      osl: '512',
      tp: '8',
    });
  });
});
