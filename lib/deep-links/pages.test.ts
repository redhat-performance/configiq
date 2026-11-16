import { describe, expect, it } from 'vitest';
import { FABRIC_NONE } from './common';
import {
  gpuExplorerLink,
  hybridSavingsLink,
  kvCacheLink,
  recommendLink,
  validationCompareLink,
  type KvCacheLinkValues,
  type RecommendLinkValues,
} from './pages';

function query(url: string): string {
  return url.slice(url.indexOf('?'));
}

describe('page link contracts', () => {
  it('parses the fields a page owns', () => {
    const parsed = recommendLink.parse(
      '?v=1&model=Qwen%2FQwen3-32B&system=h200_sxm&goal=capacity&concurrency=32&simulator=blis&backend=vllm',
    );

    expect(parsed.values).toEqual({
      model: 'Qwen/Qwen3-32B',
      system: 'h200_sxm',
      goal: 'capacity',
      concurrency: 32,
      simulator: 'blis',
      backend: 'vllm',
    });
    expect(parsed.warnings).toEqual([]);
  });

  it('reports unknown parameters instead of forwarding them', () => {
    const parsed = gpuExplorerLink.parse('?preset=performance&autorun=1');

    expect(parsed.values).toEqual({ preset: 'performance' });
    expect(parsed.warnings).toEqual(['Ignored unknown parameter "autorun".']);
  });

  it('drops an invalid value and keeps the rest of the link', () => {
    const parsed = validationCompareLink.parse(
      '?model=meta-llama%2FLlama-3.1-70B-Instruct&isl=not-a-number&tp=8',
    );

    expect(parsed.values).toEqual({ model: 'meta-llama/Llama-3.1-70B-Instruct', tp: 8 });
    expect(parsed.warnings).toEqual(['Ignored "isl": the value was not valid.']);
  });

  it('rejects values outside the field range', () => {
    expect(recommendLink.parse('?concurrency=0').values).toEqual({});
    expect(recommendLink.parse('?goal=turbo').values).toEqual({});
    expect(recommendLink.parse('?model=https%3A%2F%2Fevil.test').values).toEqual({});
  });

  it('keeps a link readable by omitting defaults', () => {
    const values: RecommendLinkValues = {
      model: 'Qwen/Qwen3-32B',
      goal: 'demand',
      isl: 2048,
      osl: 4096,
      simulator: 'blis',
    };

    expect(recommendLink.omitDefaults(values, { goal: 'demand', isl: 2048 })).toEqual({
      model: 'Qwen/Qwen3-32B',
      osl: 4096,
      simulator: 'blis',
    });
  });

  it('round-trips a built link', () => {
    const values: RecommendLinkValues = {
      model: 'Qwen/Qwen3-32B',
      system: 'h200_sxm',
      goal: 'latency',
      ttft: 800,
      tpot: 25,
      backend: 'vllm',
      backendVersion: '0.9.0',
      simulator: 'blis',
    };

    expect(recommendLink.parse(query(recommendLink.build(values)))).toEqual({
      values,
      warnings: [],
    });
  });

  it('carries provenance on pages that offer a simulator choice', () => {
    const built = hybridSavingsLink.build({
      model: 'Qwen/Qwen3-32B',
      simulator: 'aisimulate',
      backend: 'vllm',
      backendVersion: '0.9.0',
    });
    const params = new URLSearchParams(query(built));

    expect(params.get('v')).toBe('1');
    expect(params.get('simulator')).toBe('aisimulate');
    expect(params.get('backend')).toBe('vllm');
    expect(params.get('backendVersion')).toBe('0.9.0');
  });

  it('never puts a simulator in a KV cache link', () => {
    // The calculator is AISimulate-only by contract, so a simulator parameter
    // would imply a choice the page does not offer.
    const built = kvCacheLink.build({
      model: 'Qwen/Qwen3-32B',
      system: 'h200_sxm',
      backend: 'vllm',
    });
    const params = new URLSearchParams(query(built));

    expect(params.has('simulator')).toBe(false);
    expect(params.get('backend')).toBe('vllm');
  });

  it('round-trips the recommend fields it now owns', () => {
    const values: RecommendLinkValues = {
      model: 'Qwen/Qwen3-32B',
      system: 'h200_sxm',
      goal: 'latency',
      concurrency: 32,
      prefillMaxSeqLen: 8192,
      decodeMaxSeqLen: 4096,
      fabric: 'ib-400g',
      simulator: 'blis',
      backend: 'vllm',
    };

    expect(recommendLink.parse(query(recommendLink.build(values)))).toEqual({
      values,
      warnings: [],
    });
  });

  it('carries an explicit "no fabric" choice instead of dropping it', () => {
    // The page's None option is an empty string, which `build` would omit, and
    // an omitted field means "use the recipient's default" — a different view.
    const built = recommendLink.build({ fabric: FABRIC_NONE });
    const params = new URLSearchParams(query(built));

    expect(params.get('fabric')).toBe('none');
    expect(recommendLink.parse(query(built)).values).toEqual({ fabric: FABRIC_NONE });
  });

  it('accepts a fractional latency target the form can produce', () => {
    // The TTFT and E2E latency inputs accept decimals and the request contract
    // takes floats; an integer-only link field would reject the sender's view.
    const values: RecommendLinkValues = {
      goal: 'latency',
      ttft: 1.4,
      e2eLatency: 12.5,
    };

    expect(recommendLink.parse(query(recommendLink.build(values)))).toEqual({
      values,
      warnings: [],
    });
    expect(hybridSavingsLink.parse('?ttft=0.5').values).toEqual({ ttft: 0.5 });
  });

  it('drops a fraction kind the request contract does not accept', () => {
    const parsed = kvCacheLink.parse('?memFractionKind=bogus&memFractionValue=0.5');

    expect(parsed.values).toEqual({ memFractionValue: 0.5 });
    expect(parsed.warnings).toEqual(['Ignored "memFractionKind": the value was not valid.']);
  });

  it('round-trips both pool parallelism sets for the KV cache calculator', () => {
    const values: KvCacheLinkValues = {
      model: 'Qwen/Qwen3-32B',
      servingMode: 'disagg',
      prefillTp: 2,
      prefillPp: 1,
      prefillMoeTp: 2,
      prefillMoeEp: 2,
      decodeTp: 4,
      decodePp: 2,
      decodeMoeTp: 4,
      decodeMoeEp: 4,
    };

    expect(kvCacheLink.parse(query(kvCacheLink.build(values)))).toEqual({
      values,
      warnings: [],
    });
  });

  it('identifies a measured row by its configuration key', () => {
    const built = validationCompareLink.build({ configurationKey: 'fp16-tp4' });
    const params = new URLSearchParams(query(built));

    expect(params.get('configurationKey')).toBe('fp16-tp4');
    expect(validationCompareLink.parse(query(built)).values).toEqual({
      configurationKey: 'fp16-tp4',
    });
  });

  it('declares no results, so a link cannot carry an answer', () => {
    for (const link of [recommendLink, kvCacheLink, gpuExplorerLink, hybridSavingsLink, validationCompareLink]) {
      expect(Object.keys(link.fields)).not.toContain('result');
      expect(Object.keys(link.fields)).not.toContain('gpus');
    }
  });
});
