import { describe, expect, it } from 'vitest';
import {
  gpuExplorerLink,
  hybridSavingsLink,
  kvCacheLink,
  recommendLink,
  validationCompareLink,
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

  it('declares no results, so a link cannot carry an answer', () => {
    for (const link of [recommendLink, kvCacheLink, gpuExplorerLink, hybridSavingsLink, validationCompareLink]) {
      expect(Object.keys(link.fields)).not.toContain('result');
      expect(Object.keys(link.fields)).not.toContain('gpus');
    }
  });
});
