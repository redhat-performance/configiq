import { describe, expect, it } from 'vitest';
import {
  buildPredictLink,
  omitDefaults,
  parsePredictLink,
  type PredictLinkValues,
} from './predict';

describe('predict deep links', () => {
  it('parses the fields it knows', () => {
    const parsed = parsePredictLink(
      '?v=1&model=Qwen%2FQwen3-32B&system=h200_sxm&isl=4096&concurrency=32&simulator=blis',
    );

    expect(parsed.values).toEqual({
      model: 'Qwen/Qwen3-32B',
      system: 'h200_sxm',
      isl: 4096,
      concurrency: 32,
      simulator: 'blis',
    });
    expect(parsed.warnings).toEqual([]);
  });

  it('ignores unknown parameters and reports them', () => {
    const parsed = parsePredictLink('?model=Qwen%2FQwen3-32B&autorun=1');

    expect(parsed.values).toEqual({ model: 'Qwen/Qwen3-32B' });
    expect(parsed.warnings).toEqual(['Ignored unknown parameter "autorun".']);
  });

  it('keeps the rest of a link when one value is invalid', () => {
    const parsed = parsePredictLink('?model=Qwen%2FQwen3-32B&isl=not-a-number&concurrency=8');

    expect(parsed.values).toEqual({ model: 'Qwen/Qwen3-32B', concurrency: 8 });
    expect(parsed.warnings).toEqual(['Ignored "isl": the value was not valid.']);
  });

  it('rejects a value outside the contract instead of clamping silently', () => {
    const parsed = parsePredictLink('?gpuMemUtil=4&servingMode=sideways');

    expect(parsed.values).toEqual({});
    expect(parsed.warnings).toHaveLength(2);
  });

  it('omits values that match the page defaults', () => {
    const values: PredictLinkValues = {
      model: 'Qwen/Qwen3-32B',
      isl: 2048,
      osl: 128,
      concurrency: 1,
      tp: 4,
    };

    expect(
      omitDefaults(values, { isl: 2048, osl: 128, concurrency: 1, tp: 1 }),
    ).toEqual({ model: 'Qwen/Qwen3-32B', tp: 4 });
  });

  it('round-trips through the query string', () => {
    const values: PredictLinkValues = {
      model: 'meta-llama/Llama-3.1-70B-Instruct',
      system: 'h100_sxm',
      isl: 4000,
    };

    const url = buildPredictLink(values);
    expect(url.startsWith('/predict?v=1')).toBe(true);
    expect(parsePredictLink(url.slice(url.indexOf('?'))).values).toEqual({
      model: 'meta-llama/Llama-3.1-70B-Instruct',
      system: 'h100_sxm',
      isl: 4000,
    });
  });
});
