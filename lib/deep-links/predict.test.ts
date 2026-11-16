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

  it('refuses a link from a format version it does not understand', () => {
    const parsed = parsePredictLink('?v=99&model=Qwen%2FQwen3-32B');

    expect(parsed.values).toEqual({});
    expect(parsed.unsupportedVersion).toBe(true);
    expect(parsed.warnings.some((warning) => warning.includes('format version'))).toBe(true);
  });

  it('reads a link with no version marker as the current version', () => {
    const parsed = parsePredictLink('?model=Qwen%2FQwen3-32B');

    expect(parsed.values).toEqual({ model: 'Qwen/Qwen3-32B' });
    expect(parsed.unsupportedVersion).toBeUndefined();
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

  it('round-trips each disaggregated pool independently', () => {
    // A prefill pool and a decode pool can differ in every parallelism field,
    // so a shared view has to carry both rather than one aggregate pair.
    const values: PredictLinkValues = {
      model: 'Qwen/Qwen3-32B',
      system: 'h100_sxm',
      servingMode: 'disagg',
      prefillTp: 2,
      prefillPp: 1,
      prefillWorkers: 3,
      prefillBatch: 8,
      decodeTp: 4,
      decodePp: 2,
      decodeWorkers: 5,
      decodeBatch: 128,
    };

    const parsed = parsePredictLink(
      buildPredictLink(values).slice(buildPredictLink(values).indexOf('?')),
    );

    expect(parsed.values).toEqual(values);
    expect(parsed.warnings).toEqual([]);
  });

  it('round-trips the manual vLLM override panel', () => {
    // These are the values a reader pins by hand; a view that pins chunked
    // prefill is a different view from one that leaves it to the backend.
    const values: PredictLinkValues = {
      model: 'Qwen/Qwen3-32B',
      system: 'h100_sxm',
      maxNumSeqs: 64,
      chunkedPrefill: 'off',
      gpuMemUtil: 0.87,
    };

    const parsed = parsePredictLink(
      buildPredictLink(values).slice(buildPredictLink(values).indexOf('?')),
    );

    expect(parsed.values).toEqual(values);
    expect(parsed.warnings).toEqual([]);
  });

  it('round-trips a MoE weights quantisation choice', () => {
    const values: PredictLinkValues = {
      model: 'Qwen/Qwen3-235B-A22B',
      moePrecision: 'w4a8_mxfp4_mxfp8',
    };

    const parsed = parsePredictLink(
      buildPredictLink(values).slice(buildPredictLink(values).indexOf('?')),
    );

    expect(parsed.values).toEqual(values);
  });

  it('rejects a chunked-prefill choice it does not model', () => {
    const parsed = parsePredictLink('?chunkedPrefill=maybe&isl=512');

    expect(parsed.values).toEqual({ isl: 512 });
    expect(parsed.warnings).toEqual(['Ignored "chunkedPrefill": the value was not valid.']);
  });

  it('rejects a zero in a pool field instead of borrowing the other pool', () => {
    const parsed = parsePredictLink('?decodeWorkers=0&decodeBatch=16');

    expect(parsed.values).toEqual({ decodeBatch: 16 });
    expect(parsed.warnings).toEqual(['Ignored "decodeWorkers": the value was not valid.']);
  });
});
