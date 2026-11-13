import { describe, expect, it } from 'vitest';
import { supportedNumbers, verifyNumbers } from './verify';

const EVIDENCE = [
  'Llama-3.1-70B on an H200 has a KV cache that uses 1,048,576 bytes per token.',
  'The page default for ISL is 2048 tokens and OSL is 128 tokens.',
  'Two concurrent requests is the default.',
];

describe('numeric verifier', () => {
  it('accepts a number that appears in the evidence', () => {
    expect(verifyNumbers('The default ISL is 2048 tokens.', EVIDENCE).ok).toBe(true);
  });

  it('compares numbers rather than their formatting', () => {
    expect(verifyNumbers('The KV cache uses 1048576 bytes per token.', EVIDENCE).ok).toBe(true);
    expect(verifyNumbers('The KV cache uses 1,048,576 bytes per token.', EVIDENCE).ok).toBe(true);
  });

  it('rejects an invented number', () => {
    const result = verifyNumbers('You need 8 H100 GPUs for this workload.', EVIDENCE);
    expect(result.ok).toBe(false);
    expect(result.unsupported).toContain('8 H100 GPUs');
  });

  it('treats a small number with a unit as a claim', () => {
    const result = verifyNumbers('Serving adds 2 GPUs to the deployment.', EVIDENCE);
    expect(result.ok).toBe(false);
    expect(result.unsupported).toContain('2 GPUs');
  });

  it('ignores numbers that are ordinary prose', () => {
    expect(verifyNumbers('There are 3 ways to compare cost.', EVIDENCE).ok).toBe(true);
    expect(verifyNumbers('Use these 2 steps.', EVIDENCE).ok).toBe(true);
  });

  it('ignores ordered-list markers', () => {
    const answer = '1. Open Predict.\n2. Choose a model.\n3. Press Calculate.';
    expect(verifyNumbers(answer, []).ok).toBe(true);
  });

  it('ignores numbers inside URLs', () => {
    const answer = 'See https://example.com/docs/v2/2048 for details.';
    expect(verifyNumbers(answer, []).ok).toBe(true);
  });

  it('collects supported values from every evidence string', () => {
    const values = supportedNumbers(EVIDENCE);
    expect(values.has(2048)).toBe(true);
    expect(values.has(1048576)).toBe(true);
  });

  it('reports each unsupported claim once', () => {
    const result = verifyNumbers('8 GPUs here and 8 GPUs there.', EVIDENCE);
    expect(result.unsupported).toEqual(['8 GPUs']);
  });
});
