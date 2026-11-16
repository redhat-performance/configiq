import { describe, expect, it } from 'vitest';
import { supportedClaims, verifyNumbers } from './verify';

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
    expect(result.unsupported).toEqual(['8 H100 GPUs']);
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
    const values = supportedClaims(EVIDENCE);
    expect(values.has(2048)).toBe(true);
    expect(values.has(1048576)).toBe(true);
    expect(values.get(1048576)).toContain('b');
    expect(values.get(2048)).toContain('token');
  });

  it('reports each unsupported claim once', () => {
    const result = verifyNumbers('8 GPUs here and 8 GPUs there.', EVIDENCE);
    expect(result.unsupported).toEqual(['8 GPUs']);
  });
});

describe('numeric verifier units', () => {
  it('treats an attached unit as part of the claim', () => {
    expect(verifyNumbers('Each GPU has 80GB of memory.', ['Each GPU has 80GB of memory.']).ok).toBe(true);
    const result = verifyNumbers('Each GPU has 96GB of memory.', ['Each GPU has 80GB of memory.']);
    expect(result.ok).toBe(false);
    expect(result.unsupported).toEqual(['96GB']);
  });

  it('treats a percentage as a claim even though it is a small number', () => {
    const evidence = ['KV cache growth is 5% of VRAM per sequence.'];
    expect(verifyNumbers('The growth is 5% of VRAM.', evidence).ok).toBe(true);
    const result = verifyNumbers('The growth is 7% of VRAM.', evidence);
    expect(result.ok).toBe(false);
    expect(result.unsupported).toEqual(['7%']);
  });

  it('treats a currency amount as a claim even though it is a small number', () => {
    const evidence = ['A hosted A100 costs $3.50 per GPU-hour.'];
    expect(verifyNumbers('The hosted rate is $3.50 per GPU-hour.', evidence).ok).toBe(true);
    const result = verifyNumbers('The hosted rate is $9.99 per GPU-hour.', evidence);
    expect(result.ok).toBe(false);
    expect(result.unsupported).toEqual(['$9.99']);
  });

  it('rejects a claim whose unit contradicts the evidence', () => {
    const result = verifyNumbers('The default batch is 128 GPUs.', ['The default batch is 128 tokens.']);
    expect(result.ok).toBe(false);
    expect(result.unsupported).toEqual(['128 GPUs']);
  });

  it('accepts a unit written differently', () => {
    expect(verifyNumbers('ISL is 2048 tok.', ['ISL is 2048 tokens.']).ok).toBe(true);
    expect(verifyNumbers('TTFT is 240 milliseconds.', ['TTFT is 240 ms.']).ok).toBe(true);
    expect(verifyNumbers('The rate is 3 USD per hour.', ['The rate is $3.00 per hour.']).ok).toBe(true);
  });

  it('keeps units that are not equivalent apart', () => {
    const result = verifyNumbers('TTFT is 240 s.', ['TTFT is 240 ms.']);
    expect(result.ok).toBe(false);
    expect(result.unsupported).toEqual(['240 s']);

    const binary = verifyNumbers('The GPU has 80 GiB.', ['The GPU has 80 GB.']);
    expect(binary.ok).toBe(false);
    expect(binary.unsupported).toEqual(['80 GiB']);
  });

  it('accepts a claim when the evidence states the value without a unit', () => {
    // A table cell or a heading often carries the unit away from the number.
    expect(verifyNumbers('The catalog lists 8 GPUs.', ['The catalog lists 8 entries.']).ok).toBe(true);
    expect(verifyNumbers('The default is 128.', ['The default is 128.']).ok).toBe(true);
  });

  it('accepts a claim with no unit when the evidence attaches one', () => {
    expect(verifyNumbers('The page default is 2048.', EVIDENCE).ok).toBe(true);
  });

  it('does not read a unit from the sentence after a count', () => {
    expect(verifyNumbers('Use these 3 steps. GPUs are selected on Predict.', []).ok).toBe(true);
  });
});

describe('numeric verifier magnitude notation', () => {
  it('reads an attached magnitude suffix as part of the value', () => {
    expect(verifyNumbers('The context is 128k tokens.', ['The context is 128,000 tokens.']).ok).toBe(true);

    // A suffix is not a decoration: 128k means 128,000, not 128.
    const result = verifyNumbers('The context is 128k tokens.', ['The context is 128 tokens.']);
    expect(result.ok).toBe(false);
    expect(result.unsupported).toEqual(['128k tokens']);
  });

  it('reads scientific notation as a complete value', () => {
    expect(verifyNumbers('The context is 1e6 tokens.', ['The context is 1,000,000 tokens.']).ok).toBe(true);

    const result = verifyNumbers('The context is 1e6 tokens.', ['The context is 1 token.']);
    expect(result.ok).toBe(false);
    expect(result.unsupported).toEqual(['1e6 tokens']);
  });

  it('reads a magnitude word as part of the value', () => {
    expect(verifyNumbers('The corpus is 1.5 million tokens.', ['The corpus is 1,500,000 tokens.']).ok).toBe(true);
  });
});

describe('numeric verifier rate units', () => {
  it('accepts a rate written in words against one written with a slash', () => {
    expect(verifyNumbers('Throughput is 128 tokens per second.', ['Throughput is 128 tokens/s.']).ok).toBe(true);
  });

  it('rejects a rate when the evidence states only the count', () => {
    const result = verifyNumbers('Throughput is 128 tokens per second.', ['Throughput is 128 tokens.']);
    expect(result.ok).toBe(false);
    expect(result.unsupported).toEqual(['128 tokens per second']);
  });

  it('canonicalises a slash rate written with a full word', () => {
    expect(supportedClaims(['Throughput is 128 tokens/sec.']).get(128)).toContain('token/s');
  });
});
