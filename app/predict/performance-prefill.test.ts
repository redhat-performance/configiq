import { describe, expect, it } from 'vitest';
import { parsePerformancePrefill } from './performance-prefill';

describe('performance page prefill', () => {
  it('accepts the widget model and system handoff', () => {
    expect(parsePerformancePrefill(
      '?model=Qwen%2FQwen2.5-7B-Instruct&system=h200_sxm',
    )).toEqual({
      model: 'Qwen/Qwen2.5-7B-Instruct',
      system: 'h200_sxm',
    });
  });

  it('ignores malformed or oversized values', () => {
    expect(parsePerformancePrefill('?model=javascript%3Aalert(1)&system=h200%20sxm'))
      .toEqual({ model: null, system: null });
    expect(parsePerformancePrefill(`?model=org/${'a'.repeat(220)}&system=h200_sxm`))
      .toEqual({ model: null, system: 'h200_sxm' });
  });
});
