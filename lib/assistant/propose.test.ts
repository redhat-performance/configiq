import { describe, expect, it } from 'vitest';
import { kvCacheLink, recommendLink } from '@/lib/deep-links/pages';
import { predictLink } from '@/lib/deep-links/predict';
import { proposeView } from './propose';

/**
 * A proposal is only worth offering if the URL it produces is the one the page
 * will parse, so most of these tests read the link back through the page's own
 * contract instead of matching on a string.
 */

function valuesOf(
  href: string,
  parse: (search: string) => { values: object },
): Record<string, unknown> {
  return parse(href.slice(href.indexOf('?'))).values as Record<string, unknown>;
}

describe('proposeView', () => {
  it('does not read a URL path as a model name', () => {
    const proposal = proposeView({
      question:
        'How many GPUs for https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct at 4096 input tokens?',
      model: 'Qwen/Qwen3-8B',
    });

    expect(valuesOf(proposal!.link.href, recommendLink.parse).model).toBe('Qwen/Qwen3-8B');
  });

  it('offers nothing when the question names no page', () => {
    expect(proposeView({ question: 'What does max_num_seqs change?' })).toBeNull();
  });

  it('offers nothing when the page has no link contract to fill', () => {
    // `settings` is a page hint, but its state is not a shareable request.
    expect(proposeView({ question: 'Where do I change the simulator?' })).toBeNull();
  });

  it('carries the defaults when the question names no model or system', () => {
    const proposal = proposeView({
      question: 'How many GPUs for 4096 input tokens and 512 output tokens at 50 requests per second?',
      model: 'meta-llama/Llama-3.1-8B-Instruct',
      system: 'nvidia-l40s',
    });

    expect(proposal).not.toBeNull();
    const values = valuesOf(proposal!.link.href, recommendLink.parse);
    expect(values).toMatchObject({
      model: 'meta-llama/Llama-3.1-8B-Instruct',
      system: 'nvidia-l40s',
      isl: 4096,
      osl: 512,
      demandRate: 50,
      // A request rate without a latency target is the throughput goal; the
      // page would otherwise ignore the rate it was given.
      goal: 'demand',
    });
    expect(proposal!.fields).toEqual(
      expect.arrayContaining(['model', 'system', 'isl', 'osl', 'demandRate', 'goal']),
    );
  });

  it('lets a model named in the question win over the default', () => {
    const proposal = proposeView({
      question: 'How many GPUs to serve meta-llama/Llama-3.1-405B-Instruct at 8192 input tokens?',
      model: 'meta-llama/Llama-3.1-8B-Instruct',
      system: 'nvidia-h100-sxm-80gb',
    });

    const values = valuesOf(proposal!.link.href, recommendLink.parse);
    expect(values.model).toBe('meta-llama/Llama-3.1-405B-Instruct');
    expect(values.system).toBe('nvidia-h100-sxm-80gb');
    expect(values.isl).toBe(8192);
  });

  it('does not read a runtime or a ratio as a model name', () => {
    for (const question of [
      'How many GPUs for vllm/sglang at 4096 input tokens?',
      'How many GPUs for 4096/512 input/output?',
    ]) {
      const proposal = proposeView({
        question,
        model: 'meta-llama/Llama-3.1-8B-Instruct',
        system: 'nvidia-l40s',
      });
      expect(valuesOf(proposal!.link.href, recommendLink.parse).model).toBe(
        'meta-llama/Llama-3.1-8B-Instruct',
      );
    }
  });

  it('carries a latency target in milliseconds and keeps the unit the user wrote', () => {
    const proposal = proposeView({
      question: 'Recommend a size for TTFT under 2 s at 1024 input tokens.',
      model: 'meta-llama/Llama-3.1-8B-Instruct',
      system: 'nvidia-l40s',
      backend: 'vllm',
      backendVersion: '0.30.0',
      simulator: 'aisimulate',
    });

    const values = valuesOf(proposal!.link.href, recommendLink.parse);
    expect(values).toMatchObject({
      isl: 1024,
      ttft: 2000,
      goal: 'latency',
      backend: 'vllm',
      backendVersion: '0.30.0',
      simulator: 'aisimulate',
    });
    // The answer may repeat "2 s", so both forms have to be quotable.
    expect(proposal!.evidence.join(' ')).toContain('2 s');
    expect(proposal!.evidence.join(' ')).toContain('2000 ms');
  });

  it('does not guess a latency target written without a unit', () => {
    const proposal = proposeView({
      question: 'Recommend a size for TTFT under 2 at 1024 input tokens.',
      model: 'meta-llama/Llama-3.1-8B-Instruct',
    });

    expect(valuesOf(proposal!.link.href, recommendLink.parse).ttft).toBeUndefined();
  });

  it('reports a value the contract rejects instead of writing it', () => {
    const proposal = proposeView({
      question: 'How many GPUs at a concurrency of 0 requests?',
      model: 'meta-llama/Llama-3.1-8B-Instruct',
    });

    expect(proposal).not.toBeNull();
    const values = valuesOf(proposal!.link.href, recommendLink.parse);
    expect(values.concurrency).toBeUndefined();
    expect(proposal!.ignored).toEqual([
      { key: 'concurrency', value: '0', reason: 'rejected' },
    ]);
    // A goal derived from a dropped value would be a link that asks for
    // capacity without saying how much.
    expect(values.goal).toBeUndefined();
    // The panel renders this, so the drop is visible before the page opens.
    expect(proposal!.link.description).toContain(
      'Left out concurrency "0" (the page does not accept it)',
    );
  });

  it('leaves out a field the question gives two values for', () => {
    const proposal = proposeView({
      question: 'How many GPUs for input length 1024 or input length 8192 at 50 requests per second?',
      model: 'meta-llama/Llama-3.1-8B-Instruct',
    });

    const values = valuesOf(proposal!.link.href, recommendLink.parse);
    // Picking either length would present one alternative as the request.
    expect(values.isl).toBeUndefined();
    expect(values.demandRate).toBe(50);
    expect(proposal!.ignored).toEqual([
      { key: 'isl', value: '1024, 8192', reason: 'ambiguous' },
    ]);
    expect(proposal!.link.description).toContain('the question names more than one');
  });

  it('lets a GPU system named in the question win over the default', () => {
    const proposal = proposeView({
      question: 'How many GPUs for 4096 input tokens on l40s?',
      model: 'meta-llama/Llama-3.1-8B-Instruct',
      system: 'h100',
    });

    expect(valuesOf(proposal!.link.href, recommendLink.parse).system).toBe('l40s');
  });

  it('leaves the GPU system out when the question names two', () => {
    const proposal = proposeView({
      question: 'How many GPUs for 4096 input tokens on l40s or h200_sxm?',
      model: 'meta-llama/Llama-3.1-8B-Instruct',
      system: 'h100',
    });

    const values = valuesOf(proposal!.link.href, recommendLink.parse);
    // The configured system is not substituted for the one the user asked
    // about; the page keeps whatever it was showing.
    expect(values.system).toBeUndefined();
    expect(proposal!.ignored[0]).toEqual({
      key: 'system',
      value: 'l40s, h200_sxm',
      reason: 'ambiguous',
    });
  });

  it('resolves a hyphenated accelerator name to its catalog id', () => {
    const proposal = proposeView({
      question: 'How many GPUs for 4096 input tokens on an NVIDIA H100-SXM-80GB?',
      model: 'meta-llama/Llama-3.1-8B-Instruct',
      system: 'l40s',
    });

    expect(valuesOf(proposal!.link.href, recommendLink.parse).system).toBe('h100_sxm');
  });

  it('keeps the configured system when the question names only a family', () => {
    const proposal = proposeView({
      question: 'How many GPUs for 4096 input tokens on an H100?',
      model: 'meta-llama/Llama-3.1-8B-Instruct',
      system: 'l40s',
    });

    // H100 ships as SXM and PCIe; pinning either would name a machine the
    // question did not.
    expect(valuesOf(proposal!.link.href, recommendLink.parse).system).toBe('l40s');
  });

  it('reports a field the destination page does not declare', () => {
    const proposal = proposeView({
      question: 'Predict 1024 input and 128 output tokens at 50 requests per second.',
      model: 'meta-llama/Llama-3.1-8B-Instruct',
    });

    const values = valuesOf(proposal!.link.href, predictLink.parse);
    expect(values.isl).toBe(1024);
    expect(values.osl).toBe(128);
    // Predict performance has no demand input, and dropping the rate silently
    // would describe a different view from the one the answer talked about.
    expect(proposal!.ignored).toEqual([
      { key: 'demandRate', value: '50', reason: 'unsupported' },
    ]);
    expect(proposal!.link.description).toContain('Predict performance has no such field');
  });

  it('writes the context length into the field each page names it by', () => {
    const model = 'meta-llama/Llama-3.1-8B-Instruct';

    const recommend = proposeView({
      question: 'Recommend a size for a context length of 8192 with 4096 input tokens.',
      model,
    });
    expect(valuesOf(recommend!.link.href, recommendLink.parse)).toMatchObject({
      isl: 4096,
      maxSeqLen: 8192,
    });

    const predict = proposeView({
      question: 'Predict a context length of 8192 with 4096 input tokens.',
      model,
    });
    expect(valuesOf(predict!.link.href, predictLink.parse)).toMatchObject({
      isl: 4096,
      maxModelLen: 8192,
    });

    const kv = proposeView({
      question: 'Calculate the KV cache for a context length of 8192.',
      model,
    });
    expect(valuesOf(kv!.link.href, kvCacheLink.parse).maxNumTokens).toBe(8192);
  });

  it('does not read a per-token latency as an output length', () => {
    const proposal = proposeView({
      question:
        'Recommend time per output token 25 ms with output length 512 at 1024 input tokens.',
      model: 'meta-llama/Llama-3.1-8B-Instruct',
    });

    const values = valuesOf(proposal!.link.href, recommendLink.parse);
    // `output token 25` is the unit of the latency, not 25 output tokens.
    expect(values).toMatchObject({ isl: 1024, osl: 512, tpot: 25, goal: 'latency' });
    expect(proposal!.ignored).toEqual([]);
  });

  it('leaves the model out when the question names two', () => {
    const proposal = proposeView({
      question:
        'How many GPUs for meta-llama/Llama-3.1-8B-Instruct or Qwen/Qwen3-8B at 4096 input tokens?',
      model: 'Qwen/Qwen3-32B',
    });

    const values = valuesOf(proposal!.link.href, recommendLink.parse);
    // Neither alternative is chosen, and the setting is not quietly substituted
    // for the question's own models.
    expect(values.model).toBeUndefined();
    expect(values.isl).toBe(4096);
    expect(proposal!.ignored).toEqual([
      {
        key: 'model',
        value: 'meta-llama/Llama-3.1-8B-Instruct, Qwen/Qwen3-8B',
        reason: 'ambiguous',
      },
    ]);
    expect(proposal!.link.description).toContain('the question names more than one');
  });

  it('reports a rejected value even when nothing else survives', () => {
    const proposal = proposeView({ question: 'Recommend TPOT 700000 ms' });

    // The page's ceiling is 600000 ms. With no other value and no configured
    // model, the proposal has no fields left, but the refusal still has to be
    // shown rather than the whole proposal discarded.
    expect(proposal).not.toBeNull();
    expect(proposal!.fields).toEqual([]);
    expect(proposal!.ignored).toEqual([
      { key: 'tpot', value: '700000', reason: 'rejected' },
    ]);
    expect(proposal!.link.label).toBe('Open Recommend sizing');
    expect(proposal!.link.description).not.toContain('Sets ');
    expect(proposal!.link.description).toContain(
      'Left out time per output token "700000" (the page does not accept it)',
    );
  });

  it('reads a paired input/output ratio as both lengths', () => {
    const proposal = proposeView({
      question: 'How many GPUs for 4096/512 input/output tokens?',
      model: 'meta-llama/Llama-3.1-8B-Instruct',
    });

    const values = valuesOf(proposal!.link.href, recommendLink.parse);
    expect(values).toMatchObject({ isl: 4096, osl: 512 });
  });

  it('does not read the front of a magnitude as a count', () => {
    // `4k` is not 4 tokens, and guessing would propose a different question. It
    // is the only value named here, so there is nothing to propose at all.
    expect(
      proposeView({
        question: 'How many GPUs for 4k input tokens?',
        model: 'meta-llama/Llama-3.1-8B-Instruct',
      }),
    ).toBeNull();
  });

  it('does not convert a rate stated per minute', () => {
    const proposal = proposeView({
      question: 'How many GPUs for 4096 input tokens at 60 requests per minute?',
      model: 'meta-llama/Llama-3.1-8B-Instruct',
    });

    // The page stores requests per second; 60 per minute is 1 per second, and
    // writing 60 would size for sixty times the stated load.
    expect(valuesOf(proposal!.link.href, recommendLink.parse).demandRate).toBeUndefined();
  });

  it('does not read the next label\'s target as another time-to-first-token', () => {
    const proposal = proposeView({
      question: 'Recommend a size for 2 s to first token and 50 ms per output token at 1024 input tokens.',
      model: 'meta-llama/Llama-3.1-8B-Instruct',
    });

    const values = valuesOf(proposal!.link.href, recommendLink.parse);
    // `50 ms` belongs to the per-token latency, so TTFT stays unambiguous.
    expect(values.ttft).toBe(2000);
    expect(values.tpot).toBe(50);
    expect(values.goal).toBe('latency');
  });

  it('reads a latency target written without a connective', () => {
    const cases: Array<[string, string, number]> = [
      ['Recommend TTFT 2 s at 4096 input tokens.', 'ttft', 2000],
      ['Recommend TPOT 25 ms at 4096 input tokens.', 'tpot', 25],
      ['Recommend end-to-end latency 4 s at 4096 input tokens.', 'e2eLatency', 4000],
    ];
    for (const [question, field, expected] of cases) {
      const proposal = proposeView({
        question,
        model: 'meta-llama/Llama-3.1-8B-Instruct',
      });
      const values = valuesOf(proposal!.link.href, recommendLink.parse);
      expect(values[field]).toBe(expected);
      expect(values.goal).toBe('latency');
    }
  });

  it('reads a latency written before its label', () => {
    const cases: Array<[string, string, number]> = [
      ['Recommend a size for 1second TTFT at 4096 input tokens.', 'ttft', 1000],
      ['Recommend a size for 2 s to first token at 4096 input tokens.', 'ttft', 2000],
      ['Recommend a size for 100ms ITL at 4096 input tokens.', 'tpot', 100],
      ['Recommend a size for 25 ms per output token at 4096 input tokens.', 'tpot', 25],
    ];
    for (const [question, field, expected] of cases) {
      const proposal = proposeView({
        question,
        model: 'meta-llama/Llama-3.1-8B-Instruct',
      });
      const values = valuesOf(proposal!.link.href, recommendLink.parse);
      expect(values[field]).toBe(expected);
      expect(values.goal).toBe('latency');
    }
  });

  it('reads inter-token latency as time per output token', () => {
    const proposal = proposeView({
      question: 'Recommend a size for ITL of 50 ms at 4096 input tokens.',
      model: 'meta-llama/Llama-3.1-8B-Instruct',
    });

    // ITL and TPOT are the same quantity under two names; the page only has
    // the second, and leaving the number out would answer a different question.
    expect(valuesOf(proposal!.link.href, recommendLink.parse).tpot).toBe(50);
  });

  it('reads a length written with the token count first', () => {
    const proposal = proposeView({
      question: 'Recommend sizing for 4,000 token inputs and 1,000 token outputs.',
      model: 'meta-llama/Llama-3.1-8B-Instruct',
    });

    expect(valuesOf(proposal!.link.href, recommendLink.parse)).toMatchObject({
      isl: 4000,
      osl: 1000,
    });
  });

  it('resolves a bare accelerator name to its SXM part', () => {
    const proposal = proposeView({
      question: 'Recommend a size for TTFT 1 s on an H200 at 4096 input tokens.',
      model: 'meta-llama/Llama-3.1-8B-Instruct',
      system: 'l40s',
    });

    // A bare name resolves to the SXM part; a qualified one does not.
    expect(valuesOf(proposal!.link.href, recommendLink.parse).system).toBe('h200_sxm');
  });

  it('does not resolve an H200 variant to the SXM part', () => {
    for (const variant of ['H200 NVL', 'H200 PCIe']) {
      const proposal = proposeView({
        question: `Recommend a size for TTFT 1 s on an ${variant} at 4096 input tokens.`,
        model: 'meta-llama/Llama-3.1-8B-Instruct',
        system: 'l40s',
      });

      // NVL and PCIe are different machines. Pinning SXM would open the page
      // on hardware the question did not name, so the question is left to the
      // configured system instead.
      expect(proposal).not.toBeNull();
      expect(valuesOf(proposal!.link.href, recommendLink.parse).system).toBe('l40s');
    }
  });

  it('opens the reported sizing question with every value it names', () => {
    const proposal = proposeView({
      question:
        'recommend a small Qwen model that can perform 1second TTFT, 100ms ITL with a default workload and minimal H200 GPUs',
      model: 'Qwen/Qwen3-32B',
      system: 'h200_sxm',
    });

    expect(proposal).not.toBeNull();
    const values = valuesOf(proposal!.link.href, recommendLink.parse);
    expect(values).toMatchObject({
      ttft: 1000,
      tpot: 100,
      system: 'h200_sxm',
      goal: 'latency',
    });
    // "a small Qwen model" is not a model id, so the assistant must not choose
    // one; the view keeps the model the page was already set to.
    expect(values.model).toBe('Qwen/Qwen3-32B');
  });

  it('leaves out a request rate the latency goal does not read', () => {
    const proposal = proposeView({
      question: 'Recommend a size for TTFT under 2 s at 50 requests per second.',
      model: 'meta-llama/Llama-3.1-8B-Instruct',
    });

    const values = valuesOf(proposal!.link.href, recommendLink.parse);
    expect(values.ttft).toBe(2000);
    // The page writes the rate only for the demand goal, so carrying it would
    // describe an input the view never uses.
    expect(values.demandRate).toBeUndefined();
    expect(proposal!.ignored).toContainEqual({
      key: 'demandRate',
      value: '50',
      reason: 'unused',
    });
    expect(proposal!.link.description).toContain('the latency goal does not use it');
  });

  it('does not carry a negative latency target', () => {
    const proposal = proposeView({
      question: 'Recommend a size for TTFT under -2 s at 4096 input tokens.',
      model: 'meta-llama/Llama-3.1-8B-Instruct',
    });

    // A target the page would reject is reported, never written as +2000.
    expect(valuesOf(proposal!.link.href, recommendLink.parse).ttft).toBeUndefined();
  });

  it('keeps looking for a model past a runtime that looks like a path', () => {
    const proposal = proposeView({
      question: 'How many GPUs for vllm/sglang on meta-llama/Llama-3.1-70B-Instruct at 4096 input tokens?',
      model: 'Qwen/Qwen3-8B',
    });

    expect(valuesOf(proposal!.link.href, recommendLink.parse).model).toBe(
      'meta-llama/Llama-3.1-70B-Instruct',
    );
  });

  it('fills the performance page with the configuration it reads', () => {
    const proposal = proposeView({
      question: 'Predict 1024 input and 128 output tokens at concurrency 8.',
      model: 'meta-llama/Llama-3.1-8B-Instruct',
      system: 'nvidia-l40s',
      backend: 'vllm',
    });

    const values = valuesOf(proposal!.link.href, predictLink.parse);
    expect(values).toMatchObject({
      isl: 1024,
      osl: 128,
      concurrency: 8,
      backend: 'vllm',
    });
    // The page computes latency; a target from the question is not its input.
    expect(values.ttft).toBeUndefined();
  });

  it('omits provenance a page does not accept', () => {
    const proposal = proposeView({
      question: 'How much KV cache memory for 8192 context?',
      model: 'meta-llama/Llama-3.1-8B-Instruct',
      system: 'nvidia-l40s',
      simulator: 'aisimulate',
      backend: 'vllm',
    });

    const values = valuesOf(proposal!.link.href, kvCacheLink.parse);
    expect(values).toMatchObject({
      model: 'meta-llama/Llama-3.1-8B-Instruct',
      system: 'nvidia-l40s',
      backend: 'vllm',
      maxNumTokens: 8192,
    });
    // The KV cache calculator is AISimulate-only by contract.
    expect(values.simulator).toBeUndefined();
    expect(proposal!.link.href).not.toContain('simulator=');
  });

  it('says which fields it will set, and marks itself as a proposal', () => {
    const proposal = proposeView({
      question: 'How many GPUs at 4096 input and 512 output tokens?',
      model: 'meta-llama/Llama-3.1-8B-Instruct',
      system: 'nvidia-l40s',
    });

    expect(proposal!.link.kind).toBe('proposal');
    expect(proposal!.link.description).toBe(
      'Sets model, GPU system, input length, output length.',
    );
    expect(proposal!.link.href.startsWith('/recommend?v=1&')).toBe(true);
  });

  it('needs a value from the question, not just the page defaults', () => {
    const defaults = {
      model: 'meta-llama/Llama-3.1-8B-Instruct',
      system: 'nvidia-l40s',
    };
    // "memory" routes to the KV calculator, but neither question names a value,
    // so the plain page hint says everything a link could.
    expect(proposeView({ question: 'How does memory work?', ...defaults })).toBeNull();
    expect(proposeView({ question: 'How many GPUs should I use?', ...defaults })).toBeNull();
  });
});
