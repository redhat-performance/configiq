import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { GET, POST } from './route';
import type { AssistantEvent } from '@/lib/assistant/types';

type Reply = {
  content: string;
  reasoning?: string;
  reasoningContent?: string;
  finishReason?: string;
  toolCalls?: Array<{ id: string; name: string; arguments: string }>;
};

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  });
}

function completion(reply: Reply): Response {
  return jsonResponse({
    choices: [
      {
        finish_reason: reply.finishReason ?? 'stop',
        message: {
          content: reply.content,
          ...(reply.reasoning !== undefined ? { reasoning: reply.reasoning } : {}),
          ...(reply.reasoningContent !== undefined
            ? { reasoning_content: reply.reasoningContent }
            : {}),
          ...(reply.toolCalls
            ? {
                tool_calls: reply.toolCalls.map((call) => ({
                  id: call.id,
                  type: 'function',
                  function: { name: call.name, arguments: call.arguments },
                })),
              }
            : {}),
        },
      },
    ],
  });
}

function parseStream(text: string): AssistantEvent[] {
  return text
    .split('\n')
    .filter((line) => line.trim().length > 0)
    .map((line) => JSON.parse(line) as AssistantEvent);
}

async function ask(question: string, overrides: Record<string, unknown> = {}) {
  const response = await POST(
    new Request('http://localhost/api/assistant', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, route: '/predict', simulator: 'blis', ...overrides }),
    }),
  );
  return { response, events: parseStream(await response.text()) };
}

const realFetch = globalThis.fetch;

beforeEach(() => {
  vi.stubEnv('ASSISTANT_ENABLED', 'true');
  vi.stubEnv('ASSISTANT_MODEL', 'test/assistant-model');
  vi.stubEnv('ASSISTANT_INFERENCE_URL', 'http://assistant.test:8001');
  vi.stubEnv('TRITON_INFERENCE_URL', 'http://triton.test:8000');
  vi.stubEnv('AISIMULATORS_GATEWAY_URL', 'http://gateway.test');
});

afterEach(() => {
  vi.unstubAllEnvs();
  vi.unstubAllGlobals();
  globalThis.fetch = realFetch;
});

describe('assistant availability', () => {
  it('reports why the assistant is off when the flag is absent', async () => {
    vi.stubEnv('ASSISTANT_ENABLED', '');
    const response = await GET();
    const body = (await response.json()) as { enabled: boolean; reason: string };
    expect(body.enabled).toBe(false);
    expect(body.reason).toMatch(/not enabled/i);
  });

  it('reports the model and the endpoint it resolved when enabled', async () => {
    const response = await GET();
    const body = (await response.json()) as {
      enabled: boolean;
      model: string | null;
      endpoint: string | null;
      reason: string | null;
    };
    expect(body).toEqual({
      enabled: true,
      model: 'test/assistant-model',
      endpoint: 'http://assistant.test:8001',
      reason: null,
    });
  });

  it('reports the Triton URL only when no assistant endpoint is set', async () => {
    // Older deployments set one inference URL. The assistant then points at the
    // FIL classifier, which cannot answer; reporting it is what lets a deploy
    // notice instead of the first question.
    vi.stubEnv('ASSISTANT_INFERENCE_URL', '');
    const body = (await (await GET()).json()) as { enabled: boolean; endpoint: string | null };
    expect(body.enabled).toBe(true);
    expect(body.endpoint).toBe('http://triton.test:8000');
  });

  it('refuses a question when disabled instead of answering without a model', async () => {
    vi.stubEnv('ASSISTANT_ENABLED', '');
    const response = await POST(
      new Request('http://localhost/api/assistant', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: 'Where do I set concurrency?' }),
      }),
    );
    expect(response.status).toBe(503);
    const body = (await response.json()) as { code: string };
    expect(body.code).toBe('assistant_disabled');
  });

  it('rejects an empty question', async () => {
    const response = await POST(
      new Request('http://localhost/api/assistant', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: ' ' }),
      }),
    );
    expect(response.status).toBe(400);
  });
});

describe('assistant answer', () => {
  it('calls the assistant endpoint rather than the FIL classifier', async () => {
    const urls: string[] = [];
    vi.stubGlobal(
      'fetch',
      vi.fn(async (input: RequestInfo | URL) => {
        urls.push(String(input));
        return completion({ content: 'Search and documentation still work.' });
      }),
    );

    await ask('what does the costs page compare');

    expect(urls.some((url) => url.startsWith('http://assistant.test:8001/v1/chat/completions'))).toBe(true);
    expect(urls.some((url) => url.startsWith('http://triton.test:8000'))).toBe(false);
  });

  it('asks for a short reasoning trace and leaves room for the answer', async () => {
    const bodies: Array<Record<string, unknown>> = [];
    vi.stubGlobal(
      'fetch',
      vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
        if (String(input).includes('/v1/chat/completions')) {
          bodies.push(JSON.parse(String(init?.body)) as Record<string, unknown>);
          return completion({ content: 'Search and documentation still work.' });
        }
        return jsonResponse({}, 404);
      }),
    );

    await ask('what does the costs page compare');

    // A reasoning model's thinking tokens count against max_tokens, so the cap
    // has to cover the trace as well as the 120-word answer.
    expect(bodies[0].max_tokens).toBe(640);
    expect(bodies[0].chat_template_kwargs).toEqual({ reasoning_effort: 'low' });
  });

  it('never renders the model reasoning as the answer', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async (input: RequestInfo | URL) => {
        if (String(input).includes('/v1/chat/completions')) {
          return completion({
            reasoning: 'The user is asking about costs. I should compare the three options.',
            content: 'The Costs page compares hosted, rented, and purchased options.',
          });
        }
        return jsonResponse({}, 404);
      }),
    );

    const { events } = await ask('what does the costs page compare');
    const result = events.find((event) => event.type === 'result');
    if (result?.type !== 'result') throw new Error('no result event');

    expect(result.status).toBe('answered');
    expect(result.answer).not.toContain('The user is asking');
    expect(result.answer).toContain('hosted, rented, and purchased');
  });

  it('reads reasoning from the older reasoning_content field too', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async (input: RequestInfo | URL) => {
        if (String(input).includes('/v1/chat/completions')) {
          // Older OpenAI-compatible servers name the trace `reasoning_content`
          // rather than `reasoning`; either way it must not be the answer.
          return completion({
            reasoningContent: 'The user is asking about costs. I should compare the three options.',
            content: 'The Costs page compares hosted, rented, and purchased options.',
          });
        }
        return jsonResponse({}, 404);
      }),
    );

    const { events } = await ask('what does the costs page compare');
    const result = events.find((event) => event.type === 'result');
    if (result?.type !== 'result') throw new Error('no result event');

    expect(result.status).toBe('answered');
    expect(result.answer).not.toContain('The user is asking');
    expect(result.answer).toContain('hosted, rented, and purchased');
  });

  it('withholds prose when generation stops before the answer', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async (input: RequestInfo | URL) => {
        if (String(input).includes('/v1/chat/completions')) {
          // What K2-Horizon returns when the token cap lands inside its
          // reasoning block: the unfinished trace arrives as `content`.
          return completion({
            content: 'The user is asking about concurrency. I should explain the workload fields.',
            finishReason: 'length',
          });
        }
        return jsonResponse({}, 404);
      }),
    );

    const { events } = await ask('how do I set concurrency');
    const result = events.find((event) => event.type === 'result');
    if (result?.type !== 'result') throw new Error('no result event');

    expect(result.status).toBe('incomplete');
    expect(result.answer).toBe('');
    // The sources are still useful even though the prose is not.
    expect(result.citations.length).toBeGreaterThan(0);
  });

  it('keeps the action links when an answer is incomplete', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async (input: RequestInfo | URL) => {
        if (String(input).includes('/v1/chat/completions')) {
          return completion({
            content: 'The user is asking how many GPUs. I should size the deployment.',
            finishReason: 'length',
          });
        }
        return jsonResponse({}, 404);
      }),
    );

    const { events } = await ask('how many GPUs for 4096 input tokens', {
      route: '/predict',
      model: 'meta-llama/Llama-3.1-8B-Instruct',
      system: 'nvidia-l40s',
    });
    const result = events.find((event) => event.type === 'result');
    if (result?.type !== 'result') throw new Error('no result event');

    // Withholding the prose must not also withhold the view the question
    // described: the link is built from the question, not from the model.
    expect(result.status).toBe('incomplete');
    expect(result.answer).toBe('');
    expect(result.links[0].kind).toBe('proposal');
    expect(result.links[0].href).toContain('isl=4096');
  });

  it('answers from retrieved documentation and cites it', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async (input: RequestInfo | URL) => {
        const url = String(input);
        if (url.includes('/v1/chat/completions')) {
          return completion({
            content: 'The Costs page compares hosted, rented, and purchased options.',
          });
        }
        return jsonResponse({}, 404);
      }),
    );

    const { response, events } = await ask('what does the costs page compare');

    expect(response.status).toBe(200);
    expect(response.headers.get('content-type')).toContain('application/x-ndjson');

    const retrieval = events.find((event) => event.type === 'retrieval');
    expect(retrieval).toBeDefined();

    const result = events.find((event) => event.type === 'result');
    expect(result).toBeDefined();
    if (result?.type !== 'result') throw new Error('no result event');
    expect(result.status).toBe('answered');
    expect(result.answer).toContain('hosted, rented, and purchased');
    expect(result.citations.length).toBeGreaterThan(0);
    expect(result.citations[0].url.startsWith('/docs/')).toBe(true);
    expect(result.model).toBe('test/assistant-model');

    // A docs link is `/docs/<slug>#<anchor>`; adding `?v=1` after the hash
    // would put the query inside the fragment and break the anchor.
    const docLinks = result.links.filter((link) => link.href.startsWith('/docs/'));
    expect(docLinks.length).toBeGreaterThan(0);
    for (const link of docLinks) expect(link.href).not.toContain('?');
  });

  it('accepts a number the user wrote in the question', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async (input: RequestInfo | URL) => {
        const url = String(input);
        if (url.includes('/v1/chat/completions')) {
          // 4096 appears nowhere in the docs and nowhere in a tool result: it is
          // only quotable because the user wrote it.
          return completion({ content: 'The 4096 you named is the input length, also called ISL.' });
        }
        return jsonResponse({}, 404);
      }),
    );

    const { events } = await ask('what does the input length 4096 mean?');
    const result = events.find((event) => event.type === 'result');
    if (result?.type !== 'result') throw new Error('no result event');

    expect(result.status).toBe('answered');
    expect(result.answer).toContain('4096');
    // The question is evidence for the verifier, not a cited source.
    expect(result.evidence.join(' ')).not.toContain('what does the input length');
  });

  it('lists every section the model was given, so no citation is cut off', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async (input: RequestInfo | URL) => {
        const url = String(input);
        if (url.includes('/v1/chat/completions')) {
          return completion({ content: 'See the documentation for how sizing works.' });
        }
        return jsonResponse({}, 404);
      }),
    );

    const { events } = await ask('how does sizing work');
    const retrieval = events.find((event) => event.type === 'retrieval');
    if (retrieval?.type !== 'retrieval') throw new Error('no retrieval event');
    const result = events.find((event) => event.type === 'result');
    if (result?.type !== 'result') throw new Error('no result event');

    const urls = new Set(retrieval.sections.map((section) => section.url));
    // The answer may use any section it was given, so the list below it has to
    // be the whole set rather than the first few.
    expect(result.citations).toHaveLength(urls.size);
  });

  it('suppresses prose that states a number the sources do not support', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async (input: RequestInfo | URL) => {
        const url = String(input);
        if (url.includes('/v1/chat/completions')) {
          return completion({ content: 'You need 96 GB of VRAM for this model.' });
        }
        return jsonResponse({}, 404);
      }),
    );

    const { events } = await ask('what is a simulator');
    const result = events.find((event) => event.type === 'result');
    if (result?.type !== 'result') throw new Error('no result event');

    expect(result.status).toBe('suppressed');
    expect(result.answer).toBe('');
    expect(result.unsupportedNumbers.length).toBeGreaterThan(0);
    expect(result.evidence.length).toBeGreaterThan(0);
  });

  it('runs a read-only tool and never exposes a sizing tool', async () => {
    const chatBodies: Array<Record<string, unknown>> = [];
    let call = 0;
    vi.stubGlobal(
      'fetch',
      vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
        const url = String(input);
        if (url.includes('/v1/chat/completions')) {
          chatBodies.push(JSON.parse(String(init?.body)) as Record<string, unknown>);
          call += 1;
          if (call === 1) {
            return completion({
              content: '',
              toolCalls: [{ id: 'call-1', name: 'list_gpus', arguments: '{"vendor":"nvidia"}' }],
            });
          }
          return completion({ content: 'The GPU explorer compares catalog hardware.' });
        }
        if (url.startsWith('http://gateway.test/systems')) {
          return jsonResponse({
            simulator: 'aisimulate',
            systems: [
              { id: 'h200_sxm', name: 'H200 SXM', vendor: 'nvidia', memory_bytes: 141_000_000_000 },
            ],
          });
        }
        return jsonResponse({}, 404);
      }),
    );

    const { events } = await ask('which nvidia GPUs are in the catalog');

    const toolEvent = events.find((event) => event.type === 'tool');
    expect(toolEvent).toBeDefined();
    if (toolEvent?.type === 'tool') {
      expect(toolEvent.ok).toBe(true);
      expect(toolEvent.name).toBe('list_gpus');
    }

    const result = events.find((event) => event.type === 'result');
    expect(result?.type).toBe('result');
    if (result?.type === 'result') {
      expect(result.status).toBe('answered');
      expect(result.links.some((link) => link.href.startsWith('/gpu-explorer'))).toBe(true);
    }

    // The model is offered read-only tools only.
    const toolNames = (chatBodies[0].tools as Array<{ function: { name: string } }>).map(
      (tool) => tool.function.name,
    );
    expect(toolNames).toContain('list_gpus');
    expect(toolNames).not.toContain('recommend');
    expect(toolNames).not.toContain('predict');
  });

  it('returns an error event when the model is unreachable', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => {
        throw new Error('connect ECONNREFUSED');
      }),
    );

    const { events } = await ask('where do I set concurrency');
    const error = events.find((event) => event.type === 'error');
    expect(error).toBeDefined();
    if (error?.type === 'error') {
      expect(error.code).toBe('assistant_unavailable');
    }
  });

  it('adds the shared simulator to generated links', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async (input: RequestInfo | URL) => {
        const url = String(input);
        if (url.includes('/v1/chat/completions')) {
          return completion({ content: 'Use Recommend sizing to find the smallest deployment.' });
        }
        return jsonResponse({}, 404);
      }),
    );

    const { events } = await ask('how do I recommend a sizing');
    const result = events.find((event) => event.type === 'result');
    if (result?.type !== 'result') throw new Error('no result event');

    const recommendLink = result.links.find((link) => link.href.startsWith('/recommend'));
    expect(recommendLink).toBeDefined();
    expect(recommendLink?.href).toContain('simulator=blis');
    expect(recommendLink?.href).toContain('v=1');
    // A question asked on /predict should not link back to /predict.
    expect(result.links.some((link) => link.href.startsWith('/predict'))).toBe(false);
  });

  it('proposes the view the question describes, ahead of the documentation link', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async (input: RequestInfo | URL) => {
        const url = String(input);
        if (url.includes('/v1/chat/completions')) {
          return completion({ content: 'Use Recommend sizing, then press Calculate.' });
        }
        return jsonResponse({}, 404);
      }),
    );

    const { events } = await ask(
      'how many GPUs for 4096 input tokens and 512 output tokens at 50 requests per second',
      {
        route: '/predict',
        model: 'meta-llama/Llama-3.1-8B-Instruct',
        system: 'nvidia-l40s',
      },
    );
    const result = events.find((event) => event.type === 'result');
    if (result?.type !== 'result') throw new Error('no result event');

    const proposal = result.links[0];
    expect(proposal.kind).toBe('proposal');
    expect(proposal.href.startsWith('/recommend?v=1')).toBe(true);
    expect(proposal.href).toContain('isl=4096');
    expect(proposal.href).toContain('osl=512');
    expect(proposal.href).toContain('demandRate=50');
    expect(proposal.href).toContain('system=nvidia-l40s');
    expect(proposal.href).toContain('simulator=blis');
    expect(proposal.description).toContain('input length');
    // The proposal supersedes the plain hint for the same page.
    expect(result.links.filter((link) => link.href.startsWith('/recommend'))).toHaveLength(1);
  });

  it('keeps a proposal for the page the question was asked on', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async (input: RequestInfo | URL) => {
        const url = String(input);
        if (url.includes('/v1/chat/completions')) {
          return completion({ content: 'Enter those values and press Calculate.' });
        }
        return jsonResponse({}, 404);
      }),
    );

    const { events } = await ask('how many GPUs for 4096 input tokens', {
      route: '/recommend',
      model: 'meta-llama/Llama-3.1-8B-Instruct',
      system: 'nvidia-l40s',
    });
    const result = events.find((event) => event.type === 'result');
    if (result?.type !== 'result') throw new Error('no result event');

    // The plain self-link is suppressed, but the proposal carries values, so it
    // is the one case where a link to the current page is useful.
    expect(result.links[0].kind).toBe('proposal');
    expect(result.links[0].href).toContain('isl=4096');
  });

  it('keeps a tool link on the home page instead of treating every link as current', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async (input: RequestInfo | URL) => {
        if (String(input).includes('/v1/chat/completions')) {
          return completion({ content: 'Recommend sizing starts from the targets you name.' });
        }
        return jsonResponse({}, 404);
      }),
    );

    const { events } = await ask('how do I recommend a sizing', { route: '/' });
    const result = events.find((event) => event.type === 'result');
    if (result?.type !== 'result') throw new Error('no result event');

    // "/" is a prefix of every path, so a prefix comparison deleted the sizing
    // link and left only documentation — the "link that opens nothing".
    const hint = result.links.find((link) => link.href.startsWith('/recommend'));
    expect(hint).toBeDefined();
    expect(hint?.kind).toBe('page');
  });

  it('does not link the page hint back to the page it is already on', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async (input: RequestInfo | URL) => {
        if (String(input).includes('/v1/chat/completions')) {
          return completion({ content: 'Recommend sizing starts from the targets you name.' });
        }
        return jsonResponse({}, 404);
      }),
    );

    const { events } = await ask('how do I recommend a sizing', { route: '/recommend' });
    const result = events.find((event) => event.type === 'result');
    if (result?.type !== 'result') throw new Error('no result event');

    // The hint carries only provenance, so reloading /recommend would set
    // nothing. The slot goes to the documentation link instead.
    expect(result.links.some((link) => link.href.startsWith('/recommend'))).toBe(false);
    expect(result.links[0].href.startsWith('/docs/')).toBe(true);
  });

  it('does not mark a documentation link as the action', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async (input: RequestInfo | URL) => {
        if (String(input).includes('/v1/chat/completions')) {
          return completion({ content: 'The verifier checks quantitative claims.' });
        }
        return jsonResponse({}, 404);
      }),
    );

    const { events } = await ask('what does the numeric verifier check');
    const result = events.find((event) => event.type === 'result');
    if (result?.type !== 'result') throw new Error('no result event');

    expect(result.links.length).toBeGreaterThan(0);
    expect(result.links[0].href.startsWith('/docs/')).toBe(true);
    // Only a proposal or a tool page is an action; reading is not.
    expect(result.links.every((link) => link.kind === undefined)).toBe(true);
  });

  it('opens the reported sizing question with the values it names', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async (input: RequestInfo | URL) => {
        if (String(input).includes('/v1/chat/completions')) {
          return completion({ content: 'Use Recommend sizing, then press Calculate.' });
        }
        return jsonResponse({}, 404);
      }),
    );

    const { events } = await ask(
      'recommend a small Qwen model that can perform 1second TTFT, 100ms ITL with a default workload and minimal H200 GPUs',
      { route: '/', model: 'Qwen/Qwen3-32B', system: 'h200_sxm' },
    );
    const result = events.find((event) => event.type === 'result');
    if (result?.type !== 'result') throw new Error('no result event');

    const proposal = result.links[0];
    expect(proposal.kind).toBe('proposal');
    expect(proposal.href.startsWith('/recommend?v=1')).toBe(true);
    expect(proposal.href).toContain('ttft=1000');
    expect(proposal.href).toContain('tpot=100');
    expect(proposal.href).toContain('goal=latency');
    expect(proposal.href).toContain('system=h200_sxm');
  });

  it('carries the backend and never a simulator on a KV cache link', async () => {
    let call = 0;
    vi.stubGlobal(
      'fetch',
      vi.fn(async (input: RequestInfo | URL) => {
        const url = String(input);
        if (url.includes('/v1/chat/completions')) {
          call += 1;
          if (call === 1) {
            return completion({
              content: '',
              toolCalls: [
                {
                  id: 'call-memory',
                  name: 'memory',
                  arguments: '{"model":"Qwen/Qwen3-32B","system":"h200_sxm"}',
                },
              ],
            });
          }
          return completion({ content: 'The KV cache holds 62,500 tokens.' });
        }
        if (url.endsWith('/memory')) {
          return jsonResponse({
            simulator: 'aisimulate',
            total_gpu_capacity_bytes: 141_000_000_000,
            total_kv_size_tokens: 62_500,
            memory_breakdown: { weights_bytes: 70_000_000_000 },
          });
        }
        return jsonResponse({}, 404);
      }),
    );

    const { events } = await ask('how much VRAM does this model need', {
      simulator: 'aisimulate',
      backend: 'vllm',
      backendVersion: '0.10.1',
    });

    const result = events.find((event) => event.type === 'result');
    if (result?.type !== 'result') throw new Error('no result event');

    const kvLink = result.links.find((link) => link.href.includes('model='));
    expect(kvLink).toBeDefined();
    expect(kvLink?.href.startsWith('/kv-cache')).toBe(true);
    expect(kvLink?.href).toContain('v=1');
    expect(kvLink?.href).toContain('backend=vllm');
    expect(kvLink?.href).toContain('backendVersion=0.10.1');
    // /kv-cache has no simulator choice: a simulator field would imply a
    // choice the page does not offer.
    expect(kvLink?.href).not.toContain('simulator=');
  });

  it('keeps a value-carrying link to the page the question was asked on', async () => {
    let call = 0;
    vi.stubGlobal(
      'fetch',
      vi.fn(async (input: RequestInfo | URL) => {
        const url = String(input);
        if (url.includes('/v1/chat/completions')) {
          call += 1;
          if (call === 1) {
            return completion({
              content: '',
              toolCalls: [
                {
                  id: 'call-memory',
                  name: 'memory',
                  arguments: '{"model":"Qwen/Qwen3-32B","system":"h200_sxm"}',
                },
              ],
            });
          }
          return completion({ content: 'The KV cache holds 62,500 tokens.' });
        }
        if (url.endsWith('/memory')) {
          return jsonResponse({
            simulator: 'aisimulate',
            total_gpu_capacity_bytes: 141_000_000_000,
            total_kv_size_tokens: 62_500,
            memory_breakdown: { weights_bytes: 70_000_000_000 },
          });
        }
        return jsonResponse({}, 404);
      }),
    );

    // On the calculator itself, the plain page hint is redundant, but the tool
    // link carries the model and system the question named, so it still fills
    // the form and must survive.
    const { events } = await ask('how much VRAM does this model need', {
      route: '/kv-cache',
      simulator: 'aisimulate',
    });

    const result = events.find((event) => event.type === 'result');
    if (result?.type !== 'result') throw new Error('no result event');

    const kvLinks = result.links.filter((link) => link.href.startsWith('/kv-cache'));
    expect(kvLinks).toHaveLength(1);
    expect(kvLinks[0].href).toContain('model=Qwen%2FQwen3-32B');
    expect(kvLinks[0].href).toContain('system=h200_sxm');
  });

  it('answers every tool call it echoes back to the model', async () => {
    const chatBodies: Array<{ messages?: Array<Record<string, unknown>> }> = [];
    let call = 0;
    vi.stubGlobal(
      'fetch',
      vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
        const url = String(input);
        if (url.includes('/v1/chat/completions')) {
          chatBodies.push(JSON.parse(String(init?.body)) as { messages?: Array<Record<string, unknown>> });
          call += 1;
          if (call === 1) {
            // More calls than the round limit: the tail must not be echoed.
            return completion({
              content: '',
              reasoning: 'The catalog tools answer this directly.',
              toolCalls: [
                { id: 'call-1', name: 'list_gpus', arguments: '{}' },
                { id: 'call-2', name: 'list_backends', arguments: '{}' },
                { id: 'call-3', name: 'list_models', arguments: '{}' },
              ],
            });
          }
          return completion({ content: 'The catalog lists GPUs and backends.' });
        }
        return jsonResponse({});
      }),
    );

    const { events } = await ask('what is in the catalog');
    expect(events.filter((event) => event.type === 'tool')).toHaveLength(2);

    const followUp = chatBodies[1]?.messages ?? [];
    const assistant = followUp.find((message) => Array.isArray(message.tool_calls));
    // K2-Horizon's template rejects an assistant turn with no thinking field,
    // so the reasoning that produced the calls has to travel with them.
    expect(assistant?.reasoning).toBe('The catalog tools answer this directly.');
    const echoed = assistant?.tool_calls as
      | Array<{ id: string; type: string; function: { name: string; arguments: string } }>
      | undefined;
    expect(echoed?.map((entry) => entry.id)).toEqual(['call-1', 'call-2']);
    // The chat API validates the replayed call, so it must be on the wire in
    // the same shape the model produced: `{id, type, function:{name,arguments}}`.
    for (const entry of echoed ?? []) {
      expect(entry.type).toBe('function');
      expect(typeof entry.function?.name).toBe('string');
      expect(typeof entry.function?.arguments).toBe('string');
    }

    const answered = followUp
      .filter((message) => message.role === 'tool')
      .map((message) => message.tool_call_id);
    expect(answered).toEqual(['call-1', 'call-2']);
  });
});
