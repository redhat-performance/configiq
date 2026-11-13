import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { GET, POST } from './route';
import type { AssistantEvent } from '@/lib/assistant/types';

type Reply = { content: string; toolCalls?: Array<{ id: string; name: string; arguments: string }> };

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
        message: {
          content: reply.content,
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

async function ask(question: string) {
  const response = await POST(
    new Request('http://localhost/api/assistant', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, route: '/predict', simulator: 'blis' }),
    }),
  );
  return { response, events: parseStream(await response.text()) };
}

const realFetch = globalThis.fetch;

beforeEach(() => {
  vi.stubEnv('ASSISTANT_ENABLED', 'true');
  vi.stubEnv('ASSISTANT_MODEL', 'test/assistant-model');
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

  it('reports the model when it is enabled', async () => {
    const response = await GET();
    const body = (await response.json()) as { enabled: boolean; model: string | null };
    expect(body).toEqual({ enabled: true, model: 'test/assistant-model', reason: null });
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
});
