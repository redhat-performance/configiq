import { NextResponse } from 'next/server';
import { z } from 'zod';
import helpIndex from '@/lib/docs/generated/help-index.json';
import { parseHelpIndex, routeLabel } from '@/lib/help/search';
import { retrieve } from '@/lib/assistant/retrieve';
import { toolByName, toolSchemas, type ToolContext, type ToolResult } from '@/lib/assistant/tools';
import { verifyNumbers } from '@/lib/assistant/verify';
import { AssistantBusyError, assistantQueue } from '@/lib/assistant/queue';
import { SYSTEM_PROMPT, referenceBlock, toolResultBlock } from '@/lib/assistant/prompt';
import { recordAssistantAnswer, recordAssistantToolCall } from '@/lib/otel-metrics';
import { assistantConfig, type AssistantConfig } from '@/lib/assistant/config';
import type {
  AssistantCitation,
  AssistantEvent,
  AssistantLink,
  RetrievedSection,
} from '@/lib/assistant/types';

export const dynamic = 'force-dynamic';

/**
 * Assistant route.
 *
 * Server-side only: the browser posts a question, the route retrieves sections,
 * lets the model call read-only tools, and verifies every number before the
 * prose is returned. Sizing is never executed here — `recommend` and `predict`
 * are not exposed as tools, and a sizing question ends with a link the user
 * opens and submits themselves.
 *
 * The response is newline-delimited JSON so progress events (retrieval, tool
 * reads) reach the popover while the model is still working. The answer itself
 * is emitted once, after verification: showing tokens first and retracting them
 * later would mean a number could be on screen before it was checked.
 */

const MAX_QUESTION_LENGTH = 500;
const MAX_TOOL_ROUNDS = 2;
const MAX_TOOL_CALLS_PER_ROUND = 2;
const ANSWER_TOKEN_LIMIT = 320;
const MODEL_TIMEOUT_MS = 25_000;
const TOOL_TIMEOUT_MS = 8_000;

const requestSchema = z.object({
  question: z.string().trim().min(3).max(MAX_QUESTION_LENGTH),
  route: z.string().trim().max(200).optional(),
  simulator: z.string().trim().max(120).optional(),
  backend: z.string().trim().max(120).optional(),
  backendVersion: z.string().trim().max(120).optional(),
});

type AssistantRequestInput = z.infer<typeof requestSchema>;

/** Availability probe for the help popover. No secret ever leaves the server. */
export async function GET() {
  const config = assistantConfig();
  return NextResponse.json(
    {
      enabled: config.enabled,
      model: config.model,
      reason: config.reason,
    },
    { headers: { 'Cache-Control': 'no-store' } },
  );
}

function unavailable(code: string, message: string, status: number) {
  return NextResponse.json(
    { status: 'unavailable', code, message },
    { status, headers: { 'Cache-Control': 'no-store' } },
  );
}

/** A fetch helper with a timeout that returns null instead of throwing. */
function jsonFetcher(timeoutMs: number) {
  return async (url: string, init?: RequestInit): Promise<unknown | null> => {
    try {
      const response = await fetch(url, {
        ...init,
        headers: { Accept: 'application/json', ...(init?.headers ?? {}) },
        cache: 'no-store',
        signal: AbortSignal.timeout(timeoutMs),
      });
      if (!response.ok) {
        await response.body?.cancel();
        return null;
      }
      return (await response.json()) as unknown;
    } catch {
      return null;
    }
  };
}

interface ChatMessage {
  role: 'system' | 'user' | 'assistant' | 'tool';
  content: string;
  tool_calls?: unknown;
  tool_call_id?: string;
  name?: string;
}

interface ToolCall {
  id: string;
  name: string;
  arguments: string;
}

interface ChatReply {
  content: string;
  toolCalls: ToolCall[];
}

function readToolCalls(raw: unknown): ToolCall[] {
  if (!Array.isArray(raw)) return [];
  return raw.flatMap((entry): ToolCall[] => {
    if (!entry || typeof entry !== 'object') return [];
    const call = entry as Record<string, unknown>;
    const fn = call.function && typeof call.function === 'object'
      ? (call.function as Record<string, unknown>)
      : null;
    const name = typeof fn?.name === 'string' ? fn.name : null;
    const id = typeof call.id === 'string' ? call.id : null;
    if (!name || !id) return [];
    return [{
      id,
      name,
      arguments: typeof fn?.arguments === 'string' ? fn.arguments : '{}',
    }];
  });
}

async function chatCompletion(
  tritonUrl: string,
  model: string,
  messages: ChatMessage[],
  withTools: boolean,
): Promise<ChatReply | null> {
  let response: Response;
  try {
    response = await fetch(`${tritonUrl}/v1/chat/completions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
      body: JSON.stringify({
        model,
        messages,
        temperature: 0,
        max_tokens: ANSWER_TOKEN_LIMIT,
        ...(withTools
          ? { tools: toolSchemas(), tool_choice: 'auto' }
          : {}),
      }),
      cache: 'no-store',
      signal: AbortSignal.timeout(MODEL_TIMEOUT_MS),
    });
  } catch {
    return null;
  }

  if (!response.ok) {
    await response.body?.cancel();
    return null;
  }

  try {
    const payload = (await response.json()) as {
      choices?: Array<{ message?: { content?: unknown; tool_calls?: unknown } }>;
    };
    const message = payload.choices?.[0]?.message;
    return {
      content: typeof message?.content === 'string' ? message.content.trim() : '',
      toolCalls: readToolCalls(message?.tool_calls),
    };
  } catch {
    return null;
  }
}

/** Pages a question is likely to continue on, with the provenance trio added. */
const PAGE_HINTS: Array<{ pattern: RegExp; href: string; label: string }> = [
  { pattern: /\brecommend|sizing goal|how many gpus|capacity|demand\b/i, href: '/recommend', label: 'Open Recommend sizing' },
  { pattern: /\bpredict|ttft|tpot|latency|performance\b/i, href: '/predict', label: 'Open Predict performance' },
  { pattern: /\bkv cache|kv-cache|vram|memory\b/i, href: '/kv-cache', label: 'Open KV cache calculator' },
  { pattern: /\bhybrid|hosted|purchased|rented\b/i, href: '/hybrid-savings', label: 'Open Hybrid savings' },
  { pattern: /\bbenchmark|validation|measured\b/i, href: '/validation/compare', label: 'Open Benchmark comparison' },
  { pattern: /\bgpu explorer|bubble|tflops|bandwidth\b/i, href: '/gpu-explorer', label: 'Open GPU explorer' },
  { pattern: /\bsettings|simulator|backend\b/i, href: '/settings', label: 'Open Settings' },
];

/** Pages whose link contract carries the backend/version half of provenance. */
const PROVENANCE_ROUTES = new Set([
  '/predict',
  '/recommend',
  '/hybrid-savings',
  '/validation/compare',
  // AISimulate-only by contract: it carries the backend but never a simulator.
  '/kv-cache',
]);

/** Pages that also accept a `simulator` field. */
const SIMULATOR_ROUTES = new Set(['/predict', '/recommend', '/hybrid-savings', '/validation/compare']);

/** Pages with a versioned link contract. Other pages take no query at all. */
const LINK_ROUTES = new Set([...PROVENANCE_ROUTES, '/gpu-explorer']);

/**
 * Add provenance to a page link.
 *
 * Documentation links are `/docs/<slug>#<anchor>`: the fragment is the
 * address, so they are returned untouched. Appending `?v=1` after the hash
 * would move the query inside the fragment and break the anchor.
 */
function withProvenance(href: string, request: AssistantRequestInput): string {
  if (href.startsWith('/docs/') || href === '/docs') return href;

  const hashIndex = href.indexOf('#');
  const hash = hashIndex === -1 ? '' : href.slice(hashIndex);
  const withoutHash = hashIndex === -1 ? href : href.slice(0, hashIndex);
  const queryIndex = withoutHash.indexOf('?');
  const path = queryIndex === -1 ? withoutHash : withoutHash.slice(0, queryIndex);
  if (!LINK_ROUTES.has(path)) return href;

  const params = new URLSearchParams(queryIndex === -1 ? '' : withoutHash.slice(queryIndex + 1));
  if (!params.has('v')) params.set('v', '1');
  if (SIMULATOR_ROUTES.has(path) && request.simulator) params.set('simulator', request.simulator);
  if (PROVENANCE_ROUTES.has(path)) {
    if (request.backend) params.set('backend', request.backend);
    if (request.backendVersion) params.set('backendVersion', request.backendVersion);
  }
  return `${path}?${params.toString()}${hash}`;
}

function linksForAnswer(
  request: AssistantRequestInput,
  sections: RetrievedSection[],
  toolResults: ToolResult[],
): AssistantLink[] {
  const links: AssistantLink[] = [];
  const seen = new Set<string>();

  const push = (link: AssistantLink) => {
    const href = withProvenance(link.href, request);
    if (seen.has(href)) return;
    seen.add(href);
    links.push({ ...link, href });
  };

  for (const result of toolResults) {
    if (result.link) push(result.link);
  }

  for (const hint of PAGE_HINTS) {
    if (hint.pattern.test(request.question)) {
      push({ label: hint.label, href: hint.href });
      break;
    }
  }

  // A question about a page the user is already on does not need a link to it.
  if (request.route && routeLabel(request.route) !== request.route) {
    const current = links.find((link) => link.href.startsWith(request.route ?? ''));
    if (current) links.splice(links.indexOf(current), 1);
  }

  const primary = sections[0];
  if (primary && !seen.has(primary.url)) {
    push({ label: `Read ${primary.title}`, href: primary.url, description: primary.section });
  }

  return links.slice(0, 3);
}

function citationsFor(sections: RetrievedSection[]): AssistantCitation[] {
  const seen = new Set<string>();
  const citations: AssistantCitation[] = [];
  for (const section of sections) {
    if (seen.has(section.url)) continue;
    seen.add(section.url);
    citations.push({ url: section.url, title: section.title, section: section.section });
    if (citations.length === 3) break;
  }
  return citations;
}

function retrievalDocs() {
  // `parseHelpIndex` validates the generated artifact, so a malformed build
  // degrades to "no citations" instead of an exception inside the stream.
  return parseHelpIndex(helpIndex as unknown);
}

type Emit = (event: AssistantEvent) => void;

async function runAssistant(
  request: AssistantRequestInput,
  config: AssistantConfig,
  emit: Emit,
): Promise<void> {
  const docs = retrievalDocs();
  const sections = docs.length > 0 ? retrieve(request.question, docs, 6) : [];
  emit({ type: 'retrieval', sections });

  const context = [
    referenceBlock(sections),
    `The user is currently on ${routeLabel(request.route ?? '/')}.`,
  ].join('\n\n');

  const messages: ChatMessage[] = [
    { role: 'system', content: SYSTEM_PROMPT },
    { role: 'user', content: `${context}\n\nQuestion: ${request.question}` },
  ];

  const fetchJson = jsonFetcher(TOOL_TIMEOUT_MS);
  const contextForTools: ToolContext = {
    gatewayUrl: process.env.AISIMULATORS_GATEWAY_URL?.trim() || null,
    costingsUrl: process.env.AICOSTINGS_GATEWAY_URL?.trim() || null,
    simulator: request.simulator,
    fetchJson,
    fetchJsonPost: (url, body) =>
      fetchJson(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      }),
  };

  const toolResults: ToolResult[] = [];
  let reply: ChatReply | null = null;

  for (let round = 0; round <= MAX_TOOL_ROUNDS; round += 1) {
    const allowTools = round < MAX_TOOL_ROUNDS;
    reply = await chatCompletion(config.tritonUrl!, config.model!, messages, allowTools);
    if (!reply) {
      emit({
        type: 'error',
        code: 'assistant_unavailable',
        message: 'The assistant model did not answer. Search and documentation still work.',
      });
      recordAssistantAnswer('error');
      return;
    }
    if (!allowTools || reply.toolCalls.length === 0) break;

    // Only the calls we answer are echoed back. An OpenAI-compatible server
    // rejects a continuation whenever an assistant `tool_call` has no matching
    // `tool` message, so dropping the tail must happen before the echo.
    const calls = reply.toolCalls.slice(0, MAX_TOOL_CALLS_PER_ROUND);
    messages.push({ role: 'assistant', content: reply.content, tool_calls: calls });

    for (const call of calls) {
      const tool = toolByName(call.name);
      if (!tool) {
        messages.push({
          role: 'tool',
          tool_call_id: call.id,
          name: call.name,
          content: JSON.stringify({ ok: false, reason: 'Unknown tool.' }),
        });
        continue;
      }

      let args: Record<string, unknown> = {};
      try {
        const parsed = JSON.parse(call.arguments) as unknown;
        if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
          args = parsed as Record<string, unknown>;
        }
      } catch {
        // Malformed arguments are reported to the model rather than guessed at.
      }

      const result = await tool.execute(args, contextForTools);
      toolResults.push(result);
      recordAssistantToolCall(call.name, result.ok);
      emit({ type: 'tool', name: call.name, ok: result.ok, summary: result.summary });

      messages.push({
        role: 'tool',
        tool_call_id: call.id,
        name: call.name,
        content: JSON.stringify({
          ok: result.ok,
          summary: result.summary,
          evidence: result.evidence,
          reason: result.reason,
        }),
      });
    }
  }

  const draft = reply?.content?.trim() ?? '';
  if (!draft) {
    emit({
      type: 'error',
      code: 'assistant_unavailable',
      message: 'The assistant returned an empty answer. Search and documentation still work.',
    });
    recordAssistantAnswer('error');
    return;
  }

  const evidence = [
    ...sections.map((section) => `${section.title}\n${section.body}`),
    ...toolResults.flatMap((result) => result.evidence),
  ];

  const verification = verifyNumbers(draft, evidence);
  const citations = citationsFor(sections);
  const links = linksForAnswer(request, sections, toolResults);

  emit({
    type: 'result',
    status: verification.ok ? 'answered' : 'suppressed',
    answer: verification.ok ? draft : '',
    citations,
    links,
    evidence: evidence.slice(0, 8),
    unsupportedNumbers: verification.unsupported,
    model: config.model!,
  });
  recordAssistantAnswer(verification.ok ? 'answered' : 'suppressed');
}

export async function POST(request: Request) {
  const config = assistantConfig();
  if (!config.enabled) {
    return unavailable('assistant_disabled', config.reason ?? 'The assistant is not enabled.', 503);
  }

  let payload: unknown;
  try {
    payload = await request.json();
  } catch {
    return unavailable('invalid_request', 'The question was not valid JSON.', 400);
  }

  const parsed = requestSchema.safeParse(payload);
  if (!parsed.success) {
    return unavailable('invalid_request', `Ask a question between 3 and ${MAX_QUESTION_LENGTH} characters.`, 400);
  }

  let release: () => void;
  try {
    release = await assistantQueue.acquire();
  } catch (error) {
    if (error instanceof AssistantBusyError) {
      return unavailable('assistant_busy', 'The assistant is busy. Search and documentation still work.', 503);
    }
    return unavailable('assistant_unavailable', 'The assistant is unavailable right now.', 503);
  }

  const encoder = new TextEncoder();
  const stream = new ReadableStream<Uint8Array>({
    async start(controller) {
      const emit: Emit = (event) => {
        controller.enqueue(encoder.encode(`${JSON.stringify(event)}\n`));
      };
      try {
        await runAssistant(parsed.data, config, emit);
      } catch {
        emit({
          type: 'error',
          code: 'assistant_unavailable',
          message: 'The assistant could not finish this answer. Search and documentation still work.',
        });
        recordAssistantAnswer('error');
      } finally {
        release();
        controller.close();
      }
    },
    cancel() {
      // The user closed the popover; free the inference slot immediately.
      release();
    },
  });

  return new Response(stream, {
    status: 200,
    headers: {
      'Content-Type': 'application/x-ndjson; charset=utf-8',
      'Cache-Control': 'no-store',
      // nginx must not buffer progress events.
      'X-Accel-Buffering': 'no',
    },
  });
}
