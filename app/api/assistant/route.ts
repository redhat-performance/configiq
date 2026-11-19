import { NextResponse } from 'next/server';
import { z } from 'zod';
import helpIndex from '@/lib/docs/generated/help-index.json';
import { parseHelpIndex, routeLabel } from '@/lib/help/search';
import { retrieve } from '@/lib/assistant/retrieve';
import { pageHintFor } from '@/lib/assistant/intent';
import { proposeView, type ProposedView } from '@/lib/assistant/propose';
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
  AssistantResultStatus,
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
/**
 * Generation cap, not an answer cap: a reasoning model's thinking tokens count
 * against it too. K2-Horizon defaults to a long reasoning trace, so a cap sized
 * for the 120-word answer alone would end generation inside the reasoning
 * block; vLLM then has no closing tag to split on and returns the unfinished
 * thought as `content`.
 */
const GENERATION_TOKEN_LIMIT = 640;
/**
 * Reasoning effort for models whose chat template reads it. K2-Horizon accepts
 * low/medium/high and defaults to high; low keeps the trace short enough to
 * reach the answer inside the generation cap. Templates that do not define the
 * variable ignore it, so this is not a model-specific branch.
 */
const REASONING_EFFORT = 'low';
/**
 * Sections handed to the model, and therefore the most URLs that can be cited.
 * The list under an answer is what the model read, so it has to cover every
 * section the prompt allowed it to use; a shorter list would let a claim rest
 * on a source the reader was never shown.
 */
const REFERENCE_SECTIONS = 6;
// Generation now includes the reasoning trace, so the deadline covers a longer
// reply than the answer alone. Low effort keeps the common case well inside it.
const MODEL_TIMEOUT_MS = 40_000;
const TOOL_TIMEOUT_MS = 8_000;

const requestSchema = z.object({
  question: z.string().trim().min(3).max(MAX_QUESTION_LENGTH),
  route: z.string().trim().max(200).optional(),
  simulator: z.string().trim().max(120).optional(),
  backend: z.string().trim().max(120).optional(),
  backendVersion: z.string().trim().max(120).optional(),
  // The page's current choices, so a proposal can open a complete view when
  // the question names neither. Settings live in the browser, so this is the
  // only way the server learns them.
  model: z.string().trim().max(200).optional(),
  system: z.string().trim().max(120).optional(),
});

type AssistantRequestInput = z.infer<typeof requestSchema>;

/**
 * Availability probe for the help popover. No secret ever leaves the server.
 *
 * `endpoint` is the resolved base URL the assistant would call. It is an
 * internal container name, not a credential, and it is reported because a
 * deployment that sets ASSISTANT_ENABLED against an image that still reads the
 * FIL classifier URL would otherwise look healthy here and fail on the first
 * question. `deploy.sh` compares this value with the assistant it deployed.
 */
export async function GET() {
  const config = assistantConfig();
  return NextResponse.json(
    {
      enabled: config.enabled,
      model: config.model,
      endpoint: config.endpoint,
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
function jsonFetcher(timeoutMs: number, signal?: AbortSignal) {
  return async (url: string, init?: RequestInit): Promise<unknown | null> => {
    try {
      const response = await fetch(url, {
        ...init,
        headers: { Accept: 'application/json', ...(init?.headers ?? {}) },
        cache: 'no-store',
        signal: withDeadline(signal, timeoutMs),
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

/**
 * Stop work when either the caller goes away or the deadline passes.
 *
 * `AbortSignal.any` keeps both reasons without a listener that would have to be
 * removed, and it is available in every Node version this app targets.
 */
function withDeadline(signal: AbortSignal | undefined, timeoutMs: number): AbortSignal {
  const timeout = AbortSignal.timeout(timeoutMs);
  return signal ? AbortSignal.any([signal, timeout]) : timeout;
}

interface ChatToolCall {
  id: string;
  type: 'function';
  function: { name: string; arguments: string };
}

interface ChatMessage {
  role: 'system' | 'user' | 'assistant' | 'tool';
  content: string;
  /**
   * The reasoning a reasoning model produced for this turn. K2-Horizon's chat
   * template rejects an assistant message that carries none of its thinking
   * fields, so a replayed tool-call turn has to include it.
   */
  reasoning?: string;
  tool_calls?: ChatToolCall[];
  tool_call_id?: string;
  name?: string;
}

interface ToolCall {
  id: string;
  name: string;
  arguments: string;
}

interface ChatReply {
  /** Only the model's final content; reasoning is kept apart from it. */
  content: string;
  reasoning: string;
  /** `length` means generation stopped mid-turn, so `content` is not an answer. */
  finishReason: string | null;
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

/**
 * Put an assistant tool call back on the wire in the shape the chat API sent
 * it. The model's own history replay is validated by OpenAI-compatible servers,
 * and a `{id,name,arguments}` object is not a `tool_call` to them.
 */
function toWireToolCall(call: ToolCall): ChatToolCall {
  return {
    id: call.id,
    type: 'function',
    function: { name: call.name, arguments: call.arguments },
  };
}

async function chatCompletion(
  endpoint: string,
  model: string,
  messages: ChatMessage[],
  withTools: boolean,
  signal?: AbortSignal,
): Promise<ChatReply | null> {
  let response: Response;
  try {
    response = await fetch(`${endpoint}/v1/chat/completions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
      body: JSON.stringify({
        model,
        messages,
        temperature: 0,
        max_tokens: GENERATION_TOKEN_LIMIT,
        // A reasoning model applies this before it starts its trace; the cap
        // above then has room for both the trace and the answer. Chat templates
        // that do not define the variable ignore it.
        chat_template_kwargs: { reasoning_effort: REASONING_EFFORT },
        ...(withTools
          ? { tools: toolSchemas(), tool_choice: 'auto' }
          : {}),
      }),
      cache: 'no-store',
      signal: withDeadline(signal, MODEL_TIMEOUT_MS),
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
      choices?: Array<{
        message?: {
          content?: unknown;
          reasoning?: unknown;
          // Older OpenAI-compatible servers still use this name.
          reasoning_content?: unknown;
          tool_calls?: unknown;
        };
        finish_reason?: unknown;
      }>;
    };
    const choice = payload.choices?.[0];
    const message = choice?.message;
    const reasoning =
      typeof message?.reasoning === 'string'
        ? message.reasoning
        : typeof message?.reasoning_content === 'string'
          ? message.reasoning_content
          : '';
    return {
      content: typeof message?.content === 'string' ? message.content.trim() : '',
      reasoning,
      finishReason: typeof choice?.finish_reason === 'string' ? choice.finish_reason : null,
      toolCalls: readToolCalls(message?.tool_calls),
    };
  } catch {
    return null;
  }
}

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

/** The path of a page link, without its query or fragment. */
function linkPath(href: string): string {
  const withoutHash = href.split('#')[0];
  const queryIndex = withoutHash.indexOf('?');
  return queryIndex === -1 ? withoutHash : withoutHash.slice(0, queryIndex);
}

/**
 * Parameters that describe where a view came from rather than what it asks
 * for. A page link carrying only these sets nothing on the page.
 */
const PROVENANCE_PARAMS = new Set(['v', 'simulator', 'backend', 'backendVersion']);

/**
 * A plain link to the page the user is already looking at.
 *
 * It would reload the same view, so it is dropped. A link that carries values
 * is not: opening it on the page in front of the user fills the form with what
 * the question named, which is exactly what the link is for.
 */
function isRedundantSelfLink(link: AssistantLink, route: string | undefined): boolean {
  if (!route || link.kind !== 'page' || linkPath(link.href) !== route) return false;
  const query = link.href.split('#')[0].split('?')[1];
  if (!query) return true;
  return [...new URLSearchParams(query).keys()].every((key) => PROVENANCE_PARAMS.has(key));
}

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
  proposal: ProposedView | null,
): AssistantLink[] {
  const links: AssistantLink[] = [];
  const seen = new Set<string>();

  const push = (link: AssistantLink) => {
    // A proposal is already built from the page's contract, which put the
    // version marker on it; adding provenance again would rewrite values the
    // contract just validated.
    const href = link.kind === 'proposal' ? link.href : withProvenance(link.href, request);
    if (seen.has(href)) return;
    seen.add(href);
    links.push({ ...link, href });
  };

  // The view the question describes comes first: it is the answer's action,
  // the rest are places to read more.
  if (proposal) push(proposal.link);

  for (const result of toolResults) {
    // A tool link opens the page the tool read from, which is an action even
    // when it carries no values of its own.
    if (result.link) push({ ...result.link, kind: 'page' });
  }

  const hint = pageHintFor(request.question);
  // A proposal supersedes the plain hint for the same page.
  if (hint && !(proposal && proposal.link.href.startsWith(hint.href))) {
    push({ label: hint.label, href: hint.href, kind: 'page' });
  }

  // A question about a page the user is already on does not need a bare link
  // to it, but a link carrying the values the question named does: opening it
  // fills the form in front of them. A proposal is never dropped, because its
  // contract only emits a link when the question stated something to apply.
  //
  // The comparison is the parsed path, not a prefix: `"/"` is a prefix of every
  // link on the site, so comparing prefixes on the home page deleted the sizing
  // hint the question was asking for.
  //
  // This runs before the documentation link is pushed so a dropped self-link
  // frees its slot for the reading suggestion.
  for (const link of links.filter((candidate) => isRedundantSelfLink(candidate, request.route))) {
    links.splice(links.indexOf(link), 1);
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
    if (citations.length === REFERENCE_SECTIONS) break;
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
  signal: AbortSignal,
): Promise<void> {
  const docs = retrievalDocs();
  const sections = docs.length > 0 ? retrieve(request.question, docs, REFERENCE_SECTIONS) : [];
  emit({ type: 'retrieval', sections });

  // The view the question describes, built from the page's own contract. It
  // needs no model, so it is ready before the model runs and the result can
  // lead with it instead of the plain page hint.
  const proposal = proposeView({
    question: request.question,
    model: request.model,
    system: request.system,
    simulator: request.simulator,
    backend: request.backend,
    backendVersion: request.backendVersion,
  });

  const context = [
    referenceBlock(sections),
    `The user is currently on ${routeLabel(request.route ?? '/')}.`,
  ].join('\n\n');

  const messages: ChatMessage[] = [
    { role: 'system', content: SYSTEM_PROMPT },
    { role: 'user', content: `${context}\n\nQuestion: ${request.question}` },
  ];

  const fetchJson = jsonFetcher(TOOL_TIMEOUT_MS, signal);
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
    reply = await chatCompletion(config.endpoint!, config.model!, messages, allowTools, signal);
    if (!reply) {
      // A cancelled question is not a failure: the user closed the popover.
      if (signal.aborted) return;
      emit({
        type: 'error',
        code: 'assistant_unavailable',
        message: 'The assistant model did not answer. Search and documentation still work.',
      });
      recordAssistantAnswer('error');
      return;
    }
    if (signal.aborted) return;
    if (!allowTools || reply.toolCalls.length === 0) break;

    // Only the calls we answer are echoed back. An OpenAI-compatible server
    // rejects a continuation whenever an assistant `tool_call` has no matching
    // `tool` message, so dropping the tail must happen before the echo.
    const calls = reply.toolCalls.slice(0, MAX_TOOL_CALLS_PER_ROUND);
    messages.push({
      role: 'assistant',
      content: reply.content,
      // The reasoning belongs to this turn. Dropping it makes the continuation
      // invalid for K2-Horizon, whose template requires a thinking field on
      // every assistant message.
      reasoning: reply.reasoning,
      tool_calls: calls.map(toWireToolCall),
    });

    for (const call of calls) {
      if (signal.aborted) return;
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
  /**
   * Generation stopped at the token cap. For a reasoning model that usually
   * means the closing thinking tag never arrived, so the server returns the
   * unfinished trace as `content`. It reads like prose, but it is not an answer
   * and must not be shown.
   */
  const incomplete = reply?.finishReason === 'length';
  if (!draft && !incomplete) {
    if (signal.aborted) return;
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

  // A number the user wrote is quotable, so the question joins the verifier's
  // evidence along with the normalised forms a proposal adds (`2 s` and
  // `2000 ms`). Neither becomes one of the answer's sources.
  const quotable = [request.question, ...(proposal?.evidence ?? [])];
  // A tool summary is a sentence the model was given about what the tool read,
  // so a count it repeats (`10 GPU systems`) is quotable even when the rows do
  // not restate it. Summaries stay out of `evidence` because that list is the
  // sources the reader sees.
  const toolNotes = toolResults.map((result) => result.summary);
  const verification = verifyNumbers(draft, [...evidence, ...quotable, ...toolNotes]);
  const citations = citationsFor(sections);
  const links = linksForAnswer(request, sections, toolResults, proposal);

  const status: AssistantResultStatus = incomplete
    ? 'incomplete'
    : verification.ok
      ? 'answered'
      : 'suppressed';
  emit({
    type: 'result',
    status,
    answer: status === 'answered' ? draft : '',
    citations,
    links,
    evidence: evidence.slice(0, 8),
    unsupportedNumbers: incomplete ? [] : verification.unsupported,
    model: config.model!,
  });
  recordAssistantAnswer(status);
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
  // One controller for both the model request and every tool read, so closing
  // the popover stops the work instead of leaving a slot busy on the GPU.
  const inference = new AbortController();
  let cancelled = false;

  const stream = new ReadableStream<Uint8Array>({
    async start(controller) {
      const emit: Emit = (event) => {
        // A cancelled stream is closed; enqueueing after that throws.
        if (cancelled) return;
        try {
          controller.enqueue(encoder.encode(`${JSON.stringify(event)}\n`));
        } catch {
          cancelled = true;
        }
      };
      try {
        await runAssistant(parsed.data, config, emit, inference.signal);
      } catch {
        if (!cancelled) {
          emit({
            type: 'error',
            code: 'assistant_unavailable',
            message: 'The assistant could not finish this answer. Search and documentation still work.',
          });
          recordAssistantAnswer('error');
        }
      } finally {
        release();
        if (!cancelled) {
          try {
            controller.close();
          } catch {
            // Already closed by the runtime.
          }
        }
      }
    },
    cancel() {
      // Stop inference first, then free the slot: releasing while the model is
      // still generating would let a new question exceed the concurrency cap.
      cancelled = true;
      inference.abort();
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
