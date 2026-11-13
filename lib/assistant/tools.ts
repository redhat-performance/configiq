import type { AssistantLink } from './types';
import { kvCacheLink } from '@/lib/deep-links/pages';

/**
 * Read-only tools.
 *
 * The assistant may look facts up, but it may not run sizing: `recommend` and
 * `predict` are absent on purpose. A sizing question ends in a pre-filled deep
 * link and the user presses Calculate themselves, so a generated link can never
 * be mistaken for a model-produced recommendation.
 *
 * Every executor returns `evidence`: the exact sentences handed back to the
 * model and to the numeric verifier. If a fact is not in `evidence`, the
 * assistant is not allowed to state it.
 */

export interface ToolResult {
  ok: boolean;
  /** One-line description of what was read, shown in the UI. */
  summary: string;
  /** Facts the model may quote, with their numbers spelled out. */
  evidence: string[];
  /** Structured payload for the model's reasoning. */
  data: unknown;
  /** A page the user can open to continue in the UI. */
  link?: AssistantLink;
  /** Present when the tool could not answer; never fabricated data. */
  reason?: string;
}

export interface ToolContext {
  gatewayUrl: string | null;
  costingsUrl: string | null;
  simulator?: string;
  fetchJson: (url: string, init?: RequestInit) => Promise<unknown | null>;
  fetchJsonPost?: (url: string, body: unknown) => Promise<unknown | null>;
}

export interface ToolDefinition {
  name: string;
  description: string;
  parameters: {
    type: 'object';
    properties: Record<string, unknown>;
    required?: string[];
    additionalProperties: false;
  };
  execute: (args: Record<string, unknown>, context: ToolContext) => Promise<ToolResult>;
}

const GB = 1_073_741_824;

function asRecord(value: unknown): Record<string, unknown> | null {
  return value && typeof value === 'object' && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

function asArray(value: unknown): unknown[] {
  return Array.isArray(value) ? value : [];
}

function str(value: unknown): string | null {
  return typeof value === 'string' && value.trim().length > 0 ? value.trim() : null;
}

function num(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

function unavailable(summary: string, reason: string): ToolResult {
  return { ok: false, summary, reason, evidence: [], data: null };
}

function withSimulator(context: ToolContext, path: string): string {
  if (!context.simulator) return path;
  const separator = path.includes('?') ? '&' : '?';
  return `${path}${separator}simulator=${encodeURIComponent(context.simulator)}`;
}

/** Format bytes as whole GB, the unit the UI and docs both use. */
export function describeMemoryGb(bytes: number | null): string | null {
  if (bytes === null || bytes <= 0) return null;
  return `${Math.round(bytes / GB)} GB`;
}

const listGpus: ToolDefinition = {
  name: 'list_gpus',
  description:
    'List GPU systems the simulator can price, with memory, bandwidth, BF16 throughput, and GPUs per node. Use for questions about which GPUs exist or how much memory one has.',
  parameters: {
    type: 'object',
    properties: {
      vendor: { type: 'string', description: 'Optional vendor filter, for example nvidia or amd.' },
      min_memory_gb: { type: 'number', description: 'Optional minimum memory per GPU in GB.' },
      limit: { type: 'integer', minimum: 1, maximum: 40 },
    },
    additionalProperties: false,
  },
  async execute(args, context) {
    if (!context.gatewayUrl) return unavailable('GPU catalog unavailable', 'AISimulators is not configured.');
    const raw = await context.fetchJson(
      withSimulator(context, `${context.gatewayUrl}/systems?include=specs`),
    );
    const systems = asArray(asRecord(raw)?.systems);
    if (systems.length === 0) return unavailable('GPU catalog unavailable', 'The service returned no GPU systems.');

    const vendor = str(args.vendor)?.toLowerCase();
    const minBytes = typeof args.min_memory_gb === 'number' ? args.min_memory_gb * GB : 0;
    const limit = typeof args.limit === 'number' ? Math.min(Math.max(args.limit, 1), 40) : 12;

    const rows = systems
      .map(asRecord)
      .filter((system): system is Record<string, unknown> => system !== null)
      .map((system) => ({
        id: str(system.id) ?? '',
        name: str(system.name) ?? str(system.id) ?? '',
        vendor: str(system.vendor) ?? 'unknown',
        memoryBytes: num(system.memory_bytes),
        bandwidthBytes: num(system.memory_bandwidth_bytes),
        bf16Tflops: num(system.bf16_tflops),
        gpusPerNode: num(system.gpus_per_node),
      }))
      .filter((system) => system.id.length > 0)
      .filter((system) => (vendor ? system.vendor.toLowerCase() === vendor : true))
      .filter((system) => (minBytes > 0 ? (system.memoryBytes ?? 0) >= minBytes : true))
      .slice(0, limit);

    if (rows.length === 0) {
      return {
        ok: true,
        summary: 'No GPU matched the filter',
        evidence: ['No GPU system in the catalog matched that filter.'],
        data: [],
        link: { label: 'Open GPU explorer', href: '/gpu-explorer' },
      };
    }

    const evidence = rows.map((system) => {
      const memory = describeMemoryGb(system.memoryBytes);
      const bandwidth =
        system.bandwidthBytes && system.bandwidthBytes > 0
          ? `${(system.bandwidthBytes / 1e12).toFixed(1)} TB/s`
          : null;
      const parts = [
        memory ? `${system.name} (${system.id}) has ${memory} of memory per GPU` : `${system.name} (${system.id})`,
        bandwidth ? `a memory bandwidth of ${bandwidth}` : null,
        system.bf16Tflops ? `${system.bf16Tflops} BF16 TFLOPs` : null,
        system.gpusPerNode ? `${system.gpusPerNode} GPUs per node` : null,
      ].filter(Boolean);
      return `${parts.join(', ')}.`;
    });

    return {
      ok: true,
      summary: `Read ${rows.length} GPU system${rows.length === 1 ? '' : 's'}`,
      evidence,
      data: rows,
      link: { label: 'Open GPU explorer', href: '/gpu-explorer' },
    };
  },
};

const listModels: ToolDefinition = {
  name: 'list_models',
  description:
    'List models the selected simulator can price. Use before claiming a model is or is not supported.',
  parameters: {
    type: 'object',
    properties: {
      query: { type: 'string', description: 'Optional case-insensitive substring, for example llama.' },
      limit: { type: 'integer', minimum: 1, maximum: 50 },
    },
    additionalProperties: false,
  },
  async execute(args, context) {
    if (!context.gatewayUrl) return unavailable('Model catalog unavailable', 'AISimulators is not configured.');
    const raw = await context.fetchJson(withSimulator(context, `${context.gatewayUrl}/models`));
    const models = asArray(asRecord(raw)?.models)
      .map((entry) => (typeof entry === 'string' ? entry : str(asRecord(entry)?.id)))
      .filter((entry): entry is string => entry !== null);
    if (models.length === 0) return unavailable('Model catalog unavailable', 'The service returned no models.');

    const query = str(args.query)?.toLowerCase();
    const limit = typeof args.limit === 'number' ? Math.min(Math.max(args.limit, 1), 50) : 20;
    const matched = models.filter((model) => (query ? model.toLowerCase().includes(query) : true)).slice(0, limit);

    return {
      ok: true,
      summary: `Read ${matched.length} model${matched.length === 1 ? '' : 's'} from the catalog`,
      evidence: [
        matched.length > 0
          ? `The simulator can price these models: ${matched.join(', ')}.`
          : `No model in the simulator catalog contains "${query}".`,
        `The simulator catalog lists ${models.length} models.`,
      ],
      data: matched,
    };
  },
};

const listBackends: ToolDefinition = {
  name: 'list_backends',
  description:
    'List inference backends and versions the simulator supports, including whether a backend supports disaggregated serving.',
  parameters: { type: 'object', properties: {}, additionalProperties: false },
  async execute(_args, context) {
    if (!context.gatewayUrl) return unavailable('Backend catalog unavailable', 'AISimulators is not configured.');
    const raw = await context.fetchJson(withSimulator(context, `${context.gatewayUrl}/backends`));
    const backends = asArray(asRecord(raw)?.backends)
      .map(asRecord)
      .filter((backend): backend is Record<string, unknown> => backend !== null);
    if (backends.length === 0) return unavailable('Backend catalog unavailable', 'The service returned no backends.');

    const evidence = backends.slice(0, 12).map((backend) => {
      const versions = asArray(backend.versions)
        .map((version) => str(asRecord(version)?.version) ?? str(asRecord(version)?.id))
        .filter((version): version is string => version !== null);
      const name = str(backend.id) ?? str(backend.name) ?? 'unknown backend';
      const disaggregated = backend.disaggregated === true ? 'supports disaggregated serving' : null;
      return [
        `${name}`,
        versions.length > 0 ? `versions ${versions.join(', ')}` : null,
        disaggregated,
      ].filter(Boolean).join(' ') + '.';
    });

    return {
      ok: true,
      summary: `Read ${backends.length} backend${backends.length === 1 ? '' : 's'}`,
      evidence,
      data: backends,
    };
  },
};

const listNetworks: ToolDefinition = {
  name: 'list_networks',
  description:
    'List the inter-node network fabrics the selected simulator models, and its default fabric. Use for questions about RoCE, InfiniBand, or prefill/decode transfer. An empty list means the simulator does not model fabrics.',
  parameters: { type: 'object', properties: {}, additionalProperties: false },
  async execute(_args, context) {
    if (!context.gatewayUrl) return unavailable('Network catalog unavailable', 'AISimulators is not configured.');
    const raw = asRecord(await context.fetchJson(withSimulator(context, `${context.gatewayUrl}/networks`)));
    if (!raw) return unavailable('Network catalog unavailable', 'The service returned no network list.');

    const networks = asArray(raw.networks)
      .map((entry) => str(entry))
      .filter((entry): entry is string => entry !== null);
    const fallback = str(raw.default_network);

    if (networks.length === 0) {
      return {
        ok: true,
        summary: 'The selected simulator does not model network fabrics',
        evidence: ['The selected simulator does not model network fabrics, so no fabric is available.'],
        data: { networks, defaultNetwork: null },
      };
    }

    return {
      ok: true,
      summary: `Read ${networks.length} network fabric${networks.length === 1 ? '' : 's'}`,
      evidence: [
        `The selected simulator supports these network fabrics: ${networks.join(', ')}.`,
        fallback
          ? `The default network fabric is ${fallback}.`
          : 'No default network fabric is declared; the fabric must be chosen explicitly.',
      ],
      data: { networks, defaultNetwork: fallback },
    };
  },
};

const pricingOffers: ToolDefinition = {
  name: 'pricing_offers',
  description:
    'Read current cloud GPU offers from the pricing service. Use for questions about rental prices. Returns the provider, GPU, hourly price, and region.',
  parameters: {
    type: 'object',
    properties: {
      provider: { type: 'string', description: 'Optional provider filter, for example aws or azure.' },
      gpu: { type: 'string', description: 'Optional GPU name filter, for example H100.' },
      limit: { type: 'integer', minimum: 1, maximum: 25 },
    },
    additionalProperties: false,
  },
  async execute(args, context) {
    if (!context.costingsUrl) {
      return unavailable('Pricing unavailable', 'The pricing service is not configured.');
    }
    const raw = await context.fetchJson(`${context.costingsUrl}/offers`);
    const offers = asArray(asRecord(raw)?.offers)
      .map(asRecord)
      .filter((offer): offer is Record<string, unknown> => offer !== null);
    if (offers.length === 0) return unavailable('Pricing unavailable', 'The pricing service returned no offers.');

    const provider = str(args.provider)?.toLowerCase();
    const gpu = str(args.gpu)?.toLowerCase();
    const limit = typeof args.limit === 'number' ? Math.min(Math.max(args.limit, 1), 25) : 10;

    const matched = offers
      .map((offer) => ({
        provider: (str(offer.provider) ?? '').toLowerCase(),
        gpu: str(offer.gpu) ?? str(offer.accelerator) ?? str(offer.system) ?? '',
        price: num(offer.price_per_hour) ?? num(offer.hourly_price) ?? num(offer.price),
        currency: str(offer.currency) ?? 'USD',
        region: str(offer.region) ?? null,
      }))
      .filter((offer) => offer.price !== null)
      .filter((offer) => (provider ? offer.provider === provider : true))
      .filter((offer) => (gpu ? offer.gpu.toLowerCase().includes(gpu) : true))
      .slice(0, limit);

    if (matched.length === 0) {
      return {
        ok: true,
        summary: 'No offer matched the filter',
        evidence: ['No pricing offer matched that filter. Prices are unavailable rather than zero.'],
        data: [],
        link: { label: 'Open Sources', href: '/sources' },
      };
    }

    return {
      ok: true,
      summary: `Read ${matched.length} pricing offer${matched.length === 1 ? '' : 's'}`,
      evidence: matched.map(
        (offer) =>
          `${offer.provider || 'unknown provider'} offers ${offer.gpu || 'a GPU'} at ${offer.price} ${offer.currency} per hour${
            offer.region ? ` in ${offer.region}` : ''
          }.`,
      ),
      data: matched,
      link: { label: 'Open Sources', href: '/sources' },
    };
  },
};

const memoryEstimate: ToolDefinition = {
  name: 'memory',
  description:
    'Estimate memory for one model on one GPU system: weights, KV cache, and tokens of KV capacity. Read-only arithmetic from the simulator; it is not a sizing recommendation.',
  parameters: {
    type: 'object',
    properties: {
      model: { type: 'string', description: 'Hugging Face model id, for example meta-llama/Llama-3.1-70B-Instruct.' },
      system: { type: 'string', description: 'GPU system id from list_gpus.' },
      backend: { type: 'string', description: 'Inference backend id, defaults to vllm.' },
      max_num_tokens: { type: 'integer', minimum: 1 },
      max_batch_size: { type: 'integer', minimum: 1 },
    },
    required: ['model', 'system'],
    additionalProperties: false,
  },
  async execute(args, context) {
    if (!context.gatewayUrl) return unavailable('Memory service unavailable', 'AISimulators is not configured.');
    if (!context.fetchJsonPost) {
      return unavailable('Memory service unavailable', 'The memory endpoint is not available in this deployment.');
    }
    const model = str(args.model);
    const system = str(args.system);
    if (!model || !system) {
      return unavailable('Memory query incomplete', 'A model and a GPU system are required.');
    }

    const body = {
      model_path: model,
      system,
      // `backend` is a service default; forwarding it here would spread a copy
      // of that default into the browser-side contract.
      ...(str(args.backend) ? { backend: str(args.backend) } : {}),
      ...(typeof args.max_num_tokens === 'number' ? { max_num_tokens: args.max_num_tokens } : {}),
      ...(typeof args.max_batch_size === 'number' ? { max_batch_size: args.max_batch_size } : {}),
      ...(context.simulator ? { simulator: context.simulator } : {}),
    };

    const raw = asRecord(await context.fetchJsonPost(`${context.gatewayUrl}/memory`, body));
    if (!raw) return unavailable('Memory service unavailable', 'The memory endpoint did not answer.');

    // `/memory` is an AISimulate capability. If a different simulator is named
    // in the result, the numbers did not come from what the user selected, so
    // report that instead of presenting them as the requested simulator's.
    const reported = str(raw.simulator);
    if (context.simulator && reported && reported !== context.simulator) {
      return unavailable(
        'Memory estimate unavailable for the selected simulator',
        `The memory service answered as ${reported}, not ${context.simulator}. Sizing is not substituted across simulators.`,
      );
    }

    const totalCapacity = num(raw.total_gpu_capacity_bytes);
    const kvBytes = num(raw.total_kv_size_bytes);
    const perToken = num(raw.kv_size_per_token_bytes);
    const kvTokens = num(raw.total_kv_size_tokens);
    const breakdown = asRecord(raw.memory_breakdown);

    const evidence: string[] = [];
    if (reported) evidence.push(`This memory estimate was produced by ${reported}.`);
    if (totalCapacity !== null) evidence.push(`Total GPU capacity is ${describeMemoryGb(totalCapacity) ?? totalCapacity}.`);
    if (kvBytes !== null) evidence.push(`KV cache size is ${describeMemoryGb(kvBytes) ?? kvBytes}.`);
    if (perToken !== null) evidence.push(`KV cache uses ${perToken} bytes per token.`);
    if (kvTokens !== null) evidence.push(`The KV cache holds ${kvTokens} tokens.`);
    if (breakdown) {
      const weights = num(breakdown.weights_bytes) ?? num(breakdown.model_weights_bytes);
      if (weights !== null) evidence.push(`Model weights need ${describeMemoryGb(weights) ?? weights}.`);
    }

    if (evidence.length === 0) {
      return unavailable('Memory result had no readable fields', 'The service answered without memory sizes.');
    }

    return {
      ok: true,
      summary: `Read memory estimate for ${model} on ${system}`,
      evidence,
      data: raw,
      link: {
        label: `Open KV cache calculator for ${model}`,
        // The page's own contract, so the link always parses and carries the
        // version marker the recipient's page expects.
        href: kvCacheLink.build({ model, system }),
      },
    };
  },
};

export const ASSISTANT_TOOLS: ToolDefinition[] = [
  listGpus,
  listModels,
  listBackends,
  listNetworks,
  pricingOffers,
  memoryEstimate,
];

export function toolByName(name: string): ToolDefinition | undefined {
  return ASSISTANT_TOOLS.find((tool) => tool.name === name);
}

/** OpenAI function-calling descriptors; the model never sees the executors. */
export function toolSchemas() {
  return ASSISTANT_TOOLS.map((tool) => ({
    type: 'function' as const,
    function: {
      name: tool.name,
      description: tool.description,
      parameters: tool.parameters,
    },
  }));
}
