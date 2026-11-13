import { describe, expect, it } from 'vitest';
import { toolByName, type ToolContext } from './tools';
import { kvCacheLink } from '@/lib/deep-links/pages';

function memoryContext(options: {
  simulator?: string;
  response: Record<string, unknown>;
  onBody?: (body: unknown) => void;
}): ToolContext {
  return {
    gatewayUrl: 'http://gateway.test',
    costingsUrl: null,
    simulator: options.simulator,
    fetchJson: async () => null,
    fetchJsonPost: async (_url, body) => {
      options.onBody?.(body);
      return options.response;
    },
  };
}

const memoryResult = {
  simulator: 'aisimulate',
  total_gpu_capacity_bytes: 141_000_000_000,
  total_kv_size_bytes: 20_000_000_000,
  kv_size_per_token_bytes: 320_000,
  total_kv_size_tokens: 62_500,
  memory_breakdown: { weights_bytes: 70_000_000_000 },
};

async function runMemory(context: ToolContext, args: Record<string, unknown> = {}) {
  const tool = toolByName('memory');
  if (!tool) throw new Error('memory tool is registered');
  return tool.execute({ model: 'Qwen/Qwen3-32B', system: 'h200_sxm', ...args }, context);
}

describe('memory tool', () => {
  it('lets the service resolve the backend when the caller did not choose one', async () => {
    let body: unknown;
    await runMemory(memoryContext({ response: memoryResult, onBody: (value) => { body = value; } }));

    expect(body).not.toHaveProperty('backend');
  });

  it('forwards an explicitly chosen backend', async () => {
    let body: unknown;
    await runMemory(
      memoryContext({ response: memoryResult, onBody: (value) => { body = value; } }),
      { backend: 'sglang' },
    );

    expect(body).toMatchObject({ backend: 'sglang' });
  });

  it('names the simulator that produced the numbers in its evidence', async () => {
    const result = await runMemory(memoryContext({ response: memoryResult, simulator: 'aisimulate' }));
    expect(result.ok).toBe(true);
    expect(result.evidence.join(' ')).toContain('produced by aisimulate');
  });

  it('refuses a result from a different simulator instead of substituting it', async () => {
    const result = await runMemory(memoryContext({ response: memoryResult, simulator: 'blis' }));

    expect(result.ok).toBe(false);
    expect(result.reason).toMatch(/not blis/i);
    expect(result.evidence).toEqual([]);
  });

  it('links to the KV cache calculator through that page\'s own contract', async () => {
    const result = await runMemory(memoryContext({ response: memoryResult, simulator: 'aisimulate' }));
    expect(result.link).toBeDefined();

    const href = result.link?.href ?? '';
    const parsed = kvCacheLink.parse(href.slice(href.indexOf('?')));
    expect(parsed.warnings).toEqual([]);
    expect(parsed.values).toEqual({ model: 'Qwen/Qwen3-32B', system: 'h200_sxm' });
  });
});

describe('tool registry', () => {
  it('exposes read-only tools only', () => {
    for (const name of ['list_gpus', 'list_models', 'list_backends', 'list_networks', 'pricing_offers', 'memory']) {
      expect(toolByName(name)).toBeDefined();
    }
    expect(toolByName('recommend')).toBeUndefined();
    expect(toolByName('predict')).toBeUndefined();
  });
});
