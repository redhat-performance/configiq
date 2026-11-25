import { describe, expect, it } from 'vitest';
import { combinedSimulators } from './catalog-server';

describe('combinedSimulators', () => {
  const capabilities = {
    backends: [{ id: 'vllm', versions: ['1.0.0'] }],
    models: ['org/model'],
    systems: [{ id: 'h200', name: 'H200' }],
    networks: ['roce'],
    default_network: 'roce',
  };

  it('keeps an explicitly unavailable simulator unavailable', () => {
    // A combined catalogue can keep the prior capabilities in the legacy
    // support map while marking the entry itself unavailable. The explicit
    // flag wins: reading the map would report a retired simulator as serving.
    const entries = combinedSimulators({
      simulators: [
        { id: 'serving', label: 'Serving', ...capabilities },
        { id: 'retired', label: 'Retired', available: false },
      ],
      simulatorSupport: { retired: capabilities },
    });

    const retired = entries.find(entry => entry.id === 'retired');
    expect(retired).toBeDefined();
    expect(retired!.available).toBe(false);
    expect(retired!.models).toEqual([]);
    expect(retired!.backends).toEqual([]);
  });

  it('still falls back to the support map for an identity-only entry', () => {
    const entries = combinedSimulators({
      simulators: [{ id: 'listed', label: 'Listed' }],
      simulatorSupport: { listed: capabilities },
    });

    const listed = entries.find(entry => entry.id === 'listed');
    expect(listed!.available).toBe(true);
    expect(listed!.models).toEqual(['org/model']);
    expect(listed!.defaultNetwork).toBe('roce');
  });
});
