import { describe, expect, it } from 'vitest'
import { hasDefaultMemoryFractionForVersion, mapBackendOption, mapCatalogResponse, systemMemoryFraction } from './useCatalog'

describe('mapBackendOption', () => {
  it('maps backend memory defaults from the REST catalog', () => {
    expect(mapBackendOption({
      id: 'vllm',
      versions: ['0.10.0'],
      memory_fraction: 0.92,
      memory_fraction_kind: 'of_free',
    })).toMatchObject({
      id: 'vllm',
      memoryFraction: 0.92,
      memoryFractionKind: 'of_free',
    })
  })

  it('uses safe defaults when optional memory metadata is absent or invalid', () => {
    expect(mapBackendOption({ id: 'vllm', versions: [], memory_fraction_kind: 'invalid' })).toMatchObject({
      memoryFraction: null,
      memoryFractionKind: 'of_total',
      disaggregationRequiresFabric: false,
      capacityOptions: { disaggregated: false },
    })
  })

  it('reports whether capacity runs can include disaggregated options', () => {
    expect(mapBackendOption({
      id: 'vllm',
      versions: ['0.31.0'],
      capacity_options: { disaggregated: true },
    })?.capacityOptions.disaggregated).toBe(true)
    expect(mapBackendOption({
      id: 'vllm',
      versions: ['0.31.0'],
      capacity_options: { disaggregated: 'yes' },
    })?.capacityOptions.disaggregated).toBe(false)
  })

  it('reports whether a P/D topology needs a fabric', () => {
    expect(mapBackendOption({
      id: 'vllm',
      versions: ['0.31.0'],
      disaggregation_requires_fabric: true,
    })?.disaggregationRequiresFabric).toBe(true)
  })

  it('ignores malformed backend entries', () => {
    expect(mapBackendOption({ id: 'vllm' })).toBeNull()
  })

  it('maps per-system defaults so a system-specific fraction is available', () => {
    const backend = mapBackendOption({
      id: 'vllm',
      versions: ['0.14.0', '0.31.0'],
      default_version: 'current',
      memory_fraction: 0.92,
      system_defaults: {
        a100_sxm: { version: '0.14.0', memory_fraction: 0.9 },
        h200_sxm: { version: '0.24.0', memory_fraction: 0.92 },
        broken: { version: 42 },
      },
    })

    expect(backend?.defaultVersion).toBe('current')
    expect(backend?.systemDefaults).toEqual({
      a100_sxm: { version: '0.14.0', memoryFraction: 0.9 },
      h200_sxm: { version: '0.24.0', memoryFraction: 0.92 },
      broken: { version: null, memoryFraction: null },
    })
  })

  it('defaults systemDefaults to empty when the API omits it', () => {
    expect(mapBackendOption({ id: 'vllm', versions: ['0.10.0'] })?.systemDefaults).toEqual({})
  })
})

describe('systemMemoryFraction', () => {
  const backend = mapBackendOption({
    id: 'vllm',
    versions: ['0.14.0', '0.24.0'],
    memory_fraction: 0.92,
    system_defaults: {
      a100_sxm: { version: '0.14.0', memory_fraction: 0.9 },
      h200_sxm: { version: '0.24.0', memory_fraction: null },
    },
  })

  it('prefers the selected system default over the fleet figure', () => {
    expect(systemMemoryFraction(backend ?? undefined, 'a100_sxm')).toBe(0.9)
  })

  it('falls back to the fleet figure when the system has no usable default', () => {
    expect(systemMemoryFraction(backend ?? undefined, 'h200_sxm')).toBe(0.92)
    expect(systemMemoryFraction(backend ?? undefined, 'b60')).toBe(0.92)
  })

  it('returns null when the backend is unknown', () => {
    expect(systemMemoryFraction(undefined, 'a100_sxm')).toBeNull()
  })
})

describe('hasDefaultMemoryFractionForVersion', () => {
  const backend = mapBackendOption({
    id: 'vllm',
    versions: ['0.14.0', '0.24.0'],
    default_version: 'current',
    system_defaults: {
      a100_sxm: { version: '0.14.0', memory_fraction: 0.9 },
    },
  })

  it('accepts an omitted pin, current alias, or matching system version', () => {
    expect(hasDefaultMemoryFractionForVersion(backend ?? undefined, 'a100_sxm', '')).toBe(true)
    expect(hasDefaultMemoryFractionForVersion(backend ?? undefined, 'a100_sxm', 'current')).toBe(true)
    expect(hasDefaultMemoryFractionForVersion(backend ?? undefined, 'a100_sxm', '0.14.0')).toBe(true)
  })

  it('requires an explicit fraction for another version or missing per-system data', () => {
    expect(hasDefaultMemoryFractionForVersion(backend ?? undefined, 'a100_sxm', '0.24.0')).toBe(false)
    expect(hasDefaultMemoryFractionForVersion(backend ?? undefined, 'h100_sxm', '0.14.0')).toBe(false)
    expect(hasDefaultMemoryFractionForVersion(undefined, 'a100_sxm', '0.14.0')).toBe(false)
  })
})

describe('mapCatalogResponse', () => {
  const systems = [{ id: 'h200_sxm', name: 'H200 SXM', memory_bytes: 140_000_000_000 }]
  const models = [{ id: 'Qwen/Qwen3-8B', context_length: 32_768 }]
  const backends = [{ id: 'vllm', versions: ['0.31.0'], memory_fraction: 0.9 }]

  it('derives the page catalogue from the default simulator entry', () => {
    const mapped = mapCatalogResponse({
      // Stale duplicates must not override the selected entry.
      systems: [{ id: 'stale_system' }],
      models: ['stale/model'],
      backends: [],
      networks: ['stale-fabric'],
      objectives: ['stale_objective'],
      timeoutSeconds: 45,
      simulators: [
        {
          id: 'blis',
          label: 'BLIS',
          description: 'Discrete-event simulation.',
          version: 'test-version',
          revision: 'deadbee',
          default: true,
          available: true,
          recommendSearch: 'single',
          modelsExhaustive: true,
          systems,
          models,
          backends,
          networks: ['roce-200g'],
          defaultNetwork: 'roce-200g',
          objectives: ['min_gpus', 'throughput_demand'],
        },
      ],
    })

    expect(mapped.gpuOptions.map(option => option.systemId)).toEqual(['h200_sxm'])
    expect(mapped.modelOptions).toEqual(['Qwen/Qwen3-8B'])
    expect(mapped.modelSpecs.get('Qwen/Qwen3-8B')?.context_length).toBe(32_768)
    expect(mapped.backendOptions.map(backend => backend.id)).toEqual(['vllm'])
    expect(mapped.networkOptions).toEqual(['roce-200g'])
    expect(mapped.objectives).toEqual(['min_gpus', 'throughput_demand'])
    expect(mapped.timeoutSeconds).toBe(45)
    expect(mapped.simulators).toEqual([{
      id: 'blis',
      label: 'BLIS',
      description: 'Discrete-event simulation.',
      version: 'test-version',
      revision: 'deadbee',
      default: true,
    }])
  })

  it('maps the system specs the GPU picker groups and describes', () => {
    const mapped = mapCatalogResponse({
      simulators: [{
        id: 'aisimulate',
        label: 'AISimulate',
        default: true,
        available: true,
        systems: [{
          id: 'h200_sxm',
          name: 'NVIDIA H200 SXM',
          vendor: 'nvidia',
          architecture: 'hopper',
          memory_bytes: 151_397_597_184,
          memory_bandwidth_bytes: 4_800_000_000_000,
          bf16_tflops: 989,
          tdp_watts: 700,
          gpus_per_node: 8,
        }],
        models,
        backends,
        networks: [],
        objectives: ['pareto'],
      }],
    })

    expect(mapped.gpuOptions[0]).toEqual({
      systemId: 'h200_sxm',
      label: 'NVIDIA H200 SXM',
      vendor: 'nvidia',
      architecture: 'hopper',
      vramGb: 141,
      bandwidthTbps: 4.8,
      tflopsBf16: 989,
      tdpWatts: 700,
      gpusPerNode: 8,
    })
  })

  it('selects the entry flagged default, not merely the first entry', () => {
    const defaultEntry = {
      id: 'aisimulate',
      label: 'AISimulate',
      default: true,
      available: true,
      systems,
      models,
      backends,
      networks: [],
      objectives: ['pareto'],
    }
    const otherEntry = {
      id: 'blis',
      label: 'BLIS',
      default: false,
      available: true,
      systems: [{ id: 'l40s' }],
      models: [{ id: 'other/model' }],
      backends: [{ id: 'sglang', versions: ['0.5.0'] }],
      networks: ['roce-200g'],
      objectives: ['min_gpus'],
    }

    const mapped = mapCatalogResponse({ simulators: [otherEntry, defaultEntry] })

    expect(mapped.gpuOptions.map(option => option.systemId)).toEqual(['h200_sxm'])
    expect(mapped.backendOptions.map(backend => backend.id)).toEqual(['vllm'])
    expect(mapped.objectives).toEqual(['pareto'])
  })

  it('falls back to the legacy top-level catalogue when no simulator entry exists', () => {
    const mapped = mapCatalogResponse({
      systems,
      models: ['Qwen/Qwen3-8B'],
      backends,
      networks: ['ib-400g'],
      objectives: ['pareto'],
      simulators: [],
    })

    expect(mapped.gpuOptions.map(option => option.systemId)).toEqual(['h200_sxm'])
    expect(mapped.modelOptions).toEqual(['Qwen/Qwen3-8B'])
    expect(mapped.backendOptions.map(backend => backend.id)).toEqual(['vllm'])
    expect(mapped.networkOptions).toEqual(['ib-400g'])
    expect(mapped.objectives).toEqual(['pareto'])
    expect(mapped.simulators).toEqual([])
    expect(mapped.timeoutSeconds).toBeNull()
  })

  it('falls back to the top-level catalogue when the default entry is unavailable', () => {
    const mapped = mapCatalogResponse({
      systems,
      models: ['Qwen/Qwen3-8B'],
      backends,
      networks: ['ib-400g'],
      objectives: ['pareto'],
      simulators: [{
        id: 'blis',
        label: 'BLIS',
        default: true,
        available: false,
        systems: [],
        models: [],
        backends: [],
        networks: [],
        objectives: [],
      }],
    })

    expect(mapped.gpuOptions.map(option => option.systemId)).toEqual(['h200_sxm'])
    expect(mapped.objectives).toEqual(['pareto'])
    // The identity is kept so Settings can still explain why it is unavailable,
    // and the support entry stays present but empty so pages do not silently
    // present the fallback catalogue as this simulator's own.
    expect(mapped.simulators[0]).toMatchObject({ id: 'blis', default: true })
    expect(mapped.simulatorSupport.blis).toMatchObject({
      id: 'blis',
      available: false,
      models: [],
      systems: [],
      backends: [],
      objectives: [],
    })
  })

  it('keeps an unavailable non-default simulator out of the default catalogue', () => {
    const mapped = mapCatalogResponse({
      simulators: [
        {
          id: 'aisimulate',
          label: 'AISimulate',
          default: true,
          available: true,
          systems,
          models,
          backends,
          networks: [],
          objectives: ['pareto'],
        },
        {
          id: 'blis',
          label: 'BLIS',
          default: false,
          available: false,
          systems: [],
          models: [],
          backends: [],
          networks: [],
          objectives: [],
        },
      ],
    })

    // The default entry still drives the page catalogue...
    expect(mapped.gpuOptions.map(option => option.systemId)).toEqual(['h200_sxm'])
    expect(mapped.modelOptions).toEqual(['Qwen/Qwen3-8B'])
    // ...but BLIS keeps an empty, explicitly unavailable support entry rather
    // than inheriting AISimulate's models, systems, or goals.
    expect(mapped.simulatorSupport.blis).toMatchObject({
      id: 'blis',
      available: false,
      models: [],
      systems: [],
      objectives: [],
    })
  })

  it('rejects a catalogue that cannot render a page', () => {
    expect(() => mapCatalogResponse({ simulators: [] })).toThrow('empty catalog')
  })
})
