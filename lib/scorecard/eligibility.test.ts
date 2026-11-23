import { describe, expect, it } from 'vitest';
import {
  backendVersionsFor,
  eligibility,
  namesMatch,
  systemForAccelerator,
  type MeasuredRun,
  type ScorecardSimulatorSupport,
} from './eligibility';

function run(overrides: Partial<MeasuredRun> = {}): MeasuredRun {
  return {
    model: 'org/model',
    accelerator: 'H200',
    isl: 1024,
    osl: 128,
    tp: 4,
    backend: 'vllm',
    release: '0.31.0',
    servingMode: 'agg',
    topology: {
      prefillTp: 4,
      decodeTp: 4,
      prefillWorkers: 1,
      decodeWorkers: 1,
      known: false,
    },
    ...overrides,
  };
}

function support(overrides: Partial<ScorecardSimulatorSupport> = {}): ScorecardSimulatorSupport {
  return {
    id: 'sim',
    label: 'Simulator',
    available: true,
    version: '1.0.0',
    revision: null,
    models: ['org/model'],
    modelsExhaustive: true,
    systems: [
      { id: 'h200_sxm', label: 'NVIDIA H200' },
      { id: 'gb200', label: 'NVIDIA GB200' },
    ],
    backends: [{
      id: 'vllm',
      versions: ['0.29.0', '0.31.0'],
      defaultVersion: '0.31.0',
      systems: { h200_sxm: ['0.31.0'] },
      versionSupport: { '0.31.0': { disaggregated: true } },
      disaggregationRequiresFabric: false,
    }],
    networks: ['roce-200g'],
    defaultNetwork: 'roce-200g',
    ...overrides,
  };
}

describe('namesMatch', () => {
  it('matches a display prefix without matching a longer model number', () => {
    expect(namesMatch('NVIDIA H200', 'H200')).toBe(true);
    expect(namesMatch('H200', 'NVIDIA H200')).toBe(true);
    expect(namesMatch('B200', 'GB200')).toBe(false);
    expect(namesMatch('MI300X', 'MI300')).toBe(false);
  });

  it('rejects empty labels', () => {
    expect(namesMatch('', 'H200')).toBe(false);
    expect(namesMatch('H200', '   ')).toBe(false);
  });
});

describe('systemForAccelerator', () => {
  it('prefers an exact label, then a token-bounded match', () => {
    expect(systemForAccelerator([{ id: 'h200_sxm', label: 'NVIDIA H200' }], 'H200')?.id).toBe('h200_sxm');
    expect(systemForAccelerator([{ id: 'other', label: 'NVIDIA GB200' }], 'B200')).toBeNull();
  });

  it('returns null for an empty label', () => {
    expect(systemForAccelerator([{ id: 'h200_sxm', label: 'NVIDIA H200' }], '  ')).toBeNull();
  });
});

describe('backendVersionsFor', () => {
  it('uses a system-scoped list when the catalogue publishes one', () => {
    const backend = support().backends[0];
    expect(backendVersionsFor(backend, 'h200_sxm')).toEqual(['0.31.0']);
  });

  it('falls back to the shared list when the system is not scoped', () => {
    const backend = support().backends[0];
    expect(backendVersionsFor(backend, 'gb200')).toEqual(['0.29.0', '0.31.0']);
    expect(backendVersionsFor(backend, null)).toEqual(['0.29.0', '0.31.0']);
  });
});

describe('eligibility', () => {
  it('marks a missing or unavailable simulator ineligible, not unsupported', () => {
    expect(eligibility(undefined, run()).status).toBe('ineligible');
    expect(eligibility(support({ available: false }), run()).status).toBe('ineligible');
  });

  it('marks a GPU outside the catalogue ineligible', () => {
    const decision = eligibility(support(), run({ accelerator: 'MI355X' }));
    expect(decision.status).toBe('ineligible');
    if (decision.status === 'ineligible') expect(decision.reason).toContain('MI355X');
  });

  it('marks a model outside an exhaustive catalogue ineligible', () => {
    expect(eligibility(support(), run({ model: 'org/unlisted' })).status).toBe('ineligible');
  });

  it('allows an unlisted model when the catalogue is not exhaustive', () => {
    const decision = eligibility(support({ modelsExhaustive: false, models: [] }), run({ model: 'org/unlisted' }));
    expect(decision.status).toBe('eligible');
  });

  it('marks a backend outside the catalogue ineligible', () => {
    expect(eligibility(support(), run({ backend: 'sglang' })).status).toBe('ineligible');
  });

  it('sends the measured release when the catalogue lists it', () => {
    const decision = eligibility(support(), run({ release: '0.31.0' }));
    expect(decision.status).toBe('eligible');
    if (decision.status !== 'eligible') return;
    expect(decision.plan.version).toBe('0.31.0');
    expect(decision.plan.releaseMatch).toBe(true);
    expect(decision.plan.systemId).toBe('h200_sxm');
  });

  it('omits the release rather than echoing an unlisted one', () => {
    // The measured run names a release this catalogue does not list for this
    // GPU. The request must leave the field out and let the service resolve its
    // own default; the point then stays out of the matched slice.
    const decision = eligibility(support(), run({ release: '0.28.0' }));
    expect(decision.status).toBe('eligible');
    if (decision.status !== 'eligible') return;
    expect(decision.plan.version).toBeNull();
    expect(decision.plan.releaseMatch).toBe(false);
  });

  it('does not match a row that names no release', () => {
    const decision = eligibility(support(), run({ release: null }));
    expect(decision.status).toBe('eligible');
    if (decision.status !== 'eligible') return;
    expect(decision.plan.version).toBeNull();
    expect(decision.plan.releaseMatch).toBe(false);
  });

  it('uses the catalogue fabric only for disaggregated serving', () => {
    const aggregated = eligibility(support(), run({ servingMode: 'agg' }));
    const disaggregated = eligibility(support(), run({ servingMode: 'disagg' }));
    expect(aggregated.status === 'eligible' && aggregated.plan.fabric).toBeNull();
    expect(disaggregated.status === 'eligible' && disaggregated.plan.fabric).toBe('roce-200g');
  });

  it('reports a release that cannot model disaggregation as unsupported', () => {
    const decision = eligibility(
      support({ backends: [{ ...support().backends[0], versionSupport: { '0.31.0': { disaggregated: false } } }] }),
      run({ servingMode: 'disagg' }),
    );
    expect(decision.status).toBe('unsupported');
  });

  it('reports a missing fabric for a backend that requires one as unsupported', () => {
    const decision = eligibility(
      support({
        defaultNetwork: null,
        backends: [{ ...support().backends[0], disaggregationRequiresFabric: true }],
      }),
      run({ servingMode: 'disagg' }),
    );
    expect(decision.status).toBe('unsupported');
  });

  it('reports a fabric the catalogue does not list as unsupported', () => {
    const decision = eligibility(
      support({
        networks: [],
        backends: [{ ...support().backends[0], disaggregationRequiresFabric: true }],
      }),
      run({ servingMode: 'disagg' }),
    );
    expect(decision.status).toBe('unsupported');
  });

  it('does not reject a version whose capabilities the catalogue omits', () => {
    const decision = eligibility(
      support({ backends: [{ ...support().backends[0], versionSupport: {} }] }),
      run({ servingMode: 'disagg' }),
    );
    expect(decision.status).toBe('eligible');
  });
});
