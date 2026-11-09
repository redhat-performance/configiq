'use client';

import * as React from 'react';
import { loadAppConfig } from '@/lib/app-config';

export type InferenceBackend = 'vllm' | 'tensorrt-llm' | 'sglang';
export type Simulator = 'aisimulate' | 'blis';
export const DEFAULT_SIMULATOR: Simulator = 'aisimulate';
const SIMULATOR_BACKENDS: Record<Simulator, readonly InferenceBackend[]> = {
  aisimulate: ['vllm', 'tensorrt-llm', 'sglang'],
  blis: ['vllm'],
}

export function isSimulatorBackendCompatible(simulator: Simulator, backend: InferenceBackend): boolean {
  return SIMULATOR_BACKENDS[simulator].includes(backend)
}
// A hosted-model pricing feed. 'merged' is the one constant the API always
// exposes; the individual feeds (openrouter, litellm, …) are discovered at
// runtime from /health, so this is intentionally an open string rather than a
// closed union — new feeds appear (and retired ones disappear) with no
// frontend change.
export type PricingSource = string;
export const DEFAULT_PRICING_SOURCE = 'merged';

const STORAGE_KEYS = {
  defaultModel: 'settings_default_model',
  hfToken: 'hf_token',
  inferenceBackend: 'settings_inference_backend',
  simulator: 'settings_simulator',
  // v2: the old key held the fleet-wide newest version, which is not queryable
  // on every system. Dropping it clears those stale literal pins.
  backendVersion: 'settings_backend_version_v2',
  costingsEnabled: 'settings_costings_enabled',
  preferredCloudProvider: 'settings_preferred_cloud_provider',
  pricingSource: 'settings_pricing_source',
} as const;

/**
 * A version pin means something different in each simulator (an AISimulate
 * namespace vs. a literal engine release), so it is stored per simulator.
 * On load, the default simulator migrates the legacy unsuffixed key to its
 * scoped key.
 */
function versionStorageKey(simulator: Simulator): string {
  return `${STORAGE_KEYS.backendVersion}:${simulator}`;
}

interface SettingsState {
  hydrated: boolean;
  defaultModel: string;
  testedModels: string[];
  hfToken: string;
  inferenceBackend: InferenceBackend;
  simulator: Simulator;
  backendVersion: string;
  costingsEnabled: boolean;
  preferredCloudProvider: string | null;
  pricingSource: PricingSource;
  setDefaultModel: (v: string) => void;
  setHfToken: (v: string) => void;
  setInferenceBackend: (v: InferenceBackend) => void;
  setSimulator: (v: Simulator) => void;
  setBackendVersion: (v: string) => void;
  setCostingsEnabled: (v: boolean) => void;
  setPreferredCloudProvider: (v: string | null) => void;
  setPricingSource: (v: PricingSource) => void;
}

const SettingsContext = React.createContext<SettingsState>({
  hydrated: false,
  defaultModel: '',
  testedModels: [],
  hfToken: '',
  inferenceBackend: 'vllm',
  simulator: DEFAULT_SIMULATOR,
  backendVersion: '',
  costingsEnabled: false,
  preferredCloudProvider: null,
  pricingSource: 'merged',
  setDefaultModel: () => {},
  setHfToken: () => {},
  setInferenceBackend: () => {},
  setSimulator: () => {},
  setBackendVersion: () => {},
  setCostingsEnabled: () => {},
  setPreferredCloudProvider: () => {},
  setPricingSource: () => {},
});

export function SettingsProvider({ children }: { children: React.ReactNode }) {
  const [hydrated, setHydrated] = React.useState(false);
  const [defaultModel, setDefaultModelState] = React.useState('');
  const [testedModels, setTestedModels] = React.useState<string[]>([]);
  const [hfToken, setHfTokenState] = React.useState('');
  const [inferenceBackend, setInferenceBackendState] = React.useState<InferenceBackend>('vllm');
  const [simulator, setSimulatorState] = React.useState<Simulator>(DEFAULT_SIMULATOR);
  const [backendVersion, setBackendVersionState] = React.useState('');
  const [costingsEnabled, setCostingsEnabledState] = React.useState(false);
  const [preferredCloudProvider, setPreferredCloudProviderState] = React.useState<string | null>(null);
  const [pricingSource, setPricingSourceState] = React.useState<PricingSource>(DEFAULT_PRICING_SOURCE);

  React.useEffect(() => {
    async function init() {
      // Load config.json first — gives us the defaults to fall back to
      const config = await loadAppConfig();

      // localStorage overrides config; null means "never saved by user" so use config default
      const savedModel = localStorage.getItem(STORAGE_KEYS.defaultModel);
      setDefaultModelState(savedModel !== null ? savedModel : config.defaultModel);
      setTestedModels(config.testedModels);

      setHfTokenState(localStorage.getItem(STORAGE_KEYS.hfToken) ?? '');

      const savedBackend = localStorage.getItem(STORAGE_KEYS.inferenceBackend) as InferenceBackend | null;
      const activeBackend: InferenceBackend = savedBackend ?? config.defaultBackend as InferenceBackend;
      setInferenceBackendState(activeBackend);

      const savedSimulator = localStorage.getItem(STORAGE_KEYS.simulator);
      const requestedSimulator = savedSimulator === 'blis' ? 'blis' : DEFAULT_SIMULATOR;
      const activeSimulator = isSimulatorBackendCompatible(requestedSimulator, activeBackend)
        ? requestedSimulator
        : DEFAULT_SIMULATOR;
      setSimulatorState(activeSimulator);
      if (savedSimulator && savedSimulator !== activeSimulator) {
        localStorage.setItem(STORAGE_KEYS.simulator, activeSimulator);
      }

      // The legacy unsuffixed key only ever held the default simulator's pin.
      const legacyVersion = activeSimulator === DEFAULT_SIMULATOR
        ? localStorage.getItem(STORAGE_KEYS.backendVersion)
        : null;
      const savedVersion = localStorage.getItem(versionStorageKey(activeSimulator)) ?? legacyVersion;
      setBackendVersionState(savedVersion ?? '');
      if (legacyVersion !== null) {
        localStorage.setItem(versionStorageKey(activeSimulator), legacyVersion);
        localStorage.removeItem(STORAGE_KEYS.backendVersion);
      }

      const savedCostings = localStorage.getItem(STORAGE_KEYS.costingsEnabled);
      setCostingsEnabledState(savedCostings === 'true');

      setPreferredCloudProviderState(localStorage.getItem(STORAGE_KEYS.preferredCloudProvider));

      // Any non-empty string is accepted; feeds are validated at fetch time by
      // the API (which 400s on an unknown source) and the Sources selector
      // resets to merged if the persisted feed is no longer offered.
      const savedPricingSource = localStorage.getItem(STORAGE_KEYS.pricingSource);
      if (savedPricingSource) {
        setPricingSourceState(savedPricingSource);
      }

      setHydrated(true);
    }
    init();
  }, []);

  const setDefaultModel = React.useCallback((v: string) => {
    setDefaultModelState(v);
    localStorage.setItem(STORAGE_KEYS.defaultModel, v);
  }, []);

  const setHfToken = React.useCallback((v: string) => {
    setHfTokenState(v);
    if (v) {
      localStorage.setItem(STORAGE_KEYS.hfToken, v);
    } else {
      localStorage.removeItem(STORAGE_KEYS.hfToken);
    }
  }, []);

  const setInferenceBackend = React.useCallback((v: InferenceBackend) => {
    if (!isSimulatorBackendCompatible(simulator, v)) return;
    setInferenceBackendState(v);
    localStorage.setItem(STORAGE_KEYS.inferenceBackend, v);
    // Settings resolves the backend's default version from the live catalog.
    setBackendVersionState('');
    localStorage.setItem(versionStorageKey(simulator), '');
  }, [simulator]);

  const setSimulator = React.useCallback((v: Simulator) => {
    if (!isSimulatorBackendCompatible(v, inferenceBackend)) return;
    setSimulatorState(v);
    localStorage.setItem(STORAGE_KEYS.simulator, v);
    // Load the version saved for the simulator being switched to. The two
    // simulators' version namespaces are unrelated, so never carry one over.
    setBackendVersionState(localStorage.getItem(versionStorageKey(v)) ?? '');
  }, [inferenceBackend]);

  const setBackendVersion = React.useCallback((v: string) => {
    setBackendVersionState(v);
    localStorage.setItem(versionStorageKey(simulator), v);
  }, [simulator]);

  const setCostingsEnabled = React.useCallback((v: boolean) => {
    setCostingsEnabledState(v);
    localStorage.setItem(STORAGE_KEYS.costingsEnabled, String(v));
  }, []);

  const setPreferredCloudProvider = React.useCallback((v: string | null) => {
    setPreferredCloudProviderState(v);
    if (v) {
      localStorage.setItem(STORAGE_KEYS.preferredCloudProvider, v);
    } else {
      localStorage.removeItem(STORAGE_KEYS.preferredCloudProvider);
    }
  }, []);

  const setPricingSource = React.useCallback((v: PricingSource) => {
    setPricingSourceState(v);
    localStorage.setItem(STORAGE_KEYS.pricingSource, v);
  }, []);

  const value = React.useMemo<SettingsState>(
    () => ({
      hydrated, defaultModel, testedModels, hfToken, inferenceBackend, simulator, backendVersion,
      costingsEnabled, preferredCloudProvider, pricingSource,
      setDefaultModel, setHfToken, setInferenceBackend, setSimulator, setBackendVersion,
      setCostingsEnabled, setPreferredCloudProvider, setPricingSource,
    }),
    [hydrated, defaultModel, testedModels, hfToken, inferenceBackend, simulator, backendVersion,
     costingsEnabled, preferredCloudProvider, pricingSource,
     setDefaultModel, setHfToken, setInferenceBackend, setSimulator, setBackendVersion,
     setCostingsEnabled, setPreferredCloudProvider, setPricingSource],
  );

  return (
    <SettingsContext.Provider value={value}>
      {children}
    </SettingsContext.Provider>
  );
}

export function useSettings(): SettingsState {
  return React.useContext(SettingsContext);
}
