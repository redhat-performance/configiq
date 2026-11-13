'use client';

import * as React from 'react';
import { loadAppConfig } from '@/lib/app-config';
import { useCatalog } from '@/lib/hooks/useCatalog';

export type InferenceBackend = 'vllm' | 'tensorrt-llm' | 'sglang';

/**
 * Provenance carried by a shared link, applied for the current session only.
 * Omitted fields keep the user's own settings.
 */
export interface SessionViewOverride {
  simulator?: Simulator;
  backend?: InferenceBackend;
  backendVersion?: string;
}
/**
 * A simulator identifier discovered from GET /simulators. It is deliberately an
 * open string: the backend service owns the list, and a build of this app must
 * keep working when a simulator is added or retired.
 */
export type Simulator = string;
/** Empty means "no explicit choice"; the catalog's default simulator applies. */
const NO_SIMULATOR = '';

/**
 * Whether a simulator advertises support for a backend. The answer comes from
 * that simulator's own catalog entry, never from a table in this file.
 */
export function simulatorSupportsBackend(
  simulator: Simulator,
  backend: InferenceBackend,
  support: { backends: Array<{ id: string }> } | undefined,
): boolean {
  if (!simulator) return true
  return support?.backends.some(option => option.id === backend) ?? true
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
  defaultSystem: 'settings_default_system',
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
  return `${STORAGE_KEYS.backendVersion}:${simulator || '_default'}`;
}

interface SettingsState {
  hydrated: boolean;
  defaultModel: string;
  defaultSystem: string;
  hfToken: string;
  inferenceBackend: InferenceBackend;
  simulator: Simulator;
  backendVersion: string;
  costingsEnabled: boolean;
  preferredCloudProvider: string | null;
  pricingSource: PricingSource;
  setDefaultModel: (v: string) => void;
  setDefaultSystem: (v: string) => void;
  setHfToken: (v: string) => void;
  setInferenceBackend: (v: InferenceBackend) => void;
  setSimulator: (v: Simulator) => void;
  setBackendVersion: (v: string) => void;
  /**
   * Apply a shared view's provenance for this session only. Never written to
   * localStorage: opening someone else's link must not change your defaults.
   */
  applyViewForSession: (view: SessionViewOverride) => void;
  setCostingsEnabled: (v: boolean) => void;
  setPreferredCloudProvider: (v: string | null) => void;
  setPricingSource: (v: PricingSource) => void;
}

const SettingsContext = React.createContext<SettingsState>({
  hydrated: false,
  defaultModel: '',
  defaultSystem: '',
  hfToken: '',
  inferenceBackend: 'vllm',
  simulator: NO_SIMULATOR,
  backendVersion: '',
  costingsEnabled: false,
  preferredCloudProvider: null,
  pricingSource: 'merged',
  setDefaultModel: () => {},
  setDefaultSystem: () => {},
  setHfToken: () => {},
  setInferenceBackend: () => {},
  setSimulator: () => {},
  setBackendVersion: () => {},
  applyViewForSession: () => {},
  setCostingsEnabled: () => {},
  setPreferredCloudProvider: () => {},
  setPricingSource: () => {},
});

export function SettingsProvider({ children }: { children: React.ReactNode }) {
  const { simulators, isLoading: catalogLoading } = useCatalog();
  // The default is the entry the catalogue marks as such; fall back to the
  // first in service order, matching how the proxy resolves a missing flag.
  const defaultSimulator = simulators.find(entry => entry.default)?.id ?? simulators[0]?.id ?? '';
  const [hydrated, setHydrated] = React.useState(false);
  const [defaultModel, setDefaultModelState] = React.useState('');
  const [defaultSystem, setDefaultSystemState] = React.useState('');
  const [hfToken, setHfTokenState] = React.useState('');
  const [inferenceBackendState, setInferenceBackendState] = React.useState<InferenceBackend>('vllm');
  // An explicit choice; empty means the catalog's default simulator applies.
  const [simulatorChoice, setSimulatorChoiceState] = React.useState<Simulator>(NO_SIMULATOR);
  const [sessionOverride, setSessionOverride] = React.useState<SessionViewOverride | null>(null);
  const simulator = sessionOverride?.simulator ?? (simulatorChoice || defaultSimulator);
  const [backendVersionState, setBackendVersionState] = React.useState('');
  const backendVersion = sessionOverride?.backendVersion ?? backendVersionState;
  const inferenceBackend = sessionOverride?.backend ?? inferenceBackendState;
  const [costingsEnabled, setCostingsEnabledState] = React.useState(false);
  const [preferredCloudProvider, setPreferredCloudProviderState] = React.useState<string | null>(null);
  const [pricingSource, setPricingSourceState] = React.useState<PricingSource>(DEFAULT_PRICING_SOURCE);

  const initialized = React.useRef(false);
  React.useEffect(() => {
    // Wait for the catalog so a saved default model's simulator pin migrates to
    // the right per-simulator key. A failed catalog still hydrates below.
    if (catalogLoading || initialized.current) return;
    initialized.current = true;

    async function init() {
      // Load config.json first — gives us the defaults to fall back to
      const config = await loadAppConfig();

      // localStorage overrides config; null means "never saved by user" so use config default
      const savedModel = localStorage.getItem(STORAGE_KEYS.defaultModel);
      setDefaultModelState(savedModel !== null ? savedModel : config.defaultModel);

      const savedSystem = localStorage.getItem(STORAGE_KEYS.defaultSystem);
      setDefaultSystemState(savedSystem !== null ? savedSystem : config.defaultSystem);

      setHfTokenState(localStorage.getItem(STORAGE_KEYS.hfToken) ?? '');

      const savedBackend = localStorage.getItem(STORAGE_KEYS.inferenceBackend) as InferenceBackend | null;
      const activeBackend: InferenceBackend = savedBackend ?? config.defaultBackend as InferenceBackend;
      setInferenceBackendState(activeBackend);

      // The simulator id is an open string discovered from the service; keep
      // whatever was saved. An unknown or retired id surfaces the API's
      // explicit error instead of being silently replaced with another
      // simulator here.
      const savedSimulator = localStorage.getItem(STORAGE_KEYS.simulator);
      setSimulatorChoiceState(savedSimulator ?? NO_SIMULATOR);

      // The legacy unsuffixed key only ever held the default simulator's pin,
      // so migrate it to the simulator this profile actually resolves to.
      const activeSimulator = savedSimulator || defaultSimulator;
      const legacyVersion = localStorage.getItem(STORAGE_KEYS.backendVersion);
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
  }, [catalogLoading, defaultSimulator]);

  const setDefaultModel = React.useCallback((v: string) => {
    setDefaultModelState(v);
    localStorage.setItem(STORAGE_KEYS.defaultModel, v);
  }, []);

  const setDefaultSystem = React.useCallback((v: string) => {
    setDefaultSystemState(v);
    localStorage.setItem(STORAGE_KEYS.defaultSystem, v);
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
    setSessionOverride(null);
    setInferenceBackendState(v);
    localStorage.setItem(STORAGE_KEYS.inferenceBackend, v);
    // Settings resolves the backend's default version from the live catalog.
    setBackendVersionState('');
    localStorage.setItem(versionStorageKey(simulator), '');
  }, [simulator]);

  const setSimulator = React.useCallback((v: Simulator) => {
    // An explicit choice ends any shared-view override.
    setSessionOverride(null);
    setSimulatorChoiceState(v);
    localStorage.setItem(STORAGE_KEYS.simulator, v);
    // Load the version saved for the simulator being switched to. The two
    // simulators' version namespaces are unrelated, so never carry one over.
    setBackendVersionState(localStorage.getItem(versionStorageKey(v)) ?? '');
  }, []);

  const setBackendVersion = React.useCallback((v: string) => {
    setSessionOverride(null);
    setBackendVersionState(v);
    localStorage.setItem(versionStorageKey(simulator), v);
  }, [simulator]);

  const applyViewForSession = React.useCallback((view: SessionViewOverride) => {
    setSessionOverride(view);
  }, []);

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
      hydrated, defaultModel, defaultSystem, hfToken, inferenceBackend, simulator, backendVersion,
      costingsEnabled, preferredCloudProvider, pricingSource,
      setDefaultModel, setDefaultSystem, setHfToken, setInferenceBackend, setSimulator, setBackendVersion,
      applyViewForSession,
      setCostingsEnabled, setPreferredCloudProvider, setPricingSource,
    }),
    [hydrated, defaultModel, defaultSystem, hfToken, inferenceBackend, simulator, backendVersion,
     costingsEnabled, preferredCloudProvider, pricingSource,
     setDefaultModel, setDefaultSystem, setHfToken, setInferenceBackend, setSimulator, setBackendVersion,
     applyViewForSession,
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
