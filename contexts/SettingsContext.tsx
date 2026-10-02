'use client';

import * as React from 'react';
import { loadAppConfig } from '@/lib/app-config';

export type InferenceBackend = 'vllm' | 'tensorrt-llm' | 'sglang';
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
  backendVersion: 'settings_backend_version',
  costingsEnabled: 'settings_costings_enabled',
  preferredCloudProvider: 'settings_preferred_cloud_provider',
  pricingSource: 'settings_pricing_source',
} as const;

interface SettingsState {
  hydrated: boolean;
  defaultModel: string;
  hfToken: string;
  inferenceBackend: InferenceBackend;
  backendVersion: string;
  costingsEnabled: boolean;
  preferredCloudProvider: string | null;
  pricingSource: PricingSource;
  setDefaultModel: (v: string) => void;
  setHfToken: (v: string) => void;
  setInferenceBackend: (v: InferenceBackend) => void;
  setBackendVersion: (v: string) => void;
  setCostingsEnabled: (v: boolean) => void;
  setPreferredCloudProvider: (v: string | null) => void;
  setPricingSource: (v: PricingSource) => void;
}

const SettingsContext = React.createContext<SettingsState>({
  hydrated: false,
  defaultModel: '',
  hfToken: '',
  inferenceBackend: 'vllm',
  backendVersion: '',
  costingsEnabled: false,
  preferredCloudProvider: null,
  pricingSource: 'merged',
  setDefaultModel: () => {},
  setHfToken: () => {},
  setInferenceBackend: () => {},
  setBackendVersion: () => {},
  setCostingsEnabled: () => {},
  setPreferredCloudProvider: () => {},
  setPricingSource: () => {},
});

export function SettingsProvider({ children }: { children: React.ReactNode }) {
  const [hydrated, setHydrated] = React.useState(false);
  const [defaultModel, setDefaultModelState] = React.useState('');
  const [hfToken, setHfTokenState] = React.useState('');
  const [inferenceBackend, setInferenceBackendState] = React.useState<InferenceBackend>('vllm');
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

      setHfTokenState(localStorage.getItem(STORAGE_KEYS.hfToken) ?? '');

      const savedBackend = localStorage.getItem(STORAGE_KEYS.inferenceBackend) as InferenceBackend | null;
      const activeBackend: InferenceBackend = savedBackend ?? config.defaultBackend as InferenceBackend;
      setInferenceBackendState(activeBackend);

      const savedVersion = localStorage.getItem(STORAGE_KEYS.backendVersion);
      setBackendVersionState(savedVersion ?? '');

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
    setInferenceBackendState(v);
    localStorage.setItem(STORAGE_KEYS.inferenceBackend, v);
    // Settings resolves the backend's default version from the live catalog.
    setBackendVersionState('');
    localStorage.setItem(STORAGE_KEYS.backendVersion, '');
  }, []);

  const setBackendVersion = React.useCallback((v: string) => {
    setBackendVersionState(v);
    localStorage.setItem(STORAGE_KEYS.backendVersion, v);
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
      hydrated, defaultModel, hfToken, inferenceBackend, backendVersion,
      costingsEnabled, preferredCloudProvider, pricingSource,
      setDefaultModel, setHfToken, setInferenceBackend, setBackendVersion,
      setCostingsEnabled, setPreferredCloudProvider, setPricingSource,
    }),
    [hydrated, defaultModel, hfToken, inferenceBackend, backendVersion,
     costingsEnabled, preferredCloudProvider, pricingSource,
     setDefaultModel, setHfToken, setInferenceBackend, setBackendVersion,
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
