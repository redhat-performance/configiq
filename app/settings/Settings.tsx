'use client';

import * as React from 'react';
import { Button, Label, Switch } from '@patternfly/react-core';
import { EyeIcon, EyeSlashIcon, CheckCircleIcon, ExclamationTriangleIcon } from '@patternfly/react-icons';
import { useSettings, type InferenceBackend } from '@/contexts/SettingsContext';
import { getAppConfig } from '@/lib/app-config';
import { ModelInput } from '@/components/ui/ModelInput';
import { useCatalog } from '@/lib/hooks/useCatalog';
import { fetchModelConfig } from '@/lib/huggingface/fetch-config';
import styles from './Settings.module.css';

type ModelStatus = 'idle' | 'supported' | 'catalog' | 'fetching' | 'fetched' | 'error';

const BACKEND_LABELS: Record<string, string> = {
  vllm: 'vLLM',
  'tensorrt-llm': 'TensorRT-LLM',
  sglang: 'SGLang',
};

export function Settings() {
  const {
    hydrated, defaultModel, setDefaultModel, hfToken, setHfToken,
    inferenceBackend, setInferenceBackend, backendVersion, setBackendVersion,
    costingsEnabled, setCostingsEnabled,
  } = useSettings();
  const { modelOptions, backendOptions, isLoading: catalogLoading } = useCatalog();

  const [localModel, setLocalModel] = React.useState('');
  const [modelStatus, setModelStatus] = React.useState<ModelStatus>('idle');
  const [hfReveal, setHfReveal] = React.useState(false);
  const [modelSaved, setModelSaved] = React.useState(false);
  const [tokenSaved, setTokenSaved] = React.useState(false);
  const [backendSaved, setBackendSaved] = React.useState(false);
  const [costingsSaved, setCostingsSaved] = React.useState(false);
  const [costingsOpen, setCostingsOpen] = React.useState(false);
  const [validatedOpen, setValidatedOpen] = React.useState(false);
  const backendDefaultAttempted = React.useRef(false);

  // Sync local model input once context has loaded from localStorage
  const modelSynced = React.useRef(false);
  React.useEffect(() => {
    if (!hydrated || modelSynced.current) return;
    modelSynced.current = true;
    setLocalModel(defaultModel);
  }, [hydrated, defaultModel]);

  // Debounced auto-save for model (1500ms — gives the user time to finish typing)
  React.useEffect(() => {
    if (!modelSynced.current || localModel === defaultModel) return;
    const timer = setTimeout(() => {
      setDefaultModel(localModel);
      setModelSaved(true);
      const clear = setTimeout(() => setModelSaved(false), 3000);
      return () => clearTimeout(clear);
    }, 1500);
    return () => clearTimeout(timer);
  }, [localModel, defaultModel, setDefaultModel]);

  // Model catalog + HF status check (same debounced pattern as other pages)
  React.useEffect(() => {
    if (catalogLoading) { setModelStatus('idle'); return; }
    if (!localModel || !localModel.includes('/')) { setModelStatus('idle'); return; }
    const timer = setTimeout(() => {
      if (getAppConfig().testedModels.includes(localModel)) { setModelStatus('supported'); return; }
      const inCatalog = modelOptions.includes(localModel);
      if (inCatalog) { setModelStatus('catalog'); return; }
      setModelStatus('fetching');
      fetchModelConfig(localModel, hfToken).then(r => {
        setModelStatus(r.success && r.config ? 'fetched' : 'error');
      });
    }, 500);
    return () => clearTimeout(timer);
  }, [localModel, hfToken, modelOptions, catalogLoading, hydrated]);

  React.useEffect(() => {
    if (hydrated && backendOptions.length > 0 && !backendDefaultAttempted.current) {
      backendDefaultAttempted.current = true;
      const defaultVersion = backendOptions.find(b => b.id === inferenceBackend)?.defaultVersion;
      if (!backendVersion && defaultVersion) setBackendVersion(defaultVersion);
    }
  }, [hydrated, backendOptions, inferenceBackend, backendVersion, setBackendVersion]);

  const handleTokenChange = (v: string) => {
    setHfToken(v);
    if (v.startsWith('hf_')) {
      setTokenSaved(true);
      setTimeout(() => setTokenSaved(false), 2000);
    }
  };

  const handleBackendChange = (v: InferenceBackend) => {
    setInferenceBackend(v); // also resets backendVersion to the default for v
    const defaultVersion = backendOptions.find(b => b.id === v)?.defaultVersion;
    setBackendVersion(defaultVersion ?? '');
    setBackendSaved(true);
    setTimeout(() => setBackendSaved(false), 3000);
  };

  const handleVersionChange = (v: string) => {
    setBackendVersion(v);
    setBackendSaved(true);
    setTimeout(() => setBackendSaved(false), 3000);
  };

  const selectedBackend = backendOptions.find(b => b.id === inferenceBackend);

  return (
    <div className={styles.page}>
      <div className={styles.header}>
        <h1 className={styles.pageTitle}>Settings</h1>
        <p className={styles.subtitle}>
          Configure once — defaults apply across all tools automatically.
        </p>
      </div>

      <div className={styles.sections}>
        {/* ── Default model ── */}
        <div className={styles.section}>
          <div className={styles.sectionHead}>
            <div>
              <div className={styles.sectionTitle}>Default model</div>
              <div className={styles.sectionDesc}>
                Pre-filled in Predict performance, KV cache calculator, and Recommend sizing. Use any Hugging Face model ID.
              </div>
            </div>
            {modelSaved && (
              <Label color="green" icon={<CheckCircleIcon />}>Saved</Label>
            )}
          </div>
          <div className={styles.fieldWrap}>
            <ModelInput
              id="settings-model"
              model={localModel}
              onChange={setLocalModel}
              modelOptions={modelOptions}
              isLoading={catalogLoading}
              status={modelStatus}
              placeholder={catalogLoading ? 'Loading catalog…' : 'e.g. Qwen/Qwen3-32B'}
              helperText="All tools use this model by default. Type to autocomplete from the catalog."
            />
          </div>
        </div>

        {/* ── Tested models ── */}
        <div className={styles.section}>
          <div
            className={styles.sectionHead}
            style={{ cursor: 'pointer' }}
            onClick={() => setValidatedOpen(o => !o)}
          >
            <div>
              <div className={styles.sectionTitle}>
                Tested models
                <span style={{ fontSize: '11px', fontWeight: 400, marginLeft: '8px', color: '#6a6e73' }}>
                  {validatedOpen ? '▲' : '▼'}
                </span>
              </div>
              <div className={styles.sectionDesc}>
                Models tested for use with the AISimulators sizing engine.
              </div>
            </div>
            <Label color="blue" isCompact>{getAppConfig().testedModels.length} models</Label>
          </div>
          {validatedOpen && (
            <div className={styles.fieldWrap}>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px 8px' }}>
                {getAppConfig().testedModels.map(m => (
                  <Label
                    key={m}
                    color="blue"
                    isCompact
                    style={{ cursor: 'pointer' }}
                    onClick={() => {
                      setLocalModel(m);
                      setDefaultModel(m);
                      setModelSaved(true);
                      setTimeout(() => setModelSaved(false), 2000);
                    }}
                  >
                    {m.split('/').pop()}
                  </Label>
                ))}
              </div>
              {getAppConfig().modelRequestUrl && (
                <div style={{ marginTop: '12px', fontSize: '13px' }}>
                  Don&apos;t see your model?{' '}
                  <a href={getAppConfig().modelRequestUrl} target="_blank" rel="noopener" style={{ color: '#0066cc' }}>
                    Request testing →
                  </a>
                </div>
              )}
            </div>
          )}
        </div>

        {/* ── HF token ── */}
        <div className={styles.section}>
          <div className={styles.sectionHead}>
            <div>
              <div className={styles.sectionTitle}>Hugging Face token</div>
              <div className={styles.sectionDesc}>
                Required for gated or private models. Stored in this browser only — never sent to our servers.
              </div>
            </div>
            {tokenSaved && (
              <Label color="green" icon={<CheckCircleIcon />}>Saved</Label>
            )}
          </div>
          <div className={styles.fieldWrap}>
            <label className={styles.fieldLabel} htmlFor="settings-hf-token">API token</label>
            <div className={styles.tokenRow}>
              <input
                id="settings-hf-token"
                type="text"
                value={hfToken}
                onChange={e => handleTokenChange(e.target.value)}
                placeholder="hf_xxxxxxxxxxxxxxxxxxxx"
                className={styles.textInput}
                aria-label="Hugging Face API token"
                autoComplete="off"
                style={{ WebkitTextSecurity: hfReveal ? 'none' : 'disc' } as React.CSSProperties}
              />
              <Button
                variant="control"
                aria-label={hfReveal ? 'Hide token' : 'Show token'}
                onClick={() => setHfReveal(r => !r)}
                icon={hfReveal ? <EyeSlashIcon /> : <EyeIcon />}
              />
            </div>
            <div className={styles.helperText}>
              <a
                href="https://huggingface.co/settings/tokens"
                target="_blank"
                rel="noopener noreferrer"
                className={styles.link}
              >
                Get a token at huggingface.co →
              </a>
            </div>
          </div>
        </div>

        {/* ── Inference backend ── */}
        <div className={styles.section}>
          <div className={styles.sectionHead}>
            <div>
              <div className={styles.sectionTitle}>Inference backend</div>
              <div className={styles.sectionDesc}>
                Serving framework targeted when generating configuration recommendations.
              </div>
            </div>
            {backendSaved && (
              <Label color="green" icon={<CheckCircleIcon />}>Saved</Label>
            )}
          </div>
          <div className={styles.fieldWrap}>
            <div className={styles.backendRow}>
              <div className={styles.backendField}>
                <label className={styles.fieldLabel} htmlFor="settings-backend">Backend</label>
                <select
                  id="settings-backend"
                  value={inferenceBackend}
                  onChange={e => handleBackendChange(e.target.value as InferenceBackend)}
                  className={styles.selectInput}
                >
                  {backendOptions.map(b => (
                    <option key={b.id} value={b.id}>{BACKEND_LABELS[b.id] ?? b.id}</option>
                  ))}
                </select>
              </div>
              <div className={styles.versionField}>
                <label className={styles.fieldLabel} htmlFor="settings-backend-version">Version</label>
                <input
                  id="settings-backend-version"
                  type="text"
                  value={backendVersion}
                  onChange={e => handleVersionChange(e.target.value)}
                  className={styles.textInput}
                  placeholder={selectedBackend?.defaultVersion ?? ''}
                  spellCheck={false}
                  autoComplete="off"
                />
              </div>
            </div>
            {selectedBackend && (
              <div className={styles.helperText}>
                {selectedBackend.versions.length > 0
                  ? `Supported versions: ${selectedBackend.versions.join(', ')}`
                  : 'No versions reported by AISimulators'}
              </div>
            )}
          </div>
        </div>
        {/* ── Costings features ── */}
        <div className={styles.section}>
          <button
            type="button"
            className={styles.sectionHead}
            style={{ cursor: 'pointer', width: '100%', background: 'none', border: 'none', font: 'inherit', textAlign: 'left' }}
            onClick={() => setCostingsOpen(o => !o)}
            aria-expanded={costingsOpen}
            aria-controls="costings-settings-panel"
          >
            <div>
              <div className={styles.sectionTitle}>
                Costings features
                <span style={{ fontSize: '11px', fontWeight: 400, marginLeft: '8px', color: '#6a6e73' }}>
                  {costingsOpen ? '▲' : '▼'}
                </span>
              </div>
              <div className={styles.sectionDesc}>
                Experimental GPU cloud pricing, hardware costs, and on-prem cost modelling.
              </div>
            </div>
            {costingsSaved && <Label color="green" icon={<CheckCircleIcon />}>Saved</Label>}
          </button>
          {costingsOpen && (
            <div className={styles.fieldWrap} id="costings-settings-panel">
              <div className={styles.toggle}>
                <Switch
                  id="settings-costings-enabled"
                  label={costingsEnabled ? 'Enabled' : 'Disabled'}
                  isChecked={costingsEnabled}
                  onChange={(_e, checked) => {
                    setCostingsEnabled(checked);
                    setCostingsSaved(true);
                    setTimeout(() => setCostingsSaved(false), 2000);
                  }}
                />
              </div>
              {costingsEnabled && (
                <>
                </>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
