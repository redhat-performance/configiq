'use client';

import * as React from 'react';
import { Button, Label, MenuToggle, Select, SelectList, SelectOption, Switch } from '@patternfly/react-core';
import { EyeIcon, EyeSlashIcon, CheckCircleIcon } from '@patternfly/react-icons';
import { simulatorSupportsBackend, useSettings, type InferenceBackend, type Simulator } from '@/contexts/SettingsContext';
import { getAppConfig } from '@/lib/app-config';
import { useCatalog } from '@/lib/hooks/useCatalog';
import { simulatorDisplayLabel } from '@/lib/simulator-labels';
import { useTestedModels } from '@/lib/hooks/useTestedModels';
import styles from './Settings.module.css';

const BACKEND_LABELS: Record<string, string> = {
  vllm: 'vLLM',
  'tensorrt-llm': 'TensorRT-LLM',
  sglang: 'SGLang',
};

export function Settings() {
  const {
    setDefaultModel, hfToken, setHfToken,
    inferenceBackend, setInferenceBackend, backendVersion, setBackendVersion,
    simulator, setSimulator,
    costingsEnabled, setCostingsEnabled,
  } = useSettings();
  const { backendOptions, simulatorSupport, simulators, isLoading: catalogLoading } = useCatalog();
  const support = simulatorSupport[simulator];
  // The selectable backends are the union the service reports, so a simulator
  // that supports a backend the default one does not can still be used.
  const backendChoices = React.useMemo(() => {
    const byId = new Map(backendOptions.map(backend => [backend.id, backend]));
    for (const entry of Object.values(simulatorSupport)) {
      for (const backend of entry.backends) {
        if (!byId.has(backend.id)) byId.set(backend.id, backend);
      }
    }
    return [...byId.values()];
  }, [backendOptions, simulatorSupport]);
  const selectedBackend = support
    ? support.backends.find(b => b.id === inferenceBackend) ?? support.backends[0]
    : backendChoices.find(b => b.id === inferenceBackend);
  // Some simulators scope releases per system rather than reporting one shared
  // list. Offer the union so an explicit pin is still possible.
  const selectedVersions = React.useMemo(() => {
    const versions = new Set(selectedBackend?.versions ?? []);
    for (const scoped of Object.values(selectedBackend?.systems ?? {})) {
      for (const version of scoped) versions.add(version);
    }
    return [...versions];
  }, [selectedBackend]);
  const { validatedModelIds, isAvailable: testedModelsAvailable } = useTestedModels();

  const [hfReveal, setHfReveal] = React.useState(false);
  const [modelSaved, setModelSaved] = React.useState(false);
  const [tokenSaved, setTokenSaved] = React.useState(false);
  const [backendSaved, setBackendSaved] = React.useState(false);
  const [simulatorSaved, setSimulatorSaved] = React.useState(false);
  const [versionOpen, setVersionOpen] = React.useState(false);
  const [costingsSaved, setCostingsSaved] = React.useState(false);
  const [costingsOpen, setCostingsOpen] = React.useState(false);
  const [validatedOpen, setValidatedOpen] = React.useState(false);

  const saveModelAsDefault = (modelId: string) => {
    setDefaultModel(modelId);
    setModelSaved(true);
    setTimeout(() => setModelSaved(false), 2000);
  };

  const handleTokenChange = (v: string) => {
    setHfToken(v);
    if (v.startsWith('hf_')) {
      setTokenSaved(true);
      setTimeout(() => setTokenSaved(false), 2000);
    }
  };

  const handleBackendChange = (v: InferenceBackend) => {
    setInferenceBackend(v); // also clears backendVersion
    setBackendSaved(true);
    setTimeout(() => setBackendSaved(false), 3000);
  };

  const handleSimulatorChange = (v: Simulator) => {
    setSimulator(v);
    setSimulatorSaved(true);
    setTimeout(() => setSimulatorSaved(false), 3000);
  };

  const handleVersionChange = (v: string) => {
    setBackendVersion(v);
    setBackendSaved(true);
    setTimeout(() => setBackendSaved(false), 3000);
  };

  return (
    <div className={styles.page}>
      <div className={styles.header}>
        <h1 className={styles.pageTitle}>Settings</h1>
        <p className={styles.subtitle}>
          Configure once — defaults apply across all tools automatically. Your model and GPU choices follow you between tools.
        </p>
      </div>

      <div className={styles.sections}>
        {/* ── Tested models ── */}
        <div className={styles.section}>
          <button
            type="button"
            className={`${styles.sectionHead} ${styles.sectionToggle}`}
            onClick={() => setValidatedOpen(o => !o)}
            aria-expanded={validatedOpen}
            aria-controls="validated-models-panel"
          >
            <div>
              <div className={styles.sectionTitle}>
                Validated models
                <span style={{ fontSize: '11px', fontWeight: 400, marginLeft: '8px', color: '#6a6e73' }}>
                  {validatedOpen ? '▲' : '▼'}
                </span>
              </div>
              <div className={styles.sectionDesc}>
                Models with a validated performance envelope in the tested-model registry. Select one to use it as your default model.
              </div>
            </div>
            {modelSaved
              ? <Label color="green" icon={<CheckCircleIcon />}>Saved as default</Label>
              : <Label color="blue" isCompact>{testedModelsAvailable ? `${validatedModelIds.length} models` : 'Unavailable'}</Label>}
          </button>
          {validatedOpen && (
            <div className={styles.fieldWrap} id="validated-models-panel">
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px 8px' }}>
                 {validatedModelIds.map(m => (
                  <Label
                    key={m}
                    color="blue"
                    isCompact
                    style={{ cursor: 'pointer' }}
                    onClick={() => saveModelAsDefault(m)}
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

        {/* ── Simulator backend ── */}
        <div className={styles.section}>
          <div className={styles.sectionHead}>
            <div>
              <div className={styles.sectionTitle}>Simulator backend</div>
              <div className={styles.sectionDesc}>
                Simulator used by Predict performance, Recommend sizing and Hybrid savings.
              </div>
            </div>
            {simulatorSaved && <Label color="green" icon={<CheckCircleIcon />}>Saved</Label>}
          </div>
          <div className={styles.fieldWrap}>
            <div className={styles.backendField}>
              <label className={styles.fieldLabel} htmlFor="settings-simulator">Simulator</label>
              <select
                id="settings-simulator"
                value={simulator}
                onChange={e => handleSimulatorChange(e.target.value as Simulator)}
                className={styles.selectInput}
              >
                {simulators.map(entry => (
                  <option
                    key={entry.id}
                    value={entry.id}
                    disabled={!simulatorSupportsBackend(entry.id, inferenceBackend, simulatorSupport[entry.id])}
                  >
                    {simulatorDisplayLabel(entry.id, entry.label)}
                  </option>
                ))}
              </select>
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
                  {backendChoices.map(b => (
                    <option
                      key={b.id}
                      value={b.id}
                      disabled={!simulatorSupportsBackend(simulator, b.id as InferenceBackend, support)}
                    >
                      {BACKEND_LABELS[b.id] ?? b.id}
                    </option>
                  ))}
                </select>
              </div>
              <div className={styles.versionField}>
                <label className={styles.fieldLabel} htmlFor="settings-backend-version-toggle">Version</label>
                <Select
                  id="settings-backend-version"
                  isOpen={versionOpen}
                  selected={backendVersion || ''}
                  onSelect={(_event, value) => {
                    if (value == null) return;
                    handleVersionChange(String(value));
                    setVersionOpen(false);
                  }}
                  onOpenChange={setVersionOpen}
                  toggle={toggleRef => (
                    <MenuToggle
                      ref={toggleRef}
                      id="settings-backend-version-toggle"
                      aria-label="Backend version"
                      onClick={() => setVersionOpen(open => !open)}
                      isExpanded={versionOpen}
                      isFullWidth
                      className={styles.versionToggle}
                    >
                      {backendVersion || 'Simulator default'}
                    </MenuToggle>
                  )}
                  popperProps={{ width: 'trigger', maxWidth: 'trigger' }}
                >
                  <SelectList className={styles.versionList}>
                    <SelectOption value="" isSelected={!backendVersion}>
                      {selectedBackend?.defaultVersion
                        ? `Simulator default (${selectedBackend.defaultVersion})`
                        : 'Simulator default'}
                    </SelectOption>
                    {selectedVersions.map(version => (
                      <SelectOption key={version} value={version} isSelected={version === backendVersion}>
                        {version}
                      </SelectOption>
                    ))}
                  </SelectList>
                </Select>
              </div>
            </div>
            {selectedBackend ? (
              <div className={styles.helperText}>
                {selectedVersions.length > 0
                  ? support
                    ? `Supported versions: ${selectedVersions.join(', ')}. Leave blank to use the simulator's default (${selectedBackend.defaultVersion ?? 'its own default'}); Benchmark comparison then uses the newest release measured for the selected backend and GPU.`
                    : `Supported versions: ${selectedVersions.join(', ')}. Leave blank to use each system's default; Benchmark comparison then uses the newest measured release.`
                  : 'No versions reported by the server'}
              </div>
            ) : support && !catalogLoading ? (
              <div className={styles.helperText}>
                The server did not report backend support for this simulator. Predictions will be rejected.
              </div>
            ) : null}
          </div>
        </div>
        {/* ── Costings features ── */}
        <div className={styles.section}>
          <button
            type="button"
            className={`${styles.sectionHead} ${styles.sectionToggle}`}
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
                Experimental cluster-cost and routing-economics models. Hybrid savings and Sources are always available.
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
