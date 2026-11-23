'use client'

import * as React from 'react'
import {
  Alert,
  Button,
  FormSelect,
  FormSelectOption,
  Spinner,
  TextInput,
  ToggleGroup,
  ToggleGroupItem,
} from '@patternfly/react-core'
import { ComboBox, type ComboBoxItem } from '@/components/ModelComboBox/ModelComboBox'
import { GpuSystemInput } from '@/components/ui/GpuSystemInput'
import { useRecommend } from '@/contexts/RecommendContext'
import { useSettings } from '@/contexts/SettingsContext'
import { getAppConfig } from '@/lib/app-config'
import { useCatalog } from '@/lib/hooks/useCatalog'
import { useTestedModels } from '@/lib/hooks/useTestedModels'
import { workloadPresetOptions, type WorkloadPreset } from '@/lib/workload-presets'
import { buildDeployParams, pickDeployDefaults } from './deploy-request'
import styles from './deploy.module.css'

/**
 * Deploy by objective (docs/08): the user states a model, a load, and latency
 * targets; the answer card (AnswerCard.tsx) shows the deployment the simulator
 * recommends. Targets and the free-GPU count are display facts on the answer,
 * never request fields — the run itself is shaped by the load alone
 * (`sla_mode: 'none'`).
 */
export function Deploy() {
  const { isLoading, error, errorCode, result, startSizing } = useRecommend()
  const settings = useSettings()
  const catalog = useCatalog()
  const tested = useTestedModels()

  const [model, setModel] = React.useState('')
  const [systemId, setSystemId] = React.useState('')
  const [loadUnit, setLoadUnit] = React.useState<'users' | 'rps'>('users')
  const [load, setLoad] = React.useState('64')
  const [ttftTarget, setTtftTarget] = React.useState('400')
  const [tpotTarget, setTpotTarget] = React.useState('30')
  const [gpusFree, setGpusFree] = React.useState('8')
  const [workloadKey, setWorkloadKey] = React.useState<string | null>(null)

  const presets = React.useMemo<WorkloadPreset[]>(
    () => workloadPresetOptions(getAppConfig().workloadPresets),
    [],
  )
  const workload = presets.find(preset => preset.key === workloadKey) ?? presets[0]
  const modelItems = React.useMemo<ComboBoxItem[]>(
    () => catalog.modelOptions.map(option => ({
      value: option,
      label: option,
      group: option.includes('/') ? option.slice(0, option.indexOf('/')) : 'Models',
    })),
    [catalog.modelOptions],
  )
  const testedSystemIds = React.useMemo(
    () => tested.pairs.filter(pair => pair.modelId === model).map(pair => pair.systemId),
    [tested.pairs, model],
  )

  const run = React.useCallback((overrides?: { model?: string; systemId?: string }) => {
    startSizing(buildDeployParams({
      model: overrides?.model ?? model,
      systemId: overrides?.systemId ?? systemId,
      workload,
      loadUnit,
      loadValue: Number(load) || 0,
      ttftTarget: Number(ttftTarget) || 0,
      tpotTarget: Number(tpotTarget) || 0,
      gpusFree: Number(gpusFree) || 0,
      backend: settings.inferenceBackend,
      backendVersion: settings.backendVersion || undefined,
      simulator: settings.simulator,
    }))
  }, [startSizing, model, systemId, workload, loadUnit, load, ttftTarget, tpotTarget, gpusFree, settings.inferenceBackend, settings.backendVersion, settings.simulator])

  // First load runs automatically with defaults (docs/07 §4.1); edits after
  // that never auto-run — the Recommend button is for re-runs.
  const autoRan = React.useRef(false)
  const catalogReady = !catalog.isLoading && catalog.modelOptions.length > 0
  React.useEffect(() => {
    if (!catalogReady || autoRan.current) return
    const defaults = pickDeployDefaults({
      modelOptions: catalog.modelOptions,
      gpuOptions: catalog.gpuOptions,
      validatedModelIds: tested.validatedModelIds,
      pairs: tested.pairs,
    })
    if (!defaults) return
    autoRan.current = true
    setModel(defaults.model)
    setSystemId(defaults.systemId)
    run({ model: defaults.model, systemId: defaults.systemId })
  }, [catalogReady, catalog.modelOptions, catalog.gpuOptions, tested.validatedModelIds, tested.pairs, run])

  return (
    <div className={styles.page}>
      <header className={styles.heading}>
        <p className={styles.eyebrow}>Deploy</p>
        <h1>Auto-deploy a model</h1>
        <p className={styles.intro}>
          Pick a model and your load. Get a recommended deployment and run it on your cluster.
        </p>
      </header>

      <div className={styles.grid}>
        <section className={styles.inputsCard} aria-labelledby="deploy-inputs-title">
          <div className={styles.cardHead}>
            <h2 id="deploy-inputs-title">Your workload</h2>
          </div>
          <div className={styles.cardBody}>
            <div className={styles.field}>
              <span className={styles.fieldLabel}>Model</span>
              <ComboBox
                id="deploy-model"
                value={model}
                onChange={value => setModel(value)}
                items={modelItems}
                placeholder="Search models..."
                isDisabled={isLoading}
              />
            </div>
            <div className={styles.field}>
              <span className={styles.fieldLabel}>GPU system</span>
              {/* Wrapper carries the id: PatternFly Select does not render the
                  id prop as a DOM id (same workaround as Compare.tsx). */}
              <div id="deploy-gpu">
                <GpuSystemInput
                  id="deploy-gpu"
                  value={systemId}
                  onChange={setSystemId}
                  gpuOptions={catalog.gpuOptions}
                  testedSystemIds={testedSystemIds}
                  isDisabled={isLoading}
                />
              </div>
            </div>
            <div className={styles.field}>
              <span className={styles.fieldLabel}>Workload shape</span>
              <FormSelect
                id="deploy-workload"
                value={workload?.key ?? ''}
                onChange={(_event, value) => setWorkloadKey(String(value))}
                aria-label="Workload shape"
                isDisabled={isLoading}
              >
                {presets.map(preset => (
                  <FormSelectOption key={preset.key} value={preset.key}
                    label={`${preset.label} (${preset.isl.toLocaleString('en-US')} in / ${preset.osl.toLocaleString('en-US')} out)`} />
                ))}
              </FormSelect>
            </div>
            <div className={styles.field}>
              <span className={styles.fieldLabel}>Load</span>
              <div className={styles.loadRow}>
                <ToggleGroup aria-label="Load unit">
                  <ToggleGroupItem text="Concurrent users" buttonId="deploy-load-users"
                    isSelected={loadUnit === 'users'} onChange={() => setLoadUnit('users')} />
                  <ToggleGroupItem text="Requests/sec" buttonId="deploy-load-rps"
                    isSelected={loadUnit === 'rps'} onChange={() => setLoadUnit('rps')} />
                </ToggleGroup>
                <TextInput id="deploy-load" type="number" value={load} min={1}
                  onChange={(_event, value) => setLoad(value)} aria-label="Load value" className={styles.loadInput} />
              </div>
            </div>
            <div className={styles.targetGrid}>
              <div className={styles.field}>
                <label className={styles.fieldLabel} htmlFor="deploy-ttft">Mean TTFT target (ms)</label>
                <TextInput id="deploy-ttft" type="number" value={ttftTarget} min={1}
                  onChange={(_event, value) => setTtftTarget(value)} aria-label="Mean TTFT target" />
              </div>
              <div className={styles.field}>
                <label className={styles.fieldLabel} htmlFor="deploy-tpot">Mean TPOT target (ms)</label>
                <TextInput id="deploy-tpot" type="number" value={tpotTarget} min={1}
                  onChange={(_event, value) => setTpotTarget(value)} aria-label="Mean TPOT target" />
              </div>
            </div>
            <div className={styles.field}>
              <label className={styles.fieldLabel} htmlFor="deploy-gpus-free">GPUs free in your cluster</label>
              <TextInput id="deploy-gpus-free" type="number" value={gpusFree} min={0}
                onChange={(_event, value) => setGpusFree(value)} aria-label="GPUs free" />
            </div>
            <div className={styles.runRow}>
              <Button variant="primary" onClick={() => run()} isLoading={isLoading} isDisabled={isLoading || !model || !systemId}>
                Recommend
              </Button>
            </div>
          </div>
        </section>

        <section className={styles.answerColumn} aria-labelledby="deploy-answer-title">
          <h2 id="deploy-answer-title" className={styles.visuallyHidden}>Recommendation</h2>
          {isLoading && (
            <div className={styles.pending}>
              <Spinner size="lg" aria-label="Recommending" />
              <p>Running the simulator for your workload.</p>
            </div>
          )}
          {!isLoading && error && (
            <Alert variant="danger" title="Could not load the recommendation" isInline>
              <p>{errorCode ? `${errorCode}: ` : ''}{error}</p>
            </Alert>
          )}
        </section>
      </div>
    </div>
  )
}
