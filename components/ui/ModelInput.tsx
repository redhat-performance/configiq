'use client'

import * as React from 'react'
import { Label, Switch } from '@patternfly/react-core'
import CheckCircleIcon from '@patternfly/react-icons/dist/esm/icons/check-circle-icon'
import ExclamationTriangleIcon from '@patternfly/react-icons/dist/esm/icons/exclamation-triangle-icon'
import { getAppConfig } from '@/lib/app-config'
import styles from './ModelInput.module.css'

export type ModelStatus = 'idle' | 'supported' | 'catalog' | 'fetching' | 'fetched' | 'error'

interface ModelInputProps {
  id: string
  model: string
  onChange: (value: string) => void
  modelOptions: string[]
  isLoading?: boolean
  hfToken?: string
  status?: ModelStatus
  placeholder?: string
  helperText?: React.ReactNode
}

function suggestedNames(): string {
  return getAppConfig().suggestedModelNames.join(', ')
}

export function ModelInput({
  id,
  model,
  onChange,
  modelOptions,
  isLoading,
  hfToken,
  status = 'idle',
  placeholder = 'Type model name or select from dropdown...',
  helperText,
}: ModelInputProps) {
  const [testedOnly, setTestedOnly] = React.useState(false)
  const cfg = getAppConfig()
  const testedModels = cfg.testedModels ?? []
  const filteredModels = modelOptions.filter(m => testedModels.includes(m))
  const displayModels = testedOnly ? filteredModels : modelOptions
  const datalistId = `${id}-options`

  const prevModel = React.useRef(model)

  const handleToggle = (_: React.FormEvent, checked: boolean) => {
    setTestedOnly(checked)
    if (checked) {
      prevModel.current = model
      if (!testedModels.includes(model) && filteredModels.length > 0) {
        onChange(filteredModels[0])
      }
    } else {
      onChange(prevModel.current)
    }
  }

  return (
    <div className={styles.wrapper}>
      <label className={styles.label} htmlFor={id}>
        Model — Hugging Face ID
      </label>
      <div className={styles.toggle}>
        <Switch
          id={`${id}-validated-only`}
          label="Tested only"
          isChecked={testedOnly}
          onChange={handleToggle}
          isReversed
        />
      </div>

      <div className={styles.inputWrapper}>
        <input
          type="text"
          id={id}
          list={datalistId}
          value={model}
          onChange={e => onChange(e.target.value)}
          placeholder={testedOnly ? 'Select a tested model...' : placeholder}
          className={styles.input}
          spellCheck={false}
          autoComplete="off"
        />
        <datalist id={datalistId}>
          {displayModels.map(m => <option key={m} value={m} />)}
        </datalist>
        <div className={styles.badge}>
          {status === 'supported' && <Label color="blue" isCompact icon={<CheckCircleIcon />}>Tested</Label>}
          {status === 'catalog' && <Label color="green" isCompact icon={<CheckCircleIcon />}>In catalog</Label>}
          {status === 'fetching' && <Label color="grey" isCompact>Checking...</Label>}
          {status === 'fetched' && <Label color="yellow" isCompact icon={<CheckCircleIcon />}>From HuggingFace</Label>}
          {status === 'error' && <Label color="red" isCompact icon={<ExclamationTriangleIcon />}>Not found</Label>}
          {status === 'idle' && isLoading && <Label color="grey" isCompact>Loading...</Label>}
        </div>
      </div>

      <div className={styles.helperText}>
        {helperText ?? (testedOnly ? (
            <span>Tested: {suggestedNames()}, ... — type to autocomplete</span>
          ) : (
            <>
              <div>Tested: {suggestedNames()}, ... — type to autocomplete</div>
              {model && !testedModels.includes(model) && cfg.modelRequestUrl && (
                <div>New model? <a href={cfg.modelRequestUrl + encodeURIComponent(model)} target="_blank" rel="noopener" className={styles.requestLink}>Request testing →</a></div>
              )}
              {hfToken ? (
                <div style={{ color: '#0066cc', fontWeight: 500 }}>🔑 HF token active</div>
              ) : (
                <div>Gated model? <a href="/settings" className={styles.requestLink}>Add your HF token in Settings →</a></div>
              )}
            </>
          )
        )}
      </div>
    </div>
  )
}
