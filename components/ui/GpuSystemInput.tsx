'use client'

import * as React from 'react'
import { Switch } from '@patternfly/react-core'
import { getAppConfig } from '@/lib/app-config'
import type { GpuOption } from '@/lib/hooks/useAicCatalog'
import styles from './GpuSystemInput.module.css'

interface GpuSystemInputProps {
  id: string
  value: string
  onChange: (value: string) => void
  gpuOptions: GpuOption[]
}

export function GpuSystemInput({ id, value, onChange, gpuOptions }: GpuSystemInputProps) {
  const current = gpuOptions.find(g => g.systemId === value)

  const architectureGroups = React.useMemo(() => {
    const groups = new Map<string, GpuOption[]>()
    for (const g of gpuOptions) {
      const vendor = g.vendor ?? ''
      const arch = g.architecture ?? 'other'
      const archLabel = arch.charAt(0).toUpperCase() + arch.slice(1)
      const vendorLabel = vendor.charAt(0).toUpperCase() + vendor.slice(1)
      const key = vendorLabel ? `${vendorLabel} ${archLabel}` : archLabel
      const list = groups.get(key)
      if (list) list.push(g)
      else groups.set(key, [g])
    }
    return new Map(
      [...groups.entries()]
        .sort(([a], [b]) => {
          if (a === 'other') return 1
          if (b === 'other') return -1
          return a.localeCompare(b)
        })
        .map(([groupLabel, gpus]) =>
          [
            groupLabel,
            [...gpus].sort((a, b) => a.label.localeCompare(b.label)),
          ] as const,
        ),
    )
  }, [gpuOptions])

  return (
    <div className={styles.wrapper}>
      <label htmlFor={id} className={styles.label}>GPU system</label>
      <select
        id={id}
        value={value}
        onChange={e => onChange(e.target.value)}
        className={styles.select}
      >
        {gpuOptions.length === 0
          ? <option value={value} disabled>Loading GPU catalog…</option>
          : [...architectureGroups.entries()].map(([groupLabel, gpus]) => (
              <optgroup key={groupLabel} label={groupLabel}>
                {gpus.map(g => (
                  <option key={g.systemId} value={g.systemId}>
                    {g.label}
                    {g.vramGb ? ` — ${g.vramGb} GB` : ''}
                    {g.bandwidthTbps != null ? ` · ${g.bandwidthTbps} TB/s` : ''}
                    {g.tflopsBf16 != null ? ` · ${g.tflopsBf16.toFixed(0)} TFLOPS` : ''}
                  </option>
                ))}
              </optgroup>
            ))
        }
      </select>
      {current && (
        <div className={styles.helperText}>
          {current.vramGb != null && <>{current.vramGb} GB</>}
          {current.bandwidthTbps != null && <> · {current.bandwidthTbps} TB/s</>}
          {current.tflopsBf16 != null && <> · {current.tflopsBf16.toFixed(0)} TFLOPS</>}
        </div>
      )}
    </div>
  )
}