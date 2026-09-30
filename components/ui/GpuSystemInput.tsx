'use client'

import * as React from 'react'
import {
  Select,
  SelectList,
  SelectGroup,
  SelectOption,
  MenuToggle,
} from '@patternfly/react-core'
import { getAppConfig } from '@/lib/app-config'
import type { GpuOption } from '@/lib/hooks/useCatalog'
import styles from './GpuSystemInput.module.css'

interface GpuSystemInputProps {
  id: string
  value: string
  onChange: (value: string) => void
  gpuOptions: GpuOption[]
}

export function GpuSystemInput({ id, value, onChange, gpuOptions }: GpuSystemInputProps) {
  const [open, setOpen] = React.useState(false)
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
          const aOther = a.toLowerCase() === 'other' || a.toLowerCase().endsWith(' other')
          const bOther = b.toLowerCase() === 'other' || b.toLowerCase().endsWith(' other')
          if (aOther && !bOther) return 1
          if (!aOther && bOther) return -1
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

  const selectedOption = gpuOptions.find(g => g.systemId === value)
  const selectedLabel = selectedOption
    ? `${selectedOption.label}${selectedOption.vramGb ? ` — ${selectedOption.vramGb} GB` : ''}`
    : 'Select GPU system…'

  const toggle = (tRef: React.RefObject<HTMLButtonElement>) => (
    <MenuToggle
      ref={tRef}
      onClick={() => setOpen(!open)}
      isExpanded={open}
      isFullWidth
      className={styles.menuToggle}
    >
      {selectedLabel}
    </MenuToggle>
  )

  const handleSelect = (event: any, val: any) => {
    if (val) {
      onChange(String(val))
      setOpen(false)
    }
  }

  return (
    <div className={styles.wrapper}>
      <label htmlFor={id} className={styles.label}>GPU system</label>
      <Select
        id={id}
        isOpen={open}
        selected={value}
        onSelect={handleSelect}
        onOpenChange={setOpen}
        toggle={toggle}
        popperProps={{ width: 'trigger', maxWidth: 'trigger' }}
      >
        <SelectList className={styles.selectList}>
          {gpuOptions.length === 0 ? (
            <SelectOption isDisabled value="__empty__">
              Loading GPU catalog…
            </SelectOption>
          ) : (
            [...architectureGroups.entries()].map(([groupLabel, gpus]) => (
              <SelectGroup key={groupLabel} label={groupLabel}>
                {gpus.map(g => (
                  <SelectOption
                    key={g.systemId}
                    value={g.systemId}
                    isSelected={g.systemId === value}
                  >
                    <div className={styles.optionContent}>
                      <span>{g.label}</span>
                      <span className={styles.specs}>
                        {g.vramGb ? `${g.vramGb} GB` : ''}
                      </span>
                    </div>
                  </SelectOption>
                ))}
              </SelectGroup>
            ))
          )}
        </SelectList>
      </Select>
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