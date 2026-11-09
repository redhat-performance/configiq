'use client'

import * as React from 'react'
import { MenuToggle, Select, SelectList, SelectOption } from '@patternfly/react-core'
import styles from './validation-compare.module.css'

export interface CompareSelectOption {
  value: string
  label: string
  isDisabled?: boolean
}

interface CompareSelectProps {
  id: string
  label: string
  value: string
  options: CompareSelectOption[]
  placeholder: string
  placeholderIsSelectable?: boolean
  isDisabled?: boolean
  onChange: (value: string) => void
}

export function CompareSelect({ id, label, value, options, placeholder, placeholderIsSelectable = false, isDisabled = false, onChange }: CompareSelectProps) {
  const [isOpen, setIsOpen] = React.useState(false)
  const selectedOption = options.find(option => option.value === value)

  React.useEffect(() => {
    if (isDisabled) setIsOpen(false)
  }, [isDisabled])

  return (
    <Select
      isOpen={isOpen && !isDisabled}
      selected={value}
      onSelect={(_event, selected) => {
        if (selected == null) return
        onChange(String(selected))
        setIsOpen(false)
      }}
      onOpenChange={setIsOpen}
      toggle={toggleRef => (
        <MenuToggle
          ref={toggleRef}
          id={id}
          aria-label={label}
          onClick={() => setIsOpen(open => !open)}
          isExpanded={isOpen && !isDisabled}
          isDisabled={isDisabled}
          isFullWidth
          className={styles.compareMenuToggle}
        >
          <span className={selectedOption ? styles.selectedOption : styles.placeholderOption}>
            {selectedOption?.label ?? placeholder}
          </span>
        </MenuToggle>
      )}
      popperProps={{ width: 'trigger', maxWidth: 'trigger' }}
    >
      <SelectList className={styles.compareSelectList}>
        <SelectOption value="" isSelected={!value} isDisabled={!placeholderIsSelectable}>
          {placeholder}
        </SelectOption>
        {options.map(option => (
          <SelectOption
            key={option.value}
            value={option.value}
            isSelected={option.value === value}
            isDisabled={option.isDisabled}
          >
            {option.label}
          </SelectOption>
        ))}
      </SelectList>
    </Select>
  )
}
