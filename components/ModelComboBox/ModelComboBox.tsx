'use client'

import * as React from 'react'
import {
  Switch,
  Select,
  SelectList,
  SelectGroup,
  SelectOption,
  MenuToggle,
  TextInputGroup,
  TextInputGroupMain,
  TextInputGroupUtilities,
  Button,
} from '@patternfly/react-core'
import TimesIcon from '@patternfly/react-icons/dist/esm/icons/times-icon'
import { getAppConfig } from '@/lib/app-config'
import type { ModelFacets } from '@/lib/model-metadata'
import { searchModels } from '@/lib/model-search'
import styles from './ModelComboBox.module.css'

/**
 * A selectable model row. Extends ModelFacets so the derived facts (vendor,
 * family, parameter count, quantization, type) are available to the search
 * scorer and the row renderer without a second lookup — `isTested`,
 * `inCatalog` and `isHuggingFace` come from there.
 */
export interface ComboBoxItem extends ModelFacets {
  value: string
  label: string
  group: string
}

interface ComboBoxProps {
  value: string
  onChange: (value: string) => void
  items: ComboBoxItem[]
  placeholder?: string
  id?: string
  allowCustom?: boolean
  supportedModels?: string[]
  hfToken?: string
  helperText?: React.ReactNode
}

interface GroupedItems {
  group: string
  items: ComboBoxItem[]
}

function groupItems(items: ComboBoxItem[]): GroupedItems[] {
  const map = new Map<string, ComboBoxItem[]>()
  for (const item of items) {
    if (!map.has(item.group)) map.set(item.group, [])
    map.get(item.group)!.push(item)
  }
  return Array.from(map, ([group, groupItems]) => ({
    group,
    items: groupItems.toSorted((a, b) => a.label.localeCompare(b.label)),
  })).toSorted((a, b) => a.group.localeCompare(b.group))
}

/** Compact parameter count for a chip: 0.6B, 32B, 2.4T. Matches modelSizeLabel. */
function paramsChip(paramsB: number | null): string | null {
  if (paramsB === null) return null
  if (paramsB >= 1_000) return `${Number((paramsB / 1_000).toFixed(2))}T`
  if (paramsB >= 0.1) return `${Number(paramsB.toFixed(1))}B`
  return `${Math.round(paramsB * 1_000)}M`
}

/** Compact context window for a chip: 128K, 1M. */
function contextChip(contextLength: number | null): string | null {
  if (!contextLength) return null
  if (contextLength >= 1_048_576) return `${Number((contextLength / 1_048_576).toFixed(1))}M ctx`
  if (contextLength >= 1_024) return `${Math.round(contextLength / 1_024)}K ctx`
  return `${contextLength} ctx`
}

function suggestedNames(): string {
  const names = getAppConfig().suggestedModelNames
  return names.length > 0 ? names.join(', ') : 'Nemotron, DeepSeek V4, Gemma 4, Kimi'
}

function ProvenanceBadges({ item }: { item: ComboBoxItem }) {
  if (item.isTested) return <span className={styles.testedBadge}>tested</span>
  if (item.inCatalog) return <span className={styles.catalogBadge}>in catalog</span>
  if (item.isHuggingFace) return <span className={styles.hfBadge}>hugging face</span>
  return null
}

/**
 * The metadata line under a row's label. Only facts that distinguish this
 * model from its neighbours are shown — "text" is omitted because almost
 * every row is text, and a chip on every row carries no information.
 */
function RowMeta({ item }: { item: ComboBoxItem }) {
  const chips: string[] = []

  const params = paramsChip(item.paramsB)
  if (params) {
    chips.push(item.activeParamsB !== null
      ? `${params} (${paramsChip(item.activeParamsB)} active)`
      : params)
  }
  if (item.types.includes('vision')) chips.push('Vision')
  if (item.types.includes('moe')) chips.push('MoE')

  const context = contextChip(item.contextLength)
  if (context) chips.push(context)

  if (item.quantization) chips.push(item.quantization)

  if (chips.length === 0) return null
  return (
    <div className={styles.optionMeta}>
      {chips.map((chip, index) => <span key={index} className={styles.metaChip}>{chip}</span>)}
    </div>
  )
}

export function ComboBox({ value, onChange, items, placeholder, id, allowCustom = false, supportedModels, hfToken, helperText }: ComboBoxProps) {
  const [open, setOpen] = React.useState(false)
  const [filter, setFilter] = React.useState('')
  const [supportedOnly, setSupportedOnly] = React.useState(false)
  const prevModel = React.useRef(value)
  const [focusIndex, setFocusIndex] = React.useState(-1)
  const textInputRef = React.useRef<HTMLInputElement>(null)

  const validatedItems = React.useMemo(() =>
    supportedModels ? items.filter(i => supportedModels.includes(i.value)) : items,
    [items, supportedModels])

  const activeItems = supportedOnly ? validatedItems : items
  const wasAutoReplacedRef = React.useRef(false)

  const handleToggle = (_: React.FormEvent, checked: boolean) => {
    setSupportedOnly(checked)
    if (checked) {
      prevModel.current = value
      if (!validatedItems.some(i => i.value === value) && validatedItems.length > 0) {
        onChange(validatedItems[0].value)
        wasAutoReplacedRef.current = true
      } else {
        wasAutoReplacedRef.current = false
      }
    } else {
      // Only restore prevModel if enabling the filter caused an auto-replacement
      if (wasAutoReplacedRef.current) {
        onChange(prevModel.current)
      }
      wasAutoReplacedRef.current = false
    }
  }

  const selectedItem = activeItems.find(i => i.value === value)

  const isSearching = filter.trim().length > 0
  const filtered = React.useMemo(() => searchModels(activeItems, filter), [activeItems, filter])

  /**
   * Browsing groups by vendor. Searching returns one relevance-ranked list —
   * regrouping a ranked set would sort the best match back under "Q".
   */
  const groups = React.useMemo<GroupedItems[]>(
    () => {
      if (!isSearching) return groupItems(filtered)
      // A single unlabelled group, and none at all when nothing matched, so
      // the "No matches" row below still renders.
      if (filtered.length === 0) return []
      return [{ group: '', items: filtered }]
    },
    [filtered, isSearching],
  )

  const flatItems = React.useMemo(() => groups.flatMap(g => g.items), [groups])

  const exactMatch = items.some(i => i.value.toLowerCase() === filter.toLowerCase() || i.label.toLowerCase() === filter.toLowerCase())
  const showCustom = allowCustom && open && filter.trim() && !exactMatch

  function selectItem(val: string) {
    if (supportedOnly) {
      prevModel.current = val
    }
    onChange(val)
    setOpen(false)
    setFilter('')
    setFocusIndex(-1)
  }

  function handleInputChange(_event: React.FormEvent<HTMLInputElement>, val: string) {
    setFilter(val)
    setFocusIndex(-1)
    if (!open) setOpen(true)
  }

  function handleInputFocus() {
    setOpen(true)
    setFilter('')
  }

  function handleInputKeyDown(event: React.KeyboardEvent) {
    const totalItems = flatItems.length + (showCustom ? 1 : 0)

    switch (event.key) {
      case 'ArrowDown':
        event.preventDefault()
        if (totalItems === 0) { if (!open) setOpen(true); break }
        setFocusIndex(prev => (prev + 1) % totalItems)
        if (!open) setOpen(true)
        break
      case 'ArrowUp':
        event.preventDefault()
        if (totalItems === 0) { if (!open) setOpen(true); break }
        setFocusIndex(prev => (prev <= 0 ? totalItems - 1 : prev - 1))
        if (!open) setOpen(true)
        break
      case 'Enter':
        event.preventDefault()
        if (focusIndex >= 0 && focusIndex < flatItems.length) {
          selectItem(flatItems[focusIndex].value)
        } else if (focusIndex === flatItems.length && showCustom) {
          selectItem(filter.trim())
        } else if (allowCustom && filter.trim()) {
          selectItem(filter.trim())
        } else if (flatItems.length === 1) {
          selectItem(flatItems[0].value)
        }
        break
      case 'Escape':
        setOpen(false)
        setFilter('')
        setFocusIndex(-1)
        textInputRef.current?.blur()
        break
      case 'Tab':
        setOpen(false)
        setFilter('')
        setFocusIndex(-1)
        break
    }
  }

  function handleClear() {
    onChange('')
    setFilter('')
    setFocusIndex(-1)
    textInputRef.current?.focus()
  }

  function handleSelect(_event: React.MouseEvent | undefined, val: string | number | undefined) {
    if (val === '__custom__') {
      selectItem(filter.trim())
    } else if (typeof val === 'string') {
      selectItem(val)
    }
  }

  const displayValue = open ? filter : (selectedItem?.label ?? value)

  const toggle = (tRef: React.RefObject<HTMLDivElement | HTMLButtonElement>) => (
    <MenuToggle
      ref={tRef as React.RefObject<HTMLButtonElement>}
      variant="typeahead"
      aria-label="Model selector"
      onClick={() => { setOpen(prev => !prev); if (!open) setFilter('') }}
      isExpanded={open}
      isFullWidth
      className={styles.menuToggle}
    >
      <TextInputGroup isPlain>
        <TextInputGroupMain
          value={displayValue}
          onClick={() => { if (!open) { setOpen(true); setFilter('') } }}
          onChange={handleInputChange}
          onFocus={handleInputFocus}
          onKeyDown={handleInputKeyDown}
          id={id}
          autoComplete="off"
          innerRef={textInputRef}
          placeholder={placeholder ?? 'Search by name, size, type or precision...'}
          role="combobox"
          isExpanded={open}
          aria-controls={id ? `${id}-listbox` : undefined}
          aria-activedescendant={focusIndex >= 0 ? `${id}-opt-${focusIndex}` : undefined}
        />
        <TextInputGroupUtilities>
          {value && !open && selectedItem && (
            <div className={styles.selectedBadges}>
              <ProvenanceBadges item={selectedItem} />
            </div>
          )}
          {value && !open && (
            <Button variant="plain" onClick={handleClear} aria-label="Clear selection" className={styles.clearBtn}>
              <TimesIcon />
            </Button>
          )}
        </TextInputGroupUtilities>
      </TextInputGroup>
    </MenuToggle>
  )

  return (
    <div className={styles.wrapper}>
      <div className={styles.labelRow}>
        <label className={styles.label} htmlFor={id}>Model — Hugging Face ID</label>
        {supportedModels && (
          <Switch
            id={id ? `${id}-validated-only` : 'validated-only'}
            label="Tested only"
            isChecked={supportedOnly}
            onChange={handleToggle}
            isReversed
          />
        )}
      </div>

      <Select
        id={id ? `${id}-select` : undefined}
        isOpen={open}
        selected={value}
        onSelect={handleSelect}
        onOpenChange={isOpen => { setOpen(isOpen); if (!isOpen) { setFilter(''); setFocusIndex(-1) } }}
        toggle={toggle}
        shouldFocusFirstItemOnOpen={false}
        popperProps={{ width: 'trigger', maxWidth: 'trigger' }}
      >
        <SelectList id={id ? `${id}-listbox` : undefined} className={styles.selectList}>
          {groups.length === 0 && !showCustom && (
            <SelectOption isDisabled value="__empty__">
              No matches
            </SelectOption>
          )}

          {groups.map(group => {
            const options = group.items.map(item => {
              const idx = flatItems.indexOf(item)
              return (
                <SelectOption
                  key={item.value}
                  id={`${id}-opt-${idx}`}
                  value={item.value}
                  isFocused={idx === focusIndex}
                  isSelected={item.value === value}
                  onMouseEnter={() => setFocusIndex(idx)}
                >
                  <div className={styles.optionRow}>
                    <span className={styles.optionLabel}>{item.label}</span>
                    <div className={styles.badges}>
                      <ProvenanceBadges item={item} />
                    </div>
                  </div>
                  <RowMeta item={item} />
                </SelectOption>
              )
            })

            return group.group ? (
              <SelectGroup key={group.group} label={group.group}>
                {options}
              </SelectGroup>
            ) : (
              <React.Fragment key="__ungrouped__">
                {options}
              </React.Fragment>
            )
          })}

          {showCustom && (
            <SelectOption
              id={`${id}-opt-${flatItems.length}`}
              value="__custom__"
              isFocused={focusIndex === flatItems.length}
              onMouseEnter={() => setFocusIndex(flatItems.length)}
              className={styles.customOption}
            >
              <span className={styles.customOptionLabel}>Use:</span>
              {filter.trim()}
            </SelectOption>
          )}
        </SelectList>
      </Select>

      {supportedModels && (
        <div className={styles.helperText}>
          {helperText ?? (supportedOnly ? (
              <span>Tested: {suggestedNames()}, ... — type to autocomplete</span>
            ) : (
              <>
                <div>Tested: {suggestedNames()}, ... — search by size (&ldquo;70b&rdquo;), type (&ldquo;vision&rdquo;, &ldquo;moe&rdquo;) or precision (&ldquo;fp8&rdquo;)</div>
                {value && !supportedModels.includes(value) && getAppConfig().modelRequestUrl && (
                  <div>New model? <a href={getAppConfig().modelRequestUrl + encodeURIComponent(value)} target="_blank" rel="noopener" className={styles.requestLink}>Request testing →</a></div>
                )}
                {hfToken ? (
                  <div style={{ color: '#0066cc', fontWeight: 500 }}>HF token active</div>
                ) : (
                  <div>Gated model? <a href="/settings" className={styles.requestLink}>Add your HF token in Settings →</a></div>
                )}
              </>
            )
          )}
        </div>
      )}
    </div>
  )
}
