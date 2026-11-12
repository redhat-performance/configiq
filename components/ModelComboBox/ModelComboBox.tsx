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
import { contextChip, paramsChip } from '@/lib/model-chips'
import type { ModelFacets } from '@/lib/model-metadata'
import { searchModels } from '@/lib/model-search'
import styles from './ModelComboBox.module.css'

interface ComboBoxOption {
  value: string
  label: string
  group: string
  description?: string
  isTested?: boolean
  inCatalog?: boolean
  isHuggingFace?: boolean
}

/** Model rows carry searchable facets; provider/hardware rows remain plain options. */
export type ComboBoxItem = ComboBoxOption | (ComboBoxOption & ModelFacets)

function isModelItem(item: ComboBoxItem): item is ComboBoxOption & ModelFacets {
  return 'types' in item
}

/** Distinguishes a user pick from a change made by the control itself. */
export type ComboBoxChangeSource = 'user' | 'auto'

interface ComboBoxProps {
  value: string
  onChange: (value: string, source: ComboBoxChangeSource) => void
  items: ComboBoxItem[]
  placeholder?: string
  id?: string
  allowCustom?: boolean
  supportedModels?: string[]
  hfToken?: string
  helperText?: React.ReactNode
  label?: string
  portalMenu?: boolean
  preserveOrder?: boolean
  /** When set, prepend an option with an empty value that clears the filter. */
  allLabel?: string
  /** Disable typing and the menu, for example while a request is running. */
  isDisabled?: boolean
}

interface GroupedItems {
  group: string
  items: ComboBoxItem[]
}

function groupItems(items: ComboBoxItem[], preserveOrder: boolean): GroupedItems[] {
  const map = new Map<string, ComboBoxItem[]>()
  for (const item of items) {
    if (!map.has(item.group)) map.set(item.group, [])
    map.get(item.group)!.push(item)
  }
  return Array.from(map, ([group, groupItems]) => ({
    group,
    items: preserveOrder ? groupItems : groupItems.toSorted((a, b) => a.label.localeCompare(b.label)),
  })).toSorted((a, b) => a.group.localeCompare(b.group))
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
  if (!isModelItem(item)) return null
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

export function ComboBox({ value, onChange, items, placeholder, id, allowCustom = false, supportedModels, hfToken, helperText, label = 'Model — Hugging Face ID', portalMenu = false, preserveOrder = false, allLabel, isDisabled = false }: ComboBoxProps) {
  const [open, setOpen] = React.useState(false)
  const [filter, setFilter] = React.useState('')
  // Show the field's text only once the user starts typing. Until then the
  // current selection stays visible (and copyable) instead of the placeholder.
  const [editing, setEditing] = React.useState(false)
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
        onChange(validatedItems[0].value, 'auto')
        wasAutoReplacedRef.current = true
      } else {
        wasAutoReplacedRef.current = false
      }
    } else {
      // Only restore prevModel if enabling the filter caused an auto-replacement
      if (wasAutoReplacedRef.current) {
        onChange(prevModel.current, 'auto')
      }
      wasAutoReplacedRef.current = false
    }
  }

  const isSearching = filter.trim().length > 0
  const modelSearch = activeItems.every(isModelItem)
  const filtered = React.useMemo(() => {
    if (!filter.trim()) return activeItems
    if (activeItems.every(isModelItem)) return searchModels(activeItems, filter)
    const query = filter.trim().toLowerCase()
    return activeItems.filter(item => [item.label, item.description, item.value, item.group]
      .some(text => text?.toLowerCase().includes(query)))
  }, [activeItems, filter])

  const allItem = React.useMemo<ComboBoxItem | null>(
    () => (allLabel ? { value: '', label: allLabel, group: '' } : null),
    [allLabel],
  )
  const selectedItem = allItem && value === '' ? allItem : activeItems.find(i => i.value === value)

  /**
   * Browsing groups by vendor. Searching returns one relevance-ranked list —
   * regrouping a ranked set would sort the best match back under "Q". The
   * "all" option is only offered while browsing.
   */
  const groups = React.useMemo<GroupedItems[]>(
    () => {
      const visible = allItem && !isSearching ? [allItem, ...filtered] : filtered
      if (!isSearching || !modelSearch) return groupItems(visible, preserveOrder)
      // A single unlabelled group, and none at all when nothing matched, so
      // the "No matches" row below still renders.
      if (visible.length === 0) return []
      return [{ group: '', items: visible }]
    },
    [allItem, filtered, isSearching, modelSearch, preserveOrder],
  )

  const flatItems = React.useMemo(() => groups.flatMap(g => g.items), [groups])

  const exactMatch = items.some(i => i.value.toLowerCase() === filter.toLowerCase() || i.label.toLowerCase() === filter.toLowerCase())
  const showCustom = allowCustom && open && filter.trim() && !exactMatch

  function selectItem(val: string) {
    if (supportedOnly) {
      prevModel.current = val
    }
    onChange(val, 'user')
    setOpen(false)
    setEditing(false)
    setFilter('')
    setFocusIndex(-1)
  }

  function handleInputChange(_event: React.FormEvent<HTMLInputElement>, val: string) {
    // The field shows the selected label until editing begins. A keystroke at
    // the end of that text arrives already appended, so keep only the typed
    // part instead of searching for the label plus the new text.
    const displayed = selectedItem?.label ?? value
    // Only strip a single appended keystroke. A pasted ID may legitimately
    // start with the current label and must keep its full text.
    const nextValue = !editing && displayed &&
      val.length === displayed.length + 1 &&
      val.startsWith(displayed)
      ? val.slice(displayed.length)
      : val
    setEditing(true)
    setFilter(nextValue)
    setFocusIndex(-1)
    if (!open) setOpen(true)
  }

  function handleInputFocus() {
    setOpen(true)
    // Keep the selection visible, but let a keystroke replace it rather than
    // appending to it.
    if (!editing) textInputRef.current?.select()
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
        setEditing(false)
        setFilter('')
        setFocusIndex(-1)
        textInputRef.current?.blur()
        break
      case 'Tab':
        setOpen(false)
        setEditing(false)
        setFilter('')
        setFocusIndex(-1)
        break
    }
  }

  function handleClear() {
    onChange('', 'user')
    setEditing(false)
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

  const displayValue = editing ? filter : (selectedItem?.label ?? value)

  const toggle = (tRef: React.RefObject<HTMLDivElement | HTMLButtonElement>) => (
    <MenuToggle
      ref={tRef as React.RefObject<HTMLButtonElement>}
      variant="typeahead"
      aria-label={label}
      isDisabled={isDisabled}
      onClick={() => {
        if (isDisabled) return
        if (open) {
          setOpen(false)
          setEditing(false)
          setFilter('')
          setFocusIndex(-1)
        } else {
          setOpen(true)
        }
      }}
      isExpanded={open && !isDisabled}
      isFullWidth
      className={styles.menuToggle}
    >
      <TextInputGroup isPlain>
        <TextInputGroupMain
          value={displayValue}
          title={selectedItem?.label}
          disabled={isDisabled}
          onClick={() => { if (!isDisabled) setOpen(true) }}
          onChange={handleInputChange}
          onFocus={handleInputFocus}
          onKeyDown={handleInputKeyDown}
          inputId={id}
          aria-label={label}
          autoComplete="off"
          ref={textInputRef}
          placeholder={placeholder ?? (modelSearch ? 'Search by name, size, type or precision...' : 'Type or select...')}
          role="combobox"
          isExpanded={open && !isDisabled}
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
            <Button variant="plain" onClick={handleClear} isDisabled={isDisabled} aria-label="Clear selection" className={styles.clearBtn}>
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
        <label className={styles.label} htmlFor={id}>{label}</label>
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
        isOpen={open && !isDisabled}
        selected={value}
        onSelect={handleSelect}
        onOpenChange={isOpen => { if (isDisabled) return; setOpen(isOpen); if (!isOpen) { setEditing(false); setFilter(''); setFocusIndex(-1) } }}
        toggle={toggle}
        shouldFocusFirstItemOnOpen={false}
        popperProps={{ width: 'trigger', maxWidth: 'trigger', ...(portalMenu ? { appendTo: () => document.body } : {}) }}
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
                    <span className={`${styles.optionLabel} ${styles.optionText}`}>{item.label}
                      {item.description && <span className={styles.optionDescription}>{item.description}</span>}
                    </span>
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
