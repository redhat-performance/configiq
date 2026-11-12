// @vitest-environment happy-dom
import * as React from 'react'
import { act } from 'react'
import { createRoot } from 'react-dom/client'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { GpuOption } from '@/lib/hooks/useCatalog'
import { GpuSystemInput } from './GpuSystemInput'

const GPUS: GpuOption[] = [
  {
    systemId: 'h200_sxm',
    label: 'NVIDIA H200',
    vendor: 'nvidia',
    architecture: 'hopper',
    vramGb: 141,
    bandwidthTbps: 4.8,
    tflopsBf16: 989,
    tdpWatts: 700,
    gpusPerNode: 8,
  },
  {
    systemId: 'b200_sxm',
    label: 'NVIDIA B200',
    vendor: 'nvidia',
    architecture: 'blackwell',
    vramGb: 192,
    bandwidthTbps: 8,
    tflopsBf16: 2250,
    tdpWatts: 1000,
    gpusPerNode: 8,
  },
]

let container: HTMLDivElement
let root: ReturnType<typeof createRoot>

beforeEach(() => {
  (globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean })
    .IS_REACT_ACT_ENVIRONMENT = true
  container = document.createElement('div')
  document.body.append(container)
  root = createRoot(container)
})

afterEach(async () => {
  await act(async () => root.unmount())
  container.remove()
})

async function mount(props: Partial<React.ComponentProps<typeof GpuSystemInput>> = {}) {
  const onChange = vi.fn()
  await act(async () => {
    root.render(
      <GpuSystemInput id="gpu" value="h200_sxm" onChange={onChange} gpuOptions={GPUS} {...props} />,
    )
    await Promise.resolve()
  })
  return { onChange }
}

function toggle(): HTMLButtonElement {
  const button = container.querySelector<HTMLButtonElement>('.pf-v6-c-menu-toggle')
  if (!button) throw new Error('menu toggle not rendered')
  return button
}

async function open() {
  await act(async () => {
    toggle().click()
    await Promise.resolve()
  })
}

function optionContaining(text: string): HTMLElement {
  const option = [...document.body.querySelectorAll<HTMLElement>('.pf-v6-c-menu__item')]
    .find(element => element.textContent?.includes(text))
  if (!option) throw new Error(`option not found: ${text}`)
  return option
}

describe('GpuSystemInput grouping and specs', () => {
  it('groups systems by vendor and architecture', async () => {
    await mount()
    await open()
    const groupTitles = [...document.body.querySelectorAll<HTMLElement>('.pf-v6-c-menu__group-title')]
      .map(element => element.textContent)
    expect(groupTitles).toEqual(['Nvidia Blackwell', 'Nvidia Hopper'])
  })

  it('describes the selected system below the dropdown without repeating its memory', async () => {
    await mount()
    expect(container.textContent).toContain('4.8 TB/s · 989 TFLOPS')
    expect(container.textContent).not.toContain('141 GB · 4.8 TB/s')
  })
})

describe('GpuSystemInput all-option mode', () => {
  it('shows the all label for the empty value', async () => {
    await mount({ value: '', allLabel: 'All GPU systems' })
    expect(toggle().textContent).toContain('All GPU systems')
  })

  it('selects the all option with the empty value', async () => {
    const { onChange } = await mount({ allLabel: 'All GPU systems' })
    await open()
    await act(async () => {
      optionContaining('All GPU systems').click()
      await Promise.resolve()
    })
    expect(onChange).toHaveBeenCalledWith('')
  })

  it('still selects a concrete GPU system', async () => {
    const { onChange } = await mount({ allLabel: 'All GPU systems' })
    await open()
    await act(async () => {
      optionContaining('NVIDIA B200').click()
      await Promise.resolve()
    })
    expect(onChange).toHaveBeenCalledWith('b200_sxm')
  })
})
