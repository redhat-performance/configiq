// @vitest-environment happy-dom
import * as React from 'react'
import { act } from 'react'
import { createRoot } from 'react-dom/client'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { ProductTour, type TourStep } from './ProductTour'

let root: ReturnType<typeof createRoot>
let host: HTMLDivElement

const tourStep = (target: string, title: string, position?: TourStep['position']): TourStep => ({
  target,
  title,
  description: `${title} description`,
  position,
})

const dialog = () => document.querySelector<HTMLElement>('[role="dialog"]')
const title = () => document.querySelector('#product-tour-title')?.textContent ?? null

function addTarget(id: string, rect: Partial<DOMRect> = {}): HTMLElement {
  const element = document.createElement('div')
  element.id = id
  element.scrollIntoView = vi.fn()
  element.getBoundingClientRect = () =>
    ({
      top: 0,
      left: 0,
      width: 100,
      height: 40,
      right: 100,
      bottom: 40,
      x: 0,
      y: 0,
      toJSON: () => ({}),
      ...rect,
    }) as DOMRect
  document.body.append(element)
  return element
}

async function renderTour(steps: TourStep[], onComplete: () => void) {
  await act(async () => {
    root.render(<ProductTour steps={steps} onComplete={onComplete} />)
  })
}

beforeEach(() => {
  Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true })
  // Run layout measurement synchronously so tests do not depend on frame timing.
  vi.stubGlobal('requestAnimationFrame', (callback: FrameRequestCallback) => {
    callback(0)
    return 1
  })
  vi.stubGlobal('cancelAnimationFrame', () => {})
  Object.defineProperty(window, 'innerWidth', { value: 1024, configurable: true })
  Object.defineProperty(window, 'innerHeight', { value: 768, configurable: true })

  host = document.createElement('div')
  document.body.append(host)
  root = createRoot(host)
})

afterEach(async () => {
  await act(async () => root.unmount())
  host.remove()
  document.querySelectorAll('[id^="tour-target"]').forEach((element) => element.remove())
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

describe('ProductTour', () => {
  it('skips a step whose target is not in the DOM', async () => {
    addTarget('tour-target-model')
    const onComplete = vi.fn()

    await renderTour(
      [tourStep('#tour-target-missing', 'Missing step'), tourStep('#tour-target-model', 'Model step')],
      onComplete,
    )

    expect(title()).toBe('Model step')
    expect(onComplete).not.toHaveBeenCalled()
  })

  it('completes instead of stalling when no step target exists', async () => {
    const onComplete = vi.fn()

    await renderTour([tourStep('#tour-target-missing', 'Missing step')], onComplete)

    expect(onComplete).toHaveBeenCalledTimes(1)
  })

  it('completes when the final available step is dismissed with Escape', async () => {
    addTarget('tour-target-model')
    const onComplete = vi.fn()
    await renderTour([tourStep('#tour-target-model', 'Model step')], onComplete)

    await act(async () => {
      window.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }))
    })

    expect(onComplete).toHaveBeenCalledTimes(1)
  })

  it('keeps the tooltip inside the viewport when the target is near the edge', async () => {
    addTarget('tour-target-edge', { top: 900, left: 1200, width: 100, height: 50, right: 1300, bottom: 950 })
    await renderTour([tourStep('#tour-target-edge', 'Edge step', 'bottom')], vi.fn())

    const tooltip = dialog()
    expect(tooltip).not.toBeNull()
    // 1024x768 viewport, 360px fallback tooltip width, 12px margin.
    expect(tooltip!.style.top).toBe('556px')
    expect(tooltip!.style.left).toBe('652px')
    expect(tooltip!.style.visibility).toBe('visible')
  })
})
