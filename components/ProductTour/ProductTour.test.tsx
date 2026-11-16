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
/** The "N of M" counter, or null when the tooltip is not showing one. */
const progress = () => (dialog()?.textContent ?? '').match(/\d+ of \d+/)?.[0] ?? null
const action = (label: string) =>
  Array.from(dialog()?.querySelectorAll('button') ?? []).find(candidate => candidate.textContent === label)

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

  it('keeps focus inside the dialog when a step removes the focused button', async () => {
    addTarget('tour-target-model')
    addTarget('tour-target-system')
    await renderTour(
      [tourStep('#tour-target-model', 'Model step'), tourStep('#tour-target-system', 'System step')],
      vi.fn(),
    )

    const button = (label: string) =>
      Array.from(dialog()!.querySelectorAll('button')).find(candidate => candidate.textContent === label)

    await act(async () => button('Next')!.click())
    expect(title()).toBe('System step')

    // Back is the only way to reach the earlier step, and it disappears when
    // that step has no predecessor. Removing a focused button returns focus to
    // the viewport, outside the modal.
    await act(async () => button('Back')!.focus())
    expect(document.activeElement).toBe(button('Back'))

    await act(async () => button('Back')!.click())
    expect(title()).toBe('Model step')
    expect(dialog()!.contains(document.activeElement)).toBe(true)
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

  it('counts only the steps whose targets are on screen', async () => {
    // The recommend tour ends on the result tiles, which do not exist before a
    // calculation. Promising five steps and stopping at four reads as a broken
    // tour rather than a skipped one.
    addTarget('tour-target-model')
    addTarget('tour-target-goal')
    await renderTour(
      [
        tourStep('#tour-target-model', 'Model step'),
        tourStep('#tour-target-goal', 'Goal step'),
        tourStep('#tour-target-missing', 'Results step'),
      ],
      vi.fn(),
    )

    expect(title()).toBe('Model step')
    expect(progress()).toBe('1 of 2')
  })

  it('marks the last step it shows as the last one', async () => {
    addTarget('tour-target-model')
    addTarget('tour-target-goal')
    await renderTour(
      [
        tourStep('#tour-target-model', 'Model step'),
        tourStep('#tour-target-goal', 'Goal step'),
        tourStep('#tour-target-missing', 'Results step'),
      ],
      vi.fn(),
    )

    await act(async () => action('Next')!.click())

    expect(title()).toBe('Goal step')
    expect(progress()).toBe('2 of 2')
    expect(action('Got it')).toBeDefined()
  })

  it('numbers a middle skip so the end matches the count', async () => {
    // The predict tour points at result tiles between "Calculate" and the
    // assumptions, so on a first visit the fourth of five steps has no target.
    addTarget('tour-target-model')
    addTarget('tour-target-goal')
    addTarget('tour-target-calculate')
    await renderTour(
      [
        tourStep('#tour-target-model', 'Model step'),
        tourStep('#tour-target-goal', 'Goal step'),
        tourStep('#tour-target-missing', 'Results step'),
        tourStep('#tour-target-calculate', 'Calculate step'),
      ],
      vi.fn(),
    )

    expect(progress()).toBe('1 of 3')
    await act(async () => action('Next')!.click())
    expect(progress()).toBe('2 of 3')
    expect(action('Next')).toBeDefined()
    await act(async () => action('Next')!.click())

    expect(title()).toBe('Calculate step')
    expect(progress()).toBe('3 of 3')
    expect(action('Got it')).toBeDefined()
  })
})
