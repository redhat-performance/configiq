'use client'

import { useCallback, useEffect, useState } from 'react'
import type { ScorecardPoint, ScorecardSummary } from '@/lib/scorecard/types'

/**
 * Load the published scorecard through the same-origin proxy.
 *
 * The artifact is produced by a scheduled sweep, so this hook only reads. The
 * summary is enough for the default view. A row filter asks for the point
 * detail, which the same artifact already publishes, so the page can fold the
 * filtered population with the sweep's own per-configuration rule instead of
 * presenting pooled totals under a narrower table.
 */

export interface ScorecardState {
  summary: ScorecardSummary | null
  /** Published point detail, fetched the first time a row filter needs it. */
  points: ScorecardPoint[] | null
  pointsLoading: boolean
  pointsError: string | null
  /** Set when no snapshot is published yet, so the page can say so plainly. */
  notPublished: boolean
  error: string | null
  isLoading: boolean
}

const EMPTY: ScorecardState = {
  summary: null,
  points: null,
  pointsLoading: false,
  pointsError: null,
  notPublished: false,
  error: null,
  isLoading: true,
}

interface HookState extends ScorecardState {
  /** `generatedAt` of the snapshot the loaded points belong to. */
  pointsFor: string | null
}

const INITIAL: HookState = { ...EMPTY, pointsFor: null }

function errorMessage(payload: unknown, status: number): string {
  const record = payload && typeof payload === 'object' ? payload as Record<string, unknown> : null
  const error = record?.error && typeof record.error === 'object'
    ? record.error as Record<string, unknown>
    : null
  return typeof error?.message === 'string'
    ? error.message
    : `The scorecard request failed with HTTP ${status}.`
}

function summaryFrom(payload: unknown): ScorecardSummary | null {
  const record = payload && typeof payload === 'object' ? payload as Record<string, unknown> : null
  const summary = record?.summary
  return summary && typeof summary === 'object' ? summary as ScorecardSummary : null
}

export function useScorecard(): ScorecardState & { reload: () => void; loadPoints: () => void } {
  const [state, setState] = useState<HookState>(INITIAL)
  const [nonce, setNonce] = useState(0)
  const [pointsWanted, setPointsWanted] = useState(false)

  useEffect(() => {
    const controller = new AbortController()
    let active = true

    async function load() {
      setState(previous => ({ ...previous, isLoading: true, error: null }))
      try {
        const response = await fetch('/api/validation/scorecard', {
          headers: { Accept: 'application/json' },
          signal: controller.signal,
        })
        let payload: unknown = null
        try {
          payload = await response.json()
        } catch {
          payload = null
        }
        if (!active) return

        if (response.status === 404) {
          setState(previous => ({
            ...previous,
            summary: null,
            points: null,
            pointsFor: null,
            notPublished: true,
            error: null,
            isLoading: false,
          }))
          return
        }
        if (!response.ok) {
          setState(previous => ({
            ...previous,
            summary: null,
            points: null,
            pointsFor: null,
            notPublished: false,
            error: errorMessage(payload, response.status),
            isLoading: false,
          }))
          return
        }

        const summary = summaryFrom(payload)
        if (!summary) {
          setState(previous => ({
            ...previous,
            summary: null,
            points: null,
            pointsFor: null,
            notPublished: false,
            error: 'The scorecard response had no summary.',
            isLoading: false,
          }))
          return
        }
        setState(previous => {
          // A new snapshot invalidates the point detail before the next read
          // replaces it, so filtered numbers are never folded from two runs.
          const sameSnapshot = previous.summary?.generatedAt === summary.generatedAt
          return {
            ...previous,
            summary,
            points: sameSnapshot ? previous.points : null,
            pointsFor: sameSnapshot ? previous.pointsFor : null,
            notPublished: false,
            error: null,
            isLoading: false,
          }
        })
      } catch (error: unknown) {
        if (!active || controller.signal.aborted) return
        setState(previous => ({
          ...previous,
          summary: null,
          points: null,
          pointsFor: null,
          notPublished: false,
          error: error instanceof Error ? error.message : 'The scorecard could not be loaded.',
          isLoading: false,
        }))
      }
    }

    void load()
    return () => {
      active = false
      controller.abort()
    }
  }, [nonce])

  useEffect(() => {
    if (!pointsWanted || !state.summary) return
    // The default view never needs the point detail. Once a filter has asked
    // for it, keep the copy for this snapshot rather than fetching per change.
    if (state.points && state.pointsFor === state.summary.generatedAt) return

    const controller = new AbortController()
    let active = true

    async function load() {
      setState(previous => ({ ...previous, pointsLoading: true, pointsError: null }))
      try {
        const response = await fetch('/api/validation/scorecard?points=1', {
          headers: { Accept: 'application/json' },
          signal: controller.signal,
        })
        let payload: unknown = null
        try {
          payload = await response.json()
        } catch {
          payload = null
        }
        if (!active) return

        if (!response.ok) {
          setState(previous => ({
            ...previous,
            pointsLoading: false,
            pointsError: errorMessage(payload, response.status),
          }))
          return
        }

        const summary = summaryFrom(payload)
        const record = payload && typeof payload === 'object' ? payload as Record<string, unknown> : null
        const points = Array.isArray(record?.points) ? record.points as ScorecardPoint[] : null
        if (!summary || !points) {
          setState(previous => ({
            ...previous,
            pointsLoading: false,
            pointsError: 'The scorecard response had no point detail.',
          }))
          return
        }

        // The route reads both files from one snapshot and rejects a mixed
        // pair, so replacing the summary here keeps the read consistent even
        // if a new snapshot landed between the two requests.
        setState(previous => ({
          ...previous,
          summary,
          points,
          pointsFor: summary.generatedAt,
          pointsLoading: false,
          pointsError: null,
        }))
      } catch (error: unknown) {
        if (!active || controller.signal.aborted) return
        setState(previous => ({
          ...previous,
          pointsLoading: false,
          pointsError: error instanceof Error ? error.message : 'The point detail could not be loaded.',
        }))
      }
    }

    void load()
    return () => {
      active = false
      controller.abort()
    }
  }, [pointsWanted, nonce, state.summary, state.points, state.pointsFor])

  const loadPoints = useCallback(() => setPointsWanted(true), [])

  return {
    summary: state.summary,
    points: state.points,
    pointsLoading: state.pointsLoading,
    pointsError: state.pointsError,
    notPublished: state.notPublished,
    error: state.error,
    isLoading: state.isLoading,
    reload: () => setNonce(value => value + 1),
    loadPoints,
  }
}
