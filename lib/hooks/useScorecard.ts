'use client'

import { useEffect, useState } from 'react'
import type { ScorecardSummary } from '@/lib/scorecard/types'

/**
 * Load the published scorecard through the same-origin proxy.
 *
 * The artifact is produced by a scheduled sweep, so this hook only reads. It
 * never asks the browser to compute a statistic: everything it renders was
 * computed once, against one simulator release, and published with provenance.
 */

export interface ScorecardState {
  summary: ScorecardSummary | null
  /** Set when no snapshot is published yet, so the page can say so plainly. */
  notPublished: boolean
  error: string | null
  isLoading: boolean
}

const EMPTY: ScorecardState = { summary: null, notPublished: false, error: null, isLoading: true }

export function useScorecard(): ScorecardState & { reload: () => void } {
  const [state, setState] = useState<ScorecardState>(EMPTY)
  const [nonce, setNonce] = useState(0)

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
          setState({ summary: null, notPublished: true, error: null, isLoading: false })
          return
        }
        if (!response.ok) {
          const record = payload && typeof payload === 'object' ? payload as Record<string, unknown> : null
          const error = record?.error && typeof record.error === 'object'
            ? record.error as Record<string, unknown>
            : null
          setState({
            summary: null,
            notPublished: false,
            error: typeof error?.message === 'string'
              ? error.message
              : `The scorecard request failed with HTTP ${response.status}.`,
            isLoading: false,
          })
          return
        }

        const record = payload && typeof payload === 'object' ? payload as Record<string, unknown> : null
        const summary = record?.summary
        if (!summary || typeof summary !== 'object') {
          setState({ summary: null, notPublished: false, error: 'The scorecard response had no summary.', isLoading: false })
          return
        }
        setState({ summary: summary as ScorecardSummary, notPublished: false, error: null, isLoading: false })
      } catch (error: unknown) {
        if (!active || controller.signal.aborted) return
        setState({
          summary: null,
          notPublished: false,
          error: error instanceof Error ? error.message : 'The scorecard could not be loaded.',
          isLoading: false,
        })
      }
    }

    void load()
    return () => {
      active = false
      controller.abort()
    }
  }, [nonce])

  return { ...state, reload: () => setNonce(value => value + 1) }
}
