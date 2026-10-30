'use client'

import { useEffect, useState } from 'react'
import { fetchInfrastructureOffers, type InfrastructureOffers } from '@/lib/api/infrastructure-offers'
import { CLIENT_CACHE_TTL_MS } from '@/lib/cache-policy'

interface OffersState extends InfrastructureOffers {
  isLoading: boolean
  error: string | null
  stale: boolean
}
const EMPTY: InfrastructureOffers = { rented: [], purchased: [], sources: {}, rejectedRecords: 0 }
let cached: InfrastructureOffers | null = null
let cachedAt = 0
let inFlight: Promise<InfrastructureOffers> | null = null

function fetchOnce() {
  if (!inFlight) {
    inFlight = fetchInfrastructureOffers().then(data => {
      cached = data
      cachedAt = Date.now()
      return data
    }).finally(() => { inFlight = null })
  }
  return inFlight
}

/** Separate from useCostings: failure cannot poison other pages' shared cache. */
export function useInfrastructureOffers(enabled: boolean): OffersState {
  const [state, setState] = useState<OffersState>({ ...(enabled && cached ? cached : EMPTY), isLoading: enabled, error: null, stale: false })
  useEffect(() => {
    if (!enabled) { setState({ ...EMPTY, isLoading: false, error: null, stale: false }); return }
    let active = true
    let lastAttempt = 0
    const refresh = () => {
      // Retry degraded or newly configured sources promptly; healthy feeds keep
      // the normal TTL. Backend source status is separate from transport errors.
      const ttl = cached && Object.values(cached.sources).some(source => ['stale', 'unavailable', 'unconfigured'].includes(source.state))
        ? 60_000 : CLIENT_CACHE_TTL_MS.costings
      if (Date.now() - cachedAt < ttl && cached) {
        setState({ ...cached, isLoading: false, error: null, stale: false })
        return
      }
      if (lastAttempt && Date.now() - lastAttempt < 60_000) return
      lastAttempt = Date.now()
      setState(previous => ({ ...previous, isLoading: !cached, error: null }))
      void fetchOnce().then(data => {
        if (active) setState({ ...data, isLoading: false, error: null, stale: false })
      }).catch((error: unknown) => {
        if (active) setState(previous => ({ ...(cached ?? previous), isLoading: false, stale: cached !== null,
          error: error instanceof Error ? error.message : 'Infrastructure pricing unavailable.' }))
      })
    }
    const revalidate = () => { if (document.visibilityState !== 'hidden') refresh() }
    refresh()
    const timer = setInterval(revalidate, 60_000)
    window.addEventListener('focus', revalidate)
    document.addEventListener('visibilitychange', revalidate)
    return () => {
      active = false; clearInterval(timer)
      window.removeEventListener('focus', revalidate)
      document.removeEventListener('visibilitychange', revalidate)
    }
  }, [enabled])
  return state
}
