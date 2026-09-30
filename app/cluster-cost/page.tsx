'use client'

import React, { useState, useEffect, useRef, useMemo, useCallback } from 'react'
import ComingSoonRibbon from '@/components/ComingSoonRibbon/ComingSoonRibbon'
import styles from './cluster-cost.module.css'
import { fetchAllProviders, getEffectiveRate, loadUserOverrides, setUserOverride, clearUserOverride, loadSelectedGpus, saveSelectedGpu, type Provider } from '@/lib/pricing/providerPricing'
import { useCostings, resolveCloudRate, type CostingsData } from '@/lib/hooks/useCostings'
import { useSettings } from '@/contexts/SettingsContext'

interface GpuCatalogEntry {
  id: string
  label: string
  rate: number
  price: number
  tdpW: number
  mem: number
}

// GPU Catalog - matches reference exactly. rate (cloud $/hr) and price (hardware
// $) are the built-in fallbacks; when costings is enabled they are overridden
// per-GPU from live data (see buildEffectiveCatalog).
const GPU_CATALOG: GpuCatalogEntry[] = [
  { id: 'h200',    label: 'H200 141GB',   rate: 4.50, price: 42000, tdpW: 1000, mem: 141 },
  { id: 'h100',    label: 'H100 80GB',    rate: 3.89, price: 32000, tdpW: 700,  mem: 80  },
  { id: 'a100_80', label: 'A100 80GB',    rate: 3.20, price: 14000, tdpW: 400,  mem: 80  },
  { id: 'a100_40', label: 'A100 40GB',    rate: 2.40, price: 11000, tdpW: 300,  mem: 40  },
  { id: 'l40s',    label: 'L40S 48GB',    rate: 2.60, price: 8500,  tdpW: 350,  mem: 48  },
  { id: 'mi300x',  label: 'MI300X 192GB', rate: 4.20, price: 38000, tdpW: 750,  mem: 192 },
]

// Maps a cluster-cost catalog id to the aicostings/aisimulators systemId used
// by useCostings. Only ids with a confident 1:1 match are listed; a GPU with no
// entry (a100_40 40GB, mi300x — not in the costings hardware seed) keeps its
// built-in fallback values.
const CATALOG_TO_SYSTEM_ID: Record<string, string> = {
  h200:    'h200_sxm',
  h100:    'h100_sxm',
  a100_80: 'a100_sxm',
  l40s:    'l40s',
}

// Overlay live hardware costs and cloud rates onto the built-in catalog. Any
// field with no live data falls back to the hardcoded value, so the page still
// works when costings is disabled or a GPU/rate is missing. Cloud rate uses the
// shared resolver (preferred provider → cheapest on-demand → cheapest spot).
function buildEffectiveCatalog(
  costings: CostingsData,
  enabled: boolean,
  preferredCloudProvider: string | null,
): GpuCatalogEntry[] {
  if (!enabled) return GPU_CATALOG
  return GPU_CATALOG.map(g => {
    const sysId = CATALOG_TO_SYSTEM_ID[g.id]
    if (!sysId) return g
    const hw = costings.gpuHardwareCosts.get(sysId)?.new_usd
    const rate = resolveCloudRate(costings.gpuCloudRates.get(sysId), preferredCloudProvider)?.rate ?? null
    return {
      ...g,
      price: hw != null ? hw : g.price,
      rate: rate != null ? rate : g.rate,
    }
  })
}

// Backend defaults - all rates come from here
const BACKEND_DEFAULTS = {
  cloud: {
    storHot: 165,
    storWarm: 100,
    storCold: 25,
    egress: 90,
    ctrlBase: 2000,
    ctrlPerGpu: 14,
    supFrac: 0.09,
    opsFix: 1000,
    opsDbg: 1200,
    opsDbgGpu: 9,
  },
  onprem: {
    deprYrs: 5,
    powerKwh: 0.10,
    pue: 1.25,
    coloKwMo: 250,
    netCapexPerGpu: 12000,
    netAmortYrs: 3,
    staffAnnual: 200000,
    staffPerFte: 2.5,
    supFrac: 0.18,
    storHot: 150,
    storWarm: 60,
    storCold: 15,
    ctrlBase: 300,
    ctrlPerGpu: 10,
  },
  dr: {
    cloudFactor: 0.35,
    onpremFactor: 0.80,
  },
  licenses: [] as Array<{ name: string; amount: number; cycle: 'yr' | 'mo' }>,
}

// Layer metadata
const LAYER_META: Record<string, { c: string; label: string; order: number }> = {
  gpu:      { c: '#ee0000', label: 'GPU Compute',   order: 0 },
  power:    { c: '#94a3b8', label: 'Power & Colo',  order: 1 },
  network:  { c: '#0066cc', label: 'Network',        order: 2 },
  control:  { c: '#7c3aed', label: 'Control Plane',  order: 3 },
  ops:      { c: '#10b981', label: 'Support & Ops',  order: 4 },
  storage:  { c: '#f59e0b', label: 'Storage',        order: 5 },
  software: { c: '#0d9488', label: 'Software',       order: 6 },
}

const CLOUD_ORDER = ['gpu', 'network', 'control', 'ops', 'storage', 'software']
const ONPREM_ORDER = ['gpu', 'power', 'network', 'control', 'ops', 'storage', 'software']

interface Cluster {
  id: number
  name: string
  open: boolean
  nodeGroups: Array<{ gpuType: string; count: number }>
  hotTB: number
  warmTB: number
  coldTB: number
  networkEgress: number
}

interface License {
  id: number
  name: string
  amount: number
  cycle: 'yr' | 'mo'
  scope: 'both' | 'cloud' | 'onprem'
}

interface CalcResult {
  total: number
  totalGPUs: number
  effHr: number
  rawHr: number
  mult: number
  breakdown: Array<{ id: string; label: string; value: number; pct: number }>
}

// Cloud calculation - NO utilization in formula
function calcCloud(
  clusters: Cluster[],
  licMonthly: number,
  dr: boolean,
  rates: typeof BACKEND_DEFAULTS,
  activeGpuRate = 0,
  catalog: GpuCatalogEntry[] = GPU_CATALOG,
  // When true (costings enabled), each node group is priced at its own live
  // per-GPU rate from the catalog rather than the single provider-picker rate,
  // so mixed-GPU clusters price correctly and live rates actually drive the total.
  preferCatalogRate = false
): CalcResult {
  let gpuCostTotal = 0
  let totalGPUs = 0
  let storTotal = 0
  let netTotal = 0
  let ctrlTotal = 0
  let opsTotal = 0

  const r = rates.cloud

  clusters.forEach(cl => {
    let gpuCost = 0
    let ngpus = 0
    cl.nodeGroups.forEach(ng => {
      const g = catalog.find(x => x.id === ng.gpuType) || catalog[1]
      const rate = preferCatalogRate ? g.rate : (activeGpuRate > 0 ? activeGpuRate : g.rate)
      gpuCost += ng.count * rate * 720
      ngpus += ng.count
    })
    const stor = cl.hotTB * r.storHot + cl.warmTB * r.storWarm + cl.coldTB * r.storCold
    const net = cl.networkEgress * r.egress
    const ctrl = r.ctrlBase + Math.max(0, (ngpus - 8) * r.ctrlPerGpu)
    const ops = gpuCost * r.supFrac + r.opsFix + r.opsDbg + ngpus * r.opsDbgGpu

    gpuCostTotal += gpuCost
    storTotal += stor
    netTotal += net
    ctrlTotal += ctrl
    opsTotal += ops
    totalGPUs += ngpus
  })

  const base = gpuCostTotal + storTotal + netTotal + ctrlTotal + opsTotal + licMonthly
  const total = dr ? base * (1 + BACKEND_DEFAULTS.dr.cloudFactor) : base

  const avgRate = totalGPUs > 0
    ? clusters.reduce((s, cl) =>
        s + cl.nodeGroups.reduce((ss, ng) => {
          const g = catalog.find(x => x.id === ng.gpuType) || catalog[1]
          return ss + ng.count * g.rate
        }, 0), 0) / totalGPUs
    : 0

  const effHr = totalGPUs > 0 ? total / (totalGPUs * 720) : 0

  const breakdown = [
    { id: 'gpu',     label: 'GPU Compute',   value: gpuCostTotal, pct: gpuCostTotal / total * 100 },
    { id: 'storage', label: 'Storage',       value: storTotal,    pct: storTotal / total * 100 },
    { id: 'ops',     label: 'Support & Ops', value: opsTotal,     pct: opsTotal / total * 100 },
    { id: 'control', label: 'Control Plane', value: ctrlTotal,    pct: ctrlTotal / total * 100 },
    { id: 'network', label: 'Network',       value: netTotal,     pct: netTotal / total * 100 },
  ]

  if (licMonthly > 0) {
    breakdown.push({ id: 'software', label: 'Software', value: licMonthly, pct: licMonthly / total * 100 })
  }

  return {
    total,
    totalGPUs,
    effHr,
    rawHr: avgRate,
    mult: avgRate > 0 ? effHr / avgRate : 0,
    breakdown,
  }
}

// On-prem calculation - NO utilization in formula
function calcOnPrem(
  clusters: Cluster[],
  licMonthly: number,
  dr: boolean,
  rates: typeof BACKEND_DEFAULTS,
  catalog: GpuCatalogEntry[] = GPU_CATALOG
): CalcResult {
  let gpuAmortTotal = 0
  let powerTotal = 0
  let coloTotal = 0
  let netTotal = 0
  let storTotal = 0
  let opsTotal = 0
  let ctrlTotal = 0
  let totalGPUs = 0
  let gpuCapexTotal = 0

  const r = rates.onprem

  clusters.forEach(cl => {
    let gpuAmort = 0
    let gpuCapex = 0
    let gpuPow = 0
    let ngpus = 0

    cl.nodeGroups.forEach(ng => {
      const g = catalog.find(x => x.id === ng.gpuType) || catalog[1]
      gpuAmort += ng.count * g.price / (r.deprYrs * 12)
      gpuCapex += ng.count * g.price
      gpuPow += ng.count * g.tdpW
      ngpus += ng.count
    })

    const power = gpuPow / 1000 * 720 * r.powerKwh * r.pue
    const colo = gpuPow / 1000 * r.pue * r.coloKwMo
    const net = ngpus * r.netCapexPerGpu / (r.netAmortYrs * 12)
    const stor = cl.hotTB * r.storHot + cl.warmTB * r.storWarm + cl.coldTB * r.storCold
    const staff = r.staffAnnual / r.staffPerFte / 12
    const sup = gpuCapex * r.supFrac / 12
    const ops = staff + sup
    const ctrl = r.ctrlBase + ngpus * r.ctrlPerGpu

    gpuAmortTotal += gpuAmort
    powerTotal += power
    coloTotal += colo
    netTotal += net
    storTotal += stor
    opsTotal += ops
    ctrlTotal += ctrl
    totalGPUs += ngpus
    gpuCapexTotal += gpuCapex
  })

  const base = gpuAmortTotal + powerTotal + coloTotal + netTotal + storTotal + opsTotal + ctrlTotal + licMonthly
  const total = dr ? base * (1 + BACKEND_DEFAULTS.dr.onpremFactor) : base

  const avgHwPrice = totalGPUs > 0 ? gpuCapexTotal / totalGPUs : 0
  const rawHr = avgHwPrice > 0 ? avgHwPrice / (rates.onprem.deprYrs * 8760) : 0
  const effHr = totalGPUs > 0 ? total / (totalGPUs * 720) : 0

  const breakdown = [
    { id: 'gpu',     label: 'GPU Amort',     value: gpuAmortTotal,        pct: gpuAmortTotal / total * 100 },
    { id: 'power',   label: 'Power & Colo',  value: powerTotal + coloTotal, pct: (powerTotal + coloTotal) / total * 100 },
    { id: 'storage', label: 'Storage',       value: storTotal,            pct: storTotal / total * 100 },
    { id: 'ops',     label: 'Support & Ops', value: opsTotal,             pct: opsTotal / total * 100 },
    { id: 'control', label: 'Control Plane', value: ctrlTotal,            pct: ctrlTotal / total * 100 },
    { id: 'network', label: 'Network',       value: netTotal,             pct: netTotal / total * 100 },
  ]

  if (licMonthly > 0) {
    breakdown.push({ id: 'software', label: 'Software', value: licMonthly, pct: licMonthly / total * 100 })
  }

  return {
    total,
    totalGPUs,
    effHr,
    rawHr,
    mult: rawHr > 0 ? effHr / rawHr : 0,
    breakdown,
  }
}

// Count-up animation component
function CountUp({ target, dec = 0 }: { target: number; dec?: number }) {
  const [v, setV] = useState(target)
  const fr = useRef(target)

  useEffect(() => {
    const from = fr.current
    fr.current = target
    if (Math.abs(from - target) < 0.5) {
      setV(target)
      return
    }

    const t0 = performance.now()
    let raf: number

    const tick = (t: number) => {
      const p = Math.min((t - t0) / 700, 1)
      const e = 1 - Math.pow(1 - p, 4) // easeOutQuart
      setV(from + (target - from) * e)
      if (p < 1) raf = requestAnimationFrame(tick)
    }

    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [target])

  return <span>{v.toLocaleString('en-US', { minimumFractionDigits: dec, maximumFractionDigits: dec })}</span>
}

// Particle animation hook
function useParticles(ref: React.RefObject<HTMLCanvasElement>, breakdown: CalcResult['breakdown']) {
  useEffect(() => {
    const canvas = ref.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const resize = () => {
      canvas.width = canvas.offsetWidth || 200
      canvas.height = canvas.offsetHeight || 300
    }
    resize()

    const pts = Array.from({ length: 16 }, () => ({
      x: Math.random() * (canvas.width || 200),
      y: Math.random() * (canvas.height || 300),
      vy: -(0.55 + Math.random() * 0.85),
      vx: (Math.random() - 0.5) * 0.18,
      r: 1.3 + Math.random() * 1.5,
      op: 0.22 + Math.random() * 0.3,
    }))

    const getColor = (y: number, h: number) => {
      const order = [...breakdown].sort((a, b) => (LAYER_META[a.id]?.order ?? 9) - (LAYER_META[b.id]?.order ?? 9))
      let rem = h
      for (const s of order) {
        const sh = (s.pct / 100) * h
        rem -= sh
        if (y >= rem) return LAYER_META[s.id]?.c || '#151515'
      }
      return '#ee0000'
    }

    let raf: number
    const draw = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height)
      pts.forEach(p => {
        p.y += p.vy
        p.x += p.vx
        if (p.x < 0) p.x = canvas.width
        if (p.x > canvas.width) p.x = 0
        if (p.y < -6) {
          p.y = canvas.height + 4
          p.x = Math.random() * canvas.width
        }
        const col = getColor(p.y, canvas.height)
        ctx.save()
        ctx.globalAlpha = p.op
        ctx.shadowBlur = 6
        ctx.shadowColor = col
        ctx.fillStyle = col
        ctx.beginPath()
        ctx.arc(p.x, p.y, p.r, 0, 2 * Math.PI)
        ctx.fill()
        ctx.restore()
      })
      raf = requestAnimationFrame(draw)
    }
    draw()

    const ro = new ResizeObserver(resize)
    ro.observe(canvas.parentElement || canvas)

    return () => {
      cancelAnimationFrame(raf)
      ro.disconnect()
    }
  }, [breakdown, ref])
}

// Compact layer visualization - horizontal bar + rows
function LayerViz({
  breakdown,
  stackOrder,
  canvasRef,
}: {
  breakdown: CalcResult['breakdown']
  stackOrder: string[]
  canvasRef: React.RefObject<HTMLCanvasElement>
}) {
  const sorted = useMemo(
    () => stackOrder.map(id => breakdown.find(b => b.id === id)).filter(Boolean) as typeof breakdown,
    [breakdown, stackOrder]
  )

  const maxPct = Math.max(...sorted.map(s => s.pct), 1)

  return (
    <div className={styles.stackInner}>
      <canvas ref={canvasRef} className={styles.pcanvas} />

      {/* Horizontal stacked bar */}
      <div className={styles.hBarWrap}>
        <div className={styles.hBar}>
          {sorted.map(seg => {
            const m = LAYER_META[seg.id] || LAYER_META.gpu
            return (
              <div
                key={seg.id}
                className={styles.hSeg}
                style={{ flex: Math.max(seg.pct, 0.5), background: m.c }}
                title={`${m.label}: ${seg.pct.toFixed(1)}% · $${(seg.value / 1000).toFixed(1)}K`}
              />
            )
          })}
        </div>
        <div className={styles.hBarLegend}>
          {sorted.map(seg => {
            const m = LAYER_META[seg.id] || LAYER_META.gpu
            return (
              <span key={seg.id} className={styles.hblItem}>
                <span className={styles.hblDot} style={{ background: m.c }} />
                {m.label}
              </span>
            )
          })}
        </div>
      </div>

      {/* Compact rows */}
      <div className={styles.layerRows}>
        {sorted.map(seg => {
          const m = LAYER_META[seg.id] || LAYER_META.gpu
          return (
            <div key={seg.id} className={styles.lr}>
              <div className={styles.lrName}>
                <span className={styles.lrDot} style={{ background: m.c }} />
                <span className={styles.lrText}>{m.label}</span>
              </div>
              <div className={styles.lrBarWrap}>
                <div className={styles.lrBarFill} style={{ width: `${(seg.pct / maxPct) * 100}%`, background: m.c }} />
              </div>
              <span className={styles.lrVal} style={{ color: m.c }}>
                ${(seg.value / 1000).toFixed(1)}K
              </span>
              <span className={styles.lrPct}>{seg.pct.toFixed(1)}%</span>
            </div>
          )
        })}
      </div>
    </div>
  )
}

// Main component
export default function ClusterCostPage() {
  const { costingsEnabled, preferredCloudProvider, pricingSource } = useSettings()
  const costings = useCostings(costingsEnabled, pricingSource)

  const DEFAULT_CLUSTER = (): Cluster => ({
    id: Date.now(),
    name: 'Cluster 1',
    open: true,
    nodeGroups: [{ gpuType: 'h100', count: 8 }],
    hotTB: 10,
    warmTB: 50,
    coldTB: 100,
    networkEgress: 5,
  })

  const [clusters, setClusters] = useState<Cluster[]>([DEFAULT_CLUSTER()])
  const [licenses, setLicenses] = useState<License[]>([])
  const [dr, setDr] = useState(false)
  const [viewMode, setViewMode] = useState<'cloud' | 'onprem' | 'both'>('cloud')
  const [load, setLoad] = useState(75)
  const [ratesOpen, setRatesOpen] = useState(false)
  const [rates, setRates] = useState(JSON.parse(JSON.stringify(BACKEND_DEFAULTS)))
  const [toast, setToast] = useState<string | null>(null)

  // Cloud provider pricing state
  const [providers, setProviders] = useState<Provider[]>([])
  const [activeProviderId, setActiveProviderId] = useState('gcp')
  const [selectedGpus, setSelectedGpus] = useState<Record<string, string>>({})
  const [priceOverrides, setPriceOverrides] = useState<Record<string, number | undefined>>({})
  const [providerOpen, setProviderOpen] = useState(false)
  const [providerError, setProviderError] = useState<string | null>(null)
  const [providerLoading, setProviderLoading] = useState(true)

  // Panel resize state with localStorage persistence
  const [cols, setCols] = useState<[number, number]>(() => {
    if (typeof window !== 'undefined') {
      const stored = localStorage.getItem('cluster-cost-col-widths')
      if (stored) {
        try {
          const parsed = JSON.parse(stored)
          return [parsed[0] || 320, parsed[1] || 420]
        } catch {}
      }
    }
    return [320, 420]
  })

  const contentRef = useRef<HTMLDivElement>(null)
  const dragRef = useRef<{ col: number; startX: number; startW: number } | null>(null)
  const cloudRef = useRef<HTMLCanvasElement>(null)
  const onpremRef = useRef<HTMLCanvasElement>(null)

  // Persist column widths to localStorage
  useEffect(() => {
    localStorage.setItem('cluster-cost-col-widths', JSON.stringify(cols))
  }, [cols])

  // Load cloud providers on mount
  useEffect(() => {
    loadProviders()
  }, [])

  const loadProviders = async () => {
    setProviderLoading(true)
    setProviderError(null)
    try {
      const data = await fetchAllProviders()
      setProviders(data)
      setSelectedGpus(loadSelectedGpus(data))
      setPriceOverrides(loadUserOverrides())
      // Check if we're using fallback (first provider is 'gcp' with exactly those GPUs)
      const isFallback = data.length > 0 && data[0]?.id === 'gcp' && data[0]?.gpus?.length === 4
      if (isFallback) {
        setProviderError('Could not reach aicostings — live rates unavailable')
      }
    } catch (error) {
      setProviderError('Failed to load provider data')
    } finally {
      setProviderLoading(false)
    }
  }

  const startDrag = useCallback((colIndex: number, e: React.MouseEvent) => {
    e.preventDefault()
    // Fix: cols is [leftWidth, rightWidth], center is flex:1
    // colIndex 0 = left handle, colIndex 1 = right handle
    const startW = colIndex === 0 ? cols[0] : cols[1]
    dragRef.current = { col: colIndex, startX: e.clientX, startW }
    document.body.style.cursor = 'col-resize'
    document.body.style.userSelect = 'none'

    const onMove = (ev: MouseEvent) => {
      if (!dragRef.current) return
      const { col, startX, startW } = dragRef.current
      const delta = ev.clientX - startX

      setCols(prev => {
        const next: [number, number] = [...prev]
        if (col === 0) {
          next[0] = Math.max(220, Math.min(600, startW + delta))
        } else {
          next[1] = Math.max(280, Math.min(700, startW - delta))
        }
        return next
      })
    }

    const onUp = () => {
      document.body.style.cursor = ''
      document.body.style.userSelect = ''
      dragRef.current = null
      window.removeEventListener('mousemove', onMove)
      window.removeEventListener('mouseup', onUp)
    }

    window.addEventListener('mousemove', onMove)
    window.addEventListener('mouseup', onUp)
  }, [cols])

  // Calculate license costs split by scope
  const licCloud = useMemo(
    () =>
      licenses
        .filter(l => l.scope !== 'onprem')
        .reduce((s, l) => {
          const annual = l.cycle === 'yr' ? l.amount * 12 : l.amount
          return s + (isNaN(annual) ? 0 : annual / 12)
        }, 0),
    [licenses]
  )

  const licOnPrem = useMemo(
    () =>
      licenses
        .filter(l => l.scope !== 'cloud')
        .reduce((s, l) => {
          const annual = l.cycle === 'yr' ? l.amount * 12 : l.amount
          return s + (isNaN(annual) ? 0 : annual / 12)
        }, 0),
    [licenses]
  )

  const licMonthly = licCloud + licOnPrem

  // Calculate active GPU rate from provider pricing
  const activeProvider = providers.find(p => p.id === activeProviderId) || providers[0]
  const activeGpu = selectedGpus[activeProviderId] || activeProvider?.gpus[0]?.model || ''
  const activeGpuRate = activeProvider ? getEffectiveRate(activeProviderId, activeGpu, providers, priceOverrides) : null

  // Live hardware/cloud pricing overlays the built-in catalog when costings is on.
  const effectiveCatalog = useMemo(
    () => buildEffectiveCatalog(costings, costingsEnabled, preferredCloudProvider),
    [costings, costingsEnabled, preferredCloudProvider]
  )

  const cloud = useMemo(
    () => calcCloud(clusters, licCloud, dr, rates, activeGpuRate || 0, effectiveCatalog, costingsEnabled),
    [clusters, licCloud, dr, rates, activeGpuRate, effectiveCatalog, costingsEnabled]
  )
  const onprem = useMemo(
    () => calcOnPrem(clusters, licOnPrem, dr, rates, effectiveCatalog),
    [clusters, licOnPrem, dr, rates, effectiveCatalog]
  )

  useParticles(cloudRef, cloud.breakdown)
  useParticles(onpremRef, onprem.breakdown)

  const totalGPUs = clusters.reduce((s, cl) => s + cl.nodeGroups.reduce((ss, ng) => ss + ng.count, 0), 0)

  const showToast = (msg: string) => {
    setToast(msg)
    setTimeout(() => setToast(null), 2800)
  }

  const upCluster = useCallback(
    (id: number, fn: (c: Cluster) => Cluster) =>
      setClusters(cs => cs.map(c => (c.id === id ? fn(c) : c))),
    []
  )

  const addCluster = () =>
    setClusters(cs => [
      ...cs,
      { ...DEFAULT_CLUSTER(), id: Date.now(), name: `Cluster ${cs.length + 1}`, open: true },
    ])

  const removeCluster = (id: number) => setClusters(cs => (cs.length > 1 ? cs.filter(c => c.id !== id) : cs))

  const upNG = (cid: number, i: number, k: keyof Cluster['nodeGroups'][0], v: any) =>
    upCluster(cid, c => {
      const ngs = [...c.nodeGroups]
      ngs[i] = { ...ngs[i], [k]: v }
      return { ...c, nodeGroups: ngs }
    })

  const addNG = (cid: number) =>
    upCluster(cid, c => ({ ...c, nodeGroups: [...c.nodeGroups, { gpuType: 'h100', count: 4 }] }))

  const removeNG = (cid: number, i: number) =>
    upCluster(cid, c => ({
      ...c,
      nodeGroups: c.nodeGroups.length > 1 ? c.nodeGroups.filter((_, j) => j !== i) : c.nodeGroups,
    }))

  const addLicense = () => setLicenses(ls => [...ls, { id: Date.now(), name: '', amount: 0, cycle: 'yr', scope: 'both' }])
  const upLic = (id: number, k: keyof License, v: any) => setLicenses(ls => ls.map(l => (l.id === id ? { ...l, [k]: v } : l)))
  const rmLic = (id: number) => setLicenses(ls => ls.filter(l => l.id !== id))

  const upRate = (side: 'cloud' | 'onprem', k: string, v: number) =>
    setRates((r: any) => ({ ...r, [side]: { ...r[side], [k]: v } }))

  const isOv = (side: 'cloud' | 'onprem', k: string) => (rates as any)[side][k] !== (BACKEND_DEFAULTS as any)[side][k]

  const rstRate = (side: 'cloud' | 'onprem', k: string) =>
    setRates((r: any) => ({ ...r, [side]: { ...r[side], [k]: (BACKEND_DEFAULTS as any)[side][k] } }))

  const capStatus =
    load < 50
      ? { cls: 'light', msg: `💡 Lightly loaded (${load}%) — may be over-provisioned` }
      : load < 90
      ? { cls: 'ok', msg: `✓ Healthy load (${load}%)` }
      : { cls: 'hot', msg: `⚠ Near capacity (${load}%) — consider adding GPUs` }

  const saving = cloud.total - onprem.total
  const onpremCheaper = saving > 0

  const maxFive = Math.max(cloud.total * 60, onprem.total * 60, 1)
  const allIds = ['gpu', 'storage', 'ops', 'control', 'network', ...(licMonthly > 0 ? ['software'] : [])]

  const copySheets = () => {
    const rows: string[][] = []

    // Section 1: Summary
    rows.push(['Metric', 'Cloud', 'Self-hosted'])
    rows.push(['Monthly', '$' + Math.round(cloud.total), '$' + Math.round(onprem.total)])
    rows.push(['Annual', '$' + Math.round(cloud.total * 12), '$' + Math.round(onprem.total * 12)])
    rows.push(['5-year', '$' + Math.round(cloud.total * 60), '$' + Math.round(onprem.total * 60)])
    rows.push(['Eff gpu-hr', '$' + cloud.effHr.toFixed(2), '$' + onprem.effHr.toFixed(2)])
    rows.push(['Multiplier', cloud.mult.toFixed(2) + 'x', onprem.mult.toFixed(2) + 'x'])
    if (dr) {
      rows.push(['DR included', 'Cloud +35%', 'Self-hosted +80%'])
    }
    rows.push([''])

    // Section 2: Cost breakdown (monthly)
    rows.push(['COST BREAKDOWN (monthly)', 'Cloud', 'Self-hosted'])
    const allLayerIds = ['gpu', 'power', 'storage', 'ops', 'control', 'network', 'software']
    allLayerIds.forEach(id => {
      const cloudLayer = cloud.breakdown.find(b => b.id === id)
      const onpremLayer = onprem.breakdown.find(b => b.id === id)
      const cloudVal = cloudLayer?.value || 0
      const onpremVal = onpremLayer?.value || 0
      // Skip if both are zero
      if (cloudVal === 0 && onpremVal === 0) return
      const label = LAYER_META[id]?.label || id
      rows.push([label, '$' + Math.round(cloudVal), '$' + Math.round(onpremVal)])
    })
    rows.push([''])

    // Section 3: Licenses
    if (licenses.length > 0 || licCloud > 0 || licOnPrem > 0) {
      rows.push(['LICENSES', 'Annual', 'Monthly', 'Scope'])
      licenses.forEach(l => {
        const a = l.cycle === 'yr' ? l.amount * 12 : l.amount
        const scopeLabel = l.scope === 'cloud' ? 'Cloud only' : l.scope === 'onprem' ? 'On-prem only' : 'Both'
        rows.push([l.name || '(unnamed)', '$' + Math.round(a), '$' + Math.round(a / 12), scopeLabel])
      })
      if (licCloud > 0) {
        rows.push(['☁ Cloud total', '$' + Math.round(licCloud * 12), '$' + Math.round(licCloud), ''])
      }
      if (licOnPrem > 0) {
        rows.push(['🖥 On-prem total', '$' + Math.round(licOnPrem * 12), '$' + Math.round(licOnPrem), ''])
      }
      rows.push([''])
    }

    // Section 4: Clusters (only when multiple clusters)
    if (clusters.length > 1) {
      rows.push(['CLUSTERS', 'GPUs', 'Cloud/mo', 'Self-hosted/mo'])
      clusters.forEach(cl => {
        const gpus = cl.nodeGroups.reduce((s, ng) => s + ng.count, 0)
        const cCloud = calcCloud([cl], 0, false, rates, activeGpuRate || 0, effectiveCatalog, costingsEnabled).total
        const cOp = calcOnPrem([cl], 0, false, rates, effectiveCatalog).total
        rows.push([cl.name, gpus.toString(), '$' + Math.round(cCloud), '$' + Math.round(cOp)])
      })
      rows.push(['Grand total', totalGPUs.toString(), '$' + Math.round(cloud.total), '$' + Math.round(onprem.total)])
    }

    navigator.clipboard.writeText(rows.map(r => r.join('\t')).join('\n'))
    showToast('✓ Copied — paste into Google Sheets')
  }

  const CLOUD_RATES = [
    ['storHot', 'Hot storage', '$/TB/mo'],
    ['storWarm', 'Warm storage', '$/TB/mo'],
    ['storCold', 'Cold storage', '$/TB/mo'],
    ['egress', 'Egress', '$/TB'],
    ['ctrlBase', 'Control base', '$/mo'],
    ['supFrac', 'Support', 'fraction'],
  ]

  const ONPREM_RATES = [
    ['deprYrs', 'Depreciation', 'yrs'],
    ['powerKwh', 'Power rate', '$/kWh'],
    ['pue', 'PUE', '×'],
    ['coloKwMo', 'Colo/rack', '$/kW/mo'],
    ['netCapexPerGpu', 'Networking CAPEX', '$/gpu'],
    ['staffAnnual', 'Staff loaded', '$/yr'],
    ['staffPerFte', 'Clusters/FTE', ''],
    ['supFrac', 'Support/warranty', '%/yr'],
    ['storHot', 'Storage hot', '$/TB/mo'],
    ['storWarm', 'Storage warm', '$/TB/mo'],
    ['storCold', 'Storage cold', '$/TB/mo'],
    ['ctrlBase', 'Control base', '$/mo'],
    ['ctrlPerGpu', 'Control per GPU', '$/mo/gpu'],
  ]

  return (
    <ComingSoonRibbon>
    <div style={{ padding: '16px 24px 0' }}>
      <h1 style={{ fontFamily: 'var(--display, sans-serif)', fontSize: 30, fontWeight: 700, letterSpacing: '-0.01em', margin: '8px 0 4px' }}>Cluster cost</h1>
    </div>
    <div className={styles.content} ref={contentRef}>
      {/* LEFT PANEL */}
      <div className={`${styles.lp} ${styles.panel}`} style={{ width: cols[0], flexShrink: 0 }}>
        {/* Multi-cluster */}
        <div className={styles.ps}>
          <div className={styles.pst}>
            Clusters
            <span style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
              <span className={styles.chipB}>
                {clusters.length} cluster{clusters.length > 1 ? 's' : ''} · {totalGPUs} GPUs
              </span>
            </span>
          </div>
          {clusters.map(cl => (
            <div key={cl.id} className={styles.clusterAcc}>
              <div
                className={`${styles.clusterHdr} ${cl.open ? styles.open : ''}`}
                onClick={() => upCluster(cl.id, c => ({ ...c, open: !c.open }))}
              >
                <span className={styles.cn}>
                  <input
                    style={{ border: 'none', background: 'none', font: 'inherit', width: 100, cursor: 'text', padding: 0 }}
                    value={cl.name}
                    onChange={e => {
                      e.stopPropagation()
                      upCluster(cl.id, c => ({ ...c, name: e.target.value }))
                    }}
                    onClick={e => e.stopPropagation()}
                  />
                </span>
                <div className={styles.chips}>
                  <span className={styles.chipR} style={{ fontSize: 10 }}>
                    {cl.nodeGroups.reduce((s, ng) => s + ng.count, 0)} GPUs
                  </span>
                  {clusters.length > 1 && (
                    <button
                      style={{ background: 'none', border: 'none', color: '#c0c0c0', cursor: 'pointer', fontSize: 14, padding: '0 2px' }}
                      onClick={e => {
                        e.stopPropagation()
                        removeCluster(cl.id)
                      }}
                    >
                      ×
                    </button>
                  )}
                </div>
                <span className={styles.chev}>⌄</span>
              </div>
              <div className={`${styles.clusterBody} ${cl.open ? styles.open : ''}`}>
                <div className={styles.clusterInner}>
                  <div style={{ marginBottom: 8, fontFamily: 'JetBrains Mono', fontSize: 10, textTransform: 'uppercase', letterSpacing: '.08em', color: '#a0a0a0' }}>
                    GPU node groups
                  </div>
                  {cl.nodeGroups.map((ng, i) => (
                    <div key={i} className={styles.ngRow}>
                      <select className={styles.ngSel} value={ng.gpuType} onChange={e => upNG(cl.id, i, 'gpuType', e.target.value)}>
                        {effectiveCatalog.map(g => (
                          <option key={g.id} value={g.id}>
                            {g.label}
                          </option>
                        ))}
                      </select>
                      <input
                        type="number"
                        className={styles.ngCnt}
                        min={1}
                        max={512}
                        value={ng.count}
                        onChange={e => upNG(cl.id, i, 'count', Math.max(1, +e.target.value || 1))}
                      />
                      {cl.nodeGroups.length > 1 && (
                        <button className={styles.ngRm} onClick={() => removeNG(cl.id, i)}>
                          ×
                        </button>
                      )}
                    </div>
                  ))}
                  <button className={styles.btnAddNg} onClick={() => addNG(cl.id)}>
                    + Add GPU type
                  </button>

                  <div style={{ marginTop: 12, marginBottom: 6, fontFamily: 'JetBrains Mono', fontSize: 10, textTransform: 'uppercase', letterSpacing: '.08em', color: '#a0a0a0' }}>
                    Storage tiers (TB)
                  </div>
                  <div className={styles.storGrid}>
                    {([
                      ['hotTB', 'Hot', '#ee0000'],
                      ['warmTB', 'Warm', '#f59e0b'],
                      ['coldTB', 'Cold', '#0066cc'],
                    ] as const).map(([k, l, c]) => (
                      <div key={k} className={styles.storCell}>
                        <div className={styles.storLbl} style={{ color: c }}>
                          {l}
                        </div>
                        <input
                          type="number"
                          className={styles.storIn}
                          min={0}
                          value={cl[k]}
                          onChange={e => upCluster(cl.id, c2 => ({ ...c2, [k]: parseFloat(e.target.value) || 0 }))}
                        />
                      </div>
                    ))}
                  </div>

                  <div style={{ marginTop: 12, marginBottom: 5, fontFamily: 'JetBrains Mono', fontSize: 10, textTransform: 'uppercase', letterSpacing: '.08em', color: '#a0a0a0' }}>
                    Egress (TB/mo)
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <input
                      type="range"
                      style={{ flex: 1, accentColor: '#ee0000', cursor: 'pointer' }}
                      min={0}
                      max={100}
                      step={1}
                      value={cl.networkEgress}
                      onChange={e => upCluster(cl.id, c2 => ({ ...c2, networkEgress: +e.target.value }))}
                    />
                    <span className={styles.chipR}>{cl.networkEgress} TB</span>
                  </div>
                </div>
              </div>
            </div>
          ))}
          <button className={styles.btnGhost} onClick={addCluster}>
            + Add cluster
          </button>
        </div>

        {/* Cloud provider pricing */}
        <div className={styles.ps}>
          <div
            className={styles.pst}
            style={{ cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}
            onClick={() => setProviderOpen(o => !o)}
          >
            <span>
              {providerOpen ? '☁ Cloud provider pricing' : `☁ ${activeProvider?.label || 'Cloud pricing'}`}
              {!providerOpen && activeGpuRate !== null && (
                <span className={styles.chipG} style={{ marginLeft: 8 }}>
                  {activeGpu} @ ${activeGpuRate.toFixed(2)}/hr
                </span>
              )}
            </span>
            <span style={{ color: '#a0a0a0', fontSize: 11, fontFamily: 'JetBrains Mono' }}>{providerOpen ? '▲' : '▼'}</span>
          </div>
          {providerOpen && (
            <div style={{ marginTop: 12 }}>
              {providerLoading && <div style={{ fontSize: 12, color: '#8a8a8a', padding: '8px 0' }}>Loading providers...</div>}
              {providerError && (
                <div style={{
                  fontSize: 11.5,
                  color: '#92400e',
                  background: '#fffbeb',
                  border: '1px solid #fde68a',
                  borderRadius: 4,
                  padding: '8px 10px',
                  marginBottom: 10,
                  display: 'flex',
                  alignItems: 'center',
                  gap: 8
                }}>
                  <span style={{ flex: 1 }}>{providerError}</span>
                  <button
                    onClick={(e) => {
                      e.stopPropagation()
                      loadProviders()
                    }}
                    style={{
                      fontSize: 11,
                      color: '#0066cc',
                      background: 'none',
                      border: 'none',
                      cursor: 'pointer',
                      padding: '2px 6px',
                      fontFamily: 'JetBrains Mono',
                      textDecoration: 'underline'
                    }}
                  >
                    ↻ Retry
                  </button>
                </div>
              )}
              {providers.length > 0 && (
                <>
                  <div style={{ overflowX: 'auto' }}>
                    <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
                      <thead>
                        <tr style={{ borderBottom: '1px solid #e0e0e0' }}>
                          <th style={{ textAlign: 'left', padding: '6px 8px', fontWeight: 500 }}></th>
                          <th style={{ textAlign: 'left', padding: '6px 8px', fontWeight: 500 }}>Provider</th>
                          <th style={{ textAlign: 'left', padding: '6px 8px', fontWeight: 500 }}>GPU</th>
                          <th style={{ textAlign: 'left', padding: '6px 8px', fontWeight: 500 }}>$/HR</th>
                          <th style={{ textAlign: 'left', padding: '6px 8px', fontWeight: 500 }}></th>
                        </tr>
                      </thead>
                      <tbody>
                        {providers.map(p => {
                          const gpu = selectedGpus[p.id] || p.gpus[0]?.model || ''
                          const rate = getEffectiveRate(p.id, gpu, providers, priceOverrides)
                          const overrideKey = `${p.id}_${gpu}`
                          const hasOverride = priceOverrides[overrideKey] !== undefined
                          return (
                            <tr key={p.id} style={{ borderBottom: '1px solid #f0f0f0' }}>
                              <td style={{ padding: '6px 8px' }}>
                                <input
                                  type="radio"
                                  name="provider"
                                  checked={activeProviderId === p.id}
                                  onChange={() => setActiveProviderId(p.id)}
                                  style={{ cursor: 'pointer' }}
                                />
                              </td>
                              <td style={{ padding: '6px 8px', fontWeight: activeProviderId === p.id ? 600 : 400 }}>
                                {p.label}
                              </td>
                              <td style={{ padding: '6px 8px' }}>
                                <select
                                  value={gpu}
                                  onChange={e => {
                                    const newGpu = e.target.value
                                    setSelectedGpus(prev => ({ ...prev, [p.id]: newGpu }))
                                    saveSelectedGpu(p.id, newGpu)
                                  }}
                                  style={{
                                    fontSize: 11,
                                    padding: '4px 6px',
                                    border: '1px solid #e0e0e0',
                                    borderRadius: 3,
                                    background: '#fafafa',
                                    cursor: 'pointer',
                                  }}
                                >
                                  {p.gpus.map(g => (
                                    <option key={g.model} value={g.model}>
                                      {g.model}
                                    </option>
                                  ))}
                                </select>
                              </td>
                              <td style={{ padding: '6px 8px' }}>
                                <input
                                  type="number"
                                  step="0.01"
                                  min="0"
                                  value={rate ?? ''}
                                  placeholder="n/a"
                                  onChange={e => {
                                    const val = e.target.value === '' ? undefined : parseFloat(e.target.value)
                                    setPriceOverrides(prev => ({ ...prev, [overrideKey]: val }))
                                    setUserOverride(p.id, gpu, val)
                                  }}
                                  style={{
                                    width: 70,
                                    fontSize: 11,
                                    padding: '4px 6px',
                                    border: hasOverride ? '1.5px solid #0066cc' : '1px solid #e0e0e0',
                                    borderRadius: 3,
                                    background: hasOverride ? '#f0f9ff' : '#fff',
                                    fontFamily: 'JetBrains Mono',
                                  }}
                                />
                              </td>
                              <td style={{ padding: '6px 8px' }}>
                                {hasOverride && (
                                  <button
                                    onClick={() => {
                                      setPriceOverrides(prev => {
                                        const next = { ...prev }
                                        delete next[overrideKey]
                                        return next
                                      })
                                      clearUserOverride(p.id, gpu)
                                    }}
                                    style={{
                                      fontSize: 10,
                                      padding: '3px 7px',
                                      border: 'none',
                                      background: '#f0f0f0',
                                      borderRadius: 3,
                                      cursor: 'pointer',
                                      color: '#555',
                                    }}
                                  >
                                    reset
                                  </button>
                                )}
                              </td>
                            </tr>
                          )
                        })}
                      </tbody>
                    </table>
                  </div>
                  <button
                    className={styles.btnGhost}
                    onClick={() => {
                      showToast('Add provider feature coming soon')
                    }}
                    style={{ marginTop: 10 }}
                  >
                    + add provider
                  </button>
                  <div style={{ fontSize: 10, color: '#8a8a8a', marginTop: 8, fontFamily: 'JetBrains Mono' }}>
                    Last updated: {new Date().toLocaleTimeString()}
                  </div>
                </>
              )}
            </div>
          )}
        </div>

        {/* Software & licenses */}
        <div className={styles.ps}>
          <div className={styles.pst} style={{ marginBottom: 8 }}>
            Software &amp; licenses
            {licMonthly > 0 && <span className={styles.chipG}>${Math.round(licMonthly).toLocaleString()}/mo</span>}
          </div>
          <div style={{ fontSize: 11.5, color: '#8a8a8a', marginBottom: 10, lineHeight: 1.5 }}>
            Add any platform, inference server, monitoring or observability subscriptions running on this cluster.
          </div>
          {licenses.length === 0 && (
            <div className={styles.licGhost} onClick={addLicense}>
              e.g. Inference server, monitoring, orchestration…
            </div>
          )}
          {licenses.map(l => {
            const annual = l.cycle === 'yr' ? l.amount * 12 : l.amount
            const monthly = annual / 12
            return (
              <div key={l.id}>
                <div className={styles.licRow}>
                  <input
                    className={styles.licName}
                    placeholder="Software name"
                    value={l.name}
                    onChange={e => upLic(l.id, 'name', e.target.value)}
                  />
                  <input
                    type="number"
                    className={styles.licAmt}
                    min={0}
                    value={l.amount}
                    onChange={e => upLic(l.id, 'amount', parseFloat(e.target.value) || 0)}
                  />
                  <select className={styles.licUnit} value={l.cycle} onChange={e => upLic(l.id, 'cycle', e.target.value)}>
                    <option value="yr">/yr</option>
                    <option value="mo">/mo</option>
                  </select>
                  <select className={styles.licScope} value={l.scope} onChange={e => upLic(l.id, 'scope', e.target.value)}>
                    <option value="both">Cloud + On-prem</option>
                    <option value="cloud">Cloud only</option>
                    <option value="onprem">On-prem only</option>
                  </select>
                  <button className={styles.licRm} onClick={() => rmLic(l.id)}>
                    ×
                  </button>
                </div>
                <div className={styles.licEquiv}>
                  {l.cycle === 'yr' ? `= $${Math.round(monthly).toLocaleString()}/mo` : `= $${Math.round(annual).toLocaleString()}/yr`}
                </div>
              </div>
            )
          })}
          {licenses.length > 0 && (
            <button className={styles.btnGhost} onClick={addLicense}>
              + Add license
            </button>
          )}
          {licMonthly > 0 && (
            <div className={styles.licTotal}>
              <span>Total</span>
              <span className={styles.v}>${Math.round(licMonthly).toLocaleString()}/mo</span>
              <span style={{ color: '#a0a0a0' }}>·</span>
              <span className={styles.v}>${Math.round(licMonthly * 12).toLocaleString()}/yr</span>
              <span style={{ fontSize: 10, color: '#6ee7b7', marginLeft: 'auto' }}>Annual subscriptions ÷ 12</span>
            </div>
          )}
        </div>

        {/* DR */}
        <div className={styles.ps}>
          <div className={styles.pst}>Disaster recovery</div>
          <div className={`${styles.drRow} ${dr ? styles.on : ''}`} onClick={() => setDr(d => !d)}>
            <button
              className={`${styles.drSwitch} ${dr ? styles.on : styles.off}`}
              onClick={e => {
                e.stopPropagation()
                setDr(d => !d)
              }}
            />
            <div className={styles.drLabel}>
              <div className={styles.dt}>Include DR environment</div>
              <div className={styles.ds}>{dr ? 'Cloud +35% · On-prem +80% (second site)' : 'Standby cluster at secondary site'}</div>
            </div>
          </div>
        </div>

        {/* Cluster load */}
        <div className={styles.ps}>
          <div className={styles.pst}>Cluster load (capacity check)</div>
          <div style={{ fontSize: 11.5, color: '#8a8a8a', marginBottom: 8, lineHeight: 1.5 }}>
            Informational only — does not change cost. A fixed cluster costs the same whether idle or busy.
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
            <input
              type="range"
              style={{ flex: 1, accentColor: '#ee0000', cursor: 'pointer' }}
              min={10}
              max={100}
              step={5}
              value={load}
              onChange={e => setLoad(+e.target.value)}
            />
            <span className={styles.chipR}>{load}%</span>
          </div>
          <div className={`${styles.capBar} ${styles[capStatus.cls]}`} style={{ borderRadius: 5, fontSize: 12.5, padding: '7px 10px' }}>
            {capStatus.msg}
          </div>
        </div>

        {/* Rate overrides */}
        <div className={styles.ps}>
          <div className={styles.ratesAcc}>
            <div className={styles.ratesHdr} onClick={() => setRatesOpen(o => !o)}>
              <span>⚙ Override cost rates</span>
              <span style={{ color: '#a0a0a0', fontSize: 11, fontFamily: 'JetBrains Mono' }}>{ratesOpen ? '▲' : '▼'}</span>
            </div>
            <div className={`${styles.ratesBody} ${ratesOpen ? styles.open : ''}`}>
              <div className={styles.rateGroup}>☁ Cloud other rates</div>
              {CLOUD_RATES.map(([k, lbl, unit]) => (
                <div key={k} className={styles.rateRow}>
                  <span className={styles.rateK}>{lbl}</span>
                  <input
                    className={`${styles.rateIn} ${isOv('cloud', k) ? styles.ov : ''}`}
                    type="number"
                    value={(rates as any).cloud[k]}
                    onChange={e => upRate('cloud', k, parseFloat(e.target.value) || 0)}
                  />
                  <span className={styles.rateU}>{unit}</span>
                  {isOv('cloud', k) && (
                    <button className={styles.rateRst} onClick={() => rstRate('cloud', k)}>
                      reset
                    </button>
                  )}
                </div>
              ))}
              <div className={styles.rateGroup}>🖥 On-prem GPU prices (CAPEX $)</div>
              {effectiveCatalog.map(g => (
                <div key={g.id} className={styles.rateRow}>
                  <span className={styles.rateK}>{g.label}</span>
                  <input className={styles.rateIn} type="number" value={Math.round(g.price)} readOnly style={{ width: '100%' }} />
                  <span className={styles.rateU}>$</span>
                  <span />
                </div>
              ))}
              <div className={styles.rateGroup}>🖥 On-prem other costs</div>
              {ONPREM_RATES.map(([k, lbl, unit]) => (
                <div key={k} className={styles.rateRow}>
                  <span className={styles.rateK}>{lbl}</span>
                  <input
                    className={`${styles.rateIn} ${isOv('onprem', k) ? styles.ov : ''}`}
                    type="number"
                    value={(rates as any).onprem[k]}
                    onChange={e => upRate('onprem', k, parseFloat(e.target.value) || 0)}
                  />
                  <span className={styles.rateU}>{unit}</span>
                  {isOv('onprem', k) && (
                    <button className={styles.rateRst} onClick={() => rstRate('onprem', k)}>
                      reset
                    </button>
                  )}
                </div>
              ))}
              <div style={{ padding: '10px 12px' }}>
                <button
                  onClick={() => setRates(JSON.parse(JSON.stringify(BACKEND_DEFAULTS)))}
                  style={{ width: '100%', padding: 7, borderRadius: 4, border: '1.5px solid #e0e0e0', background: '#fafafa', fontSize: 12, cursor: 'pointer', color: '#707070' }}
                >
                  Reset all rates to defaults
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* RESIZE HANDLE 1 */}
      <div className={styles.resizer} onMouseDown={(e) => startDrag(0, e)} title="Drag to resize" />

      {/* CENTER PANEL */}
      <div className={styles.cp} style={{ flex: 1, minWidth: 240, overflow: 'hidden', height: '100%' }}>
        <div className={styles.cpHdr}>
          <span className={styles.cpTitle}>Infrastructure cost layers</span>
          <div className={styles.viewTabs}>
            {([
              ['cloud', '☁ Cloud'],
              ['onprem', '🖥 Self-hosted'],
              ['both', '⊞ Both'],
            ] as const).map(([id, lbl]) => (
              <button key={id} className={`${styles.vt} ${viewMode === id ? styles.on : ''}`} onClick={() => setViewMode(id)}>
                {lbl}
              </button>
            ))}
          </div>
        </div>
        <div className={styles.stacksRow} style={{ gridTemplateColumns: viewMode === 'both' ? '1fr 1fr' : '1fr' }}>
          {(viewMode === 'cloud' || viewMode === 'both') && (
            <div className={styles.stackCol}>
              <div className={`${styles.stackHdr} ${styles.cloud}`}>
                ☁ <span>Cloud ({activeProvider?.label || 'Cloud'})</span>
                <span style={{ marginLeft: 'auto', fontFamily: 'JetBrains Mono', fontSize: 11, opacity: 0.7 }}>
                  ${Math.round(cloud.total / 1000)}K/mo
                </span>
              </div>
              <LayerViz breakdown={cloud.breakdown} stackOrder={CLOUD_ORDER} canvasRef={cloudRef} />
            </div>
          )}
          {(viewMode === 'onprem' || viewMode === 'both') && (
            <div className={styles.stackCol}>
              <div className={`${styles.stackHdr} ${styles.onprem}`}>
                🖥 <span>Self-hosted</span>
                <span style={{ marginLeft: 'auto', fontFamily: 'JetBrains Mono', fontSize: 11, opacity: 0.7 }}>
                  ${Math.round(onprem.total / 1000)}K/mo
                </span>
              </div>
              <LayerViz breakdown={onprem.breakdown} stackOrder={ONPREM_ORDER} canvasRef={onpremRef} />
            </div>
          )}
        </div>
        <div className={styles.flowbar}>
          <div className={styles.fnode}>
            <span style={{ color: '#0066cc' }}>⬇</span>inputs
          </div>
          <span className={styles.farrow}>→</span>
          <div className={styles.fnode}>
            <span style={{ color: '#ee0000' }}>◈</span>cost layers
          </div>
          <span className={styles.farrow}>→</span>
          <div className={styles.fnode}>
            <span style={{ color: '#10b981' }}>$</span>comparison
          </div>
        </div>
      </div>

      {/* RESIZE HANDLE 2 */}
      <div className={styles.resizer} onMouseDown={(e) => startDrag(1, e)} title="Drag to resize" />

      {/* RIGHT PANEL */}
      <div className={`${styles.rp} ${styles.panel}`} style={{ width: cols[1], flexShrink: 0 }}>
        {/* Capacity status */}
        <div className={`${styles.capBar} ${styles[capStatus.cls]}`} style={{ borderBottom: '1px solid #f0f0f0' }}>
          {capStatus.msg}
        </div>

        {/* Hero */}
        <div className={styles.hero}>
          <div style={{ fontFamily: 'JetBrains Mono', fontSize: 10, letterSpacing: '.1em', textTransform: 'uppercase', color: '#555', marginBottom: 10 }}>
            Monthly cost{dr ? ' · DR included' : ''}
          </div>
          <div className={styles.cmpGrid}>
            {([
              { cls: 'cloud', lbl: '☁ Cloud', spotColor: '#0066cc', side: cloud },
              { cls: 'onprem', lbl: '🖥 Self-hosted', spotColor: '#10b981', side: onprem },
            ] as const).map(({ cls, lbl, spotColor, side }) => (
              <div key={cls} className={`${styles.cmpCol} ${styles[cls]}`}>
                <div className={styles.cmpLbl}>
                  <span className={styles.cmpSpot} style={{ background: spotColor }} />
                  {lbl}
                </div>
                <div className={styles.cmpNum}>
                  <span className={styles.cmpPre}>$</span>
                  <CountUp target={Math.round(side.total)} />
                </div>
                <div className={styles.cmpSub}>
                  <span className={styles.sv}>${side.effHr.toFixed(2)}/gpu-hr</span> eff
                  <br />
                  {side.mult.toFixed(1)}× {cls === 'cloud' ? 'headline' : 'hardware'} price
                </div>
                {cls === 'cloud' && licCloud > 0 && (
                  <div className={styles.cmpLic}>
                    ⬡ +${Math.round(licCloud).toLocaleString()}/mo software
                  </div>
                )}
                {cls === 'onprem' && licOnPrem > 0 && (
                  <div className={styles.cmpLic}>
                    ⬡ +${Math.round(licOnPrem).toLocaleString()}/mo software
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>

        {/* 5-Year Cost Difference */}
        <div className={styles.costDiff}>
          <div className={styles.costDiffTitle}>5-Year Cost Difference</div>
          <div className={styles.costDiffRows}>
            <div className={styles.costDiffRow}>
              <span className={styles.costDiffLabel}>☁ Cloud</span>
              <span className={styles.costDiffVal} style={{ color: '#0066cc' }}>
                {cloud.total * 60 >= 1e6
                  ? `$${((cloud.total * 60) / 1e6).toFixed(2)}M`
                  : `$${Math.round((cloud.total * 60) / 1000)}K`}
              </span>
            </div>
            <div className={styles.costDiffRow}>
              <span className={styles.costDiffLabel}>🖥 Self-hosted</span>
              <span className={styles.costDiffVal} style={{ color: '#10b981' }}>
                {onprem.total * 60 >= 1e6
                  ? `$${((onprem.total * 60) / 1e6).toFixed(2)}M`
                  : `$${Math.round((onprem.total * 60) / 1000)}K`}
              </span>
            </div>
            <div className={`${styles.costDiffRow} ${styles.costDiffTotal}`}>
              <span className={styles.costDiffLabel}>Difference</span>
              <span className={styles.costDiffVal} style={{ color: '#151515' }}>
                {Math.abs(saving * 60) >= 1e6
                  ? `$${(Math.abs(saving * 60) / 1e6).toFixed(2)}M`
                  : `$${Math.round(Math.abs(saving * 60) / 1000)}K`}
              </span>
            </div>
          </div>
        </div>

        {/* Multi-cluster table */}
        {clusters.length > 1 && (
          <div className={styles.mct}>
            <div className={styles.sectLbl}>All clusters</div>
            <table>
              <thead>
                <tr>
                  <th>Cluster</th>
                  <th>GPUs</th>
                  <th>Cloud/mo</th>
                  <th>Self-hosted/mo</th>
                </tr>
              </thead>
              <tbody>
                {clusters.map(cl => {
                  const gpus = cl.nodeGroups.reduce((s, ng) => s + ng.count, 0)
                  const cCloud = calcCloud([cl], 0, false, rates, activeGpuRate || 0, effectiveCatalog, costingsEnabled).total
                  const cOp = calcOnPrem([cl], 0, false, rates, effectiveCatalog).total
                  return (
                    <tr key={cl.id}>
                      <td>{cl.name}</td>
                      <td>
                        <span style={{ fontFamily: 'JetBrains Mono' }}>{gpus}</span>
                      </td>
                      <td>${Math.round(cCloud / 1000)}K</td>
                      <td>${Math.round(cOp / 1000)}K</td>
                    </tr>
                  )
                })}
                <tr className={styles.total}>
                  <td>Grand total</td>
                  <td>
                    <span style={{ fontFamily: 'JetBrains Mono' }}>{totalGPUs}</span>
                  </td>
                  <td style={{ color: '#0066cc' }}>${Math.round(cloud.total / 1000)}K</td>
                  <td style={{ color: '#10b981' }}>${Math.round(onprem.total / 1000)}K</td>
                </tr>
              </tbody>
            </table>
          </div>
        )}

        {/* DR note */}
        {dr && (
          <div className={styles.drNote}>
            🔄 <span>DR included — Cloud +35%, Self-hosted +80% for a standby site. Adjust factors in rate overrides.</span>
          </div>
        )}

        {/* 5yr */}
        <div className={styles.fiveyr}>
          <div className={styles.sectLbl}>5-year total cost</div>
          {([
            { lbl: '☁ Cloud', v: cloud.total * 60, c: '#0066cc' },
            { lbl: '🖥 Self-hosted', v: onprem.total * 60, c: '#10b981' },
          ] as const).map(r => (
            <div key={r.lbl} className={styles.fyRow}>
              <div className={styles.fyHd}>
                <span style={{ fontWeight: 500 }}>{r.lbl}</span>
                <span className={styles.fyHdV} style={{ color: r.c }}>
                  {r.v >= 1e6 ? '$' + (r.v / 1e6).toFixed(2) + 'M' : '$' + (r.v / 1e3).toFixed(0) + 'K'}
                </span>
              </div>
              <div className={styles.fyTrack}>
                <div className={styles.fyFill} style={{ width: `${(r.v / maxFive) * 100}%`, background: r.c }} />
              </div>
            </div>
          ))}
        </div>

        {/* Per-layer breakdown */}
        <div className={styles.brkdown}>
          <div className={styles.sectLbl} style={{ marginBottom: 8 }}>
            Per-layer breakdown{' '}
            <span style={{ float: 'right', fontSize: 9, letterSpacing: '.04em', textTransform: 'none', fontFamily: 'normal' }}>
              ☁ cloud &nbsp; 🖥 on-prem
            </span>
          </div>
          {allIds.map(id => {
            const m = LAYER_META[id] || LAYER_META.gpu
            const cS = cloud.breakdown.find(b => b.id === id) || { value: 0, pct: 0 }
            const oS = onprem.breakdown.find(b => b.id === id) || { value: 0, pct: 0 }
            const maxV = Math.max(cS.value, oS.value, 1)
            return (
              <div key={id} className={styles.bkRow}>
                <span className={styles.bkLbl}>{m.label}</span>
                <div className={styles.bkBars}>
                  <div
                    className={styles.bkBar}
                    style={{ height: `${(cS.value / maxV) * 100}%`, width: 12, background: m.c, opacity: 0.9 }}
                    title={`Cloud $${Math.round(cS.value).toLocaleString()}`}
                  />
                  <div
                    className={styles.bkBar}
                    style={{ height: `${(oS.value / maxV) * 100}%`, width: 12, background: m.c, opacity: 0.35 }}
                    title={`On-prem $${Math.round(oS.value).toLocaleString()}`}
                  />
                </div>
                <div className={styles.bkNote}>
                  <span style={{ color: '#0066cc' }}>{cS.pct.toFixed(0)}%</span>
                  <span style={{ color: '#10b981' }}>{oS.pct.toFixed(0)}%</span>
                </div>
              </div>
            )
          })}
        </div>

        {/* Multipliers */}
        <div className={styles.multGrid}>
          {([
            { lbl: '☁ Cloud', sc: '#0066cc', side: cloud, suffix: 'headline' },
            { lbl: '🖥 Self-hosted', sc: '#10b981', side: onprem, suffix: 'amort' },
          ] as const).map(({ lbl, sc, side, suffix }) => (
            <div key={lbl} className={styles.mc}>
              <div className={styles.mcLbl}>
                <span className={styles.mcSpot} style={{ background: sc }} />
                {lbl}
              </div>
              <div className={styles.mcVal} style={{ color: side.mult >= 2.5 ? '#dc2626' : '#d97706' }}>
                {side.mult.toFixed(2)}×
              </div>
              <div className={styles.mcSub}>
                ${side.rawHr.toFixed(2)}/hr {suffix}
                <br />→ ${side.effHr.toFixed(2)} all-in/hr
              </div>
            </div>
          ))}
        </div>

        <div className={styles.warn}>
          ⚠ <span>Cloud uses public list prices. On-prem uses CAPEX ÷ {rates.onprem.deprYrs}yr depreciation + power + colo + staff. Override any rate in the left panel. On-prem excludes facilities buildout and management overhead.</span>
        </div>

        <div className={styles.exp}>
          <div className={styles.sectLbl}>Export</div>
          <button className={`${styles.ebtn} ${styles.p}`} onClick={copySheets}>
            📊 Copy comparison for Google Sheets
          </button>
          <button
            className={`${styles.ebtn} ${styles.s}`}
            onClick={() => {
              navigator.clipboard.writeText(
                JSON.stringify(
                  {
                    clusters: clusters.map(c => ({
                      name: c.name,
                      nodeGroups: c.nodeGroups,
                      storage: { hotTB: c.hotTB, warmTB: c.warmTB, coldTB: c.coldTB },
                      networkEgress: c.networkEgress,
                    })),
                    licenses: licenses.map(l => ({ name: l.name, amount: l.amount, cycle: l.cycle })),
                    dr,
                  },
                  null,
                  2
                )
              )
              showToast('✓ API request body copied')
            }}
          >
            &lt;/&gt; Copy API request
          </button>
        </div>
        {toast && <div className={styles.toast}>✓ {toast}</div>}
      </div>
    </div>
    </ComingSoonRibbon>
  )
}
